from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
import random
from scipy.special import logsumexp
from scipy.stats import mode, multivariate_normal
from snorkel.labeling.model import LabelModel
import itertools
from numpy.linalg import matrix_rank, svd

from tensor_decomp import mixture_tensor_decomp_full, mse_perm


def learn_structure(sigma_O, gamma=3, solver=cp.SCS, verbose=True, **solver_kwargs):
    """Sparse + low-rank precision decomposition with optional solver diagnostics."""
    A = np.asarray(sigma_O)
    M = A.shape[0]
    lam = 4e-3/np.sqrt(M)

    O = 0.5 * (A + A.T)
    if verbose:
        print("finite:", np.isfinite(O).all(), "nans:", np.isnan(O).sum(), "||O||_F:", np.linalg.norm(O))
    try:
        eigvals = np.linalg.eigvalsh(0.5 * (O + O.T))
        if verbose:
            print("eig(O) min/max:", eigvals.min(), eigvals.max())
    except Exception as exc:
        if verbose:
            print("eig error:", exc)

    O_root = np.real(sp.linalg.sqrtm(O))
    if not np.isfinite(O_root).all() and verbose:
        print("sqrtm produced non-finite entries")

    L_cvx = cp.Variable((M, M), PSD=True)
    S = cp.Variable((M, M), PSD=True)
    R = cp.Variable((M, M), PSD=True)

    objective = cp.Minimize(0.5 * cp.norm(R @ O_root, 'fro') ** 2 - cp.trace(R) + lam * (gamma * cp.pnorm(S, 1) + cp.norm(L_cvx, "nuc")))
    problem = cp.Problem(objective, [R == S - L_cvx, L_cvx >> 0])

    try:
        if solver is None:
            problem.solve(verbose=verbose, **solver_kwargs)
        else:
            problem.solve(verbose=verbose, solver=solver, **solver_kwargs)
    except Exception as exc:
        if verbose:
            print("solve exception:", repr(exc))

    if verbose:
        print("status:", problem.status, "value:", problem.value)
    stats = problem.solver_stats
    if stats is not None:
        if verbose:
            print("solver:", getattr(stats, 'solver_name', None),
                  "time:", getattr(stats, 'solve_time', None),
                  "iters:", getattr(stats, 'num_iters', None))
        extra = getattr(stats, 'extra_stats', None)
        if extra and verbose:
            try:
                print("extra_stats keys:", list(extra.keys()))
            except Exception:
                pass
    if verbose:
        print("S is None?", S.value is None, "L is None?", L_cvx.value is None)
    return S.value, L_cvx.value

def get_weights(L_est, penalize_dependency=True):
    eigenval, eigenvec = np.linalg.eig(L_est)
    idx = np.argsort(eigenval)[::-1]
    eigenval, eigenvec = eigenval[idx], eigenvec[:, idx]
    
     # 2. take the leading factor (assumed = Q) as starting weights
    weights = eigenvec[:, 0] * np.sqrt(eigenval[0])

    # # 3. Gram–Schmidt: subtract the projection on every retained spurious factor
    # for i in range(1, len(eigenval)):
    #     if eigenval[i] < threshold:        # skip small / negative factors
    #         continue
    #     spurious = eigenvec[:, i] * np.sqrt(eigenval[i])
    #     alpha = np.dot(spurious, weights) / np.dot(spurious, spurious)
    #     weights = weights - alpha * spurious

    # weights = weights / np.linalg.norm(weights)
    
    # # Weird rule that works better. Sad
    if penalize_dependency:
        for i in range(1, len(eigenval)):
            if eigenval[i] < 0:        # skip small / negative factors
                continue
            spurious = eigenvec[:, i] * np.sqrt(eigenval[i])
            weights = weights - spurious

    return weights

def majority_vote(df):
    # mode() returns a DataFrame: one column per tied mode,
    # so we just grab the first
    majority_votes = df.mode(axis=1)[0]
    return majority_votes

def ws_aggregate(df, seed=123, n_epochs=1000, decimals=6, class_balance=None):
    """Snorkel label model aggregation with optional class-balance prior."""
    encoded_df, inverse_mapping = encode_for_label_models(df, decimals=decimals)
    ws_indices = run_label_model(
        encoded_df,
        seed=seed,
        n_epochs=n_epochs,
        class_balance=class_balance,
    )
    return np.array([inverse_mapping.get(idx, np.nan) for idx in ws_indices], dtype=float)


def sanitize_correlation(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite entries and enforce unit diagonal in correlation matrices."""
    sanitized = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    np.fill_diagonal(sanitized.values, 1.0)
    return sanitized


def caresl_aggregate(
    df,
    gamma=1,
    solver=cp.SCS,
    verbose=False,
    corr_matrix=None,
    weights=None,
    return_weights=False,
    penalize_dependency=True,
    **solver_kwargs,
):
    """CARESL aggregation with tunable sparsity weight ``gamma``.

    When ``weights`` are supplied the combination is applied directly without
    solving for a new structure. When ``return_weights`` is True the function
    returns a ``(scores, weights)`` tuple so callers can reuse the learned
    combination on held-out data.
    """
    if weights is None:
        if corr_matrix is None:
            corr_matrix = df.corr()
        S_est, L_est = learn_structure(
            corr_matrix,
            gamma=gamma,
            solver=solver,
            verbose=verbose,
            **solver_kwargs,
        )
        weights = get_weights(L_est, penalize_dependency=penalize_dependency)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != df.shape[1]:
        raise ValueError(
            f"CARESL weights length {weights.shape[0]} does not match frame width {df.shape[1]}"
        )
    denominator = np.sum(weights)
    if np.isclose(denominator, 0.0):
        raise ValueError("Computed CARESL weights sum to zero; cannot normalize")
    weighted_avg = sum(weights[i] * df.iloc[:, i].to_numpy() for i in range(len(weights))) / denominator
    if return_weights:
        return weighted_avg, weights
    return weighted_avg

class UWS:
    def __init__(self, n_voters: int, dim=1):
        """
        Initializes the Smoothie class.

        Args:
            n_voters (int): number of generators. This can be the number of models or the number of prompts.
            dim (int): dimension of the embeddings
        """
        self.n_voters = n_voters
        self.dim = dim
        self.theta = np.ones(n_voters)


    def fit(self, lambda_arr: np.ndarray):
        """
        Fits weights using triplet method.

        Args:
            lambda (np.ndarray): embeddings from noisy voters. Has shape (n_samples, n_voters, dim)

        """
        n_samples, n_voters = lambda_arr.shape
        dim = self.dim

        diff = np.zeros(n_voters)  # E[||\lambda_i - y||^2]
        for i in range(n_voters):
            # Consider all other voters and select two at random
            other_idxs = np.delete(np.arange(n_voters), i)
            # Generate all unique pairs of indices
            rows, cols = np.triu_indices(len(other_idxs), k=1)
            pairs = np.vstack((other_idxs[rows], other_idxs[cols])).T

            index_diffs = []
            for j, k in pairs:
                index_diffs.append(
                    triplet(
                        lambda_arr[:, i], lambda_arr[:, j], lambda_arr[:, k]
                    )
                )

            # Set the difference to the average of all the differences
            diff[i] = np.mean(index_diffs)

        # Convert to cannonical parameters
        self.theta = dim / (2 * diff)
        self.theta = self.theta / self.theta.sum()


    def predict(self, lambda_arr: np.ndarray):
        """
        Predicts the true embedding using the weights

        Args:
            lambda_arr (np.ndarray): embeddings from noisy voters. Has shape (n_voters, dim)

        Returns:
            y_pred (np.ndarray): predicted true embedding. Has shape (dim)
        """
        predicted_y = 1 / self.theta.sum() * lambda_arr.dot(self.theta)
        return predicted_y


def triplet(i_arr: np.ndarray, j_arr: np.ndarray, k_arr: np.ndarray):
    """
    Applies triplet method to compute the difference between three voters

    Args:
        i_arr (np.ndarray): embeddings from voter i. Has shape (n_samples, dim)
        j_arr (np.ndarray): embeddings from voter j. Has shape (n_samples, dim)
        k_arr (np.ndarray): embeddings from voter k. Has shape (n_samples, dim)

    Returns:
        diff (float): difference between the three voters
    """
    diff_ij = (np.linalg.norm(i_arr - j_arr, ord=2) ** 2).mean()
    diff_ik = (np.linalg.norm(i_arr - k_arr, ord=2) ** 2).mean()
    diff_jk = (np.linalg.norm(j_arr - k_arr, ord=2) ** 2).mean()
    return 0.5 * (diff_ij + diff_ik - diff_jk)

class ContinuousLabelModel():
    def __init__(self, use_triplets=True):
        self.use_triplets = use_triplets  # only choice right now

    def fit(self, L_train, var_Y, median=True, seed=10):
        self.n, self.m = L_train.shape
        n, m = self.n, self.m
        self.O = np.transpose(L_train) @ L_train / self.n
        self.Sigma_hat = np.zeros([m + 1, m + 1])
        self.Sigma_hat[:m, :m] = self.O

        random.seed(seed)

        if median:
            # Init dict to collect accuracies in triplets
            acc_collection = {}
            for i in range(m):
                acc_collection[i] = []

            # Collect triplet results
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        acc_i = np.sqrt(self.O[i, j] * self.O[i, k] * var_Y / self.O[j, k])
                        acc_j = np.sqrt(self.O[j, i] * self.O[j, k] * var_Y / self.O[i, k])
                        acc_k = np.sqrt(self.O[k, i] * self.O[k, j] * var_Y / self.O[i, j])
                        acc_collection[i].append(acc_i)
                        acc_collection[j].append(acc_j)
                        acc_collection[k].append(acc_k)

            # Take medians
            for i in range(m):
                self.Sigma_hat[i, m] = np.median(acc_collection[i])
                self.Sigma_hat[m, i] = np.median(acc_collection[i])
        else:
            for i in range(m):
                idxes = set(range(m))
                idxes.remove(i)
                # triplet is now i,j,k
                [j, k] = random.sample(idxes, 2)
                # solve from triplet using conditional independence
                acc = np.sqrt(self.O[i, j] * self.O[i, k] * var_Y / self.O[j, k])
                self.Sigma_hat[i, m] = acc
                self.Sigma_hat[m, i] = acc

        # we filled in all but the right-bottom corner, add it in
        self.Sigma_hat[m, m] = var_Y
        return

    def predict(self, L):
        n, m = self.n, self.m
        self.Y_hat = np.zeros(self.n)
        for i in range(self.n):
            self.Y_hat[i] = np.expand_dims(self.Sigma_hat[m, :m], axis=0) \
                            @ np.linalg.inv(self.Sigma_hat[:m, :m]) \
                            @ np.expand_dims(L[i, :self.m], axis=1)
        return self.Y_hat

    def score(self, Y_samples, metric="mse"):
        err = 0
        for i in range(self.n):
            err += (Y_samples[i] - self.Y_hat[i]) ** 2
        return err / self.n

def uws_aggregate(df):
    n_voters = df.shape[1]
    uws = UWS(n_voters)
    uws.fit(df.to_numpy())
    return uws.predict(df.to_numpy())

# def uws_aggregate(df, mean_est, var_est):
#     n_voters = df.shape[1]
#     df_array = df.to_numpy()
#     normalized_df_array = (df_array - mean_est) / np.sqrt(var_est)
#     clm = ContinuousLabelModel()
#     clm.fit(normalized_df_array, var_Y=1)
#     normalized_pred = clm.predict(normalized_df_array)
#     unnormalized_pred = normalized_pred * np.sqrt(var_est) + mean_est
#     return unnormalized_pred

# def uws_aggregate(df, mean_est, var_est):
#     n_voters = df.shape[1]
#     df_array = df.to_numpy()
#     df_array = df_array
#     clm = ContinuousLabelModel()
#     clm.fit(df_array, var_Y=var_est)
#     normalized_pred = clm.predict(df_array)
#     unnormalized_pred = normalized_pred + mean_est
#     return unnormalized_pred

# function for align columns
def find_best_permutation(mu_hat, mu_true):
    k = mu_hat.shape[1]
    best_perm = None
    best_cost = np.inf

    for perm in itertools.permutations(range(k)):
        # total cost for this alignment
        cost = 0.0
        for j in range(k):
            cost += np.linalg.norm(mu_hat[:, perm[j]] - mu_true[:, j])
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return list(best_perm)

def assert_dependency(mu_full, indep):
    for j, latent_list in indep.items():          # loop over all judges we modify
        if 'Q' in latent_list:
            # make the two rows that differ only in Q identical
            mu_full[1, j] = mu_full[0, j]           # (C=0) rows: (0,0) -> (0,1)
            mu_full[3, j] = mu_full[2, j]           # (C=1) rows: (1,0) -> (1,1)
        if 'C' in latent_list:
            # make the two rows that differ only in C identical
            mu_full[2, j] = mu_full[0, j]           # (Q=0) rows: (0,0) -> (1,0)
            mu_full[3, j] = mu_full[1, j]           # (Q=1) rows: (0,1) -> (1,1)

    return mu_full

# def for generating conditional mean given number of judges per group
def generate_mu_views(g, 
                      indep_structure=[
                        {0: ['Q'], 1: ['C']},
                        {0: ['C']},
                        {2: ['C'], 3: ['Q']}
                        ]):
    """
    Return a list [μ¹, μ², μ³] with shape (4, g) each,
    satisfying the three independence patterns you gave.
    Works for any g ≥ 4.
    """
    
    views = []
    thresh = 1.0                       # minimum singular value you want
    for struct in indep_structure:
        while True:
            M = np.random.uniform(1, 4, size=(4, g))   # each rows represent one combination of (C,Q)
            if matrix_rank(M) < 4:
                continue

            M = assert_dependency(M, struct)           # enforce independencies

            # use singular values (rectangular-safe) or eigenvalues of MMᵀ
            s_min = svd(M, compute_uv=False)[-1]       # smallest σ_i ≥ 0
            # alternatively: s_min = np.sqrt(eigvalsh(M @ M.T).min())

            if s_min >= thresh:
                views.append(M)
                break
    return views


def find_three_groups_auto(S: np.ndarray):
    """
    Finds three disjoint groups of size ⌊n/3⌋, ⌊n/3⌋, and n - 2*⌊n/3⌋
    whose maximum cross-block |S[i,j]| is as small as possible.
    
    Uses binary search on threshold with connected components and DP.
    Much faster than exhaustive search while remaining exact.

    Returns
    -------
    (g1, g2, g3, threshold)
      g1, g2, g3 : lists of indices (sizes s1,s2,s3)
      threshold : the minimal possible max|S[i,j]| between any two groups
    """
    n = S.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 judges to form three groups.")
    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be square.")

    # Determine group sizes
    base = n // 3
    rem  = n - 3*base
    s1, s2, s3 = base, base, base + rem

    # Use symmetric absolute similarities, zero diagonal
    absS = np.maximum(np.abs(S), np.abs(S.T)).copy()
    np.fill_diagonal(absS, 0.0)

    # Get unique threshold candidates (upper triangular values)
    triu_indices = np.triu_indices(n, 1)
    candidates = np.unique(absS[triu_indices])
    
    # Binary search for minimal feasible threshold
    left, right = -1, len(candidates) - 1
    
    # Check if largest threshold is feasible (should always be true)
    if not _is_feasible_partition(absS, candidates[right], s1, s2, s3):
        return None
    
    # Binary search for minimal feasible threshold
    while right - left > 1:
        mid = (left + right) // 2
        if _is_feasible_partition(absS, candidates[mid], s1, s2, s3):
            right = mid
        else:
            left = mid
    
    # Reconstruct the actual partition
    groups = _find_partition(absS, candidates[right], s1, s2, s3)
    if groups is None:
        return None
        
    g1, g2, g3 = groups
    return g1, g2, g3, float(candidates[right])

def _is_feasible_partition(absS, threshold, s1, s2, s3):
    """
    Check if a partition with given group sizes is feasible for threshold.
    Returns True if feasible, False otherwise.
    """
    n = absS.shape[0]
    
    # Create adjacency matrix: edges where |S[i,j]| > threshold must stay together
    adj = absS > threshold
    
    # Find connected components
    components = _find_connected_components(adj)
    
    # Check if components can be partitioned into required sizes
    return _can_partition_components(components, s1, s2, s3)


def _find_partition(absS, threshold, s1, s2, s3):
    """
    Find the actual partition for the given threshold.
    Returns (g1, g2, g3) or None if not feasible.
    """
    n = absS.shape[0]
    
    # Create adjacency matrix
    adj = absS > threshold
    
    # Find connected components
    components = _find_connected_components(adj)
    
    # Try to partition components into required sizes
    partition = _partition_components(components, s1, s2, s3)
    if partition is None:
        return None
    
    # Map component assignments back to node indices
    g1, g2, g3 = [], [], []
    for i, comp in enumerate(components):
        if partition[i] == 0:
            g1.extend(comp)
        elif partition[i] == 1:
            g2.extend(comp)
        else:
            g3.extend(comp)
    
    return sorted(g1), sorted(g2), sorted(g3)


def _find_connected_components(adj):
    """Find connected components using BFS."""
    from collections import deque
    
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    
    for i in range(n):
        if not visited[i]:
            # BFS from node i
            queue = deque([i])
            visited[i] = True
            component = [i]
            
            while queue:
                u = queue.popleft()
                # Find neighbors
                neighbors = np.where(adj[u])[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        component.append(v)
            
            components.append(component)
    
    return components


def _can_partition_components(components, s1, s2, s3):
    """Check if components can be partitioned into required sizes using DP."""
    sizes = [len(comp) for comp in components]
    total = sum(sizes)
    
    # Quick check
    if total != s1 + s2 + s3 or any(size > max(s1, s2, s3) for size in sizes):
        return False
    
    # DP: dp[i][j][k] = can we achieve size j in group1 and size k in group2 using first i components?
    m = len(sizes)
    
    dp = np.zeros((m + 1, s1 + 1, s2 + 1), dtype=bool)
    dp[0][0][0] = True
    
    for i in range(1, m + 1):
        size = sizes[i - 1]
        for j in range(s1 + 1):
            for k in range(s2 + 1):
                if dp[i-1][j][k]:
                    # Place in group1
                    if j + size <= s1:
                        dp[i][j + size][k] = True
                    # Place in group2  
                    if k + size <= s2:
                        dp[i][j][k + size] = True
                    # Place in group3
                    dp[i][j][k] = True
    
    return dp[m][s1][s2]


def _partition_components(components, s1, s2, s3):
    """Find actual partition of components into groups using DP backtracking."""
    sizes = [len(comp) for comp in components]
    m = len(sizes)
    
    # DP table
    dp = np.zeros((m + 1, s1 + 1, s2 + 1), dtype=bool)
    dp[0][0][0] = True
    
    for i in range(1, m + 1):
        size = sizes[i - 1]
        for j in range(s1 + 1):
            for k in range(s2 + 1):
                if dp[i-1][j][k]:
                    if j + size <= s1:
                        dp[i][j + size][k] = True
                    if k + size <= s2:
                        dp[i][j][k + size] = True
                    dp[i][j][k] = True
    
    if not dp[m][s1][s2]:
        return None
    
    # Backtrack to find actual assignment
    assignment = [0] * m
    j, k = s1, s2
    
    for i in range(m, 0, -1):
        size = sizes[i - 1]
        
        # Try group1 first
        if j >= size and dp[i-1][j - size][k]:
            assignment[i-1] = 0
            j -= size
        # Then group2
        elif k >= size and dp[i-1][j][k - size]:
            assignment[i-1] = 1
            k -= size
        # Finally group3
        else:
            assignment[i-1] = 2
    
    return assignment


STATE_VERSION = "1"


class PrecisionProblem:
    """Reusable sparse + low-rank precision decomposition."""

    def __init__(
        self,
        sigma_O: np.ndarray,
        *,
        solver: Optional[cp.solvers.solver.Solver] = cp.SCS,
        warm_start: bool = True,
        solver_kwargs: Optional[dict] = None,
    ) -> None:
        sigma = np.asarray(sigma_O, dtype=float)
        if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
            raise ValueError("sigma_O must be a square matrix")

        self._dim = sigma.shape[0]
        symmetric = 0.5 * (sigma + sigma.T)
        sqrtm = np.real(sp.linalg.sqrtm(symmetric))
        if not np.isfinite(sqrtm).all():
            raise ValueError("sqrtm produced non-finite entries")

        self._O_root = cp.Constant(sqrtm)

        self._L = cp.Variable((self._dim, self._dim), PSD=True)
        self._S = cp.Variable((self._dim, self._dim), PSD=True)
        self._R = cp.Variable((self._dim, self._dim), PSD=True)
        self._lam_S = cp.Parameter(nonneg=True)
        self._lam_L = cp.Parameter(nonneg=True)

        objective = cp.Minimize(
            0.5 * cp.norm(self._R @ self._O_root, "fro") ** 2
            - cp.trace(self._R)
            + self._lam_S * cp.pnorm(self._S, 1)
            + self._lam_L * cp.norm(self._L, "nuc")
        )
        constraints = [self._R == self._S - self._L, self._L >> 0]

        self._problem = cp.Problem(objective, constraints)
        self._solver = solver
        base_kwargs = {"warm_start": warm_start}
        if solver_kwargs:
            base_kwargs.update(solver_kwargs)
        self._solver_kwargs = base_kwargs
        self._warm_start = warm_start

    def solve(self, lam_S: float, lam_L: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve for requested lambdas and return ``(S_hat, L_hat, objective)``."""

        self._lam_S.value = float(lam_S)
        self._lam_L.value = float(lam_L)

        solve_kwargs = dict(self._solver_kwargs)
        solve_kwargs.setdefault("warm_start", self._warm_start)
        solve_kwargs.setdefault("verbose", False)

        if self._solver is None:
            result = self._problem.solve(**solve_kwargs)
        else:
            result = self._problem.solve(solver=self._solver, **solve_kwargs)
        status = self._problem.status
        if status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"precision decomposition failed with status {status}")

        S_est = np.asarray(self._S.value, dtype=float)
        L_est = np.asarray(self._L.value, dtype=float)

        S_est = 0.5 * (S_est + S_est.T)
        L_est = 0.5 * (L_est + L_est.T)
        return S_est, L_est, float(result)

    def warm_start_with(self, S_est: np.ndarray, L_est: np.ndarray) -> None:
        """Seed CVXPY variables with externally provided solutions."""

        S_arr = np.asarray(S_est, dtype=float)
        L_arr = np.asarray(L_est, dtype=float)
        if S_arr.shape != self._S.shape:
            raise ValueError("Warm-start S array has incorrect shape")
        if L_arr.shape != self._L.shape:
            raise ValueError("Warm-start L array has incorrect shape")
        self._S.value = S_arr
        self._L.value = L_arr
        self._R.value = S_arr - L_arr


@dataclass
class _TensorCacheEntry:
    ranks: int
    w_rec: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    mu3: np.ndarray
    sigma1: np.ndarray
    sigma2: np.ndarray
    sigma3: np.ndarray
    X1: np.ndarray
    X2: np.ndarray
    X3: np.ndarray
    err: float


class FastCaretAggregator:
    """CARET implementation with warm-started precision and tensor caching."""

    def __init__(
        self,
        judge_scores: np.ndarray,
        *,
        class_balance: float,
        ranks: Sequence[int],
        tensor_opts: Optional[dict] = None,
        precision_problem: Optional[PrecisionProblem] = None,
        solver: Optional[cp.solvers.solver.Solver] = cp.SCS,
        solver_kwargs: Optional[dict] = None,
        ridge: float = 1e-6,
        dataset_name: Optional[str] = None,
        state_cache_dir: Optional[Path] = None,
    ) -> None:
        scores = np.asarray(judge_scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError("judge_scores must be a 2D array")
        self._J = scores
        self._n, self._p = scores.shape
        self._class_balance = float(class_balance)
        self._ranks = tuple(int(r) for r in ranks)
        default_tensor_opts = {
            "max_iters": 6,
            "early_stop_patience": 50,
            "improvement_tol": 1e-3,
        }
        self._tensor_opts = {**default_tensor_opts, **(tensor_opts or {})}
        self._ridge = float(ridge)

        default_solver_kwargs = {"max_iters": 2000, "eps": 1e-4, "verbose": False}
        if solver_kwargs:
            default_solver_kwargs.update(solver_kwargs)
        self._solver_kwargs = default_solver_kwargs

        sigma_hat = np.cov(self._J.T, bias=False)
        self._precision = precision_problem or PrecisionProblem(
            sigma_hat,
            solver=solver,
            solver_kwargs=self._solver_kwargs,
        )

        self._tensor_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], _TensorCacheEntry] = {}

        self._dataset_name = dataset_name
        if dataset_name and state_cache_dir is not None:
            state_dir = Path(state_cache_dir) / dataset_name
            state_dir.mkdir(parents=True, exist_ok=True)
            self._state_dir: Optional[Path] = state_dir
        else:
            self._state_dir = None

    def predict(self, lam_L: float, lam_S: float) -> Tuple[np.ndarray, dict]:
        """Run CARET with requested lambdas and return integer predictions."""

        precision_cache_hit = False
        S_hat: Optional[np.ndarray] = None
        L_hat: Optional[np.ndarray] = None
        objective: Optional[float] = None
        groups: Optional[Tuple[Sequence[int], Sequence[int], Sequence[int], float]] = None

        if self._state_dir is not None:
            cached = self._load_precision_state(lam_L, lam_S)
            if cached is not None:
                S_hat, L_hat, groups, objective, cached_entry = cached
                precision_cache_hit = True
                if groups is not None:
                    G1, G2, G3, _ = groups
                    if cached_entry is not None:
                        key = self._tensor_cache_key(G1, G2, G3)
                        self._tensor_cache[key] = cached_entry
                if S_hat is not None and L_hat is not None:
                    try:
                        self._precision.warm_start_with(S_hat, L_hat)
                    except ValueError:
                        precision_cache_hit = False

        if not precision_cache_hit:
            S_hat, L_hat, objective = self._precision.solve(lam_S=lam_S, lam_L=lam_L)
            groups = find_three_groups_auto(S_hat)
        elif groups is None:
            groups = find_three_groups_auto(S_hat)

        if groups is None:
            raise RuntimeError("failed to recover judge groups from S_hat")

        G1, G2, G3, threshold = groups
        if not (G1 and G2 and G3):
            raise RuntimeError("judge grouping produced an empty slice")

        tensor_entry, tensor_cache_hit = self._get_tensor_entry(G1, G2, G3)

        w_rec = tensor_entry.w_rec
        mu_hat_1 = tensor_entry.mu1
        mu_hat_2 = tensor_entry.mu2
        mu_hat_3 = tensor_entry.mu3
        rank_count = tensor_entry.ranks

        X1 = tensor_entry.X1
        X2 = tensor_entry.X2
        X3 = tensor_entry.X3

        Sigma1_hat = tensor_entry.sigma1
        Sigma2_hat = tensor_entry.sigma2
        Sigma3_hat = tensor_entry.sigma3

        def comp_ll(j: int) -> np.ndarray:
            return (
                np.log(max(w_rec[j], 1e-12))
                + multivariate_normal.logpdf(
                    X1,
                    mean=mu_hat_1[:, j],
                    cov=Sigma1_hat,
                    allow_singular=True,
                )
                + multivariate_normal.logpdf(
                    X2,
                    mean=mu_hat_2[:, j],
                    cov=Sigma2_hat,
                    allow_singular=True,
                )
                + multivariate_normal.logpdf(
                    X3,
                    mean=mu_hat_3[:, j],
                    cov=Sigma3_hat,
                    allow_singular=True,
                )
            )

        log_like = np.vstack([comp_ll(r) for r in range(rank_count)])
        post = np.exp(log_like - logsumexp(log_like, axis=0, keepdims=True))

        evals, evecs = np.linalg.eigh(L_hat)
        idx = int(np.argmax(evals.real))
        v = evecs[:, idx].real
        if v.sum() < 0:
            v = -v

        mu_full = np.zeros((self._p, rank_count))
        mu_full[G1, :] = mu_hat_1
        mu_full[G2, :] = mu_hat_2
        mu_full[G3, :] = mu_hat_3

        scores = v @ mu_full
        median_threshold = float(np.median(scores))
        q1 = np.where(scores >= median_threshold)[0]
        if q1.size == 0:
            q1 = np.array([int(np.argmax(scores))])
        elif q1.size == rank_count:
            q1 = np.array([int(np.argmax(scores))])

        p_hat = post[q1, :].sum(axis=0)
        cutoff = np.percentile(p_hat, self._class_balance)
        y_pred = (p_hat > cutoff).astype(int)

        metadata = {
            "lam_L": float(lam_L),
            "lam_S": float(lam_S),
            "objective": float(objective) if objective is not None else float("nan"),
            "group_threshold": float(threshold),
            "precision_cache_hit": precision_cache_hit,
            "tensor_cache_hit": tensor_cache_hit,
            "q1_size": int(q1.size),
        }

        if self._state_dir is not None and not precision_cache_hit:
            self._persist_precision_state(
                lam_L,
                lam_S,
                S_hat,
                L_hat,
                (G1, G2, G3, threshold),
                objective,
                tensor_entry,
            )

        return y_pred, metadata

    @staticmethod
    def _tensor_cache_key(
        G1: Sequence[int], G2: Sequence[int], G3: Sequence[int]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return tuple(tuple(int(i) for i in group) for group in (G1, G2, G3))

    def _state_file_path(self, lam_L: float, lam_S: float) -> Path:
        if self._state_dir is None:
            raise RuntimeError("State caching directory not configured")
        filename = f"lamL_{self._format_lambda(lam_L)}_lamS_{self._format_lambda(lam_S)}.npz"
        return self._state_dir / filename

    def _persist_precision_state(
        self,
        lam_L: float,
        lam_S: float,
        S_hat: Optional[np.ndarray],
        L_hat: Optional[np.ndarray],
        groups: Tuple[Sequence[int], Sequence[int], Sequence[int], float],
        objective: Optional[float],
        tensor_entry: _TensorCacheEntry,
    ) -> None:
        if self._state_dir is None or S_hat is None or L_hat is None:
            return

        G1, G2, G3, threshold = groups
        path = self._state_file_path(lam_L, lam_S)
        tmp_path = path.with_suffix(".tmp.npz")

        payload = {
            "state_version": STATE_VERSION,
            "lam_L": float(lam_L),
            "lam_S": float(lam_S),
            "objective": float(objective) if objective is not None else float("nan"),
            "threshold": float(threshold),
            "S_hat": np.asarray(S_hat, dtype=float),
            "L_hat": np.asarray(L_hat, dtype=float),
            "G1": np.asarray(G1, dtype=np.int64),
            "G2": np.asarray(G2, dtype=np.int64),
            "G3": np.asarray(G3, dtype=np.int64),
            "rank": np.int64(tensor_entry.ranks),
            "w_rec": np.asarray(tensor_entry.w_rec, dtype=float),
            "mu1": np.asarray(tensor_entry.mu1, dtype=float),
            "mu2": np.asarray(tensor_entry.mu2, dtype=float),
            "mu3": np.asarray(tensor_entry.mu3, dtype=float),
            "sigma1": np.asarray(tensor_entry.sigma1, dtype=float),
            "sigma2": np.asarray(tensor_entry.sigma2, dtype=float),
            "sigma3": np.asarray(tensor_entry.sigma3, dtype=float),
            "tensor_err": float(tensor_entry.err),
        }

        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            tmp_path.replace(path)
        except Exception as exc:  # pragma: no cover
            if tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            print(f"Warning: failed to persist CARET state to {path}: {exc}")

    def _load_precision_state(
        self, lam_L: float, lam_S: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[Sequence[int], Sequence[int], Sequence[int], float], float, Optional[_TensorCacheEntry]]]:
        if self._state_dir is None:
            return None
        path = self._state_file_path(lam_L, lam_S)
        if not path.exists():
            return None

        try:
            with np.load(path) as data:
                version = str(data["state_version"])
                if version != STATE_VERSION:
                    return None

                S_hat = np.asarray(data["S_hat"], dtype=float)
                L_hat = np.asarray(data["L_hat"], dtype=float)
                G1 = data["G1"].astype(int).tolist()
                G2 = data["G2"].astype(int).tolist()
                G3 = data["G3"].astype(int).tolist()
                threshold = float(data["threshold"])
                objective = float(data["objective"])
                groups = (G1, G2, G3, threshold)

                files = set(data.files)
                required = {"rank", "w_rec", "mu1", "mu2", "mu3", "sigma1", "sigma2", "sigma3", "tensor_err"}
                tensor_entry: Optional[_TensorCacheEntry] = None
                if required.issubset(files):
                    rank = int(data["rank"])
                    w_rec = np.asarray(data["w_rec"], dtype=float)
                    mu1 = np.asarray(data["mu1"], dtype=float)
                    mu2 = np.asarray(data["mu2"], dtype=float)
                    mu3 = np.asarray(data["mu3"], dtype=float)
                    sigma1 = np.asarray(data["sigma1"], dtype=float)
                    sigma2 = np.asarray(data["sigma2"], dtype=float)
                    sigma3 = np.asarray(data["sigma3"], dtype=float)
                    err = float(data["tensor_err"])

                    X1 = self._J[:, G1]
                    X2 = self._J[:, G2]
                    X3 = self._J[:, G3]

                    tensor_entry = _TensorCacheEntry(
                        ranks=rank,
                        w_rec=w_rec,
                        mu1=mu1,
                        mu2=mu2,
                        mu3=mu3,
                        sigma1=sigma1,
                        sigma2=sigma2,
                        sigma3=sigma3,
                        X1=X1,
                        X2=X2,
                        X3=X3,
                        err=err,
                    )

                return S_hat, L_hat, groups, objective, tensor_entry
        except Exception:  # pragma: no cover
            return None

        return None

    @staticmethod
    def _format_lambda(value: float) -> str:
        formatted = format(float(value), ".6g")
        formatted = formatted.replace("+", "p").replace("-", "m").replace(".", "d")
        return formatted

    def _get_tensor_entry(
        self,
        G1: Sequence[int],
        G2: Sequence[int],
        G3: Sequence[int],
    ) -> Tuple[_TensorCacheEntry, bool]:
        key = self._tensor_cache_key(G1, G2, G3)
        if key in self._tensor_cache:
            return self._tensor_cache[key], True

        X1 = self._J[:, G1]
        X2 = self._J[:, G2]
        X3 = self._J[:, G3]

        T_emp = np.einsum("ni,nj,nk->ijk", X1, X2, X3) / self._n

        best_entry: Optional[_TensorCacheEntry] = None
        best_err: Optional[float] = None
        for rank in self._ranks:
            try:
                w_rec, mu1, mu2, mu3 = mixture_tensor_decomp_full(
                    w=np.ones(self._n) / self._n,
                    x1=X1.T,
                    x2=X2.T,
                    x3=X3.T,
                    k=rank,
                    debug=False,
                    **self._tensor_opts,
                )
            except Exception:
                continue

            T_hat = np.einsum("i,ji,ki,li->jkl", w_rec, mu1, mu2, mu3)
            err = mse_perm(T_emp, T_hat, return_perm=False)
            if best_err is None or err < best_err:
                Sigma1 = self._regularised_covariance(X1)
                Sigma2 = self._regularised_covariance(X2)
                Sigma3 = self._regularised_covariance(X3)

                best_entry = _TensorCacheEntry(
                    ranks=int(rank),
                    w_rec=np.asarray(w_rec, dtype=float),
                    mu1=np.asarray(mu1, dtype=float),
                    mu2=np.asarray(mu2, dtype=float),
                    mu3=np.asarray(mu3, dtype=float),
                    sigma1=Sigma1,
                    sigma2=Sigma2,
                    sigma3=Sigma3,
                    X1=X1,
                    X2=X2,
                    X3=X3,
                    err=float(err),
                )
                best_err = float(err)

        if best_entry is None:
            raise RuntimeError("tensor decomposition failed for the selected groups")

        self._tensor_cache[key] = best_entry
        return best_entry, False

    def _regularised_covariance(self, X: np.ndarray) -> np.ndarray:
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        cov += self._ridge * np.eye(cov.shape[0])
        return cov

    def cache_info(self) -> dict:
        """Expose lightweight stats about the tensor cache."""
        return {
            "tensor_cache_entries": len(self._tensor_cache),
            "state_cache_enabled": self._state_dir is not None,
        }


def caret_grid_search(
    aggregator: FastCaretAggregator,
    lam_L_grid: Iterable[float],
    lam_S_grid: Iterable[float],
) -> Dict[Tuple[float, float], Tuple[np.ndarray, dict]]:
    """Evaluate ``aggregator`` on the cartesian product of both grids."""

    results: Dict[Tuple[float, float], Tuple[np.ndarray, dict]] = {}
    for lam_L in lam_L_grid:
        for lam_S in lam_S_grid:
            preds, meta = aggregator.predict(lam_L=lam_L, lam_S=lam_S)
            results[(float(lam_L), float(lam_S))] = (preds, meta)
    return results


def caret_aggregate(
    J,
    lam_S=0.1,
    lam_L=0.001,
    class_balance=50,
    ranks=(2, 3, 4),
    tensor_opts: dict | None = None,
    solver_kwargs: dict | None = None,
    dataset_name: str | None = None,
    state_cache_dir=None,
):
    """Run CARET aggregation using the integrated fast implementation."""

    aggregator = FastCaretAggregator(
        np.asarray(J, dtype=float),
        class_balance=class_balance,
        ranks=ranks,
        tensor_opts=tensor_opts,
        solver_kwargs=solver_kwargs,
        dataset_name=dataset_name,
        state_cache_dir=state_cache_dir,
    )
    y_pred, _ = aggregator.predict(lam_L=lam_L, lam_S=lam_S)
    return y_pred


def encode_for_label_models(df, decimals=6):
    """Discretize continuous judge outputs for label-model training."""
    rounded = df.round(decimals)
    unique_values = pd.unique(rounded.values.ravel())
    unique_values = [val for val in unique_values if not pd.isna(val)]
    if not unique_values:
        raise ValueError('No valid judge scores to encode.')
    unique_values = np.array(unique_values, dtype=float)
    unique_values.sort()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    encoded = rounded.replace(mapping).astype(int)
    inverse_mapping = {idx: val for val, idx in mapping.items()}
    return encoded, inverse_mapping


def run_label_model(encoded_df, seed=123, n_epochs=1000, class_balance=None):
    """Train a Snorkel label model and return hard predictions."""
    labels = encoded_df.to_numpy().astype(int)
    cardinality = int(labels.max()) + 1
    model = LabelModel(cardinality=cardinality, verbose=False)
    class_balance_arr = None
    if class_balance is not None:
        class_balance = np.asarray(class_balance, dtype=float)
        if class_balance.ndim == 0:
            class_balance = np.array([1.0 - float(class_balance), float(class_balance)])
        if class_balance.shape[0] != cardinality:
            raise ValueError(
                f"class_balance length {class_balance.shape[0]} does not match label cardinality {cardinality}"
            )
        class_balance_arr = class_balance / class_balance.sum()
    model.fit(
        L_train=labels,
        n_epochs=n_epochs,
        log_freq=100,
        seed=seed,
        class_balance=class_balance_arr,
    )
    return model.predict(labels, tie_break_policy='random').astype(int)
    return y_pred
