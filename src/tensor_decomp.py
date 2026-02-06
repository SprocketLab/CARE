import numpy as np
import itertools
import os
from itertools import permutations


def mse(a, b):
    """"""
    return ((a - b) ** 2).mean()


def max_ae(a, b):
    """"""
    return np.max(np.abs(a - b))


def mse_perm(a, b, return_perm=False):
    valid_perms = []
    a_shape = tuple(a.shape)
    for perm in itertools.permutations(range(b.ndim)):
        if tuple(b.shape[p] for p in perm) == a_shape:
            valid_perms.append(perm)

    if not valid_perms:
        raise ValueError(
            f"No permutation of b with shape {b.shape} matches a.shape {a_shape}."
        )

    errs = []
    for perm in valid_perms:
        errs.append(mse(a, b.transpose(perm)))

    ix = np.argmin(errs)
    if return_perm:
        return errs[ix], valid_perms[ix]
    return errs[ix]


def max_ae_perm(a, b, return_perm=False):
    errs = [
        max_ae(a, b.transpose(perm)) for perm in itertools.permutations(np.arange(3))
    ]
    ix = np.argmin(errs)
    if return_perm:
        return errs[ix], list(itertools.permutations(np.arange(3)))[ix]
    return errs[ix]


def _ensure_column(a: np.ndarray) -> np.ndarray:
    """Return ``a`` as a 2-D column matrix without modifying the input."""

    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    return a


def multimap(A, V_array):
    """Vectorised multilinear map ``T ×_1 V_1^T ×_2 …`` (pg. 2778, Section 2)."""

    if not V_array:
        return np.asarray(A)

    result = np.asarray(A)
    for V in V_array:
        Vc = _ensure_column(V)
        result = np.tensordot(result, Vc, axes=([0], [0]))
    return result


def two_tensor_prod(w, x, y):
    """
    A type of outer product
    """
    r = x.shape[0]
    M2 = np.zeros([r, r])

    for a in range(w.shape[0]):
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                M2[i, j] += w[a] * x[i, a] * y[j, a]

    return M2


def three_tensor_prod(w, x, y, z):
    """
    Three-way outer product
    """
    r = x.shape[0]
    M3 = np.zeros([r, r, r])

    if len(w.shape) == 0:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                for k in range(z.shape[0]):
                    M3[i, j, k] += w * x[i] * y[j] * z[k]
    else:
        for a in range(w.shape[0]):
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    for k in range(z.shape[0]):
                        M3[i, j, k] += w[a] * x[i, a] * y[j, a] * z[k, a]

    return M3


def T_map(T, u):
    """Vectorised power-method update ``T(u, u)`` (pg. 2790, equation (5))."""

    return np.tensordot(np.tensordot(T, u, axes=(2, 0)), u, axes=([1], [0]))


def tensor_decomp(
    M2,
    M3,
    comps,
    *,
    power_iters: int = 10,
    max_restarts: int = 1000,
    early_stop_patience: int | None = None,
    improvement_tol: float = 0.0,
    random_state: np.random.RandomState | None = None,
):
    """Tensor Decomposition Algorithm (pg. 2795, Algorithm 1)
    This is combined with reduction (4.3.1)

    Parameters
    ----------
    M2
        Symmetric matrix to aid the decomposition
    M3
        Symmetric tensor to be decomposed
    comps
        Number of eigencomponents to return

    Returns
    -------
    mu_rec
        Recovered eigenvectors (a matrix with #comps eigenvectors)
    lam_rec
        Recovered eigenvalues (a vector with #comps eigenvalues)

    """
    lam_rec = np.zeros(comps)
    mu_rec = np.zeros((M2.shape[0], comps))

    for b in range(comps):
        # initial eigendecomposition used in reduction (4.3.1)
        lam, v = np.linalg.eigh(M2)
        idx = lam.argsort()[::-1]
        lam = lam[idx]
        v = v[:, idx]

        # keep only the positive eigenvalues
        n_eigpos = np.sum(lam > 1e-2)
        if n_eigpos > 0:
            lam_pos = lam[:n_eigpos]
            sqrt_lam = np.sqrt(np.abs(lam_pos))
            W = v[:, :n_eigpos] / sqrt_lam

            B = v[:, :n_eigpos] * sqrt_lam
            M3_tilde = multimap(M3, [W, W, W])  # reduction complete

            rng = random_state or np.random
            tau_star = -np.inf
            u_star = np.zeros(n_eigpos)
            no_improve = 0

            for _restart in range(max_restarts):
                u = rng.randn(n_eigpos)
                u_norm = np.linalg.norm(u)
                if u_norm == 0:
                    continue
                u /= u_norm

                for _ in range(power_iters):
                    u = T_map(M3_tilde, u)
                    norm = np.linalg.norm(u)
                    if norm == 0:
                        break
                    u /= norm

                tau_candidate = float(multimap(M3_tilde, [u, u, u]))

                if tau_candidate > tau_star + improvement_tol:
                    tau_star = tau_candidate
                    u_star = u
                    no_improve = 0
                else:
                    no_improve += 1
                    if early_stop_patience is not None and no_improve >= early_stop_patience:
                        break

            u = u_star
            for _ in range(power_iters):
                u = T_map(M3_tilde, u)
                norm = np.linalg.norm(u)
                if norm == 0:
                    break
                u /= norm

            # recovered modified (post-reduction) eigenvalue
            lamb = (T_map(M3_tilde, u) / u)[0]

            # recover original eigenvector and eigenvalue pair
            mu_rec[:, b] = lamb * B @ u
            lam_rec[b] = 1 / lamb**2

            # deflation: remove component, repeat
            M2 -= lam_rec[b] * np.outer(mu_rec[:, b], mu_rec[:, b])
            M3 -= three_tensor_prod(
                np.array(lam_rec[b]), mu_rec[:, b], mu_rec[:, b], mu_rec[:, b]
            )

    return mu_rec, lam_rec


def lowrank(x, k):
    u, s, vh = np.linalg.svd(x)
    s_abs = np.abs(s)
    inds = np.argsort(s_abs)[::-1][:k]
    rec = np.zeros_like(x)
    for i in inds:
        rec += s[i] * np.outer(u.T[i], vh[i])
    return rec


def tensor_decomp_x3(
    w, x1, x2, x3, k=None, debug=False, return_errs=False, savedir=None
):
    if k is None:
        k = w.shape[0]

    ex32 = lowrank(np.einsum("i,ji,ki->jk", w, x3, x2), k=k)
    ex12 = lowrank(np.einsum("i,ji,ki->jk", w, x1, x2), k=k)

    ex12_inv = np.linalg.pinv(ex12)  # TODO what's going on here?

    ex31 = lowrank(np.einsum("i,ji,ki->jk", w, x3, x1), k=k)
    ex21 = lowrank(np.einsum("i,ji,ki->jk", w, x2, x1), k=k)

    ex21_inv = np.linalg.pinv(ex21)  # TODO what's going on here?

    x_tilde_1 = ex32 @ ex12_inv @ x1
    x_tilde_2 = ex31 @ ex21_inv @ x2
    M2 = np.einsum("i,ji,ki->jk", w, x_tilde_1, x_tilde_2)
    M3 = np.einsum("i,ji,ki,li->jkl", w, x_tilde_1, x_tilde_2, x3)

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir + "/" + "ex32", ex32)
        np.save(savedir + "/" + "ex12", ex12)
        np.save(savedir + "/" + "ex12_inv", ex12_inv)
        np.save(savedir + "/" + "ex31", ex31)
        np.save(savedir + "/" + "ex21", ex21)
        np.save(savedir + "/" + "ex21_inv", ex21_inv)
        np.save(savedir + "/" + "x_tilde_1", x_tilde_1)
        np.save(savedir + "/" + "x_tilde_2", x_tilde_2)
        np.save(savedir + "/" + "M2", M2)
        np.save(savedir + "/" + "M3", M3)
        print("saved")

    factors, weights = tensor_decomp(M2, M3, k)

    try:
        err = mse(factors, x3)
    except ValueError:
        # print("[TENSOR_DECOMP] cannot compute error due to rank...")
        err = -1.0
    if debug:
        print(f"[TENSOR_DECOMP] error:", err)
    if return_errs:
        return weights, factors, err
    return weights, factors


def mixture_tensor_decomp_full_inner(
    w, x1, x2, x3, k=None, debug=False, return_errs=False, savedir=None
):
    w_rec_1, x3_rec, err_3_12 = tensor_decomp_x3(
        w, x1, x2, x3, k=k, debug=debug, return_errs=True, savedir=savedir
    )
    #print('1', w_rec_1)
    w_rec_2, x2_rec, err_2_13 = tensor_decomp_x3(
        w, x1, x3, x2, k=k, debug=debug, return_errs=True, savedir=savedir
    )
    #print('2', w_rec_2)
    w_rec_3, x1_rec, err_1_23 = tensor_decomp_x3(
        w, x2, x3, x1, k=k, debug=debug, return_errs=True, savedir=savedir
    )
    #print('3', w_rec_3)
    w_rec = np.mean([w_rec_1,w_rec_2,w_rec_3],axis=0)

    return w_rec, x1_rec, x2_rec, x3_rec


def mixture_tensor_decomp_full(
    w,
    x1,
    x2,
    x3,
    k=None,
    debug=False,
    savedir=None,
    *,
    max_iters: int = 10,
    early_stop_patience: int | None = None,
    improvement_tol: float = 0.0,
):
    T = np.einsum("i,ji,ki,li->jkl", w, x1, x2, x3)
    T_hat = np.zeros_like(T)
    if debug:
        print(x1.shape, x2.shape, x3.shape)
    eps = 1e-2
    err_best = np.inf
    w_rec_best, x1_rec_best, x2_rec_best, x3_rec_best = None, None, None, None
    no_improve = 0
    for i in range(max_iters):
        #print(f"ITERATION {i}")
        w_rec, x1_rec, x2_rec, x3_rec = mixture_tensor_decomp_full_inner(
            w, x1, x2, x3, k=k, debug=False, savedir=savedir
        )
        T_hat = np.einsum("i,ji,ki,li->jkl", w_rec, x1_rec, x2_rec, x3_rec)

        err, perm = mse_perm(T, T_hat, return_perm=True)
        if err + improvement_tol < err_best:
            w_rec_best, x1_rec_best, x2_rec_best, x3_rec_best = (
                w_rec,
                x1_rec,
                x2_rec,
                x3_rec,
            )
            err_best = err
            no_improve = 0
        else:
            no_improve += 1
        #print(err, perm)
        if err <= eps:
            break
        if early_stop_patience is not None and no_improve >= early_stop_patience:
            break

    return w_rec_best, x1_rec_best, x2_rec_best, x3_rec_best


def main():
    # TODO should have a tolerance parameter...

    dim = 10
    eps = 0.2
    for p in np.arange(0, 1.0 + eps, eps):
        w = np.array([p, 1.0 - p])
        k = len(w)
        x1 = np.random.normal(size=(dim, k)) + 1
        x2 = np.random.normal(size=(dim, k)) + 2
        x3 = np.random.normal(size=(dim, k)) + 3

        w_rec, x1_rec, x2_rec, x3_rec = mixture_tensor_decomp_full(
            w, x1, x2, x3, debug=True
        )

        T = np.einsum("i,ji,ki,li->jkl", w, x1, x2, x3)
        T_rec = np.einsum("i,ji,ki,li->jkl", w_rec, x1_rec, x2_rec, x3_rec)

        # print(np.sort(w_rec)[0], np.sort(w)[0])
        print(mse_perm(T, T_rec, return_perm=True))


if __name__ == "__main__":
    main()
