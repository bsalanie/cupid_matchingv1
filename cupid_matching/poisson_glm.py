"""Estimates the semilinear Choo and Siow homoskedastic (2006) model
using Poisson GLM.
"""

from typing import Optional
from math import sqrt
import numpy as np
import scipy.linalg as spla
import scipy.sparse as spr
from sklearn import linear_model

from .matching_utils import Matching, _make_XY_K_mat, _variance_muhat
from .poisson_glm_utils import PoissonGLMResults, _prepare_data


def choo_siow_poisson_glm(
    muhat: Matching,
    phi_bases: np.ndarray,
    tol: Optional[float] = 1e-12,
    max_iter: Optional[int] = 10000,
    verbose: Optional[int] = 1,
) -> PoissonGLMResults:
    """Estimates the semilinear Choo and Siow homoskedastic (2006) model
        using Poisson GLM.

    Args:
        muhat: the observed Matching
        phi_bases: an (X, Y, K) array of bases
        tol: tolerance level for `linear_model.PoissonRegressor.fit`
        max_iter: maximum number of iterations
            for `linear_model.PoissonRegressor.fit`
        verbose: defines how much output we want (0 = least)

    Returns:
        a `PoissonGLMResults` instance

    Example:
        ```py
        n_households = 1e6
        X, Y, K = 4, 3, 6
        # we setup a quadratic set of basis functions
        phi_bases = np.zeros((X, Y, K))
        phi_bases[:, :, 0] = 1
        for x in range(X):
            phi_bases[x, :, 1] = x
            phi_bases[x, :, 3] = x * x
            for y in range(Y):
                phi_bases[x, y, 4] = x * y
        for y in range(Y):
            phi_bases[:, y, 2] = y
            phi_bases[:, y, 5] = y * y

        lambda_true = np.random.randn(K)
        phi_bases = np.random.randn(X, Y, K)
        Phi = phi_bases @ lambda_true

        # we simulate a Choo and Siow sample from a population
        #  with equal numbers of men and women of each type
        n = np.ones(X)
        m = np.ones(Y)
        choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
        mus_sim = choo_siow_instance.simulate(n_households)
        muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

        results = choo_siow_poisson_glm(mus_sim, phi_bases)

        # compare true and estimated parameters
        results.print_results(
            lambda_true,
            u_true=-np.log(mux0_sim / n_sim),
            v_true=-np.log(mu0y_sim / m_sim)
        )
        ```

    """
    try_sparse = False
    X, Y, K = phi_bases.shape
    XY = X * Y
    n_rows = XY + X + Y
    n_cols = X + Y + K

    # the vector of weights for the Poisson regression
    w = np.concatenate((2 * np.ones(XY), np.ones(X + Y)))
    # reshape the bases
    phi_mat = _make_XY_K_mat(phi_bases)

    if try_sparse:
        w_mat = spr.csr_matrix(
            np.concatenate((2 * np.ones(XY, n_cols), np.ones(X + Y, n_cols)))
        )

        # construct the Z matrix
        ones_X = spr.csr_matrix(np.ones((X, 1)))
        ones_Y = spr.csr_matrix(np.ones((Y, 1)))
        zeros_XK = spr.csr_matrix(np.zeros((X, K)))
        zeros_YK = spr.csr_matrix(np.zeros((Y, K)))
        zeros_XY = spr.csr_matrix(np.zeros((X, Y)))
        zeros_YX = spr.csr_matrix(np.zeros((Y, X)))
        id_X = spr.csr_matrix(np.eye(X))
        id_Y = spr.csr_matrix(np.eye(Y))
        Z_unweighted = spr.vstack(
            [
                spr.hstack([-spr.kron(id_X, ones_Y), -spr.kron(ones_X, id_Y), phi_mat]),
                spr.hstack([-id_X, zeros_XY, zeros_XK]),
                spr.hstack([zeros_YX, -id_Y, zeros_YK]),
            ]
        )
        Z = Z_unweighted / w_mat
    else:
        ones_X = np.ones((X, 1))
        ones_Y = np.ones((Y, 1))
        zeros_XK = np.zeros((X, K))
        zeros_YK = np.zeros((Y, K))
        zeros_XY = np.zeros((X, Y))
        zeros_YX = np.zeros((Y, X))
        id_X = np.eye(X)
        id_Y = np.eye(Y)
        Z_unweighted = np.vstack(
            [
                np.hstack([-np.kron(id_X, ones_Y), -np.kron(ones_X, id_Y), phi_mat]),
                np.hstack([-id_X, zeros_XY, zeros_XK]),
                np.hstack([zeros_YX, -id_Y, zeros_YK]),
            ]
        )
        Z = Z_unweighted / w.reshape((-1, 1))

    _, _, _, n, m = muhat.unpack()
    var_muhat, var_munm = _variance_muhat(muhat)
    (
        muxyhat_norm,
        var_muhat_norm,
        var_munm_norm,
        n_households,
        n_individuals,
    ) = _prepare_data(muhat, var_muhat, var_munm)

    clf = linear_model.PoissonRegressor(
        fit_intercept=False, tol=tol, verbose=verbose, alpha=0, max_iter=max_iter
    )
    clf.fit(Z, muxyhat_norm, sample_weight=w)
    gamma_est = clf.coef_

    # we compute the variance-covariance of the estimator
    nr, nc = Z.shape
    exp_Zg = np.exp(Z @ gamma_est).reshape(n_rows)
    A_hat = np.zeros((nc, nc))
    B_hat = np.zeros((nc, nc))
    for i in range(nr):
        Zi = Z[i, :]
        wi = w[i]
        A_hat += wi * exp_Zg[i] * np.outer(Zi, Zi)
        for j in range(nr):
            Zj = Z[j, :]
            B_hat += wi * w[j] * var_muhat_norm[i, j] * np.outer(Zi, Zj)

    A_inv = spla.inv(A_hat)
    variance_gamma = A_inv @ B_hat @ A_inv
    stderrs_gamma = np.sqrt(np.diag(variance_gamma))

    beta_est = gamma_est[-K:]
    beta_std = stderrs_gamma[-K:]
    Phi_est = phi_bases @ beta_est

    # we correct for the effect of the normalization
    n_norm = n / n_individuals
    m_norm = m / n_individuals
    u_est = gamma_est[:X] + np.log(n_norm)
    v_est = gamma_est[X:-K] + np.log(m_norm)

    # since u = a + log(n_norm) we also need to adjust the estimated variance
    z_unweighted_T = Z_unweighted.T
    u_std = np.zeros(X)
    ix = XY
    for x in range(X):
        n_norm_x = n_norm[x]
        A_inv_x = A_inv[x, :]
        var_log_nx = var_munm_norm[ix, ix] / n_norm_x / n_norm_x
        slice_x = slice(x * Y, (x + 1) * Y)
        covar_term = var_muhat_norm[:, ix] + np.sum(var_muhat_norm[:, slice_x], 1)
        cov_a_lognx = (A_inv_x @ z_unweighted_T @ covar_term) / n_norm_x
        ux_var = variance_gamma[x, x] + var_log_nx + 2.0 * cov_a_lognx
        u_std[x] = sqrt(ux_var)
        ix += 1

    v_std = stderrs_gamma[X:-K]
    iy, jy = X, XY + X
    for y in range(Y):
        m_norm_y = m_norm[y]
        A_inv_y = A_inv[iy, :]
        var_log_my = var_munm_norm[jy, jy] / m_norm_y / m_norm_y
        slice_y = slice(y, XY, Y)
        covar_term = var_muhat_norm[:, jy] + np.sum(var_muhat_norm[:, slice_y], 1)
        cov_b_logmy = (A_inv_y @ z_unweighted_T @ covar_term) / m_norm_y
        vy_var = variance_gamma[iy, iy] + var_log_my + 2.0 * cov_b_logmy
        v_std[y] = sqrt(vy_var)
        iy += 1
        jy += 1

    results = PoissonGLMResults(
        X=X,
        Y=Y,
        K=K,
        number_households=n_households,
        number_individuals=n_individuals,
        estimated_gamma=gamma_est,
        estimated_Phi=Phi_est,
        estimated_beta=beta_est,
        estimated_u=u_est,
        estimated_v=v_est,
        variance_gamma=variance_gamma,
        stderrs_gamma=stderrs_gamma,
        stderrs_beta=beta_std,
        stderrs_u=u_std,
        stderrs_v=v_std,
    )

    return results
