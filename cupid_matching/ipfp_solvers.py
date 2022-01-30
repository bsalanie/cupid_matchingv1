"""Implementations of the IPFP algorithm to solve for equilibrium
and do comparative statics
in several variants of the
`Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model:

 * homoskedastic with singles (as in Choo and Siow 2006)
 * homoskedastic without singles
 * gender-heteroskedastic: with a scale parameter on the error term for women
 * gender- and type-heteroskedastic: with a scale parameter on the error term
   for each gender and type
 * two-level nested logit, with nests and nest parameters that do not depend on the type,
   and {0} as the first nest

Each solver, when fed the joint surplus and margins, returns the equilibrium matching patterns,
the adding-up errors on the margins,
and if requested (using `gr=True`) the derivatives of the matching patterns
in all primitives.
"""
from math import sqrt
from typing import Tuple, Union

import numpy as np
import scipy.linalg as spla

from .utils import (
    bs_error_abort,
    der_nppow,
    npexp,
    npmaxabs,
    nppow,
    nprepeat_col,
    nprepeat_row,
    test_vector,
)

TripleArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]
IPFPnoGradientResults = Tuple[TripleArrays, np.ndarray, np.ndarray, np.ndarray]
IPFPGradientResults = Tuple[
    TripleArrays, np.ndarray, np.ndarray, np.ndarray, TripleArrays
]
IPFPResults = Union[IPFPnoGradientResults, IPFPGradientResults]


def _ipfp_check_sizes(
    men_margins: np.ndarray, women_margins: np.ndarray, Phi: np.ndarray
) -> Tuple[int, int]:
    """checks that the margins and surplus have the correct shapes and sizes"""
    X = test_vector(men_margins)
    Y = test_vector(women_margins)
    if Phi.shape != (X, Y):
        bs_error_abort(f"The shape of Phi should be ({X}, {Y}")
    return X, Y


def ipfp_homoskedastic_nosingles_solver(
    Phi: np.array,
    men_margins: np.array,
    women_margins: np.array,
    tol: float = 1e-9,
    gr: bool = False,
    verbose: bool = False,
    maxiter: int = 1000,
) -> IPFPResults:
    """Solves for equilibrium in a Choo and Siow market without singles,
    given systematic surplus and margins

    Args:
        Phi: matrix of systematic surplus, shape (X, Y)
        men_margins: vector of men margins, shape (X)
        women_margins: vector of women margins, shape (Y)
        tol: tolerance on change in solution
        gr: if `True`, also evaluate derivatives of $(\mu_{xy})$ wrt $\Phi$
        verbose: if `True`, prints information
        maxiter: maximum number of iterations

    Returns:
         muxy: the matching patterns, shape (X, Y)
         marg_err_x, marg_err_y: the errors on the margins
         and the gradients of $(\mu_{xy})$ wrt $\Phi$ if `gr` is `True`
    """
    X, Y = _ipfp_check_sizes(men_margins, women_margins, Phi)
    n_couples = np.sum(men_margins)

    # check that there are as many men as women
    if np.abs(np.sum(women_margins) - n_couples) > n_couples * tol:
        bs_error_abort("There should be as many men as women")

    ephi2, der_ephi2 = npexp(Phi / 2.0, deriv=True)
    ephi2T = ephi2.T

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   starting with a reasonable initial point for tx and ty: : tx = ty = bigc
    #   it is important that it fit the number of individuals
    #############################################################################
    bigc = sqrt(n_couples / np.sum(ephi2))
    txi = np.full(X, bigc)
    tyi = np.full(Y, bigc)

    err_diff = bigc
    tol_diff = tol * err_diff
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        sx = ephi2 @ tyi
        tx = men_margins / sx
        sy = ephi2T @ tx
        ty = women_margins / sy
        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y
        txi, tyi = tx, ty
        niter += 1
    muxy = ephi2 * np.outer(txi, tyi)
    marg_err_x = np.sum(muxy, 1) - men_margins
    marg_err_y = np.sum(muxy, 0) - women_margins
    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return muxy, marg_err_x, marg_err_y
    else:
        sxi = ephi2 @ tyi
        syi = ephi2T @ txi
        n_sum_categories = X + Y
        n_prod_categories = X * Y

        # start with the LHS of the linear system
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:X, :X] = np.diag(sxi)
        lhs[:X, X:] = ephi2 * txi.reshape((-1, 1))
        lhs[X:, X:] = np.diag(syi)
        lhs[X:, :X] = ephi2T * tyi.reshape((-1, 1))

        # now fill the RHS
        n_cols_rhs = n_prod_categories
        rhs = np.zeros((n_sum_categories, n_cols_rhs))

        #  to compute derivatives of (txi, tyi) wrt Phi
        der_ephi2 /= 2.0 * ephi2  # 1/2 with safeguards
        ivar = 0
        for iman in range(X):
            rhs[iman, ivar: (ivar + Y)] = -muxy[iman, :] * der_ephi2[iman, :]
            ivar += Y
        ivar1 = X
        ivar2 = 0
        for iwoman in range(Y):
            rhs[ivar1, ivar2:n_cols_rhs:Y] = -muxy[:, iwoman] * der_ephi2[:, iwoman]
            ivar1 += 1
            ivar2 += 1
        # solve for the derivatives of txi and tyi
        dt_dT = spla.solve(lhs, rhs)
        dt = dt_dT[:X, :]
        dT = dt_dT[X:, :]
        # now construct the derivatives of muxy
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        ivar = 0
        for iman in range(X):
            dt_man = dt[iman, :]
            dmuxy[ivar: (ivar + Y), :] = np.outer((ephi2[iman, :] * tyi), dt_man)
            ivar += Y
        for iwoman in range(Y):
            dT_woman = dT[iwoman, :]
            dmuxy[iwoman:n_prod_categories:Y, :] += np.outer(
                (ephi2[:, iwoman] * txi), dT_woman
            )
        # add the term that comes from differentiating ephi2
        muxy_vec2 = (muxy * der_ephi2).reshape(n_prod_categories)
        dmuxy += np.diag(muxy_vec2)
        return muxy, marg_err_x, marg_err_y, dmuxy


def ipfp_homoskedastic_solver(
    Phi: np.array,
    men_margins: np.array,
    women_margins: np.array,
    tol: float = 1e-9,
    gr: bool = False,
    verbose: bool = False,
    maxiter: int = 1000,
) -> IPFPResults:
    """Solves for equilibrium in a Choo and Siow market with singles,
    given systematic surplus and margins

    Args:
        Phi: matrix of systematic surplus, shape (X, Y)
        men_margins: vector of men margins, shape (X)
        women_margins: vector of women margins, shape (Y)
        tol: tolerance on change in solution
        gr: if `True`, also evaluate derivatives of the matching patterns
        verbose: if `True`, prints information
        maxiter: maximum number of iterations

    Returns:
         (muxy, mux0, mu0y): the matching patterns
         marg_err_x, marg_err_y: the errors on the margins
         and the gradients of the matching patterns wrt (men_margins, women_margins, Phi)
         if `gr` is `True`


    Example:
        ```py
        # we generate a Choo and Siow homoskedastic matching
        X = Y = 20
        n_sum_categories = X + Y
        n_prod_categories = X * Y

        mu, sigma = 0.0, 1.0
        n_bases = 4
        bases_surplus = np.zeros((X, Y, n_bases))
        x_men = (np.arange(X) - X / 2.0) / X
        y_women = (np.arange(Y) - Y / 2.0) / Y

        bases_surplus[:, :, 0] = 1
        for iy in range(Y):
            bases_surplus[:, iy, 1] = x_men
        for ix in range(X):
            bases_surplus[ix, :, 2] = y_women
        for ix in range(X):
            for iy in range(Y):
                bases_surplus[ix, iy, 3] = (x_men[ix] - y_women[iy]) * (
                    x_men[ix] - y_women[iy]
                )

        men_margins = np.random.uniform(1.0, 10.0, size=X)
        women_margins = np.random.uniform(1.0, 10.0, size=Y)

        # np.random.normal(mu, sigma, size=n_bases)
        true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
        true_surplus_matrix = bases_surplus @ true_surplus_params

        mus, marg_err_x, marg_err_y = ipfp_homoskedastic_solver(
            true_surplus_matrix, men_margins, women_margins, tol=1e-12
        )
        ```
    """
    X, Y = _ipfp_check_sizes(men_margins, women_margins, Phi)

    ephi2, der_ephi2 = npexp(Phi / 2.0, deriv=True)

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   where mux0=tx**2  and mu0y=ty**2
    #   starting with a reasonable initial point for tx and ty: tx = ty = bigc
    #   it is important that it fit the number of individuals
    #############################################################################

    ephi2T = ephi2.T
    nindivs = np.sum(men_margins) + np.sum(women_margins)
    bigc = sqrt(nindivs / (X + Y + 2.0 * np.sum(ephi2)))
    txi = np.full(X, bigc)
    tyi = np.full(Y, bigc)

    err_diff = bigc
    tol_diff = tol * bigc
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        sx = ephi2 @ tyi
        tx = (np.sqrt(sx * sx + 4.0 * men_margins) - sx) / 2.0
        sy = ephi2T @ tx
        ty = (np.sqrt(sy * sy + 4.0 * women_margins) - sy) / 2.0
        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y
        txi = tx
        tyi = ty
        niter += 1
    mux0 = txi * txi
    mu0y = tyi * tyi
    muxy = ephi2 * np.outer(txi, tyi)
    marg_err_x = mux0 + np.sum(muxy, 1) - men_margins
    marg_err_y = mu0y + np.sum(muxy, 0) - women_margins
    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y
    else:  # we compute the derivatives
        sxi = ephi2 @ tyi
        syi = ephi2T @ txi
        n_sum_categories = X + Y
        n_prod_categories = X * Y
        # start with the LHS of the linear system
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:X, :X] = np.diag(2.0 * txi + sxi)
        lhs[:X, X:] = ephi2 * txi.reshape((-1, 1))
        lhs[X:, X:] = np.diag(2.0 * tyi + syi)
        lhs[X:, :X] = ephi2T * tyi.reshape((-1, 1))
        # now fill the RHS
        n_cols_rhs = n_sum_categories + n_prod_categories
        rhs = np.zeros((n_sum_categories, n_cols_rhs))
        #  to compute derivatives of (txi, tyi) wrt men_margins
        rhs[:X, :X] = np.eye(X)
        #  to compute derivatives of (txi, tyi) wrt women_margins
        rhs[X:n_sum_categories, X:n_sum_categories] = np.eye(Y)
        #  to compute derivatives of (txi, tyi) wrt Phi
        der_ephi2 /= 2.0 * ephi2  # 1/2 with safeguards
        ivar = n_sum_categories
        for iman in range(X):
            rhs[iman, ivar: (ivar + Y)] = -muxy[iman, :] * der_ephi2[iman, :]
            ivar += Y
        ivar1 = X
        ivar2 = n_sum_categories
        for iwoman in range(Y):
            rhs[ivar1, ivar2:n_cols_rhs:Y] = -muxy[:, iwoman] * der_ephi2[:, iwoman]
            ivar1 += 1
            ivar2 += 1
        # solve for the derivatives of txi and tyi
        dt_dT = spla.solve(lhs, rhs)
        dt = dt_dT[:X, :]
        dT = dt_dT[X:, :]
        # now construct the derivatives of the mus
        dmux0 = 2.0 * (dt * txi.reshape((-1, 1)))
        dmu0y = 2.0 * (dT * tyi.reshape((-1, 1)))
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        ivar = 0
        for iman in range(X):
            dt_man = dt[iman, :]
            dmuxy[ivar: (ivar + Y), :] = np.outer((ephi2[iman, :] * tyi), dt_man)
            ivar += Y
        for iwoman in range(Y):
            dT_woman = dT[iwoman, :]
            dmuxy[iwoman:n_prod_categories:Y, :] += np.outer(
                (ephi2[:, iwoman] * txi), dT_woman
            )
        # add the term that comes from differentiating ephi2
        muxy_vec2 = (muxy * der_ephi2).reshape(n_prod_categories)
        dmuxy[:, n_sum_categories:] += np.diag(muxy_vec2)
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y, (dmuxy, dmux0, dmu0y)


def ipfp_gender_heteroskedastic_solver(
    Phi: np.array,
    men_margins: np.array,
    women_margins: np.array,
    tau: float,
    tol: float = 1e-9,
    gr: bool = False,
    verbose: bool = False,
    maxiter: int = 1000,
) -> IPFPResults:
    """Solves for equilibrium in a in a gender-heteroskedastic Choo and Siow market
    given systematic surplus and margins and a scale parameter `tau`

    Args:
        Phi: matrix of systematic surplus, shape (X, Y)
        men_margins: vector of men margins, shape (X)
        women_margins: vector of women margins, shape (Y)
        tau: the standard error for all women
        tol: tolerance on change in solution
        gr: if `True`, also evaluate derivatives of the matching patterns
        verbose: if `True`, prints information
        maxiter: maximum number of iterations

    Returns:
         (muxy, mux0, mu0y): the matching patterns
         marg_err_x, marg_err_y: the errors on the margins
         and the gradients of the matching patterns
         wrt (men_margins, women_margins, Phi, tau) if `gr` is `True`
    """
    X, Y = _ipfp_check_sizes(men_margins, women_margins, Phi)

    if tau <= 0:
        bs_error_abort(f"We need a positive tau, not {tau}")

    #############################################################################
    # we use ipfp_heteroxy_solver with sigma_x = 1 and tau_y = tau
    #############################################################################

    sigma_x = np.ones(X)
    tau_y = np.full(Y, tau)

    if gr:
        mus_hxy, marg_err_x, marg_err_y, dmus_hxy = ipfp_heteroskedastic_solver(
            Phi,
            men_margins,
            women_margins,
            sigma_x,
            tau_y,
            tol=tol,
            gr=True,
            maxiter=maxiter,
            verbose=verbose,
        )
        muxy, mux0, mu0y = mus_hxy
        dmus_xy, dmus_x0, dmus_0y = dmus_hxy
        n_sum_categories = X + Y
        n_prod_categories = X * Y
        n_cols = n_sum_categories + n_prod_categories
        itau_y = n_cols + X
        dmuxy = np.zeros((n_prod_categories, n_cols + 1))
        dmuxy[:, :n_cols] = dmus_xy[:, :n_cols]
        dmuxy[:, -1] = np.sum(dmus_xy[:, itau_y:], 1)
        dmux0 = np.zeros((X, n_cols + 1))
        dmux0[:, :n_cols] = dmus_x0[:, :n_cols]
        dmux0[:, -1] = np.sum(dmus_x0[:, itau_y:], 1)
        dmu0y = np.zeros((Y, n_cols + 1))
        dmu0y[:, :n_cols] = dmus_0y[:, :n_cols]
        dmu0y[:, -1] = np.sum(dmus_0y[:, itau_y:], 1)
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y, (dmuxy, dmux0, dmu0y)

    else:
        return ipfp_heteroskedastic_solver(
            Phi,
            men_margins,
            women_margins,
            sigma_x,
            tau_y,
            tol=tol,
            gr=False,
            maxiter=maxiter,
            verbose=verbose,
        )


def ipfp_heteroskedastic_solver(
    Phi: np.array,
    men_margins: np.array,
    women_margins: np.array,
    sigma_x: np.array,
    tau_y: np.array,
    tol: float = 1e-9,
    gr: bool = False,
    verbose: bool = False,
    maxiter: int = 1000,
) -> IPFPResults:
    """Solves for equilibrium in a in a fully heteroskedastic Choo and Siow market
    given systematic surplus and margins
    and standard errors `sigma_x` and `tau_y`

    Args:
        Phi: matrix of systematic surplus, shape (X, Y)
        men_margins: vector of men margins, shape (X)
        women_margins: vector of women margins, shape (Y)
        sigma_x: the vector of standard errors for the X types of men
        sigma_x: the vector of standard errors for Y types of women
        tol: tolerance on change in solution
        gr: if `True`, also evaluate derivatives of the matching patterns
        verbose: if `True`, prints information
        maxiter: maximum number of iterations

    Returns:
         (muxy, mux0, mu0y): the matching patterns
         marg_err_x, marg_err_y: the errors on the margins
         and the gradients of the matching patterns
         wrt (men_margins, women_margins, Phi, sigma_x, tau_y)
         if `gr` is `True`
    """

    X, Y = _ipfp_check_sizes(men_margins, women_margins, Phi)

    if np.min(sigma_x) <= 0.0:
        bs_error_abort("All elements of sigma_x must be positive")
    if np.min(tau_y) <= 0.0:
        bs_error_abort("All elements of tau_y must be positive")

    sumxy1 = 1.0 / np.add.outer(sigma_x, tau_y)
    ephi2, der_ephi2 = npexp(Phi * sumxy1, deriv=True)

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   with tx = mux0^(sigma_x/(sigma_x + tau_max))
    #   and ty = mu0y^(tau_y/(sigma_max + tau_y))
    #   starting with a reasonable initial point for tx and ty: tx = ty = bigc
    #   it is important that it fit the number of individuals
    #############################################################################

    nindivs = np.sum(men_margins) + np.sum(women_margins)
    bigc = nindivs / (X + Y + 2.0 * np.sum(ephi2))
    # we find the largest values of sigma_x and tau_y
    xmax = np.argmax(sigma_x)
    sigma_max = sigma_x[xmax]
    ymax = np.argmax(tau_y)
    tau_max = tau_y[ymax]
    # we use tx = mux0^(sigma_x/(sigma_x + tau_max))
    #    and ty = mu0y^(tau_y/(sigma_max + tau_y))
    sig_taumax = sigma_x + tau_max
    txi = np.power(bigc, sigma_x / sig_taumax)
    sigmax_tau = tau_y + sigma_max
    tyi = np.power(bigc, tau_y / sigmax_tau)
    err_diff = bigc
    tol_diff = tol * bigc
    tol_newton = tol
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):  # IPFP main loop
        # Newton iterates for men
        err_newton = bigc
        txin = txi.copy()
        mu0y_in = np.power(np.power(tyi, sigmax_tau), 1.0 / tau_y)
        while err_newton > tol_newton:
            txit = np.power(txin, sig_taumax)
            mux0_in = np.power(txit, 1.0 / sigma_x)
            out_xy = np.outer(np.power(mux0_in, sigma_x), np.power(mu0y_in, tau_y))
            muxy_in = ephi2 * np.power(out_xy, sumxy1)
            errxi = mux0_in + np.sum(muxy_in, 1) - men_margins
            err_newton = npmaxabs(errxi)
            txin -= errxi / (
                sig_taumax * (mux0_in / sigma_x + np.sum(sumxy1 * muxy_in, 1)) / txin
            )
        tx = txin

        # Newton iterates for women
        err_newton = bigc
        tyin = tyi.copy()
        mux0_in = np.power(np.power(tx, sig_taumax), 1.0 / sigma_x)
        while err_newton > tol_newton:
            tyit = np.power(tyin, sigmax_tau)
            mu0y_in = np.power(tyit, 1.0 / tau_y)
            out_xy = np.outer(np.power(mux0_in, sigma_x), np.power(mu0y_in, tau_y))
            muxy_in = ephi2 * np.power(out_xy, sumxy1)
            erryi = mu0y_in + np.sum(muxy_in, 0) - women_margins
            err_newton = npmaxabs(erryi)
            tyin -= erryi / (
                sigmax_tau * (mu0y_in / tau_y + np.sum(sumxy1 * muxy_in, 0)) / tyin
            )

        ty = tyin

        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y

        txi = tx
        tyi = ty

        niter += 1

    mux0 = mux0_in
    mu0y = mu0y_in
    muxy = muxy_in
    marg_err_x = mux0 + np.sum(muxy, 1) - men_margins
    marg_err_y = mu0y + np.sum(muxy, 0) - women_margins

    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y
    else:  # we compute the derivatives
        n_sum_categories = X + Y
        n_prod_categories = X * Y
        # we work directly with (mux0, mu0y)
        sigrat_xy = sumxy1 * sigma_x.reshape((-1, 1))
        taurat_xy = 1.0 - sigrat_xy
        mux0_mat = nprepeat_col(mux0, Y)
        mu0y_mat = nprepeat_row(mu0y, X)
        # muxy = axy * bxy * ephi2
        axy = nppow(mux0_mat, sigrat_xy)
        bxy = nppow(mu0y_mat, taurat_xy)
        der_axy1, der_axy2 = der_nppow(mux0_mat, sigrat_xy)
        der_bxy1, der_bxy2 = der_nppow(mu0y_mat, taurat_xy)
        der_axy1_rat, der_axy2_rat = der_axy1 / axy, der_axy2 / axy
        der_bxy1_rat, der_bxy2_rat = der_bxy1 / bxy, der_bxy2 / bxy

        # start with the LHS of the linear system on (dmux0, dmu0y)
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:X, :X] = np.diag(1.0 + np.sum(muxy * der_axy1_rat, 1))
        lhs[:X, X:] = muxy * der_bxy1_rat
        lhs[X:, X:] = np.diag(1.0 + np.sum(muxy * der_bxy1_rat, 0))
        lhs[X:, :X] = (muxy * der_axy1_rat).T

        # now fill the RHS (derivatives wrt men_margins, then men_margins,
        #    then Phi, then sigma_x and tau_y)
        n_cols_rhs = n_sum_categories + n_prod_categories + X + Y
        rhs = np.zeros((n_sum_categories, n_cols_rhs))

        #  to compute derivatives of (mux0, mu0y) wrt men_margins
        rhs[:X, :X] = np.eye(X)
        #  to compute derivatives of (mux0, mu0y) wrt women_margins
        rhs[X:, X:n_sum_categories] = np.eye(Y)

        #   the next line is sumxy1 with safeguards
        sumxy1_safe = sumxy1 * der_ephi2 / ephi2

        big_a = muxy * sumxy1_safe
        big_b = der_axy2_rat - der_bxy2_rat
        b_mu_s = big_b * muxy * sumxy1
        a_phi = Phi * big_a
        big_c = sumxy1 * (a_phi - b_mu_s * tau_y)
        big_d = sumxy1 * (a_phi + b_mu_s * sigma_x.reshape((-1, 1)))

        #  to compute derivatives of (mux0, mu0y) wrt Phi
        ivar = n_sum_categories
        for iman in range(X):
            rhs[iman, ivar: (ivar + Y)] = -big_a[iman, :]
            ivar += Y
        ivar1 = X
        ivar2 = n_sum_categories
        iend_phi = n_sum_categories + n_prod_categories
        for iwoman in range(Y):
            rhs[ivar1, ivar2:iend_phi:Y] = -big_a[:, iwoman]
            ivar1 += 1
            ivar2 += 1

        #  to compute derivatives of (mux0, mu0y) wrt sigma_x
        iend_sig = iend_phi + X
        der_sigx = np.sum(big_c, 1)
        rhs[:X, iend_phi:iend_sig] = np.diag(der_sigx)
        rhs[X:, iend_phi:iend_sig] = big_c.T
        #  to compute derivatives of (mux0, mu0y) wrt tau_y
        der_tauy = np.sum(big_d, 0)
        rhs[X:, iend_sig:] = np.diag(der_tauy)
        rhs[:X, iend_sig:] = big_d

        # solve for the derivatives of mux0 and mu0y
        dmu0 = spla.solve(lhs, rhs)
        dmux0 = dmu0[:X, :]
        dmu0y = dmu0[X:, :]

        # now construct the derivatives of muxy
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        der1 = ephi2 * der_axy1 * bxy
        ivar = 0
        for iman in range(X):
            dmuxy[ivar: (ivar + Y), :] = np.outer(der1[iman, :], dmux0[iman, :])
            ivar += Y
        der2 = ephi2 * der_bxy1 * axy
        for iwoman in range(Y):
            dmuxy[iwoman:n_prod_categories:Y, :] += np.outer(
                der2[:, iwoman], dmu0y[iwoman, :]
            )

        # add the terms that comes from differentiating ephi2
        #  on the derivative wrt Phi
        i = 0
        j = n_sum_categories
        for iman in range(X):
            for iwoman in range(Y):
                dmuxy[i, j] += big_a[iman, iwoman]
                i += 1
                j += 1
        #  on the derivative wrt sigma_x
        ivar = 0
        ix = iend_phi
        for iman in range(X):
            dmuxy[ivar: (ivar + Y), ix] -= big_c[iman, :]
            ivar += Y
            ix += 1
        # on the derivative wrt tau_y
        iy = iend_sig
        for iwoman in range(Y):
            dmuxy[iwoman:n_prod_categories:Y, iy] -= big_d[:, iwoman]
            iy += 1

        return (muxy, mux0, mu0y), marg_err_x, marg_err_y, (dmuxy, dmux0, dmu0y)
