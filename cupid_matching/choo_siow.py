"""The components of the derivative of the entropy
for the Choo and Siow homoskedastic model.
"""
from typing import Optional, Tuple, Union
import numpy as np

from .utils import bs_error_abort
from .matching_utils import Matching
from .entropy import EntropyFunctions, EntropyHessianComponents


def _entropy_choo_siow(
    muhat: Matching, deriv: Optional[int] = 0
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Returns the values of $\mathcal{E}$
    and the first (if `deriv` is 1 or 2) and second (if `deriv` is 2) derivatives
    for the Choo and Siow model

    Args:
        muhat: a Matching
        deriv: if equal 1, we compute the first derivatives too;
               if equals 2, also the hessian

    Returns:
        the value of the generalized entropy
        if deriv = 1 or 2, the (X,Y) matrix of the first derivative of the entropy
        if deriv = 2, the (X,Y,X,Y) array of the second derivative
            wrt $(\mu,\mu)$
          and the (X,Y,X+Y) second derivatives
            wrt $(\mu,(n,m))$
    """
    muxy, mux0, mu0y, n, m = muhat.unpack()

    logxy = np.log(muxy)
    logx0 = np.log(mux0)
    log0y = np.log(mu0y)

    val_entropy = (
        -2.0 * np.sum(muxy * logxy)
        - np.sum(mux0 * logx0)
        - np.sum(mu0y * log0y)
        + np.sum(n * np.log(n))
        + np.sum(m * np.log(m))
    )

    if deriv == 0:
        return val_entropy
    if deriv in [1, 2]:
        der_xy = -2.0 * logxy + log0y
        der_xy += logx0.reshape((-1, 1))
        if deriv == 1:
            return val_entropy, der_xy
        else:  # we compute the Hessians
            X, Y = muxy.shape
            derlogxy = 1.0 / muxy
            derlogx0 = 1.0 / mux0
            derlog0y = 1.0 / mu0y
            der2_xyzt = np.zeros((X, Y, X, Y))
            der2_xyr = np.zeros((X, Y, X + Y))
            for x in range(X):
                dlogx0 = derlogx0[x]
                for y in range(Y):
                    d2xy = np.zeros((X, Y))
                    d2xy[x, :] = -dlogx0
                    d2xy[:, y] -= derlog0y[y]
                    d2xy[x, y] -= 2.0 * derlogxy[x, y]
                    der2_xyzt[x, y, :, :] = d2xy
                    der2_xyr[x, y, x] = derlogx0[x]
                    der2_xyr[x, y, X + y] = derlog0y[y]
            return val_entropy, der_xy, der2_xyzt, der2_xyr
    else:
        bs_error_abort("deriv should be 0, 1, or 2")


def e0_fun_choo_siow(muhat: Matching) -> np.ndarray:
    """Returns the values of $e_0$ for the Choo and Siow model.

    Args:
        muhat: a Matching

    Returns:
        the (X,Y) matrix of the first derivative of the entropy
    """
    entropy_res = _entropy_choo_siow(muhat, deriv=1)
    return entropy_res[1]


def e0_derivative_choo_siow(muhat: Matching) -> EntropyHessianComponents:
    """Returns the derivatives of $e_0$ for the Choo and Siow model.

    Args:
        muhat: a Matching

    Returns:
        the three components of the hessian wrt $(\mu,\mu)$ of the entropy
        and the two components of the hessian wrt $(\mu,r)$
    """
    entropy_res = _entropy_choo_siow(muhat, deriv=2)
    hessmumu = entropy_res[2]
    hessmur = entropy_res[3]
    muxy, *_ = muhat.unpack()
    X, Y = muxy.shape
    hess_x = np.zeros((X, Y, Y))
    hess_y = np.zeros((X, Y, X))
    hess_xy = np.zeros((X, Y))
    hess_nx = np.zeros((X, Y))
    hess_my = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            d2xy = hessmumu[x, y, :, :]
            d2r = hessmur[x, y, :]
            hess_x[x, y, :] = d2xy[x, :]
            hess_y[x, y, :] = d2xy[:, y]
            hess_xy[x, y] = d2xy[x, y]
            hess_nx[x, y] = d2r[x]
            hess_my[x, y] = d2r[X + y]
    return (hess_x, hess_y, hess_xy), (hess_nx, hess_my)


entropy_choo_siow = EntropyFunctions(
    e0_fun=e0_fun_choo_siow,
    hessian="provided",
    e0_derivative=e0_derivative_choo_siow,
    description="Choo and Siow homoskedastic with analytic Hessian",
)

entropy_choo_siow_numeric = EntropyFunctions(
    e0_fun=e0_fun_choo_siow,
    description="Choo and Siow homoskedastic with numerical Hessian",
)
