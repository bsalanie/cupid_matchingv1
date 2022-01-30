"""Entropies of some useful models. """
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple
import numpy as np

from .utils import _EPS, _TWO_EPS, bs_error_abort
from .matching_utils import Matching


EntropyGradient = Callable[[Matching, Optional[List]], np.ndarray]
"""The type of a function that takes in a Matching
   and possibly a list of other parameters
   and returns an array with the gradient of entropy wrt $\mu$.
"""

EntropyHessianComponentsMuMu = Tuple[np.ndarray, np.ndarray, np.ndarray]
"""The type of a tuple of the three components of the hessian of the entropy
   wrt $(\mu,\mu)$.
"""

EntropyHessianMuMu = Callable[[Matching], EntropyHessianComponentsMuMu]
"""The type of a function that takes in a Matching
   and returns the three components of the hessian of the entropy
   wrt $(\mu,\mu)$.
"""


EntropyHessianComponentsMuR = Tuple[np.ndarray, np.ndarray]
"""The type of a tuple of the two components of the hessian of the entropy
   wrt $(\mu,n)$ and $(\mu, m))$.
"""

EntropyHessianMuR = Callable[[Matching], EntropyHessianComponentsMuR]
"""The type of a function that takes in a Matching
   and returns the two components of the hessian of the entropy
   wrt $(\mu,n)$ and $(\mu, m))$.
"""

EntropyHessianComponents = Tuple[
    EntropyHessianComponentsMuMu, EntropyHessianComponentsMuR
]
""" combines the tuples of components of the hessians"""

EntropyHessians = Tuple[EntropyHessianMuMu, EntropyHessianMuR]
""" combines the hessians """


@dataclass
class EntropyFunctions:
    """Defines the entropy used, via the derivative $e_0 + e \cdot \\alpha$

    Attributes:
        e0_fun: required
        parameter_dependent:  if `True`, the entropy depends on parameters.
            Defaults to `False`
        e_fun: only in entropies that depend on parameters.
            Defaults to `None`
        hessian: defaults to `"numeric"`
            * if `"provide"`, we provide the hessian of the entropy.
            * if `"numerical"`, it is computed by central differences.
        e0_derivative: the derivative of `e0_fun`, if available.
            Defaults to `None`
        e_derivative: the derivative of `e_fun`, if available.
            Defaults to `None`
        more_params: additional parameters
            that define the distribution of errors.
            Defaults to `None`
        description: some text describing the model.
            Defaults to `None`

    Examples:
        See `entropy_choo_siow` in `choo_siow.py`
    """

    e0_fun: EntropyGradient
    parameter_dependent: Optional[bool] = False
    e_fun: Optional[EntropyGradient] = None
    hessian: Optional[str] = "numerical"
    e0_derivative: Optional[EntropyHessians] = None
    e_derivative: Optional[EntropyHessians] = None
    more_params: Optional[List] = None
    description: Optional[str] = None

    def __post_init__(self):
        if not self.parameter_dependent:
            if self.hessian == "provided" and self.e0_derivative is None:
                bs_error_abort(
                    "You claim to provide the hessian "
                    + "but you did not provide the e0_derivative."
                )
        else:
            if self.e_fun is None:
                bs_error_abort(
                    "Your entropy is parameter dependent "
                    + " but you did not provide the e_fun."
                )
            if self.hessian == "provided" and self.e_derivative is None:
                bs_error_abort(
                    "Your entropy is parameter dependent, "
                    + "you claim to provide the hessian,\n"
                    + " but I do not see the e_derivative."
                )


def entropy_gradient(
    entropy: EntropyFunctions,
    muhat: Matching,
    alpha: Optional[np.ndarray] = None,
    more_params: Optional[List] = None,
) -> np.ndarray:
    """Computes the derivative of the entropy wrt $\mu$
     at $(\mu, n, m, \alpha)$

    Args:
        entropy: the `EntropyFunctions` object
        muhat: a Matching
        alpha: a vector of parameters of the derivative of the entropy, if any
        more_params: a list of additional parameters, if any

    Returns:
        the derivative of the entropy wrt $\mu$ at $(\mu, n, m, \alpha)$.
    """
    e0_fun = entropy.e0_fun
    if more_params is None:
        e0_vals = e0_fun(muhat)
    else:
        e0_vals = e0_fun(muhat, more_params=more_params)
    parameter_dependent = entropy.parameter_dependent
    if parameter_dependent:
        if alpha is None:
            bs_error_abort("alpha should be specified for this model")
        e_fun = entropy.e_fun
        if more_params is None:
            e_vals = e_fun(muhat)
        else:
            e_vals = e_fun(muhat, more_params=more_params)
        return e0_vals + e_vals @ alpha
    else:
        return e0_vals


def _numeric_hessian(
    entropy: EntropyFunctions,
    muhat: Matching,
    alpha: Optional[np.ndarray] = None,
    more_params: Optional[List] = None,
) -> EntropyHessianComponents:
    """Evaluates numerically the components of the hessians of the entropy
    wrt $(\mu,\mu)$ and $(\mu,(n,m))$

    Args:
        entropy: the `EntropyFunctions` object
        muhat: a Matching
        alpha: a vector of parameters of the derivative of the entropy, if any
        more_params: a list of additional parameters, if any

    Returns:
        the hessians of the entropy wrt $(\mu,\mu)$ and $(\mu,(n,m))$.
    """
    parameter_dependent = entropy.parameter_dependent
    if not parameter_dependent:
        if more_params is None:
            entropy_deriv = entropy.e0_fun
        else:
            entropy_deriv = partial(entropy_gradient, entropy, more_params=more_params)
    else:
        if more_params is None:
            entropy_deriv = partial(entropy_gradient, entropy, alpha=alpha)
        else:
            entropy_deriv = partial(
                entropy_gradient, entropy, alpha=alpha, more_params=more_params
            )

    muxyhat, _, _, n, m = muhat.unpack()
    X, Y = muxyhat.shape

    # start with the hessian wrt (mu, mu)
    hessian_x = np.zeros((X, Y, Y))
    hessian_y = np.zeros((X, Y, X))
    hessian_xy = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            for t in range(Y):
                muxy = muxyhat.copy().astype(float)
                muxy[x, t] += _EPS
                mus = Matching(muxy, n, m)
                der_entropy_plus = entropy_deriv(mus)
                muxy[x, t] -= _TWO_EPS
                mus = Matching(muxy, n, m)
                der_entropy_minus = entropy_deriv(mus)
                hessian_x[x, y, t] = (
                    der_entropy_plus[x, y] - der_entropy_minus[x, y]
                ) / _TWO_EPS
            for z in range(X):
                muxy = muxyhat.copy().astype(float)
                muxy[z, y] += _EPS
                mus = Matching(muxy, n, m)
                der_entropy_plus = entropy_deriv(mus)
                muxy[z, y] -= _TWO_EPS
                mus = Matching(muxy, n, m)
                der_entropy_minus = entropy_deriv(mus)
                hessian_y[x, y, z] = (
                    der_entropy_plus[x, y] - der_entropy_minus[x, y]
                ) / _TWO_EPS

            muxy = muxyhat.copy().astype(float)
            muxy[x, y] += _EPS
            mus = Matching(muxy, n, m)
            der_entropy_plus = entropy_deriv(mus)
            muxy[x, y] -= _TWO_EPS
            mus = Matching(muxy, n, m)
            der_entropy_minus = entropy_deriv(mus)
            hessian_xy[x, y] = (
                der_entropy_plus[x, y] - der_entropy_minus[x, y]
            ) / _TWO_EPS

    components_mumu = (hessian_x, hessian_y, hessian_xy)

    # now the hessian wrt (mu, r)
    hessian_n = np.zeros((X, Y))
    hessian_m = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            n1 = n.copy().astype(float)
            n1[x] += _EPS
            mus = Matching(muxyhat, n1, m)
            der_entropy_plus = entropy_deriv(mus)
            n1[x] -= _TWO_EPS
            mus = Matching(muxy, n1, m)
            der_entropy_minus = entropy_deriv(mus)
            hessian_n[x, y] = (
                der_entropy_plus[x, y] - der_entropy_minus[x, y]
            ) / _TWO_EPS

            m1 = m.copy().astype(float)
            m1[y] += _EPS
            mus = Matching(muxyhat, n, m1)
            der_entropy_plus = entropy_deriv(mus)
            m1[y] -= _TWO_EPS
            mus = Matching(muxyhat, n, m1)
            der_entropy_minus = entropy_deriv(mus)
            hessian_m[x, y] = (
                der_entropy_plus[x, y] - der_entropy_minus[x, y]
            ) / _TWO_EPS

    components_mur = (hessian_n, hessian_m)

    return components_mumu, components_mur


def _fill_hessianMuMu_from_components(
    hessian_components: EntropyHessianComponentsMuMu,
) -> np.ndarray:
    """Fills the hessian of the entropy wrt $(\mu,\mu)$

    Args:
        hessian_components: the three components of the hessian

    Returns:
        the (XY,XY) matrix of the hessian
    """
    hess_x, hess_y, hess_xy = hessian_components
    X, Y = hess_xy.shape
    XY = X * Y
    hessian = np.zeros((XY, XY))

    i = 0
    ix = 0
    for x in range(X):
        for y in range(Y):
            hessian[i, ix : (ix + Y)] = hess_x[x, y, :]
            slice_y = slice(y, XY, Y)
            hessian[i, slice_y] = hess_y[x, y, :]
            hessian[i, i] = hess_xy[x, y]
            i += 1
        ix += Y

    return hessian


def _fill_hessianMuR_from_components(
    hessian_components: EntropyHessianComponentsMuR,
) -> np.ndarray:
    """Fills the hessian of the entropy wrt $(\mu,(n,m))$

    Args:
        hessian_components: the two components of the hessian

    Returns:
        the (XY,X+Y) matrix of the hessian
    """
    hess_nx, hess_my = hessian_components
    X, Y = hess_nx.shape[:2]
    XY = X * Y
    hessian = np.zeros((XY, X + Y))

    i = 0
    for x in range(X):
        iy = X
        for y in range(Y):
            hessian[i, x] = hess_nx[x, y]
            hessian[i, iy] = hess_my[x, y]
            i += 1
            iy += 1

    return hessian
