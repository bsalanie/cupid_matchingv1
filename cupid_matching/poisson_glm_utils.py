"""Utilities for Poisson GLM.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .utils import print_stars, npmaxabs
from .matching_utils import Matching


@dataclass
class PoissonGLMResults:
    """Stores and formats the estimation results.

    Args:
        X: int
        Y: int
        K: int
        number_households: int
        number_individuals: int
        estimated_gamma: np.ndarray
        variance_gamma: np.ndarray
        stderrs_gamma: np.ndarray
        estimated_beta: np.ndarray
        estimated_u: np.ndarray
        estimated_v: np.ndarray
        stderrs_beta: np.ndarray
        stderrs_u: np.ndarray
        stderrs_v: np.ndarray
        estimated_Phi: np.ndarray
    """

    X: int
    Y: int
    K: int
    number_households: int
    number_individuals: int
    estimated_gamma: np.ndarray
    variance_gamma: np.ndarray
    stderrs_gamma: np.ndarray
    estimated_beta: np.ndarray
    estimated_u: np.ndarray
    estimated_v: np.ndarray
    stderrs_beta: np.ndarray
    stderrs_u: np.ndarray
    stderrs_v: np.ndarray
    estimated_Phi: np.ndarray

    def __str__(self):
        line_stars = "*" * 80 + "\n"
        print_stars("Estimating a Choo and Siow model by Poisson GLM.")
        model_str = f"The data has {self.number_households} households\n\n"
        model_str += f"We use {self.K} basis functions.\n\n"
        repr_str = line_stars + model_str
        repr_str += (
            "The estimated basis coefficients (and their standard errors) are\n\n"
        )
        for i in range(self.K):
            repr_str += (
                f"   base_{i + 1}: {self.estimated_beta[i]: > 10.3f}  "
                + f"({self.stderrs_beta[i]: .3f})\n"
            )
        repr_str += "The estimated utilities of men (and their standard errors) are\n\n"
        for i in range(self.X):
            repr_str += (
                f"   u_{i + 1}: {self.estimated_u[i]: > 10.3f}  "
                + f"({self.stderrs_u[i]: .3f})\n"
            )
        repr_str += (
            "The estimated utilities of women (and their standard errors) are\n\n"
        )
        for i in range(self.Y):
            repr_str += (
                f"   v {i + 1}: {self.estimated_v[i]: > 10.3f}  "
                + f"({self.stderrs_v[i]: .3f})\n"
            )
        return repr_str + line_stars

    def print_results(
        self,
        lambda_true: Optional[np.ndarray] = None,
        u_true: Optional[np.ndarray] = None,
        v_true: Optional[np.ndarray] = None,
    ) -> float:
        estimates_beta = self.estimated_beta
        stderrs_beta = self.stderrs_beta

        if lambda_true is None:
            repr_str = "The  estimated coefficients "
            repr_str += "(and their standard errors) are\n\n"
            for i, coeff in enumerate(estimates_beta):
                repr_str += f" {coeff: > 10.3f}  ({stderrs_beta[i]: > 10.3f})\n"
            print_stars(repr_str)
        else:
            repr_str = "The  true and estimated coefficients "
            repr_str += "(and their standard errors) are\n\n"
            for i, coeff in enumerate(estimates_beta):
                repr_str += f"   base {i + 1}: {lambda_true[i]: > 10.3f} "
                repr_str += f" {coeff: > 10.3f}  ({stderrs_beta[i]: > 10.3f})\n"
            print_stars(repr_str)

        estimates_u = self.estimated_u
        stderrs_u = self.stderrs_u

        if u_true is None:
            repr_str = "The estimated utilities for men  "
            repr_str += "(and their standard errors) are:\n\n"
            for i, coeff in enumerate(estimates_u):
                repr_str += f" {coeff: > 10.3f}  ({stderrs_u[i]: > 10.3f})\n"
            print_stars(repr_str)
        else:
            repr_str = "The true and estimated utilities for men"
            repr_str += "(and their standard errors) are:\n\n"
            for i, coeff in enumerate(estimates_u):
                repr_str += f"   u_{i + 1}: {u_true[i]: > 10.3f} "
                repr_str += f" {coeff: > 10.3f}  ({stderrs_u[i]: > 10.3f})\n"
            print_stars(repr_str)

        estimates_v = self.estimated_v
        stderrs_v = self.stderrs_v
        if v_true is None:
            repr_str = "The estimated utilities for women  "
            repr_str += "(and their standard errors) are:\n\n"
            for i, coeff in enumerate(estimates_v):
                repr_str += f" {coeff: > 10.3f}  ({stderrs_v[i]: > 10.3f})\n"
            print_stars(repr_str)
        else:
            repr_str = "The true and estimated utilities for women"
            repr_str += "(and their standard errors) are:\n\n"
            for i, coeff in enumerate(estimates_v):
                repr_str += f"   v_{i + 1}: {v_true[i]: > 10.3f} "
                repr_str += f" {coeff: > 10.3f}  ({stderrs_v[i]: > 10.3f})\n"
            print_stars(repr_str)

        if lambda_true is not None:
            discrepancy = npmaxabs(lambda_true - estimates_beta)
            print_stars(f"The true-estimated discrepancy is {discrepancy}")
            return discrepancy

def _prepare_data(
    muhat: Matching,
    var_muhat: np.ndarray,
    var_munm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Normalizes the matching patterns and stacks them.
    We rescale the data so that the total number of individuals is one.

    Args:
        muhat: the observed Matching
        var_muhat: the variance-covariance of $(\mu_{xy}, \mu_{x0},\mu_{0y})$
        var_munm: the variance-covariance of $(\mu_{xy},n_x,m_y)$
        phi_bases: an (X, Y, K) array of bases

    Returns:
        the stacked muxy, mux0, mu0y
        the corresponding variance-covariance matrix
        the number of households
        the number of individuals
    """
    muxy, mux0, mu0y, _, _ = muhat.unpack()
    n_couples = np.sum(muxy)
    n_households = n_couples + np.sum(mux0) + np.sum(mu0y)
    n_individuals = n_households + n_couples
    # rescale the data so that the total number of individuals is one
    muhat_norm = np.concatenate([muxy.flatten(), mux0, mu0y]) / n_individuals
    var_muhat_norm = var_muhat / n_individuals / n_individuals
    var_munm_norm = var_munm / n_individuals / n_individuals
    return muhat_norm, var_muhat_norm, var_munm_norm, n_households, n_individuals
