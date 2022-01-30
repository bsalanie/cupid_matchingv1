"""
To test the quality of the estimators, we generate data
both from a semilinear Choo and Siow  model
and from a semilinear  nested logit model.
We use both the Poisson estimator and the minimum-distance estimator
on the former model, and only the minimum-distance estimator on the latter.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
import seaborn as sns

from .utils import nprepeat_col, nprepeat_row, print_stars
from .choo_siow import entropy_choo_siow
from .entropy import EntropyFunctions
from .min_distance_utils import MDEResults
from .min_distance import estimate_semilinear_mde
from .model_classes import ChooSiowPrimitives, NestedLogitPrimitives
from .nested_logit import setup_standard_nested_logit
from .poisson_glm import PoissonGLMResults, choo_siow_poisson_glm


def choo_siow_simul(
    phi_bases: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
    true_coeffs: np.ndarray,
    n_households: int,
    n_simuls: int,
    seed: int = None,
) -> Tuple[List[MDEResults], List[PoissonGLMResults]]:
    """
    Monte Carlo simulation of the minimum distance and Poisson estimators
    for the Choo and Siow model

    Args:
        phi_bases: an (X,Y,K) array of bases
        n: an X-vector, margins for men
        m: an Y-vector, margins for women
        true_coeffs: a K-vector, the true values of
            the coefficients of the bases
        n_households: the number of households
        n_simuls: the number of samples for the simulation
        seed: an integer seed for the random number generator

    Returns:
        the lists of results for the min distance estimator
        and the Poisson GLM estimator

    """

    Phi = phi_bases @ true_coeffs
    choo_siow_instance = ChooSiowPrimitives(Phi, n, m)

    ss = SeedSequence(seed)
    child_seeds = ss.spawn(n_simuls)

    min_distance_results = []
    poisson_results = []
    for s in range(n_simuls):
        mus_sim = choo_siow_instance.simulate(n_households, child_seeds[s])
        mde_results = estimate_semilinear_mde(mus_sim, phi_bases, entropy_choo_siow)
        min_distance_results.append(mde_results)
        poisson_glm_results = choo_siow_poisson_glm(mus_sim, phi_bases, verbose=1)
        poisson_results.append(poisson_glm_results)
        print(f"\nChoo-Siow: estimates for sample {s}:")
        print("    MDE:")
        print(mde_results.estimated_coefficients)
        print("    Poisson:")
        print(poisson_glm_results.estimated_beta)
    return min_distance_results, poisson_results


def nested_logit_simul(
    phi_bases: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
    entropy_model: EntropyFunctions,
    true_alphas: np.ndarray,
    true_betas: np.ndarray,
    n_households: int,
    n_simuls: int,
    seed: int = None,
) -> List[MDEResults]:
    """
    Monte Carlo simulation of the minimum distance estimator
    for the nested logit

    Args:
        phi_bases: an (X,Y,K) array of bases
        n: an X-vector, margins for men
        m: an Y-vector, margins for women
        entropy_model: the nested logit specification
        true_alphas: an (n_rhos+n_deltas)-vector,
            the true values of the nests parameters
        true_betas: a K-vector,
            the true values of the coefficients of the bases
        n_households: the number of households
        n_simuls: the number of samples for the simulation
        seed: an integer seed for the random number generator

    Returns:
        the list of results for the min distance estimator
    """

    Phi = phi_bases @ true_betas
    nests_for_each_x, nests_for_each_y = entropy_model.more_params
    nested_logit_instance = NestedLogitPrimitives(
        Phi, n, m, nests_for_each_x, nests_for_each_y, true_alphas
    )

    ss = SeedSequence(seed)
    child_seeds = ss.spawn(n_simuls)

    min_distance_results = []
    for s in range(n_simuls):
        mus_sim = nested_logit_instance.simulate(n_households, child_seeds[s])
        mde_result = estimate_semilinear_mde(
            mus_sim,
            phi_bases,
            entropy_model,
            more_params=entropy_model.more_params,
        )
        min_distance_results.append(mde_result)
        print(f"\nNested logit: MDE estimates for sample {s}:")
        print(mde_result.estimated_coefficients)
    return min_distance_results


if __name__ == "__main__":

    """we draw n_simuls samples of n_households households"""
    n_households = 100_000
    n_simuls = 1000

    run_choo_siow = True
    run_nested_logit = False

    plot_choo_siow = True
    plot_nested_logit = False

    # integer to select a variant; None to do the central scenario
    do_variant = None

    X, Y, K = 20, 20, 8
    # set of 8 basis functions
    phi_bases = np.zeros((X, Y, K))
    phi_bases[:, :, 0] = 1.0
    vec_x = np.arange(X)
    vec_y = np.arange(Y)
    phi_bases[:, :, 1] = nprepeat_col(vec_x, Y)
    phi_bases[:, :, 2] = nprepeat_row(vec_y, X)
    phi_bases[:, :, 3] = phi_bases[:, :, 1] * phi_bases[:, :, 1]
    phi_bases[:, :, 4] = phi_bases[:, :, 1] * phi_bases[:, :, 2]
    phi_bases[:, :, 5] = phi_bases[:, :, 2] * phi_bases[:, :, 2]
    for i in range(X):
        for j in range(i, Y):
            phi_bases[i, j, 6] = 1
            phi_bases[i, j, 7] = i - j
    true_betas = np.array([1.0, 0.0, 0.0, -0.01, 0.02, -0.01, 0.5, 0.0])
    str_variant = ""

    if do_variant is not None:
        if do_variant == 1:
            X, Y, K = 10, 10, 4
            phi_bases = phi_bases[:X, :Y, :K]
            true_betas = true_betas[:K]
        elif do_variant == 2:
            X, Y, K = 4, 5, 6
            phi_bases = phi_bases[:X, :Y, :K]
            true_betas = true_betas[:K]
        elif do_variant == 3:
            X, Y, K = 20, 20, 2
            phi_bases = phi_bases[:X, :Y, :K]
            true_betas = true_betas[:K]
        str_variant = f"_v{do_variant}"

    t = 0.2
    n = np.logspace(start=0, base=1 - t, stop=X - 1, num=X)
    m = np.logspace(start=0, base=1 - t, stop=Y - 1, num=Y)

    beta_names = [f"beta[{i}]" for i in range(1, K + 1)]

    seed = 5456456

    if run_choo_siow:
        choo_siow_results_file = (
            "choo_siow_simul_results"
            + f"{str_variant}_N{n_households}_seed{seed}"
            + ".csv"
        )
        min_distance_results, poisson_results = choo_siow_simul(
            phi_bases, n, m, true_betas, n_households, n_simuls, seed
        )
        mde_estimated_beta = np.zeros((n_simuls, K))
        poisson_estimated_beta = np.zeros((n_simuls, K))
        mde_stderrs_beta = np.zeros((n_simuls, K))
        poisson_stderrs_beta = np.zeros((n_simuls, K))
        K2 = 2 * K
        n_rows = K2 * n_simuls
        simulation = np.zeros(n_rows, dtype=int)
        estimate = np.zeros(n_rows)
        stderrs = np.zeros(n_rows)
        estimator = []
        coefficient = []
        beg_s = 0
        for s in range(n_simuls):
            mde_resus_s = min_distance_results[s]
            poisson_resus_s = poisson_results[s]
            mde_estimated_beta[s, :] = mde_resus_s.estimated_coefficients
            poisson_estimated_beta[s, :] = poisson_resus_s.estimated_beta
            mde_stderrs_beta[s, :] = mde_resus_s.stderrs_coefficients
            poisson_stderrs_beta[s, :] = poisson_resus_s.stderrs_beta
            slice_K2 = slice(beg_s, beg_s + K2)
            simulation[slice_K2] = s
            estimator.extend(["Minimum distance"] * K)
            estimator.extend(["Poisson"] * K)
            coefficient.extend(beta_names)
            coefficient.extend(beta_names)
            slice_K = slice(beg_s, beg_s + K)
            estimate[slice_K] = mde_estimated_beta[s, :]
            stderrs[slice_K] = mde_stderrs_beta[s, :]
            slice_K_K2 = slice(beg_s + K, beg_s + K2)
            estimate[slice_K_K2] = poisson_estimated_beta[s, :]
            stderrs[slice_K_K2] = poisson_stderrs_beta[s, :]
            beg_s += K2
        true_values = np.tile(true_betas, 2 * n_simuls)

        choo_siow_results = pd.DataFrame(
            {
                "Simulation": simulation,
                "Estimator": estimator,
                "Parameter": coefficient,
                "Estimate": estimate,
                "Standard Error": stderrs,
                "True value": true_values,
            }
        )

        choo_siow_results.to_csv(choo_siow_results_file)

    if plot_choo_siow:
        choo_siow_results_file = (
            "choo_siow_simul_results"
            + f"{str_variant}_N{n_households}_seed{seed}"
            + ".csv"
        )
        choo_siow_results = pd.read_csv(choo_siow_results_file)

        # discard outliers
        beta_err = np.array([2.0, 0.5, 0.5, 0.1, 0.1, 0.1, 2.0, 0.5])[:K]
        beta_min = true_betas - beta_err
        beta_max = true_betas + beta_err
        beta_min2 = np.tile(beta_min, 2)
        beta_max2 = np.tile(beta_max, 2)
        beta_min_large = np.tile(beta_min, 2 * n_simuls)
        beta_max_large = np.tile(beta_max, 2 * n_simuls)

        beta_estimates = choo_siow_results.Estimate.values
        outlier_mask = (beta_estimates < beta_min_large) | (
            beta_estimates > beta_max_large
        )
        outlier_simuls = choo_siow_results["Simulation"][outlier_mask].unique()

        print_stars(
            f"We have a total of {len(outlier_simuls)} outliers"
            + f" out of {n_simuls} simulations."
        )
        # for s in outlier_simuls:
        #     print(f" Outlier for simulation {s}:")
        #     mask_s = choo_siow_results["Simulation"] == s
        #     outlier_s = choo_siow_results[mask_s]
        #     estimates_s = outlier_s.Estimate.values
        #     estimators_s = outlier_s.Estimator.values
        #     parameters_s = outlier_s.Parameter.values
        #     wrong_s = (estimates_s > beta_max2) | (estimates_s < beta_min2)
        #     for i, w in enumerate(wrong_s):
        #         if w:
        #             print(
        #                 f"{estimators_s[i]} failed for {parameters_s[i]}"
        #                 + f" with value {estimates_s[i]}"
        #             )

        choo_siow_cleaned_results = choo_siow_results[
            ~choo_siow_results.Simulation.isin(outlier_simuls)
        ]

        g = sns.FacetGrid(
            data=choo_siow_cleaned_results,
            sharex=False,
            sharey=False,
            hue="Estimator",
            col="Parameter",
            col_wrap=2,
        )
        g.map(sns.kdeplot, "Estimate")
        g.set_titles("{col_name}")
        for true_val, ax in zip(true_betas, g.axes.ravel()):
            ax.vlines(true_val, *ax.get_ylim(), color="k", linestyles="dashed")
        g.add_legend()

        plt.savefig(
            "choo_siow_simul_results"
            + f"{str_variant}_N{n_households}_seed{seed}"
            + ".png"
        )

        gs = sns.FacetGrid(
            data=choo_siow_cleaned_results,
            sharex=False,
            sharey=False,
            hue="Estimator",
            col="Parameter",
            col_wrap=2,
        )
        gs.map(sns.kdeplot, "Standard Error")
        gs.set_titles("{col_name}")
        gs.add_legend()

        plt.savefig(
            "choo_siow_simul_results_stderrs"
            + f"{str_variant}_N{n_households}_seed{seed}"
            + ".png"
        )

    if run_nested_logit:
        # Nests and nest parameters for our two-level nested logit
        #  0 is the first nest;
        #   all other nests and nest parameters are type-independent
        # each x has the same nests over 1, ..., Y
        nests_for_each_y = [
            list(range(1, Y // 2 + 1)),
            list(range(Y // 2 + 1, Y + 1)),
        ]
        # each y has the same nests over 1, ..., X
        nests_for_each_x = [
            list(range(1, X // 2 + 1)),
            list(range(X // 2 + 1, X + 1)),
        ]

        entropy_nested_logit, _ = setup_standard_nested_logit(
            nests_for_each_x, nests_for_each_y
        )

        n_rhos, n_deltas = len(nests_for_each_x), len(nests_for_each_y)
        n_alphas = n_rhos + n_deltas

        true_alphas = np.full(n_alphas, 0.5)

        nested_logit_results_file = (
            f"nested_logit_simul_results_N{n_households}_seed{seed}.csv"
        )

        min_distance_results = nested_logit_simul(
            phi_bases,
            n,
            m,
            entropy_nested_logit,
            true_alphas,
            true_betas,
            n_households,
            n_simuls,
        )

        true_coeffs = np.concatenate((true_alphas, true_betas))
        n_pars = n_alphas + K

        alpha_names = [f"rho[{i+1}]" for i in range(n_rhos)] + [
            f"delta[{i+1}]" for i in range(n_deltas)
        ]
        alphabeta_names = alpha_names + beta_names

        mde_estimated_coeffs = np.zeros((n_simuls, n_pars))
        mde_stderrs = np.zeros((n_simuls, n_pars))
        n_rows = n_pars * n_simuls
        simulation = np.zeros(n_rows, dtype=int)
        estimate = np.zeros(n_rows)
        stderrs = np.zeros(n_rows)
        coefficient = []
        beg_s = 0
        for s in range(n_simuls):
            resus_s = min_distance_results[s]
            mde_estimated_coeffs[s, :] = resus_s.estimated_coefficients
            mde_stderrs[s, :] = resus_s.stderrs_coefficients
            slice_s = slice(beg_s, beg_s + n_pars)
            simulation[slice_s] = s
            coefficient.extend(alphabeta_names)
            estimate[slice_s] = mde_estimated_coeffs[s, :]
            stderrs[slice_s] = mde_stderrs[s, :]
            beg_s += n_pars

        true_values = np.tile(true_coeffs, n_simuls)

        nested_logit_results = pd.DataFrame(
            {
                "Simulation": simulation,
                "Parameter": coefficient,
                "Estimate": estimate,
                "Standard Error": stderrs,
                "True value": true_values,
            }
        )

        nested_logit_results.to_csv(nested_logit_results_file)

    if plot_nested_logit:
        nested_logit_results_file = (
            f"nested_logit_simul_results_N{n_households}_seed{seed}.csv"
        )

        nested_logit_results = pd.read_csv(nested_logit_results_file)

        g = sns.FacetGrid(
            data=nested_logit_results,
            sharex=False,
            sharey=False,
            col="Parameter",
            col_wrap=2,
        )
        g.map(sns.kdeplot, "Estimate")
        g.set_titles("{col_name}")
        for true_val, ax in zip(true_values, g.axes.ravel()):
            ax.vlines(true_val, *ax.get_ylim(), color="k", linestyles="dashed")
        g.add_legend()

        plt.savefig(f"nested_logit_results_N{n_households}_seed{seed}.png")

        gs = sns.FacetGrid(
            data=nested_logit_results,
            sharex=False,
            sharey=False,
            col="Parameter",
            col_wrap=2,
        )
        gs.map(sns.kdeplot, "Standard Error")
        gs.set_titles("{col_name}")
        gs.add_legend()

        plt.savefig(f"nested_logit_results_stderrs_N{n_households}_seed{seed}.png")
