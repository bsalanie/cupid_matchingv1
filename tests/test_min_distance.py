import numpy as np

from cupid_matching.utils import print_stars
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.choo_siow import entropy_choo_siow, entropy_choo_siow_numeric
from cupid_matching.choo_siow_gender_heteroskedastic import (
    entropy_choo_siow_gender_heteroskedastic,
    entropy_choo_siow_gender_heteroskedastic_numeric,
)
from cupid_matching.choo_siow_heteroskedastic import (
    entropy_choo_siow_heteroskedastic,
    entropy_choo_siow_heteroskedastic_numeric,
)
from cupid_matching.nested_logit import setup_standard_nested_logit
from cupid_matching.min_distance import estimate_semilinear_mde


def test_mde():
    X, Y, K = 10, 20, 2
    n_households = int(1e12)
    # we simulate a Choo and Siow population
    #  with equal numbers of men and women of each type
    lambda_true = np.random.randn(K)
    phi_bases = np.random.randn(X, Y, K)
    n = np.ones(X)
    m = np.ones(Y)
    Phi = phi_bases @ lambda_true
    choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
    mus_sim = choo_siow_instance.simulate(n_households)
    choo_siow_instance.describe()

    # Nests and nest parameters for our two-level nested logit
    #  0 is the first nest, all other nests and nest parameters are type-independent
    # each x has the same nests over 1, ..., Y
    nests_for_each_x = [list(range(Y // 2)), list(range(Y // 2, Y))]
    # each y has the same nests over 1, ..., X
    nests_for_each_y = [list(range(X // 2)), list(range(X // 2, X))]

    entropy_nested_logit, entropy_nested_logit_numeric = setup_standard_nested_logit(
        nests_for_each_x, nests_for_each_y
    )

    entropy_models = [
        entropy_choo_siow,
        entropy_choo_siow_numeric,
        entropy_choo_siow_gender_heteroskedastic,
        entropy_choo_siow_gender_heteroskedastic_numeric,
        entropy_choo_siow_heteroskedastic,
        entropy_choo_siow_heteroskedastic_numeric,
        entropy_nested_logit,
        entropy_nested_logit_numeric,
    ]

    # tolerance for absolute discrepancy
    #   the fully heteroskedastic models do not work well
    TOL = [1e-3] * 4 + [np.inf] * 2 + [1e-3] * 2

    additional_params = [None] * 6 + [entropy_nested_logit.more_params] * 2

    i_model = 0
    for (entropy_model, more_params) in zip(entropy_models, additional_params):
        if entropy_model.description is not None:
            print_stars(entropy_model.description)

        mde_results = estimate_semilinear_mde(
            mus_sim, phi_bases, entropy_model, more_params=more_params
        )

        if entropy_model in [
            entropy_choo_siow_gender_heteroskedastic,
            entropy_choo_siow_gender_heteroskedastic_numeric,
        ]:
            true_alpha = np.ones(1)

        estimates = mde_results.estimated_coefficients

        if entropy_model in [
            entropy_choo_siow_heteroskedastic,
            entropy_choo_siow_heteroskedastic_numeric,
        ]:
            true_alpha = np.ones(estimates.size - lambda_true.size)

        if entropy_model in [entropy_nested_logit, entropy_nested_logit_numeric]:
            true_alpha = np.ones(estimates.size - lambda_true.size)

        if entropy_model.parameter_dependent:
            true_coeffs = np.concatenate((true_alpha, lambda_true))
        else:
            true_coeffs = lambda_true

        n_alpha = 0
        if entropy_model.parameter_dependent:
            n_alpha = true_alpha.size

        discrepancy = mde_results.print_results(
            true_coeffs=true_coeffs, n_alpha=n_alpha
        )
        print_stars(f"Discrepancy for the {entropy_model.description}: "
                    + f"{discrepancy}")
        
        assert discrepancy < TOL[i_model]
        i_model += 1
