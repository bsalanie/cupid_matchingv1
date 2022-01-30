import numpy as np

from cupid_matching.utils import npmaxabs, print_stars, describe_array
from cupid_matching.ipfp_solvers import ipfp_homoskedastic_solver, ipfp_homoskedastic_nosingles_solver, \
    ipfp_heteroskedastic_solver, ipfp_gender_heteroskedastic_solver

def test_ipfp():
    do_test_gradient_gender_heteroskedastic = True
    do_test_gradient_heteroskedastic = True

    TOL = 1e-9
    TOL_GRADIENT = 1e-6

    # we generate a Choo and Siow homoskedastic matching
    X = Y = 20
    n_sum_categories = X + Y
    n_prod_categories = X * Y

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

    print_stars("Testing ipfp homoskedastic:")
    mus, marg_err_x, marg_err_y = ipfp_homoskedastic_solver(
        true_surplus_matrix, men_margins, women_margins, tol=1e-12
    )
    total_error = npmaxabs(marg_err_x) + npmaxabs(marg_err_y)
    assert total_error < TOL

    # and we test ipfp_gender_heteroskedastic for tau = 1
    tau = 1.0
    print_stars("Testing ipfp gender-heteroskedastic for tau = 1:")
    mus_tau, marg_err_x_tau, marg_err_y_tau = ipfp_gender_heteroskedastic_solver(
        true_surplus_matrix, men_margins, women_margins, tau
    )
    total_error = npmaxabs(marg_err_x_tau) + npmaxabs(marg_err_y_tau)
    assert total_error < TOL

    # and we test ipfp heteroxy for sigma = tau = 1
    print_stars("Testing ipfp heteroskedastic for sigma_x and tau_y = 1:")

    sigma_x = np.ones(X)
    tau_y = np.ones(Y)

    mus_hxy, marg_err_x_hxy, marg_err_y_hxy = ipfp_heteroskedastic_solver(
        true_surplus_matrix, men_margins, women_margins, sigma_x, tau_y
    )
    muxy_hxy, _, _ = mus_hxy
    total_error = npmaxabs(marg_err_x_hxy) + npmaxabs(marg_err_y_hxy)
    assert total_error < TOL

    # and we test ipfp homo w/o singles
    print_stars("Testing ipfp homoskedastic w/o singles:")
    # we need as many women as men
    women_margins_nosingles = women_margins * (
        np.sum(men_margins) / np.sum(women_margins)
    )
    muxy_nos, marg_err_x_nos, marg_err_y_nos = ipfp_homoskedastic_nosingles_solver(
        true_surplus_matrix, men_margins, women_margins_nosingles, gr=False
    )
    total_error = npmaxabs(marg_err_x_nos) + npmaxabs(marg_err_y_nos)
    assert total_error < TOL

    # we check the gradient
    iman = 3
    iwoman = 17

    GRADIENT_STEP = 1e-6

    if do_test_gradient_heteroskedastic:
        mus_hxy, marg_err_x_hxy, marg_err_y_hxy, dmus_hxy = ipfp_heteroskedastic_solver(
            true_surplus_matrix, men_margins, women_margins, sigma_x, tau_y, gr=True
        )
        muij = mus_hxy[0][iman, iwoman]
        muij_x0 = mus_hxy[1][iman]
        muij_0y = mus_hxy[2][iwoman]
        gradij = dmus_hxy[0][iman * Y + iwoman, :]
        gradij_x0 = dmus_hxy[1][iman, :]
        gradij_0y = dmus_hxy[2][iwoman, :]
        n_cols_rhs = n_prod_categories + 2 * n_sum_categories
        gradij_numeric = np.zeros(n_cols_rhs)
        gradij_numeric_x0 = np.zeros(n_cols_rhs)
        gradij_numeric_0y = np.zeros(n_cols_rhs)
        icoef = 0
        for ix in range(X):
            men_marg = men_margins.copy()
            men_marg[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroskedastic_solver(
                true_surplus_matrix, men_marg, women_margins, sigma_x, tau_y
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(Y):
            women_marg = women_margins.copy()
            women_marg[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroskedastic_solver(
                true_surplus_matrix, men_margins, women_marg, sigma_x, tau_y
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for i1 in range(X):
            for i2 in range(Y):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = ipfp_heteroskedastic_solver(
                    surplus_mat, men_margins, women_margins, sigma_x, tau_y
                )
                gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
                gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
                gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
                icoef += 1
        for ix in range(X):
            sigma = sigma_x.copy()
            sigma[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroskedastic_solver(
                true_surplus_matrix, men_margins, women_margins, sigma, tau_y
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(Y):
            tau = tau_y.copy()
            tau[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroskedastic_solver(
                true_surplus_matrix, men_margins, women_margins, sigma_x, tau
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1

        diff_gradients = gradij_numeric - gradij
        error_gradient = np.abs(diff_gradients)

        assert npmaxabs(error_gradient) < TOL_GRADIENT

        diff_gradients_x0 = gradij_numeric_x0 - gradij_x0
        error_gradient_x0 = np.abs(diff_gradients_x0)

        assert npmaxabs(error_gradient_x0) < TOL_GRADIENT

        diff_gradients_0y = gradij_numeric_0y - gradij_0y
        error_gradient_0y = np.abs(diff_gradients_0y)

        assert npmaxabs(error_gradient_0y) < TOL_GRADIENT

    if do_test_gradient_gender_heteroskedastic:
        tau = 1.0
        mus_h, marg_err_x_h, marg_err_y_h, dmus_h = ipfp_gender_heteroskedastic_solver(
            true_surplus_matrix, men_margins, women_margins, tau, gr=True
        )
        muij = mus_h[0][iman, iwoman]
        gradij = dmus_h[0][iman * Y + iwoman, :]
        n_cols_rhs = n_prod_categories + n_sum_categories + 1
        gradij_numeric = np.zeros(n_cols_rhs)
        icoef = 0
        for ix in range(X):
            men_marg = men_margins.copy()
            men_marg[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_gender_heteroskedastic_solver(
                true_surplus_matrix, men_marg, women_margins, tau
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            icoef += 1
        for iy in range(Y):
            women_marg = women_margins.copy()
            women_marg[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_gender_heteroskedastic_solver(
                true_surplus_matrix, men_margins, women_marg, tau
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            icoef += 1
        for i1 in range(X):
            for i2 in range(Y):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = ipfp_gender_heteroskedastic_solver(
                    surplus_mat, men_margins, women_margins, tau
                )
                gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
                icoef += 1
        tau_plus = tau + GRADIENT_STEP
        mus, marg_err_x, marg_err_y = ipfp_gender_heteroskedastic_solver(
            true_surplus_matrix, men_margins, women_margins, tau_plus
        )
        gradij_numeric[-1] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP

        error_gradient = np.abs(gradij_numeric - gradij)

        assert npmaxabs(error_gradient) < TOL_GRADIENT
