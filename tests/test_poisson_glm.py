import numpy as np

from cupid_matching.matching_utils import Matching
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm
from cupid_matching.ipfp_solvers import ipfp_homoskedastic_solver


def test_infinite_population():
    # Example 1: using IPFP to solve for the matching
    TOL = 1e-4
    X, Y, K = 4, 3, 6
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

    nx = np.array([5, 6, 5, 8]) * 10000
    my = np.array([4, 8, 6]) * 10000
    (muxy, mux0, mu0y), err_x, err_y = ipfp_homoskedastic_solver(Phi, nx, my)
    results = choo_siow_poisson_glm(Matching(muxy, nx, my), phi_bases)

    discrepancy = results.print_results(
        lambda_true, u_true=-np.log(mux0 / nx), v_true=-np.log(mu0y / my)
    )
    assert discrepancy < TOL



def test_finite_population():
    # Example 2: simulating many individuals
    TOL = 1e-4
    n_households = 1e12
    # we simulate a Choo and Siow population
    #  with equal numbers of men and women of each type
    X, Y, K = 4, 3, 6
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
    n = np.ones(X)
    m = np.ones(Y)
    Phi = phi_bases @ lambda_true
    choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
    mus_sim = choo_siow_instance.simulate(n_households)
    muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

    results = choo_siow_poisson_glm(mus_sim, phi_bases)

    discrepancy = results.print_results(
        lambda_true, u_true=-np.log(mux0_sim / n_sim), 
        v_true=-np.log(mu0y_sim / m_sim)
    )

    assert discrepancy < TOL
