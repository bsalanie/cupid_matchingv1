# cupid_matching

[![image](https://img.shields.io/pypi/v/cupid_matching.svg)](https://pypi.python.org/pypi/cupid_matching)
[![image](https://github.com/bsalanie/cupid_matching/workflows/docs/badge.svg)](https://cupid_matching.gishub.org)
[![image](https://github.com/bsalanie/cupid_matching/workflows/build/badge.svg)](https://github.com/bsalanie/cupid_matching/actions?query=workflow%3Abuild)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python package to solve, simulate and estimate separable matching models**


-   Free software: MIT license
-   Documentation: https://bsalanie.github.io/cupid_matching
    
## Installation


```
pip install -U cupid_matching
```

## Accessing the code
For instance:
```py
from cupid_matching.min_distance import estimate_semilinear_mde
```

## An example
We create a Choo-Siow market; we solve for the stable matching 
in an infinite ppulation using IPFP; we simulate a sample drawn
from the stable matching and we estimate the coefficients
of the basis functions using both minimum distance
and Poisson GLM estimators.
```py
import numpy as np
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.choo_siow import entropy_choo_siow
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.poisson_glm import choo_siow_poisson_glm

X, Y, K = 10, 20, 2
# we simulate a Choo and Siow population
#  with 10 types of men and 20 types of women
#  with equal numbers of men and women of each type
#  and two random basis functions
lambda_true = np.random.randn(K)
phi_bases = np.random.randn(X, Y, K)
n = np.ones(X)
m = np.ones(Y)
Phi = phi_bases @ lambda_true
choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
matching_popu = choo_siow_instance.ipfp_solve()
muxy_popu, mux0_popu, mu0y_popu, n_popu, m_popu \
    = matching_popu.unpack()

# we simulate the market on a finite population
n_households = int(1e6)
mus_sim = choo_siow_instance.simulate(n_households)
choo_siow_instance.describe()

# We estimate the parameters using minimum distance
mde_results = estimate_semilinear_mde(
            mus_sim, phi_bases, entropy_choo_siow, 
            more_params=None
        )

# we print and check the results
mde_discrepancy = mde_results.print_results(
            true_coeffs=lambda_true, n_alpha=0
        )

# we also estimate using Poisson GLM 
poisson_results = choo_siow_poisson_glm(mus_sim, phi_bases)

muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim \
    = mus_sim.unpack()

poisson_discrepancy = poisson_results.print_results(
        lambda_true, 
        u_true=-np.log(mux0_sim/ n_sim), 
        v_true=-np.log(mu0y_sim / m_sim)
    )
```

