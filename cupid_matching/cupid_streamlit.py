""" Interactive Streamlit application that solves for the stable matching
    and estimates the parameters of the joint surplus
    in a `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model
    (homoskedastic, with singles)
"""
from math import pow

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .utils import nprepeat_col, nprepeat_row
from .choo_siow import entropy_choo_siow
from .min_distance import estimate_semilinear_mde
from .model_classes import ChooSiowPrimitives
from .poisson_glm import choo_siow_poisson_glm


def _make_margins(nmen, ncat_men, scenario="Constant"):
    nx_constant = nmen / ncat_men
    if scenario == "Constant":
        nx = np.full(ncat_men, nx_constant)
    elif scenario_men == "Increasing":
        lambda_men = pow(2.0, 1.0 / (ncat_men - 1))
        n1 = nmen * (lambda_men - 1.0) / (pow(lambda_men, ncat_men) - 1.0)
        nx = n1 * np.logspace(base=lambda_men, start=0, stop=ncat_men - 1, num=ncat_men)
    elif scenario_men == "Decreasing":
        lambda_men = pow(2.0, -1.0 / (ncat_men - 1))
        n1 = nmen * (lambda_men - 1.0) / (pow(lambda_men, ncat_men) - 1.0)
        nx = n1 * np.logspace(base=lambda_men, start=0, stop=ncat_men - 1, num=ncat_men)
    return nx


def _table_estimates(coeff_names, true_coeffs, estimates, stderrs):
    st.write("The coefficients are:")
    df_coeffs_estimates = pd.DataFrame(
        {"True": true_coeffs, "Estimated": estimates, "Standard errors": stderrs},
        index=coeff_names,
    )
    return st.table(df_coeffs_estimates)


def _plot_heatmap(mat, str_tit=None):
    ncat_men, ncat_women = mat.shape
    mat_arr = np.empty((mat.size, 4))
    mat_min = np.min(mat)
    i = 0
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            m = mat[ix, iy]
            s = m - mat_min + 1
            mat_arr[i, :] = np.array([ix, iy, m, s])
            i += 1

    mat_df = pd.DataFrame(mat_arr, columns=["Men", "Women", "Value", "Size"])
    mat_df = mat_df.astype(
        dtype={"Men": int, "Women": int, "Value": float, "Size": float}
    )
    base = alt.Chart(mat_df).encode(x="Men:O", y=alt.Y("Women:O", sort="descending"))
    mat_map = base.mark_circle(opacity=0.4).encode(
        size=alt.Size("Size:Q", legend=None, scale=alt.Scale(range=[1000, 10000])),
        color=alt.Color("Value:Q"),
        # tooltip=alt.Tooltip('Value', format=".2f")
    )
    text = base.mark_text(baseline="middle", fontSize=16).encode(
        text=alt.Text("Value:Q", format=".2f"),
    )
    if str_tit is None:
        both = (mat_map + text).properties(width=500, height=500)
    else:
        both = (mat_map + text).properties(title=str_tit, width=400, height=400)
    return both


def _gender_bars(xvals, str_gender):
    ncat = xvals.size
    str_cat = "x" if str_gender == "men" else "y"
    str_val = f"Single {str_gender}"
    source = pd.DataFrame({str_cat: np.arange(ncat), str_val: xvals})

    g_bars = alt.Chart(source).mark_bar().encode(x=str_cat, y=str_val)
    return g_bars.properties(width=300, height=300)


def _plot_bars(mux0, mu0y):
    men_bars = _gender_bars(mux0, "men")
    women_bars = _gender_bars(mu0y, "women")
    return (men_bars & women_bars).properties(title="Singles")


def _plot_matching(mus):
    muxy, mux0, mu0y, _, _ = mus.unpack()
    plotxy = _plot_heatmap(muxy, str_tit="Marriages")
    plotsingles = _plot_bars(mux0, mu0y)
    return plotxy | plotsingles


st.title("Separable matching with transfers")

st.markdown(
    """
> This solves for equilibrium in, and estimates the parameters of,
a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model
with transferable utilities.
> It relies on the IPFP algorithm in
[Galichon and Salanié 2021a](http://bsalanie.com/wp-content/uploads/2021/06/2021-06-1_Cupids.pdf)
and on the estimation methods in Galichon and Salanié (2021b).

> See also the [cupidpython](https://pypi.org/project/cupidpython/) package.
"""
)

expander_bar = st.expander("More information")
expander_bar.markdown(
    """
The app lets you choose the total numbers of men and women in a marriage market; the number of types of each;
the proportions of men and women in each type; and the parameters of a quadratic joint surplus function:
"""
)
expander_bar.latex(r"""$\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2$""")

expander_bar.markdown("It plots the resulting joint surplus matrix")

expander_bar.markdown(
    """
Then it solves for the large market equilibrium in a simulated Choo and Siow market,
and it fits the simulated data using the two estimators in Galichon-Salanié (2021b):

a minimum distance estimator and a Poisson GLM estimator.
"""
)

list_nhh = [1000, 10000, 100000]
st.sidebar.subheader("First, choose the total number of households")
n_households = st.sidebar.radio("Number of households", list_nhh)

list_ncat = [5, 10]
st.sidebar.subheader("Now, the numbers of types of each gender")
ncat_men = st.sidebar.radio("Number of categories of men", list_ncat)
ncat_women = st.sidebar.radio("Number of categories of women", list_ncat)

# nx = np.zeros(ncat_men)
# my = np.zeros(ncat_women)
# st.subheader("Second, choose the numbers of men and women in each category")
# for iman in range(ncat_men):
#     nx[iman] = st.slider(f"Number of men in category {iman+1}",
#                          min_value=1, max_value=10, step=1)
# for iwoman in range(ncat_women):
#     my[iwoman] = st.slider(f"Number of women in category {iwoman+1}",
#                            min_value=1, max_value=10, step=1)
#
st.sidebar.markdown(
    """
By default there are as many men as women.
You can also change the proportion.
"""
)
proportion_men = st.sidebar.slider(
    "Proportion of men", min_value=0.05, max_value=0.95, value=0.5
)


st.sidebar.markdown(
    """
By default each category within a gender has the same number of individuals.
You can also have the number increase by a factor two across categories, or decrease.
"""
)

list_scenarii = ["Constant", "Increasing", "Decreasing"]
scenario_men = st.sidebar.radio("Profile across categories for men", list_scenarii)
scenario_women = st.sidebar.radio("Profile across categories for women", list_scenarii)

nx = _make_margins(proportion_men, ncat_men, scenario_men)
my = _make_margins(1.0 - proportion_men, ncat_women, scenario_women)


st.sidebar.write("Finally, choose the coefficients of the 6 basis functions")
st.sidebar.latex(r"$\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2$")
min_c = np.array([-3.0] + [-2.0 / ncat_men] * 5)
max_c = np.array([3.0] + [2.0 / ncat_women] * 5)
true_coeffs = np.zeros(6)
coeff_names = [f"c[{i}]" for i in range(6)]


if "randoms" not in st.session_state:
    random_coeffs = np.round(min_c + (max_c - min_c) * np.random.rand(6), 2)
    st.session_state.randoms = random_coeffs

random_coeffs = st.session_state["randoms"]
for i in range(6):
    val_i = float(random_coeffs[i])
    true_coeffs[i] = st.sidebar.slider(
        coeff_names[i], min_value=min_c[i], max_value=max_c[i], value=val_i
    )

xvals = np.arange(ncat_men) + 1
yvals = np.arange(ncat_women) + 1

bases = np.zeros((ncat_men, ncat_women, 6))
bases[:, :, 0] = 1.0
xvals_mat = nprepeat_col(xvals, ncat_women)
yvals_mat = nprepeat_row(yvals, ncat_men)
bases[:, :, 1] = xvals_mat
bases[:, :, 2] = yvals_mat
bases[:, :, 3] = xvals_mat * xvals_mat
bases[:, :, 4] = np.outer(xvals, yvals)
bases[:, :, 5] = yvals_mat * yvals_mat

Phi = bases @ true_coeffs
st.markdown("Here is your joint surplus by categories:")
st.altair_chart(_plot_heatmap(Phi))

cs_market = ChooSiowPrimitives(Phi, nx, my)

st.subheader(
    f"Here are the stable matching patterns in a sample of {n_households} households:"
)

mus_sim = cs_market.simulate(n_households)
muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

st.altair_chart(_plot_matching(mus_sim))


st.subheader("Estimating the parameters.")

if st.button("Estimate"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "Below: the minimum distance estimator in Galichon and Salanié (2021b)."
        )
        st.write("It also gives us a specification test.")
        mde_results = estimate_semilinear_mde(mus_sim, bases, entropy_choo_siow)
        mde_estimates = mde_results.estimated_coefficients
        mde_stderrs = mde_results.stderrs_coefficients

        _table_estimates(coeff_names, true_coeffs, mde_estimates, mde_stderrs)

        specif_test_stat = round(mde_results.test_statistic, 2)
        specif_test_pval = round(mde_results.test_pvalue, 2)
        st.write(
            f"Test statistic: chi2({mde_results.ndf}) = {specif_test_stat} has p-value {specif_test_pval}"
        )

    with col2:
        st.markdown(
            "Here is the Poisson GLM estimator in Galichon and Salanié (2021b)."
        )
        st.write(
            "It also gives us the estimates of the expected utilities $u_x$ and $v_y$."
        )

        pglm_results = choo_siow_poisson_glm(mus_sim, bases)

        u = pglm_results.estimated_u
        v = pglm_results.estimated_v
        pglm_estimates = pglm_results.estimated_coefficients
        pglm_stderrs = pglm_results.stderrs_coefficients

        _table_estimates(coeff_names, true_coeffs, pglm_estimates, pglm_stderrs)

        x_names = [str(x) for x in range(ncat_men)]
        y_names = [str(y) for y in range(ncat_women)]

        st.write("The expected utilities are:")
        df_u_estimates = pd.DataFrame(
            {"Estimated": u, "True": -np.log(mux0_sim / n_sim)}, index=x_names
        )
        st.table(df_u_estimates)
        df_v_estimates = pd.DataFrame(
            {"Estimated": v, "True": -np.log(mu0y_sim / m_sim)}, index=y_names
        )
        st.table(df_v_estimates)
