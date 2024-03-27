"""
"""

from jax import config
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .jax_powerlaw import powerlaw_pdf

config.update("jax_enable_x64", True)

LGQH_MIN = 48.0  # log10(photon/s)
LGQH_MAX = 52.0  # log10(photon/s)

SANTORO22_ALPHA = -1.73
NPHOTONS_SINGLE_OSTAR = 10**LGQH_MIN
LINE_LUM_NORM = 1e30


@jjit
def _avg_Q_H_kern(x, alpha, logQ_H_min=LGQH_MIN, logQ_H_max=LGQH_MAX):
    weights = _get_qh_weights(x, alpha, logQ_H_min, logQ_H_max)
    return jnp.sum(weights * x)


@jjit
def _qh_lumfuncweight_scalar_kern(x, alpha, logQ_H_min, logQ_H_max):
    """
    Parameters
    ----------
    x : float or array
        Q_H/1e48

    alpha : float
        pdf(x) propto x^α

    Returns
    -------
    pdf : float or array

    """
    a = 1.0
    b = 10 ** (logQ_H_max - logQ_H_min)
    g = alpha + 1.0
    pdf = powerlaw_pdf(x, a, b, g)  # assumes g!=0
    return pdf


@jjit
def _get_qh_weights(x, alpha, logQ_H_min=LGQH_MIN, logQ_H_max=LGQH_MAX):
    """
    Parameters
    ----------
    x : array of shape (n, )
        Q_H/1e48

    alpha : float
        pdf(x) propto x^α

    Returns
    -------
    weights : array of shape (n, )

    """
    weights = _qh_lumfuncweight_scalar_kern(x, alpha, logQ_H_min, logQ_H_max)
    weights = weights / jnp.sum(weights)
    return weights


@jjit
def _get_pdf_weighted_htwo_singleline_kern(
    Q_H_grid, alpha, htwo_grid_singleline, grid_normalization_table
):
    n_qh = len(Q_H_grid)
    x_grid = Q_H_grid / NPHOTONS_SINGLE_OSTAR
    avg_Q_H = _avg_Q_H_kern(x_grid, alpha) * NPHOTONS_SINGLE_OSTAR

    qh_weights = _get_qh_weights(x_grid, alpha).reshape((n_qh, 1, 1))
    pdf_weighted_line = jnp.sum(qh_weights * htwo_grid_singleline, axis=0)

    integrand = (pdf_weighted_line * grid_normalization_table) / avg_Q_H
    data = (qh_weights, pdf_weighted_line, avg_Q_H)

    return integrand, data


_A = (None, None, 0, None)
_get_pdf_weighted_htwo_singleline_vmap = jjit(
    vmap(_get_pdf_weighted_htwo_singleline_kern, in_axes=_A)
)


@jjit
def _get_pdf_weighted_htwo_grid(Q_H_grid, alpha, htwo_grid, grid_norms):
    """

    Parameters
    ----------
    Q_H_grid : array, shape (n_qh, )
        grid in photon/s
        e.g. np.array((1e48, 1e49, 1e50, 1e51, 1e52))

    alpha : float
        alpha (α) is the power law index that determines the Q_H_grid weighting:
        φ(Q_H) ~ Q_H^α

    htwo_grid : array, shape (n_lines, n_qh, n_met, n_age)

    grid_norms : array, shape (n_lines, n_met, n_age)

    Returns
    -------
    pdf_weighted_htwo_grid : array, shape (n_lines, n_met, n_age)
        X_ij stores an HII emission line table, after averaging over φ(Q_H)

        The emission lines of the galaxy are computed as a
        weighted sum of the returned X_ij, where the weights M_ij
        are determined by the history of star formation and metallicity

        X_ij is defined as follows:

            L_ij = ∑_k φ_k L_ijk / ∑_k φ_k

            ⟨Q_H⟩ = ∑_k φ_k Q_H_k / ∑_k φ_k

            Now define:
            X_ij := L_ij q_ij / ⟨Q_H⟩

            If M_ij is the SFH, then we compute the galaxy luminosity L as:
            L = ∑_ij X_ij*M_ij


    data : sequence
        data = qh_weights, pdf_weighted_line, avg_Q_H

            qh_weights : array, shape (n_lines, n_qh)

            pdf_weighted_line : array, shape (n_lines, n_met, n_age)

            avg_Q_H : array, shape (n_lines, )

    """
    args = (Q_H_grid, alpha, htwo_grid, grid_norms)
    pdf_weighted_htwo_grid, data = _get_pdf_weighted_htwo_singleline_vmap(*args)
    return pdf_weighted_htwo_grid, data
