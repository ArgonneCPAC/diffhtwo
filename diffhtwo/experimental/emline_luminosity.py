# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
import jax.numpy as jnp
from diffsky import diffndhist_lomem
from jax import jit as jjit


@jjit
def get_emline_luminosity_func(
    L_emline_cgs,
    weights,
    dlgL_bin=0.2,
    lgL_min=38.0,
    lgL_max=45.0,
    sig=None,
    lgL_bin_edges=None,
):
    """
    Calculates the emline LF

    Parameters
    ----------
    L_emline_cgs : array of shape (n,) or (n, 1)
        h-alpha Luminosities in [erg/s]

    weights : array of shape (n,)
        weights to multiply with L_emline_cgs

    sig : array of shape (nbins, 1)
        bin dependent sigma for diffndhist

    Returns
    -------
    lgL_bin_edges : array of luminosity function bin edges in log10-space
        defined using default arguments of this function.
    tw_hist_weighted:
        luminosity function - weighted histogram counts using diffsky.diffndhist
    """

    n_L = L_emline_cgs.size
    L_emline = L_emline_cgs.reshape(n_L, 1)

    # mask: valid (strictly positive & finite)
    valid = jnp.isfinite(L_emline) & (L_emline > 0)
    L_emline = jnp.where(valid, L_emline, 10)
    lgL_emline = jnp.log10(L_emline)

    # weights: zero-out invalids
    w = jnp.where(
        valid.reshape(
            n_L,
        ),
        weights.reshape(
            n_L,
        ),
        0.0,
    )

    if lgL_bin_edges is None:
        lgL_bin_edges = jnp.arange(lgL_min, lgL_max, dlgL_bin)

    lgL_bin_lo = lgL_bin_edges[:-1].reshape(lgL_bin_edges[:-1].size, 1)
    lgL_bin_hi = lgL_bin_edges[1:].reshape(lgL_bin_edges[1:].size, 1)

    if sig is None:
        sig = jnp.zeros_like(lgL_bin_lo) + (dlgL_bin / 2)

    tw_hist_weighted = diffndhist_lomem.tw_ndhist_weighted(
        lgL_emline, sig, w, lgL_bin_lo, lgL_bin_hi
    )

    return lgL_bin_edges, tw_hist_weighted
