import jax.numpy as jnp
from diffsky import diffndhist_lomem
from jax import jit as jjit

from .lc_spec_kern import mc_photline_kern_merging_wrapper


@jjit
def N_linelum(
    ran_key,
    line_wave_table,
    lg_Lbin_edges,
    lc_data,
    param_collection,
):
    (
        phot_kern_results,
        phot_randoms,
        photline_kern_results,
    ) = mc_photline_kern_merging_wrapper(
        ran_key,
        param_collection,
        lc_data,
        line_wave_table,
    )

    lg_linelum_weighted = jnp.log10(photline_kern_results.linelum_weighted)
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    sig = jnp.diff(lg_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    lg_Lbin_edges = lg_Lbin_edges.reshape(lg_Lbin_edges.size, 1)

    Lbin_lo = lg_Lbin_edges[:-1]
    Lbin_hi = lg_Lbin_edges[1:]

    N_linelum = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_weighted,
        sig,
        gal_weight,
        Lbin_lo,
        Lbin_hi,
    )

    return N_linelum
