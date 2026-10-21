import jax.numpy as jnp
from diffsky import diffndhist_lomem
from jax import jit as jjit

from .lc_photline_kern import mc_photline_kern_merging_wrapper


@jjit
def N_linelum(
    ran_key,
    line_wave_table,
    lg_Lbin_edges,
    lc_data,
    param_collection,
):
    (
        phot_data,
        phot_randoms,
        photline_data,
    ) = mc_photline_kern_merging_wrapper(
        ran_key,
        param_collection,
        lc_data,
        line_wave_table,
    )

    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    sig = jnp.diff(lg_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    lg_Lbin_edges = lg_Lbin_edges.reshape(lg_Lbin_edges.size, 1)

    Lbin_lo = lg_Lbin_edges[:-1]
    Lbin_hi = lg_Lbin_edges[1:]

    weight_q = phot_data.frac_q
    weight_ms = (1 - weight_q) * (1 - phot_data.burstiness_info_ms.p_burst)
    weight_bursty = (1 - weight_q) * phot_data.burstiness_info_ms.p_burst

    # ms
    lg_linelum_ms = jnp.log10(photline_data.linelum_ms)
    N_linelum_ms = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_ms,
        sig,
        gal_weight * weight_ms,
        Lbin_lo,
        Lbin_hi,
    )

    # q
    lg_linelum_q = jnp.log10(photline_data.linelum_q)
    N_linelum_q = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_q,
        sig,
        gal_weight * weight_q,
        Lbin_lo,
        Lbin_hi,
    )

    # bursty
    lg_linelum_bursty = jnp.log10(photline_data.linelum_bursty)
    N_linelum_bursty = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_bursty,
        sig,
        gal_weight * weight_bursty,
        Lbin_lo,
        Lbin_hi,
    )

    N_linelum_composite = N_linelum_ms + N_linelum_q + N_linelum_bursty

    return N_linelum_composite


@jjit
def N_linelum2(
    ran_key,
    line_wave_table,
    z_data,
    param_collection,
):
    (
        phot_data,
        phot_randoms,
        photline_data,
    ) = mc_photline_kern_merging_wrapper(
        ran_key,
        param_collection,
        z_data.lc_data,
        line_wave_table,
    )

    gal_weight = z_data.lc_data.cen_weight * z_data.lc_data.sat_weight

    weight_q = phot_data.frac_q
    weight_ms = (1 - weight_q) * (1 - phot_data.burstiness_info_ms.p_burst)
    weight_bursty = (1 - weight_q) * phot_data.burstiness_info_ms.p_burst

    # ms
    lg_linelum_ms = jnp.log10(photline_data.linelum_ms)
    N_linelum_ms = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_ms,
        z_data.lf_data.sig,
        gal_weight * weight_ms,
        z_data.lf_data.bin_lo,
        z_data.lf_data.bin_hi,
    )

    # q
    lg_linelum_q = jnp.log10(photline_data.linelum_q)
    N_linelum_q = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_q,
        z_data.lf_data.sig,
        gal_weight * weight_q,
        z_data.lf_data.bin_lo,
        z_data.lf_data.bin_hi,
    )

    # ms
    lg_linelum_bursty = jnp.log10(photline_data.linelum_bursty)
    N_linelum_bursty = diffndhist_lomem.tw_ndhist_weighted(
        lg_linelum_bursty,
        z_data.lf_data.sig,
        gal_weight * weight_bursty,
        z_data.lf_data.bin_lo,
        z_data.lf_data.bin_hi,
    )

    N_linelum_composite = N_linelum_ms + N_linelum_q + N_linelum_bursty

    return N_linelum_composite
