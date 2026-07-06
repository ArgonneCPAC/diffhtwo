import jax.numpy as jnp
from diffsky.burstpop import freqburst_mono
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY

from .. import emline_luminosity
from .gehrels_err import N_0, N_FLOOR
from .lc_spec_kern import lc_spec_kern


def get_halpha_LF_q_ms_burst(
    ran_key,
    param_collection,
    halpha_wave_aa,
    lg_halpha_Lbin_edges,
    halpha_LF_z,
    halpha_LF_delta_z,
    ssp_data,
    tcurves,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=100,
    sky_area_degsq=10000,
    n_z_phot_table=15,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    z_min = halpha_LF_z - (halpha_LF_delta_z / 2)
    z_max = halpha_LF_z + (halpha_LF_delta_z / 2)
    _res = lc_spec_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        halpha_wave_aa,
    )
    phot_kern_results, phot_randoms, spec_kern_results, lc_data = _res

    linelum_gal = spec_kern_results.linelum_gal
    linelum_gal_in_situ = spec_kern_results.linelum_gal_in_situ
    gal_weight = lc_data.cen_weight * lc_data.sat_weight
    p_merge = phot_kern_results.p_merge

    # get mc_is for q, ms, and burst
    mc_is_q, mc_is_ms, mc_is_burst = get_mc_is(
        param_collection, phot_randoms, phot_kern_results
    )

    # composite in-situ
    lg_halpha_LF_in_situ = get_lf_from_linelum(
        linelum_gal_in_situ, gal_weight, lg_halpha_Lbin_edges, lc_data
    )

    # composite
    lg_halpha_LF = get_lf_from_linelum(
        linelum_gal, gal_weight * (1 - p_merge), lg_halpha_Lbin_edges, lc_data
    )

    # q
    lg_halpha_LF_q = get_lf_from_linelum(
        linelum_gal[mc_is_q],
        gal_weight[mc_is_q] * (1 - p_merge[mc_is_q]),
        lg_halpha_Lbin_edges,
        lc_data,
    )

    # ms
    lg_halpha_LF_ms = get_lf_from_linelum(
        linelum_gal[mc_is_ms],
        gal_weight[mc_is_ms] * (1 - p_merge[mc_is_ms]),
        lg_halpha_Lbin_edges,
        lc_data,
    )

    # burst
    lg_halpha_LF_burst = get_lf_from_linelum(
        linelum_gal[mc_is_burst],
        gal_weight[mc_is_burst] * (1 - p_merge[mc_is_burst]),
        lg_halpha_Lbin_edges,
        lc_data,
    )

    lgL_bin_centers = 0.5 * (lg_halpha_Lbin_edges[1:] + lg_halpha_Lbin_edges[:-1])

    return (
        lgL_bin_centers,
        lg_halpha_LF,
        lg_halpha_LF_q,
        lg_halpha_LF_ms,
        lg_halpha_LF_burst,
        lg_halpha_LF_in_situ,
        phot_kern_results,
        spec_kern_results,
        lg_halpha_Lbin_edges,
        lc_data,
    )


def get_lf_from_linelum(linelum_gal, gal_weight, lg_emline_Lbin_edges, lc_data):
    sig = jnp.diff(lg_emline_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    _, emline_N = emline_luminosity.get_emline_luminosity_func(
        linelum_gal,
        gal_weight,
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    # take care of bins with low/zero number counts in a similar way
    # to n_mag.get_n_data_err(), using same N_floor and N_0:
    emline_N = jnp.where(emline_N > N_FLOOR, emline_N, N_0)
    lg_emline_LF = jnp.log10(emline_N / lc_data.lc_tot_vol_mpc3)

    return lg_emline_LF


def get_mc_is(param_collection, phot_randoms, phot_kern_results):
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        phot_kern_results.logsm_obs,
        phot_kern_results.logssfr_obs,
    )

    mc_is_q = phot_randoms.mc_is_q
    mc_is_ms = ~mc_is_q

    mc_is_burst = phot_randoms.uran_pburst < p_burst
    mc_is_burst = (mc_is_ms) & (mc_is_burst)
    mc_is_ms = (mc_is_ms) & (~mc_is_burst)

    return mc_is_q, mc_is_ms, mc_is_burst
