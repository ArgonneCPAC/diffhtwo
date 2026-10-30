import jax.numpy as jnp
from diffsky.burstpop import freqburst_mono

from .. import emline_luminosity
from .gehrels_err import N_0, N_FLOOR
from .lc_photline_kern import lc_photline_kern


def get_halpha_LF_q_ms_burst(
    ran_key,
    param_collection,
    halpha_wave_aa,
    lg_halpha_Lbin_edges,
    halpha_LF_z,
    halpha_LF_delta_z,
    ssp_data,
    tcurves,
    lgmp_min=10.5,
    lgmp_max=15.0,
    num_halos=100,
    sky_area_degsq=10000,
    n_z_phot_table=15,
    mc_merge=1,
):
    z_min = halpha_LF_z - (halpha_LF_delta_z / 2)
    z_max = halpha_LF_z + (halpha_LF_delta_z / 2)

    line_wave_table = jnp.array([halpha_wave_aa])

    _res = lc_photline_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        line_wave_table,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        mc_merge=mc_merge,
    )
    phot_data, phot_randoms, photline_data, lc_data = _res
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    linelum_gal = photline_data.linelum_gal
    linelum_gal_in_situ = photline_data.linelum_gal_in_situ

    # get mc_is for q, ms, and burst
    mc_is_q, mc_is_ms, mc_is_burst = get_mc_is(
        param_collection, phot_randoms, phot_data
    )

    # in-situ
    lg_halpha_LF_in_situ = get_lf_from_linelum(
        linelum_gal_in_situ, gal_weight, lg_halpha_Lbin_edges, lc_data
    )

    # in+ex-situ
    unmerged = phot_data.p_merge < 0.5
    lg_halpha_LF = get_lf_from_linelum(
        linelum_gal[unmerged], gal_weight[unmerged], lg_halpha_Lbin_edges, lc_data
    )

    # q
    lg_halpha_LF_q = get_lf_from_linelum(
        linelum_gal[mc_is_q & unmerged],
        gal_weight[mc_is_q & unmerged],
        lg_halpha_Lbin_edges,
        lc_data,
    )

    # ms
    lg_halpha_LF_ms = get_lf_from_linelum(
        linelum_gal[mc_is_ms & unmerged],
        gal_weight[mc_is_ms & unmerged],
        lg_halpha_Lbin_edges,
        lc_data,
    )

    # burst
    lg_halpha_LF_burst = get_lf_from_linelum(
        linelum_gal[mc_is_burst & unmerged],
        gal_weight[mc_is_burst & unmerged],
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
        phot_data,
        photline_data,
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
