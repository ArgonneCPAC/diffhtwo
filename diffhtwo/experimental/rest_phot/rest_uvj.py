from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky.burstpop import freqburst_mono
from diffsky.experimental import mc_diffstarpop_wrappers as mcdw
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.kernels import mc_randoms
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..lightcone_generators import generate_lc_data
from . import rest_phot_kernels_merging as rpkm


def uvj_q_ms_burst(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    uvj_tcurves,
    lgmp_min=10,
    lgmp_max=15,
    n_z_phot_table=15,
    num_halos=100,
    sky_area_degsq=1000,
    z_kcorrect=0.1,
):
    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        uvj_tcurves,
        z_phot_table,
    )

    lc_data = generate_lc_data(*lc_args)

    precomputed_ssp_restflux_table = psspp.get_ssp_restflux_table(
        ssp_data, uvj_tcurves, z_kcorrect
    )
    n_bands, n_met, n_age = precomputed_ssp_restflux_table.shape
    precomputed_ssp_restflux_table = precomputed_ssp_restflux_table.reshape(
        1, n_bands, n_met, n_age
    )

    rest_mags, weights = rest_mag_kern(
        ran_key,
        param_collection,
        lc_data,
        precomputed_ssp_restflux_table,
    )

    mc_is_q, mc_is_ms, mc_is_burst = mc_is_q_ms_burst(
        ran_key,
        param_collection,
        lc_data,
    )

    uv = rest_mags[:, 0] - rest_mags[:, 1]
    vj = rest_mags[:, 1] - rest_mags[:, 2]

    UVJ = namedtuple("UVJ", ["uv", "vj", "mc_is_q", "mc_is_ms", "mc_is_bursty"])
    uvj = UVJ(uv, vj, mc_is_q, mc_is_ms, mc_is_burst)

    return uvj


def rest_mag_kern(
    ran_key,
    param_collection,
    lc_data,
    precomputed_ssp_restflux_table,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    phot_kern_results, phot_randoms = rpkm._mc_phot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        precomputed_ssp_restflux_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *param_collection,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
        mc_merge,
    )

    rest_mags = phot_kern_results.obs_mags
    weights = jnp.where(
        lc_data.is_central, lc_data.nhalos, lc_data.nhalos * lc_data.nhalos_host
    )

    return rest_mags, weights


def mc_is_q_ms_burst(
    ran_key,
    param_collection,
    lc_data,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    n_t_table=100,
):
    # get randoms
    phot_randoms, sfh_params, merging_randoms = mc_randoms.get_mc_phot_merge_randoms(
        ran_key, param_collection.diffstarpop_params, lc_data.mah_params, cosmo_params
    )

    # get logsm_obs, logssfr_obs from diffstarpop
    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        lc_data.mah_params, sfh_params, lc_data.t_obs, cosmo_params, fb, n_t_table
    )

    # get p_burst for logsm_obs, logssfr_obs and freqburst_params
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        logsm_obs,
        logssfr_obs,
    )

    # get mc_is for q, ms, and burst
    mc_is_q = phot_randoms.mc_is_q
    mc_is_ms = ~mc_is_q

    mc_is_burst = phot_randoms.uran_pburst < p_burst
    mc_is_burst = (mc_is_ms) & (mc_is_burst)
    mc_is_ms = (mc_is_ms) & (~mc_is_burst)

    return mc_is_q, mc_is_ms, mc_is_burst
