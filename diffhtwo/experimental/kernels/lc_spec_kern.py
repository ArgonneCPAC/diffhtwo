from diffsky.experimental.kernels import gd_specphot_kernels_merging as gspkm
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit


@jjit
def mc_specphot_kern_merging_wrapper(
    ran_key,
    param_collection,
    lc_data,
    line_wave_table,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    _res = gspkm._mc_specphot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        *param_collection,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )

    (phot_kern_results, phot_randoms, spec_kern_results) = _res

    return spec_kern_results
