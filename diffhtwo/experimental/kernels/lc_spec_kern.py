import jax.numpy as jnp
from diffsky.experimental.kernels import gd_specphot_kernels_merging as gspkm
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from ..lightcone_generators import generate_lc_data


def lc_spec_kern(
    ran_key,
    param_collection,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    lgmp_min=10,
    lgmp_max=15,
    lc_sky_area_degsq=1000,
    n_z_phot_table=15,
):
    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )

    lc_args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = generate_lc_data(*lc_args)

    line_wave_table = jnp.array([halpha_wave_aa])

    _res = mc_specphot_kern_merging_wrapper(
        ran_key,
        param_collection,
        lc_data,
        line_wave_table,
    )
    (phot_kern_results, phot_randoms, spec_kern_results) = _res

    return phot_kern_results, phot_randoms, spec_kern_results, lc_data


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

    return phot_kern_results, phot_randoms, spec_kern_results
