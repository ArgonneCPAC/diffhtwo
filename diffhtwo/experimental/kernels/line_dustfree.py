import jax.numpy as jnp
from diffsky.experimental.kernels import mc_randoms, phot_kernels, phot_kernels_in_situ
from diffsky.experimental.kernels import ssp_weight_kernels as sspwk
from diffsky.merging import compute_x_tot_from_x_in_situ, merging_model
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit


@jjit
def linelum_gal_dustfree(
    ran_key,
    lc_data,
    ssp_data,
    line_wave_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    merging_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    upid = jnp.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall

    (
        phot_randoms,
        diffstarpop_results,
        merging_randoms,
    ) = mc_randoms.get_phot_merge_randoms(
        ran_key,
        diffstarpop_params,
        lc_data.mah_params,
        upid,
        lgmu_infall,
        lc_data.logmhost_infall,
        gyr_since_infall,
        cosmo_params,
    )
    t_infall = lc_data.t_obs - gyr_since_infall

    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_obs,
        t_infall,
        upid,
    )
    phot_kern_results = phot_kernels_in_situ._phot_kern(
        phot_randoms,
        diffstarpop_results,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        p_merge_smooth,
        ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )

    n_gal = phot_kern_results.logsm_obs.size
    n_line = line_wave_table.size
    n_age = ssp_data.ssp_lg_age_gyr.size

    # no dust attenuation, all ones
    dust_ftrans_lines = jnp.ones((n_gal, n_line, n_age))

    dustfree_linelum_gal_in_situ = sspwk._compute_linelum_from_weights(
        phot_kern_results.logsm_obs,
        dust_ftrans_lines,
        ssp_data,
        phot_kern_results.ssp_weights,
    )

    # update quantities with merging
    _res = phot_kernels._get_phot_kern_merging_quantities(
        phot_kern_results,
        merging_randoms,
        p_merge_smooth,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )
    mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, flux_obs_weighted, p_merge = _res

    args = (
        phot_kern_results,
        mstar_in_situ,
        mstar_obs,
        flux_in_situ,
        flux_obs,
        flux_obs_weighted,
        p_merge,
        merging_randoms.uran_pmerge,
    )
    func = phot_kernels._update_phot_kern_results_with_merging
    phot_kern_results = func(*args)

    dustfree_linelum_gal = compute_x_tot_from_x_in_situ(
        dustfree_linelum_gal_in_situ,
        phot_kern_results.p_merge[:, jnp.newaxis],
        lc_data.sat_weight[:, jnp.newaxis],
        lc_data.halo_indx,
    )

    return dustfree_linelum_gal
