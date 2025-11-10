import jax.numpy as jnp
from diffsky import diffndhist
from diffsky.experimental import lc_phot_kern
from jax import jit as jjit
from jax.debug import print


@jjit
def nd_mag_kern(
    diffstarpop_params,
    spspop_params,
    ran_key,
    lc_halopop,
    lc_vol,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    ssp_err_pop_params,
    tcurves,
    lh_centroids,
    dmag,
):
    """Kernel for calculating number density in 4D mag space based on
    diffstarpop/bursty/dust parameters

    Parameters
    ----------
    lc_halopop : dict of halo lightcone output of
                 diffsky.experimental.mc_lightcone_halos.mc_weighted_halo_lightcone()

    tcurves : list of dsps.data_loaders.defaults.TransmissionCurve objects

    lh_centroids: Latin Hypercube centroids in 4D mag space based on data
                     array with shape (n_centroids, n_dim)

    Returns
    -------
    nd : array of number counts weighted by pop fracs and nhalos in n_centroids bins
            centered on lh_centroids shape (n_centroids,)
    """

    args = (
        ran_key,
        jnp.array(lc_halopop["z_obs"]),
        lc_halopop["t_obs"],
        lc_halopop["mah_params"],
        lc_halopop["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        # spspop_params:
        # Diffburstpop = DiffburstPopParams(freqburst_params,
        #                   fburstpop_params, tburstpop_params)
        # Dustpop =
        scatter_params,
        ssp_err_pop_params,
    )
    lc_phot = lc_phot_kern.multiband_lc_phot_kern(*args)

    n_bands = len(tcurves)

    for band in range(0, n_bands):
        print("band:{}", band)

    i_q = lc_phot.obs_mags_q[:, 0]
    ig_q = lc_phot.obs_mags_q[:, 0] - lc_phot.obs_mags_q[:, 1]
    ir_q = lc_phot.obs_mags_q[:, 0] - lc_phot.obs_mags_q[:, 2]
    iz_q = lc_phot.obs_mags_q[:, 0] - lc_phot.obs_mags_q[:, 3]
    colors_q = jnp.vstack((i_q, ig_q, ir_q, iz_q)).T

    i_smooth_ms = lc_phot.obs_mags_smooth_ms[:, 0]
    ig_smooth_ms = lc_phot.obs_mags_smooth_ms[:, 0] - lc_phot.obs_mags_smooth_ms[:, 1]
    ir_smooth_ms = lc_phot.obs_mags_smooth_ms[:, 0] - lc_phot.obs_mags_smooth_ms[:, 2]
    iz_smooth_ms = lc_phot.obs_mags_smooth_ms[:, 0] - lc_phot.obs_mags_smooth_ms[:, 3]
    colors_smooth_ms = jnp.vstack(
        (i_smooth_ms, ig_smooth_ms, ir_smooth_ms, iz_smooth_ms)
    ).T

    i_bursty_ms = lc_phot.obs_mags_bursty_ms[:, 0]
    ig_bursty_ms = lc_phot.obs_mags_bursty_ms[:, 0] - lc_phot.obs_mags_bursty_ms[:, 1]
    ir_bursty_ms = lc_phot.obs_mags_bursty_ms[:, 0] - lc_phot.obs_mags_bursty_ms[:, 2]
    iz_bursty_ms = lc_phot.obs_mags_bursty_ms[:, 0] - lc_phot.obs_mags_bursty_ms[:, 3]
    colors_bursty_ms = jnp.vstack(
        (i_bursty_ms, ig_bursty_ms, ir_bursty_ms, iz_bursty_ms)
    ).T

    sig = jnp.zeros(lc_phot.obs_mags_q.shape) + 0.1

    lh_centroids_lo = lh_centroids - (dmag / 2)
    lh_centroids_hi = lh_centroids + (dmag / 2)

    nd_q = diffndhist.tw_ndhist_weighted(
        colors_q,
        sig,
        lc_phot.weights_q * lc_halopop["nhalos"],
        lh_centroids_lo,
        lh_centroids_hi,
    )

    nd_smooth_ms = diffndhist.tw_ndhist_weighted(
        colors_smooth_ms,
        sig,
        lc_phot.weights_smooth_ms * lc_halopop["nhalos"],
        lh_centroids_lo,
        lh_centroids_hi,
    )

    nd_bursty_ms = diffndhist.tw_ndhist_weighted(
        colors_bursty_ms,
        sig,
        lc_phot.weights_bursty_ms * lc_halopop["nhalos"],
        lh_centroids_lo,
        lh_centroids_hi,
    )

    nd_model = (nd_q + nd_smooth_ms + nd_bursty_ms) / lc_vol

    return nd_model, colors_q, colors_smooth_ms, colors_bursty_ms
