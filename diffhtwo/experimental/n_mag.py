import jax.numpy as jnp
from diffsky import diffndhist
from diffsky.experimental import lc_phot_kern
from jax import jit as jjit


@jjit
def n_mag_kern(
    diffstarpop_params,
    spspop_params,
    ran_key,
    lc_halopop,
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
    """Kernel for calculating number density in N-dimensional mag-color space based on
    diffstarpop/bursty/dust parameters

    Parameters
    ----------
    lc_halopop : dict of halo lightcone output of
                 diffsky.experimental.mc_lightcone_halos.mc_weighted_halo_lightcone()

    tcurves : list of dsps.data_loaders.defaults.TransmissionCurve objects

    lh_centroids: Latin Hypercube centroids in mag-color space based on data
                     array with shape (n_centroids, n_bands)

    Returns
    -------
    n : array of number counts weighted by pop fracs and nhalos in n_centroids bins
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

    # shape = number of halos in lightcone
    lc_phot = lc_phot_kern.multiband_lc_phot_kern(*args)

    num_halos, n_bands = lc_phot.obs_mags_q.shape

    mag_colors_q = lc_phot.obs_mags_q[:, 0]
    mag_colors_q = mag_colors_q.reshape(1, mag_colors_q.size)
    for band in range(0, n_bands - 1):
        color = lc_phot.obs_mags_q[:, band] - lc_phot.obs_mags_q[:, band + 1]
        mag_colors_q = jnp.vstack((mag_colors_q, color))
    mag_colors_q = mag_colors_q.T

    mag_colors_smooth_ms = lc_phot.obs_mags_smooth_ms[:, 0]
    mag_colors_smooth_ms = mag_colors_smooth_ms.reshape(1, mag_colors_smooth_ms.size)
    for band in range(0, n_bands - 1):
        color = (
            lc_phot.obs_mags_smooth_ms[:, band]
            - lc_phot.obs_mags_smooth_ms[:, band + 1]
        )
        mag_colors_smooth_ms = jnp.vstack((mag_colors_smooth_ms, color))
    mag_colors_smooth_ms = mag_colors_smooth_ms.T

    mag_colors_bursty_ms = lc_phot.obs_mags_bursty_ms[:, 0]
    mag_colors_bursty_ms = mag_colors_bursty_ms.reshape(1, mag_colors_bursty_ms.size)
    for band in range(0, n_bands - 1):
        color = (
            lc_phot.obs_mags_bursty_ms[:, band]
            - lc_phot.obs_mags_bursty_ms[:, band + 1]
        )
        mag_colors_bursty_ms = jnp.vstack((mag_colors_bursty_ms, color))
    mag_colors_bursty_ms = mag_colors_bursty_ms.T

    sig = jnp.zeros(mag_colors_q.shape) + (dmag / 2)

    lh_centroids_lo = lh_centroids - (dmag / 2)
    lh_centroids_hi = lh_centroids + (dmag / 2)

    n_q = diffndhist.tw_ndhist_weighted(
        mag_colors_q,
        sig,
        lc_phot.weights_q * lc_halopop["nhalos"],
        lh_centroids_lo,
        lh_centroids_hi,
    )

    n_smooth_ms = diffndhist.tw_ndhist_weighted(
        mag_colors_smooth_ms,
        sig,
        lc_phot.weights_smooth_ms * lc_halopop["nhalos"],
        lh_centroids_lo,
        lh_centroids_hi,
    )

    n_bursty_ms = diffndhist.tw_ndhist_weighted(
        mag_colors_bursty_ms,
        sig,
        lc_phot.weights_bursty_ms * lc_halopop["nhalos"],
        lh_centroids_lo,
        lh_centroids_hi,
    )

    n_model = (n_q + n_smooth_ms + n_bursty_ms) / lc_halopop["lc_vol_mp3"]

    return n_model
