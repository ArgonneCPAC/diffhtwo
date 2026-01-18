# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


import jax.numpy as jnp
from diffsky import diffndhist
from diffsky.experimental import lc_phot_kern
from jax import jit as jjit
from jax import vmap


@jjit
def n_mag_kern(
    diffstarpop_params,
    spspop_params,
    ran_key,
    lc_z_obs,
    lc_t_obs,
    lc_mah_params,
    lc_logmp0,
    lc_nhalos,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    ssp_err_pop_params,
    lh_centroids,
    dmag,
    mag_column,
    cosmo_params,
    fb,
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
        lc_z_obs,
        lc_t_obs,
        lc_mah_params,
        lc_logmp0,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    # shape = number of halos in lightcone
    lc_phot = lc_phot_kern.multiband_lc_phot_kern(*args)

    num_halos, n_bands = lc_phot.obs_mags_q.shape

    obs_colors_mag_q = []
    obs_colors_mag_smooth_ms = []
    obs_colors_mag_bursty_ms = []

    for i in range(n_bands - 1):
        obs_color_q = lc_phot.obs_mags_q[:, i] - lc_phot.obs_mags_q[:, i + 1]
        obs_colors_mag_q.append(obs_color_q)

        obs_color_smooth_ms = (
            lc_phot.obs_mags_smooth_ms[:, i] - lc_phot.obs_mags_smooth_ms[:, i + 1]
        )
        obs_colors_mag_smooth_ms.append(obs_color_smooth_ms)

        obs_color_bursty_ms = (
            lc_phot.obs_mags_bursty_ms[:, i] - lc_phot.obs_mags_bursty_ms[:, i + 1]
        )
        obs_colors_mag_bursty_ms.append(obs_color_bursty_ms)

    """mag_column"""
    obs_mag_q = lc_phot.obs_mags_q[:, mag_column]
    obs_colors_mag_q.append(obs_mag_q)
    obs_colors_mag_q = jnp.asarray(obs_colors_mag_q).T

    obs_mag_smooth_ms = lc_phot.obs_mags_smooth_ms[:, mag_column]
    obs_colors_mag_smooth_ms.append(obs_mag_smooth_ms)
    obs_colors_mag_smooth_ms = jnp.asarray(obs_colors_mag_smooth_ms).T

    obs_mag_bursty_ms = lc_phot.obs_mags_bursty_ms[:, mag_column]
    obs_colors_mag_bursty_ms.append(obs_mag_bursty_ms)
    obs_colors_mag_bursty_ms = jnp.asarray(obs_colors_mag_bursty_ms).T

    sig = jnp.zeros(obs_colors_mag_q.shape) + (dmag / 2)

    lh_centroids_lo = lh_centroids - (dmag / 2)
    lh_centroids_hi = lh_centroids + (dmag / 2)

    N_q = diffndhist.tw_ndhist_weighted(
        obs_colors_mag_q,
        sig,
        lc_phot.weights_q * lc_nhalos,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    N_smooth_ms = diffndhist.tw_ndhist_weighted(
        obs_colors_mag_smooth_ms,
        sig,
        lc_phot.weights_smooth_ms * lc_nhalos,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    N_bursty_ms = diffndhist.tw_ndhist_weighted(
        obs_colors_mag_bursty_ms,
        sig,
        lc_phot.weights_bursty_ms * lc_nhalos,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    N_model = N_q + N_smooth_ms + N_bursty_ms

    lg_n, lg_n_avg_err = get_n_data_err(N_model, lc_vol_mpc3)

    return lg_n, lg_n_avg_err


_N = (
    None,
    None,
    None,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    None,
    0,
    0,
    0,
    None,
    None,
    None,
    0,
    None,
    None,
    None,
    None,
)
n_mag_kern_multi_z = jjit(
    vmap(
        n_mag_kern,
        in_axes=_N,
    )
)


@jjit
def n_mag_kern_1d(
    diffstarpop_params,
    spspop_params,
    ran_key,
    lc_z_obs,
    lc_t_obs,
    lc_mah_params,
    lc_logmp0,
    lc_nhalos,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    ssp_err_pop_params,
    bin_centers_1d,
    dmag,
    mag_column,
    cosmo_params,
    fb,
):
    """Kernel for calculating number density in N-dimensional mag-color space based on
    diffstarpop/bursty/dust parameters

    Parameters
    ----------
    lc_halopop : dict of halo lightcone output of
                 diffsky.experimental.mc_lightcone_halos.mc_weighted_halo_lightcone()

    tcurve : list of dsps.data_loaders.defaults.TransmissionCurve objects

    bin_centers_1d: Latin Hypercube centroids in mag-color space based on data
                     array with shape (n_centroids, n_bands)

    Returns
    -------
    lg_n_model_1d : shape (ndim, 2, nbins)
    """

    args = (
        ran_key,
        lc_z_obs,
        lc_t_obs,
        lc_mah_params,
        lc_logmp0,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    # shape = number of halos in lightcone
    lc_phot = lc_phot_kern.multiband_lc_phot_kern(*args)

    num_halos, n_bands = lc_phot.obs_mags_q.shape

    lg_n_model_1d_err = []
    for i in range(n_bands - 1):
        obs_color_q = lc_phot.obs_mags_q[:, i] - lc_phot.obs_mags_q[:, i + 1]
        obs_color_q = obs_color_q.reshape(obs_color_q.size, 1)

        obs_color_smooth_ms = (
            lc_phot.obs_mags_smooth_ms[:, i] - lc_phot.obs_mags_smooth_ms[:, i + 1]
        )
        obs_color_smooth_ms = obs_color_smooth_ms.reshape(obs_color_smooth_ms.size, 1)

        obs_color_bursty_ms = (
            lc_phot.obs_mags_bursty_ms[:, i] - lc_phot.obs_mags_bursty_ms[:, i + 1]
        )
        obs_color_bursty_ms = obs_color_bursty_ms.reshape(obs_color_bursty_ms.size, 1)

        sig = jnp.zeros(obs_color_q.shape) + (dmag / 2)

        bin_centers_1d_lo = bin_centers_1d[i] - (dmag / 2)
        bin_centers_1d_hi = bin_centers_1d[i] + (dmag / 2)

        bin_centers_1d_lo = bin_centers_1d_lo.reshape(bin_centers_1d_lo.size, 1)
        bin_centers_1d_hi = bin_centers_1d_hi.reshape(bin_centers_1d_hi.size, 1)

        N_q = diffndhist.tw_ndhist_weighted(
            obs_color_q,
            sig,
            lc_phot.weights_q * lc_nhalos,
            bin_centers_1d_lo,
            bin_centers_1d_hi,
        )

        N_smooth_ms = diffndhist.tw_ndhist_weighted(
            obs_color_smooth_ms,
            sig,
            lc_phot.weights_smooth_ms * lc_nhalos,
            bin_centers_1d_lo,
            bin_centers_1d_hi,
        )

        N_bursty_ms = diffndhist.tw_ndhist_weighted(
            obs_color_bursty_ms,
            sig,
            lc_phot.weights_bursty_ms * lc_nhalos,
            bin_centers_1d_lo,
            bin_centers_1d_hi,
        )

        N_model = N_q + N_smooth_ms + N_bursty_ms
        lg_n_model_1d_err.append(get_n_data_err(N_model, lc_vol_mpc3))

    """mag_column"""
    obs_mags_q = lc_phot.obs_mags_q[:, mag_column]
    obs_mags_q = obs_mags_q.reshape(obs_mags_q.size, 1)

    obs_mags_smooth_ms = lc_phot.obs_mags_smooth_ms[:, mag_column]
    obs_mags_smooth_ms = obs_mags_smooth_ms.reshape(obs_mags_smooth_ms.size, 1)

    obs_mags_bursty_ms = lc_phot.obs_mags_bursty_ms[:, mag_column]
    obs_mags_bursty_ms = obs_mags_bursty_ms.reshape(obs_mags_bursty_ms.size, 1)

    sig = jnp.zeros(obs_mags_q.shape) + (dmag / 2)

    bin_centers_1d_lo = bin_centers_1d[-1] - (dmag / 2)
    bin_centers_1d_hi = bin_centers_1d[-1] + (dmag / 2)

    bin_centers_1d_lo = bin_centers_1d_lo.reshape(bin_centers_1d_lo.size, 1)
    bin_centers_1d_hi = bin_centers_1d_hi.reshape(bin_centers_1d_hi.size, 1)

    N_q = diffndhist.tw_ndhist_weighted(
        obs_mags_q,
        sig,
        lc_phot.weights_q * lc_nhalos,
        bin_centers_1d_lo,
        bin_centers_1d_hi,
    )

    N_smooth_ms = diffndhist.tw_ndhist_weighted(
        obs_mags_smooth_ms,
        sig,
        lc_phot.weights_smooth_ms * lc_nhalos,
        bin_centers_1d_lo,
        bin_centers_1d_hi,
    )

    N_bursty_ms = diffndhist.tw_ndhist_weighted(
        obs_mags_bursty_ms,
        sig,
        lc_phot.weights_bursty_ms * lc_nhalos,
        bin_centers_1d_lo,
        bin_centers_1d_hi,
    )

    N_model = N_q + N_smooth_ms + N_bursty_ms

    lg_n_model_1d_err.append(get_n_data_err(N_model, lc_vol_mpc3))

    return lg_n_model_1d_err


_N_1d = (
    None,
    None,
    None,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    None,
    0,
    0,
    0,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
)
n_mag_kern_1d_multi_z = jjit(
    vmap(
        n_mag_kern_1d,
        in_axes=_N_1d,
    )
)
"""
Gehrels Poisson error 
"""


@jjit
def Gehrels_upp_eq9(Ngal):
    """
    upper limit approximation - Eq. 9 Gehrels (1986); 1-sigma
    """
    Ngal = jnp.asarray(Ngal, dtype=float)

    return (Ngal + 1) * (
        1 - (1 / (9 * (Ngal + 1))) + (1 / (3 * jnp.sqrt(Ngal + 1)))
    ) ** 3


@jjit
def Gehrels_low_eq12(Ngal):
    """
    lower limit approximation - Eq. 12 Gehrels (1986); 1-sigma
    """
    Ngal = jnp.asarray(Ngal, dtype=float)

    # use a safe placeholder for N=0 to avoid div/0 in the formula
    N_safe = jnp.where(Ngal > 0.0, Ngal, 1.0)

    low_raw = (
        N_safe * (1.0 - 1.0 / (9.0 * N_safe) - 1.0 / (3.0 * jnp.sqrt(N_safe))) ** 3
    )

    # now overwrite N=0 with 0.0
    return jnp.where(Ngal > 0.0, low_raw, 0.0)


@jjit
def get_n_data_err(N, vol, N_floor=0.5):
    N_0 = 1e-12

    N = jnp.where(N > N_floor, N, N_0)
    lg_n = jnp.log10(N / vol)

    # upper limit approximation - Eq. 9 Gehrels (1986); 1-sigma
    N_upp = Gehrels_upp_eq9(N)
    lg_n_upp = jnp.log10(N_upp / vol)
    lg_n_upp_err = lg_n_upp - lg_n

    # lower limit approximation - Eq. 12 Gehrels (1986); 1-sigma
    N_low = Gehrels_low_eq12(N)
    N_low = jnp.where(N > N_floor, N_low, N_0)
    lg_n_low = jnp.log10(N_low / vol)

    lg_n_low_err = lg_n - lg_n_low

    lg_n_avg_err = (lg_n_low_err + lg_n_upp_err) / 2
    # just the upper limit for N < N_floor
    lg_n_avg_err = jnp.where(N > N_floor, lg_n_avg_err, lg_n_upp_err)

    return lg_n, lg_n_avg_err
