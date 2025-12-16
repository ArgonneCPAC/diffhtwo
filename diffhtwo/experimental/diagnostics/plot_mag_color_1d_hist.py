import jax.numpy as jnp
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental import precompute_ssp_phot as psspp
from diffstar.defaults import FB, T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from jax import random as jran

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_n_ugriz(
    diffstarpop_params,
    spspop_params,
    tcurves,
    mag_column,
    dmag,
    ran_key,
    zmin,
    zmax,
    ssp_data,
    mzr_params,
    scatter_params,
    ssp_err_pop_params,
    lgmp_min=10.0,
    sky_area_degsq=1.0,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """mc lightcone"""
    ran_key, lc_key = jran.split(ran_key, 2)
    lc_args = (lc_key, lgmp_min, zmin, zmax, sky_area_degsq, cosmo_params)
    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*lc_args)

    n_z_phot_table = 15

    z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = jnp.log10(t_0)
    t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
    )

    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    ran_key, phot_key = jran.split(ran_key, 2)
    phot_args = (
        phot_key,
        lc_halopop["z_obs"],
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
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    lc_phot = lc_phot_kern.multiband_lc_phot_kern(*phot_args)

    # color_bin_edges = np.arange(-3. - dmag / 2, 4., dmag)
    # mag_bin_edges = np.arange(18. - dmag / 2, 26., dmag)

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

    fig, ax = plt.subplots(1, 5, figsize=(10, 3))
    fig.subplots_adjust(left=0.275, hspace=0, top=0.95, right=0.87, wspace=0.3)

    for i in range(0, len(tcurves) + 1):
        ax[i].hist(
            obs_colors_mag_q[:, i],
            weights=lc_phot.weights_q,
            width=dmag,
            facecolor="orange",
            alpha=0.7,
            edgecolor="none",
            label="data",
        )

    # ax[0].set_ylabel("log$_{10}$ (1 + N)")
    ax[0].set_xlabel("MegaCam_uS - HSC_g [AB mag]")
    ax[1].set_xlabel("HSC_g- HSC_r [AB mag]")
    ax[2].set_xlabel("HSC_r - HSC_i [AB mag]")
    ax[3].set_xlabel("HSC_i - HSC_z [AB mag]")
    ax[4].set_xlabel("HSC_i [AB mag]")

    plt.legend()
    plt.tight_layout()
    plt.show()
