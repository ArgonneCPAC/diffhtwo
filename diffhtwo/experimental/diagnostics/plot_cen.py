import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..lightcone_generators import generate_lc_data
from ..n_specphot import mag_kern


def plot_massive_cen_colors(
    dataset,
    param_collection,
    dimension_labels,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    model_nickname,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=5000,
    lc_sky_area_degsq=1000,
    n_z_phot_table=30,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        dataset.filter_info.tcurves,
        z_phot_table,
    )

    obs_color_mag, weights, phot_kern_results = mag_kern(
        ran_key,
        param_collection,
        lc_data,
        dataset.filter_info.mag_thresh,
        dataset.frac_cat,
    )

    sm_cut = (phot_kern_results.logsm_obs > 11.5) & (lc_data.is_central == 1)

    obs_mags = phot_kern_results.obs_mags
    obs_mags_in_situ = phot_kern_results.obs_mags_in_situ

    n_gals, n_bands = obs_mags.shape

    obs_colors = obs_mags[:, 0 : n_bands - 1] - obs_mags[:, 1:n_bands]
    obs_colors_in_situ = (
        obs_mags_in_situ[:, 0 : n_bands - 1] - obs_mags_in_situ[:, 1:n_bands]
    )

    n_colors = obs_colors.shape[1]
    fig, ax = plt.subplots(1, n_colors, figsize=(14, 3))
    fig.subplots_adjust(wspace=0.3, bottom=0.22, left=0.05, right=0.99, top=0.85)

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    fig.suptitle("centrals w/ logsm > 11.5 | " + z_min_label + " < z < " + z_max_label)
    for c in range(0, n_colors):
        std = np.std(obs_colors_in_situ[:, c][sm_cut])
        med = np.median(obs_colors_in_situ[:, c][sm_cut])
        bins = np.linspace(
            med - (3 * std),
            med + (3 * std),
            20,
        )
        ax[c].hist(
            obs_colors_in_situ[:, c][sm_cut],
            weights=weights[sm_cut],
            bins=bins,
            density=True,
            alpha=0.5,
            label="in-situ",
        )

        ax[c].hist(
            obs_colors[:, c][sm_cut],
            weights=weights[sm_cut],
            bins=bins,
            density=True,
            alpha=0.5,
            label="in+ex-situ",
        )
        ax[c].set_xlabel(dimension_labels[c])
    ax[0].set_ylabel("PDF")
    plt.legend(fontsize=8)

    fig.savefig(
        savedir
        + "/cen_"
        + model_nickname
        + "_colors_massive_z"
        + z_min_label
        + "-"
        + z_max_label
        + ".png"
    )
    plt.close()
