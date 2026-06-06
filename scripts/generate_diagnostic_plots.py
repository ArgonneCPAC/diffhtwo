import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.diagnostics import plot_cosmos_merging as pcm
from diffsky.experimental.diagnostics import check_smhm
from diffsky.ssp_err_model.diagnostics import plot_ssp_err_model as psspem
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps import load_ssp_templates
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from diffhtwo.experimental.data_loaders import load_feniks, load_hizels, load_sdss
from diffhtwo.experimental.defaults import (
    FENIKS_Z_MAX,
    FENIKS_Z_MIN,
    SDSS_Z_MAX,
    SDSS_Z_MIN,
)
from diffhtwo.experimental.diagnostics.plot_avpop_mono import (
    make_avpop_mono_comparison_plots,
)
from diffhtwo.experimental.diagnostics.plot_burstpop import (
    make_fburstpop_comparison_plot,
    plot_lgfburst_mh_z,
)
from diffhtwo.experimental.diagnostics.plot_cen import plot_massive_cen_colors
from diffhtwo.experimental.diagnostics.plot_halpha import (
    plot_halpha,
    plot_halpha_insitu_exsitu,
    plot_halpha_ms_q_burst,
    plot_halpha_sfr,
    plot_halpha_ssfr,
)
from diffhtwo.experimental.diagnostics.plot_insitu_sm import plot_insitu_sm
from diffhtwo.experimental.diagnostics.plot_phot import (
    plot_app_mag_funcs,
    plot_color_pdfs,
    plot_n_colors_mag,
    plot_n_mags,
)
from diffhtwo.experimental.diagnostics.plot_restframe_colors import plot_uvj
from diffhtwo.experimental.diagnostics.plot_sat import plot_merging_sat_colors
from diffhtwo.experimental.diagnostics.plot_satquench import (
    generate_sat_plots,
    plot_satquench_model,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_diagnostics.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # get directories/files
    os.environ["COSMOS20_DRN"] = cfg["cosmos20_drn"]
    os.environ["DSPS_DRN"] = cfg["dsps_drn"]

    sdss_drn = cfg["sdss_drn"]
    feniks_drn = cfg["feniks_drn"]
    hizels_drn = Path(cfg["hizels_drn"])

    ssp_filename = cfg["ssp_file"]
    fit_diagnostics_save_drn = cfg["fit_diagnostics_save_drn"]
    param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
        cfg["model_drn"],
        cfg["model_nickname"],
    )

    num_halos = cfg["plots"]["num_halos"]

    # get ssp data
    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    halpha_wave_aa = jnp.array(ssp_data.ssp_emline_wave[0])
    ran_key = jran.key(0)

    if cfg["plots"]["plot_satquench_model"]:
        plot_satquench_model(
            param_collection_fit.diffstarpop_params,
            "fit",
            fit_diagnostics_save_drn,
            plt_show=False,
        )

    if cfg["plots"]["plot_insitu_smhm"]:
        # Plot in-situ SMHM
        print("Generating in-situ SMHM plot...")
        check_smhm.plot_diffstarpop_insitu_smhm(
            DEFAULT_DIFFSTARPOP_PARAMS,
            param_collection_fit.diffstarpop_params,
            fit_diagnostics_save_drn + "/insitu_smhm_w_default.png",
        )
    # Plot feniks Avpop
    if cfg["plots"]["plot_avpop"]:
        print("Generating Avpop plot...")
        _ = make_avpop_mono_comparison_plots(
            param_collection_fit.spspop_params.dustpop_params.avpop_params,
            fname=fit_diagnostics_save_drn + "/avpop_mono.png",
        )

    if cfg["plots"]["plot_burstpop"]:
        print("Generating fburstpop plot...")
        _ = make_fburstpop_comparison_plot(
            param_collection_fit.spspop_params.burstpop_params.fburstpop_params,
            fname=fit_diagnostics_save_drn + "/burstpop.png",
            label1="fit",
        )

    """
    Plot HiZELS
    """
    if cfg["plot_hizels"]:
        feniks = load_feniks.get_feniks_data(feniks_drn, ran_key, ssp_data)
        hizels_drn = Path(hizels_drn)
        hizels_label = "hizels"
        hizels = load_hizels.get_hizels_data(
            hizels_drn,
            ran_key,
            ssp_data,
            feniks.filter_info.tcurves,
            halpha_wave_aa,
        )

        print("Generating h-alpha LF plot...")
        plot_halpha(
            ran_key,
            hizels,
            param_collection_fit,
            ssp_data,
            feniks.filter_info.tcurves,
            halpha_wave_aa,
            hizels_label,
            fit_diagnostics_save_drn,
            num_halos=num_halos,
            plt_show=False,
        )

        print("Generating h-alpha LF ms/q/burst plot...")
        plot_halpha_ms_q_burst(
            ran_key,
            hizels,
            param_collection_fit,
            ssp_data,
            feniks.filter_info.tcurves,
            halpha_wave_aa,
            hizels_label,
            fit_diagnostics_save_drn,
            num_halos=num_halos,
            plt_show=False,
        )

        print("Generating h-alpha LF ssfr plot...")
        plot_halpha_ssfr(
            ran_key,
            hizels,
            param_collection_fit,
            ssp_data,
            feniks.filter_info.tcurves,
            halpha_wave_aa,
            hizels_label,
            fit_diagnostics_save_drn,
            num_halos=num_halos,
            plt_show=False,
        )

        print("Generating h-alpha LF sfr plot...")
        plot_halpha_sfr(
            ran_key,
            hizels,
            param_collection_fit,
            ssp_data,
            feniks.filter_info.tcurves,
            halpha_wave_aa,
            hizels_label,
            fit_diagnostics_save_drn,
            num_halos=num_halos,
            plt_show=False,
        )

        print("Generating h-alpha LF in-situ/ex-situ plot...")
        plot_halpha_insitu_exsitu(
            ran_key,
            hizels,
            param_collection_fit,
            ssp_data,
            feniks.filter_info.tcurves,
            halpha_wave_aa,
            hizels_label,
            fit_diagnostics_save_drn,
            num_halos=num_halos,
            plt_show=False,
        )

    """
    Plot FENIKS
    """
    if cfg["plot_feniks"]:
        feniks_label = "feniks"  # + cfg["model_nickname"].split("_")[0]
        feniks = load_feniks.get_feniks_data(feniks_drn, ran_key, ssp_data)
        feniks_zbins = feniks.zbins

        if cfg["plots"]["plot_app_mag_funcs"]:
            print("Generating FENIKS app mag funcs plot...")
            plot_app_mag_funcs(
                feniks,
                feniks_label,
                param_collection_fit,
                ran_key,
                ssp_data,
                fit_diagnostics_save_drn,
                num_halos=num_halos,
                plt_show=False,
            )

        if cfg["plots"]["plot_uvj"]:
            print("Generating FENIKS UVJ plot...")
            plot_uvj(
                ran_key,
                param_collection_fit,
                ssp_data,
                feniks_drn,
                feniks_label,
                fit_diagnostics_save_drn,
                num_halos=num_halos,
            )

        if cfg["plots"]["plot_exsitu_frac"]:
            print("Generating FENIKS ex-situ frac plot...")
            pdata = pcm.get_plotting_data(
                **param_collection_fit._asdict(),
                z_min=FENIKS_Z_MIN,
                z_max=FENIKS_Z_MAX,
                num_halos=num_halos,
            )
            pcm.plot_ex_situ_fraction(
                pdata=pdata,
                model_nickname=feniks_label
                + "_z"
                + str(FENIKS_Z_MIN)
                + "-"
                + str(FENIKS_Z_MAX),
                drn_out=fit_diagnostics_save_drn,
            )

        if cfg["plots"]["plot_fburst_mh_z"]:
            print("Generating FENIKS lgfburst plot...")
            plot_lgfburst_mh_z(
                ran_key,
                param_collection_fit,
                FENIKS_Z_MIN,
                FENIKS_Z_MAX,
                ssp_data,
                feniks.filter_info.tcurves,
                feniks_label,
                fit_diagnostics_save_drn,
                # mag_thresh=feniks.filter_info.mag_thresh,
                # frac_cat=feniks.frac_cat,
                num_halos=num_halos,
                plt_show=False,
            )

        for zbin in range(0, len(feniks_zbins)):
            z_min = feniks_zbins[zbin][0]
            z_max = feniks_zbins[zbin][1]

            if cfg["plots"]["plot_insitu_sm"]:
                print(
                    f"Generating FENIKS in-situ sm plot for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                plot_insitu_sm(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    feniks.dataset_dim_labels,
                    ssp_data,
                    feniks.filter_info.tcurves,
                    feniks_label,
                    fit_diagnostics_save_drn,
                    num_halos=num_halos,
                    plt_show=False,
                )

            if cfg["plots"]["plot_color_pdfs"]:
                print(
                    f"Generating FENIKS color PDFs for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                plot_color_pdfs(
                    feniks,
                    feniks_label,
                    param_collection_fit,
                    ran_key,
                    z_min,
                    z_max,
                    ssp_data,
                    fit_diagnostics_save_drn,
                )

            if cfg["plots"]["plot_colors_mags"]:
                print(
                    f"Generating FENIKS photometry plots for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                plot_n_colors_mag(
                    feniks,
                    feniks_label,
                    param_collection_fit,
                    ran_key,
                    z_min,
                    z_max,
                    ssp_data,
                    fit_diagnostics_save_drn,
                )

            if cfg["plots"]["plot_mags"]:
                plot_n_mags(
                    feniks,
                    feniks_label,
                    param_collection_fit,
                    ran_key,
                    z_min,
                    z_max,
                    ssp_data,
                    fit_diagnostics_save_drn,
                )

            if cfg["plots"]["plot_ssperr"]:
                print(
                    f"Generating FENIKS ssp error plot for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                z_obs = np.median((z_min, z_max))
                psspem.plot_ssp_err_model_delta_mag_vs_wavelength(
                    ssp_err_params=param_collection_fit.ssperr_params,
                    z_obs=z_obs,
                    drn_out=fit_diagnostics_save_drn,
                    model_nickname=feniks_label,
                )

            if cfg["plots"]["plot_massive_cen_colors"]:
                print(
                    f"Generating FENIKS massive central colors plot for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                plot_massive_cen_colors(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    feniks.dataset_dim_labels,
                    ssp_data,
                    feniks.filter_info.tcurves,
                    feniks_label,
                    fit_diagnostics_save_drn,
                    num_halos=num_halos,
                    plt_show=False,
                )

            if cfg["plots"]["plot_merging_sat_colors"]:
                print(
                    f"Generating FENIKS merging sat colors plot for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                plot_merging_sat_colors(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    feniks.dataset_dim_labels,
                    ssp_data,
                    feniks.filter_info.tcurves,
                    feniks_label,
                    fit_diagnostics_save_drn,
                    num_halos=num_halos,
                    plt_show=False,
                )

                # print(
                #     f"Generating massive central colors plot for {zbin+1}/{len(feniks_zbins)} z-bin with feniks specific additional weights that take into account magthresh and frac_cat..."
                # )
                # plot_massive_cen_colors(
                #     ran_key,
                #     param_collection_fit,
                #     z_min,
                #     z_max,
                #     feniks.dataset_dim_labels,
                #     ssp_data,
                #     tcurves,
                #     feniks_label + "_weighted",
                #     fit_diagnostics_save_drn,
                #     mag_thresh=mag_thresh,
                #     frac_cat=frac_cat,
                #     num_halos=num_halos,
                #     plt_show=False,
                # )

            if cfg["plots"]["plot_satquench"]:
                print(
                    f"Generating FENIKS satquench plots for {zbin+1}/{len(feniks_zbins)} z-bin..."
                )
                generate_sat_plots(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    ssp_data,
                    feniks.filter_info.tcurves,
                    feniks_label,
                    fit_diagnostics_save_drn,
                    mag_thresh=feniks.filter_info.mag_thresh,
                    frac_cat=feniks.frac_cat,
                    num_halos=num_halos,
                    plt_show=False,
                )

    """
    Plot SDSS
    """
    if cfg["plot_sdss"]:
        sdss_label = "sdss"  # + cfg["model_nickname"].split("_")[0]
        sdss = load_sdss.get_sdss_data(sdss_drn, ran_key, ssp_data)
        sdss_zbins = np.array(
            [
                [0.02, 0.06],
                [0.06, 0.1],
                [0.1, 0.14],
                [0.14, 0.18],
                [0.18, 0.2],
            ]
        )

        if cfg["plots"]["plot_exsitu_frac"]:
            print("Generating SDSS ex-situ frac plot...")
            pdata = pcm.get_plotting_data(
                **param_collection_fit._asdict(),
                z_min=SDSS_Z_MIN,
                z_max=SDSS_Z_MAX,
                num_halos=num_halos,
            )
            pcm.plot_ex_situ_fraction(
                pdata=pdata,
                model_nickname=sdss_label
                + "_z"
                + str(SDSS_Z_MIN)
                + "-"
                + str(SDSS_Z_MAX),
                drn_out=fit_diagnostics_save_drn,
            )

        if cfg["plots"]["plot_fburst_mh_z"]:
            print("Generating SDSS lgfburst plot...")
            plot_lgfburst_mh_z(
                ran_key,
                param_collection_fit,
                SDSS_Z_MIN,
                SDSS_Z_MAX,
                ssp_data,
                sdss.filter_info.tcurves,
                sdss_label,
                fit_diagnostics_save_drn,
                # mag_thresh=sdss.filter_info.mag_thresh,
                # frac_cat=sdss.frac_cat,
                num_halos=num_halos,
                plt_show=False,
            )

        for zbin in range(0, len(sdss_zbins)):
            z_min = sdss_zbins[zbin][0]
            z_max = sdss_zbins[zbin][1]

            if cfg["plots"]["plot_insitu_sm"]:
                print(
                    f"Generating SDSS in-situ sm plot for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                plot_insitu_sm(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    sdss.dataset_dim_labels,
                    ssp_data,
                    sdss.filter_info.tcurves,
                    sdss_label,
                    fit_diagnostics_save_drn,
                    num_halos=num_halos,
                    plt_show=False,
                )

            if cfg["plots"]["plot_color_pdfs"]:
                print(
                    f"Generating SDSS color PDFs for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                plot_color_pdfs(
                    sdss,
                    sdss_label,
                    param_collection_fit,
                    ran_key,
                    z_min,
                    z_max,
                    ssp_data,
                    fit_diagnostics_save_drn,
                )

            if cfg["plots"]["plot_colors_mags"]:
                print(
                    f"Generating SDSS photometry plots for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                plot_n_colors_mag(
                    sdss,
                    sdss_label,
                    param_collection_fit,
                    ran_key,
                    z_min,
                    z_max,
                    ssp_data,
                    fit_diagnostics_save_drn,
                )

            if cfg["plots"]["plot_mags"]:
                plot_n_mags(
                    sdss,
                    sdss_label,
                    param_collection_fit,
                    ran_key,
                    z_min,
                    z_max,
                    ssp_data,
                    fit_diagnostics_save_drn,
                )

            if cfg["plots"]["plot_ssperr"]:
                print(
                    f"Generating SDSS ssp error plot for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                z_obs = np.median((z_min, z_max))
                psspem.plot_ssp_err_model_delta_mag_vs_wavelength(
                    ssp_err_params=param_collection_fit.ssperr_params,
                    z_obs=z_obs,
                    drn_out=fit_diagnostics_save_drn,
                    model_nickname=sdss_label,
                )

            if cfg["plots"]["plot_massive_cen_colors"]:
                print(
                    f"Generating SDSS massive central colors plot for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                plot_massive_cen_colors(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    sdss.dataset_dim_labels,
                    ssp_data,
                    sdss.filter_info.tcurves,
                    sdss_label,
                    fit_diagnostics_save_drn,
                    num_halos=num_halos,
                    plt_show=False,
                )

            if cfg["plots"]["plot_merging_sat_colors"]:
                print(
                    f"Generating SDSS merging sat colors plot for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                plot_merging_sat_colors(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    sdss.dataset_dim_labels,
                    ssp_data,
                    sdss.filter_info.tcurves,
                    sdss_label,
                    fit_diagnostics_save_drn,
                    num_halos=num_halos,
                    plt_show=False,
                )

            if cfg["plots"]["plot_satquench"]:
                print(
                    f"Generating SDSS satquench plots for {zbin+1}/{len(sdss_zbins)} z-bin..."
                )
                generate_sat_plots(
                    ran_key,
                    param_collection_fit,
                    z_min,
                    z_max,
                    ssp_data,
                    sdss.filter_info.tcurves,
                    sdss_label,
                    fit_diagnostics_save_drn,
                    mag_thresh=sdss.filter_info.mag_thresh,
                    frac_cat=sdss.frac_cat,
                    num_halos=num_halos,
                    plt_show=False,
                )
