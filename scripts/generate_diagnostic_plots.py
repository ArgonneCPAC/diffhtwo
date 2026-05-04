import os

import jax.numpy as jnp
import numpy as np
from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.diagnostics import plot_cosmos_merging as pcm
from diffsky.experimental.diagnostics import check_smhm
from diffsky.ssp_err_model.diagnostics import plot_ssp_err_model as psspem
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps import load_ssp_templates
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from diffhtwo.experimental.data_loaders import load_feniks
from diffhtwo.experimental.diagnostics.plot_avpop_mono import (
    make_avpop_mono_comparison_plots,
)
from diffhtwo.experimental.diagnostics.plot_phot import plot_n_colors_mag, plot_n_mags
from diffhtwo.experimental.diagnostics.plot_restframe_colors import plot_uvj

if __name__ == "__main__":
    os.environ["COSMOS20_DRN"] = "/Users/kumail/diffdir/data/COSMOS20"
    os.environ[
        "DSPS_DRN"
    ] = "/Users/kumail/diffdir/data/COSMOS20/portal.nersc.gov/project/hacc/aphearin/DSPS_data"
    os.environ["SDSS_DRN"] = "/Users/kumail/diffdir/sdss"
    os.environ["FENIKS_DRN"] = "/Users/kumail/diffdir/feniks"
    os.environ["HIZELS_DRN"] = "/Users/kumail/diffdir/hizels"

    os.environ[
        "SSP_DATA"
    ] = "/Users/kumail/diffdir/ssp_data/ssp_w_emlines/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"

    feniks_drn = os.environ["FENIKS_DRN"]
    ssp_fn = os.environ["SSP_DATA"]

    ran_key = jran.key(0)

    fit_runid = "run48"
    fit_type = "diffstarpop+spspop+merging"

    param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
        "/Users/kumail/diffdir/fits/" + fit_runid + "/",
        fit_runid + "_" + fit_type,
    )

    savedir = (
        "/Users/kumail/diffdir/fits/" + fit_runid + "/diagnostic_plots/" + fit_type
    )

    os.makedirs(savedir, exist_ok=True)

    check_smhm.plot_diffstarpop_insitu_smhm(
        DEFAULT_DIFFSTARPOP_PARAMS,
        param_collection_fit.diffstarpop_params,
        savedir + "/insitu_smhm_" + fit_type + "_w_default.png",
    )

    # get ssp data
    ssp_data = load_ssp_templates(fn=ssp_fn)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    emline_wave_aa = jnp.array(ssp_data.ssp_emline_wave[0])
    emline_wave_table = jnp.array([emline_wave_aa])

    # set dim labels for plotting
    sdss_dim_labels = [
        r"$u - g$",
        r"$g - r$",
        r"$r - i$",
        r"$i - z$",
        r"$r$",
    ]

    feniks_dim_labels = [
        r"$uS_{MegaCam} - g_{HSC}$",
        r"$g_{HSC} - r_{HSC}$",
        r"$r_{HSC} - i_{HSC}$",
        r"$i_{HSC} - z_{HSC}$",
        r"$z_{HSC} - Y_{VIDEO}$",
        r"$Y_{VIDEO} - J_{UDS}$",
        r"$J_{UDS} - H_{UDS}$",
        r"$H_{UDS} - K_{UDS}$",
        r"$uS_{MegaCam}$",
        r"$K_{UDS}$",
    ]

    sdss_mag_labels = [
        r"$u$",
        r"$g$",
        r"$r$",
        r"$i$",
        r"$z$",
    ]

    feniks_mag_labels = [
        r"$uS_{MegaCam}$",
        r"$g_{HSC}$",
        r"$r_{HSC}$",
        r"$i_{HSC}$",
        r"$z_{HSC}$",
        r"$Y_{VIDEO}$",
        r"$J_{UDS}$",
        r"$H_{UDS}$",
        r"$K_{UDS}$",
    ]

    # get feniks dataset
    FENIKS = load_feniks.get_feniks_data(
        feniks_drn,
        ran_key,
        ssp_data,
    )

    # Plot feniks fitted color-mag space and app mag funcs
    feniks_zbins = np.array(
        [
            [0.2, 0.5],
            [0.5, 1.0],
            [1.0, 1.25],
            [1.25, 1.5],
            [1.5, 1.75],
            [1.75, 2.25],
            [2.25, 2.75],
            [2.75, 3.5],
        ]
    )

    for zbin in range(0, len(feniks_zbins)):
        z_min = feniks_zbins[zbin][0]
        z_max = feniks_zbins[zbin][1]
        print(
            f"Generating FENIKS photometry plots for {zbin+1}/{len(feniks_zbins)} z-bin..."
        )
        plot_n_colors_mag(
            FENIKS,
            "FENIKS",
            param_collection_fit,
            "diffsky",
            feniks_dim_labels,
            ran_key,
            z_min,
            z_max,
            ssp_data,
            suptitle="K < " + str(FENIKS.mag_thresh) + " AB",
            savedir=savedir + "/feniks",
        )

        plot_n_mags(
            FENIKS,
            "FENIKS",
            param_collection_fit,
            "diffsky",
            feniks_mag_labels,
            ran_key,
            z_min,
            z_max,
            ssp_data,
            suptitle="K < " + str(FENIKS.mag_thresh) + " AB",
            savedir=savedir + "/feniks",
        )

        print(
            f"Generating FENIKS ssp error plot for {zbin+1}/{len(feniks_zbins)} z-bin..."
        )
        z_obs = np.median((z_min, z_max))
        psspem.plot_ssp_err_model_delta_mag_vs_wavelength(
            ssp_err_params=param_collection_fit.ssperr_params,
            z_obs=z_obs,
            drn_out=savedir,
            model_nickname="feniks_" + fit_runid + "_" + fit_type + "_z" + str(z_obs),
        )

    # Plot feniks UVJ
    print("Generating FENIKS UVJ plot...")
    plot_uvj(
        ran_key,
        param_collection_fit,
        ssp_data,
        feniks_drn,
        savedir + "/feniks",
        num_halos=10000,
    )

    # Plot feniks ex-situ frac
    print("Generating FENIKS ex-situ frac plot...")
    pdata = pcm.get_plotting_data(
        **param_collection_fit._asdict(), z_min=0.2, z_max=4.0, num_halos=10000
    )
    pcm.plot_ex_situ_fraction(
        pdata=pdata,
        model_nickname="feniks_" + fit_runid + "_" + fit_type,
        drn_out=savedir,
    )

    # Plot feniks Avpop
    print("Generating FENIKS Avpop plot...")
    _ = make_avpop_mono_comparison_plots(
        param_collection_fit.spspop_params.dustpop_params.avpop_params,
        fname=savedir + "/feniks_avpop_mono.png",
    )
