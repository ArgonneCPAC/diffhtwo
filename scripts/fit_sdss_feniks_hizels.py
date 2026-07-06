import argparse
import os
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.merging.merging_model import DEFAULT_MERGE_PARAMS
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_PARAMS
from diffsky.ssp_err_model.defaults import ZERO_SSPERR_PARAMS
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)
from dsps import load_ssp_templates
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from diffhtwo.experimental import param_utils as pu
from diffhtwo.experimental.data_loaders import load_feniks, load_hizels, load_sdss
from diffhtwo.experimental.optimizers import Np_specphot_opt

DIFFSTARPOP_GALACTICUS_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash[
    "galacticus_in_plus_ex_situ"
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_sdss_feniks_hizels.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sdss_drn = cfg["base_path"] + "/sdss"
    feniks_drn = cfg["base_path"] + "/feniks"
    hizels_drn = Path(cfg["base_path"] + "/hizels")
    ssp_filename = (
        cfg["base_path"]
        + "/ssp_data/ssp_w_emlines/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )

    # get ssp data
    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    halpha_wave_aa = jnp.array(ssp_data.ssp_emline_wave[0])

    # start fit dirs
    fit_start_drn = cfg["base_path"] + "/fits/" + cfg["start_runid"] + "/"
    param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
        fit_start_drn,
        cfg["start_runid"] + "_" + cfg["start_fit_type"],
    )
    if cfg["defaults"]["diffstarpop"]:
        param_collection_fit = param_collection_fit._replace(
            diffstarpop_params=DIFFSTARPOP_GALACTICUS_exsitu
        )
    if cfg["defaults"]["spspop"]:
        param_collection_fit = param_collection_fit._replace(
            spspop_params=DEFAULT_SPSPOP_PARAMS
        )
    if cfg["defaults"]["ssperr"]:
        param_collection_fit = param_collection_fit._replace(
            ssperr_params=ZERO_SSPERR_PARAMS
        )
    if cfg["defaults"]["merging"]:
        param_collection_fit = param_collection_fit._replace(
            merging_params=DEFAULT_MERGE_PARAMS
        )

    u_theta_fit = pu.get_u_theta_from_param_collection(param_collection_fit)

    # fit dirs
    trainable_params = pu.get_trainable_params(fit_type=cfg["fit_type"])
    fit_save_drn = cfg["base_path"] + "/fits/" + cfg["fit_runid"] + "/"
    fit_diagnostics_save_drn = (
        cfg["base_path"]
        + "/fits/"
        + cfg["fit_runid"]
        + "/diagnostic_plots/"
        + cfg["fit_type"]
    )
    os.makedirs(fit_diagnostics_save_drn + "/loss", exist_ok=True)
    os.makedirs(fit_diagnostics_save_drn + "/lh_N_z", exist_ok=True)

    os.system(f"cp {args.config} {fit_diagnostics_save_drn}")

    with open(fit_diagnostics_save_drn + "/" + args.config) as f:
        cfg = yaml.safe_load(f)

    initial_pts = []
    start = time.time()
    ran_key = jran.key(0)
    for epoch in range(0, cfg["epoch"]["n_it"]):
        print(f'Running Epoch {epoch+1}/{cfg["epoch"]["n_it"]}...')

        # load sdss data
        sdss = load_sdss.get_sdss_data(
            sdss_drn,
            ran_key,
            ssp_data,
            num_halos_coarse_zbins=cfg["sdss"]["num_halos_coarse_zbins"],
            num_halos_fine_zbins=cfg["sdss"]["num_halos_fine_zbins"],
        )
        remove = {"dataset_dim_labels", "mags_labels"}
        SdssFitting = namedtuple("Sdss", [s for s in sdss._fields if s not in remove])
        sdss_fitting_data = SdssFitting(
            **{s: getattr(sdss, s) for s in SdssFitting._fields}
        )

        # load feniks data
        feniks_fitting_data = load_feniks.get_feniks_fitting_data(
            feniks_drn,
            ran_key,
            ssp_data,
            num_halos_coarse_zbins=cfg["feniks"]["num_halos_coarse_zbins"],
            num_halos_fine_zbins=cfg["feniks"]["num_halos_fine_zbins"],
        )

        # load hizels data
        hizels_fitting_data = load_hizels.get_hizels_data(
            hizels_drn,
            ran_key,
            ssp_data,
            feniks_fitting_data.filter_info.tcurves,
            halpha_wave_aa,
            num_halos=cfg["hizels"]["num_halos"],
        )

        (
            loss_hist,
            loss_sdss_hist,
            loss_feniks_hist,
            loss_hizels_hist,
            u_theta_fit,
        ) = Np_specphot_opt.fit_sdss_feniks_hizels(
            u_theta_fit,
            trainable_params,
            ran_key,
            sdss_fitting_data,
            feniks_fitting_data,
            hizels_fitting_data,
            n_steps=cfg["epoch"]["n_steps"],
            step_size=cfg["epoch"]["step_size"],
            w_sdss=cfg["w_sdss"],
            w_feniks=cfg["w_feniks"],
            w_hizels=cfg["w_hizels"],
        )

        jax.clear_caches()

        param_collection_fit = pu.get_param_collection_from_u_theta(u_theta_fit)
        lc_mock.write_diffsky_param_collection_merging(
            fit_save_drn,
            cfg["fit_runid"] + "_" + cfg["fit_type"],
            param_collection_fit,
        )
        if epoch == 0:
            STEPS = np.arange(1, cfg["epoch"]["n_steps"] + 1, 1)
            LOSS_HIST = loss_hist
            LOSS_SDSS_HIST = loss_sdss_hist
            LOSS_FENIKS_HIST = loss_feniks_hist
            LOSS_HIZELS_HIST = loss_hizels_hist
            initial_pts.append((STEPS[0], LOSS_HIST[0]))
        else:
            steps = np.arange(STEPS[-1] + 1, STEPS[-1] + cfg["epoch"]["n_steps"] + 1, 1)
            initial_pts.append((steps[0], loss_hist[0]))
            STEPS = np.concatenate((STEPS, steps))
            LOSS_HIST = np.concatenate((LOSS_HIST, loss_hist))
            LOSS_SDSS_HIST = np.concatenate((LOSS_SDSS_HIST, loss_sdss_hist))
            LOSS_FENIKS_HIST = np.concatenate((LOSS_FENIKS_HIST, loss_feniks_hist))
            LOSS_HIZELS_HIST = np.concatenate((LOSS_HIZELS_HIST, loss_hizels_hist))
    end = time.time()
    elapsed = end - start
    print(
        f'Gradient descent took {elapsed/60:.3f} minutes for {cfg["epoch"]["n_steps"]*cfg["epoch"]["n_it"]} steps.'
    )
    print(f'speed: {elapsed/(cfg["epoch"]["n_steps"]*cfg["epoch"]["n_it"]):.3f} s/it')
    # gradient descent figure
    fig_loss, ax_loss = plt.subplots(1)
    start_step = [s[0] for s in initial_pts]
    start_loss = [s[1] for s in initial_pts]
    ax_loss.scatter(start_step, start_loss, s=50, c="k")
    ax_loss.plot(STEPS, LOSS_HIST, c="k", label="total")
    ax_loss.plot(
        STEPS,
        LOSS_SDSS_HIST,
        c="#0a7a80",
        linestyle="--",
        alpha=0.7,
        label="sdss",
    )
    ax_loss.plot(
        STEPS,
        LOSS_FENIKS_HIST,
        c="#c87820",
        linestyle="--",
        alpha=0.7,
        label="feniks",
    )
    ax_loss.plot(
        STEPS,
        LOSS_HIZELS_HIST,
        c="#DA291C",
        linestyle="--",
        alpha=0.7,
        label="hizels",
    )
    ax_loss.legend()
    ax_loss.set_ylabel("Poisson Negative Log-Likelihood")
    ax_loss.set_xlabel("steps")
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    plt.savefig(fit_diagnostics_save_drn + "/loss/loss_" + ts + ".png")
    plt.close()
