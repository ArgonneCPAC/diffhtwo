import argparse
import os
import time
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from dsps import load_ssp_templates
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from diffhtwo.experimental import param_utils as pu
from diffhtwo.experimental.data_loaders import load_minerva
from diffhtwo.experimental.optimizers.Np_minerva_opt import fit_minerva

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_minerva.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    minerva_phot_drn = cfg["minerva_path"] + "/n3.0_v1.2"
    minerva_halpha_drn = cfg["minerva_path"] + "/halpha"

    ssp_filename = (
        cfg["base_path"]
        + "/ssp_data/ssp_w_emlines/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )

    # get ssp data
    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    halpha_wave_aa = ssp_data.ssp_emline_wave[0]

    # start fit dirs
    fit_start_drn = cfg["base_path"] + "/fits/" + cfg["start_runid"] + "/"
    param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
        fit_start_drn,
        cfg["start_runid"] + "_" + cfg["start_fit_type"],
    )
    if cfg["defaults"]["diffstarpop"]:
        param_collection_fit = param_collection_fit._replace(
            diffstarpop_params=DEFAULT_PARAM_COLLECTION.diffstarpop_params
        )
    if cfg["defaults"]["spspop"]:
        param_collection_fit = param_collection_fit._replace(
            spspop_params=DEFAULT_PARAM_COLLECTION.spspop_params
        )
    if cfg["defaults"]["ssperr"]:
        param_collection_fit = param_collection_fit._replace(
            ssperr_params=DEFAULT_PARAM_COLLECTION.ssperr_params
        )
    if cfg["defaults"]["merging"]:
        param_collection_fit = param_collection_fit._replace(
            merging_params=DEFAULT_PARAM_COLLECTION.merging_params
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

        # load phot data
        minerva_phot = load_minerva.get_minerva_phot_fitting_data(
            minerva_phot_drn,
            ran_key,
            ssp_data,
            num_halos=cfg["phot"]["num_halos"],
            lgmp_min=cfg["lgmp_min"],
            lgmp_max=cfg["lgmp_max"],
        )

        # load halpha data
        minerva_halpha = load_minerva.get_minerva_halpha(
            minerva_halpha_drn,
            minerva_phot_drn,
            ran_key,
            ssp_data,
        )

        (loss_hist, loss_phot_hist, loss_halpha_hist, u_theta_fit) = fit_minerva(
            u_theta_fit,
            trainable_params,
            ran_key,
            minerva_phot,
            minerva_halpha,
            halpha_wave_aa,
            n_steps=cfg["epoch"]["n_steps"],
            step_size=cfg["epoch"]["step_size"],
            w_phot=cfg["w_phot"],
            w_halpha=cfg["w_halpha"],
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
            LOSS_PHOT_HIST = loss_phot_hist
            LOSS_HALPHA_HIST = loss_halpha_hist

            initial_pts.append((STEPS[0], LOSS_HIST[0]))
        else:
            steps = np.arange(STEPS[-1] + 1, STEPS[-1] + cfg["epoch"]["n_steps"] + 1, 1)
            initial_pts.append((steps[0], loss_hist[0]))
            STEPS = np.concatenate((STEPS, steps))
            LOSS_HIST = np.concatenate((LOSS_HIST, loss_hist))
            LOSS_PHOT_HIST = np.concatenate((LOSS_PHOT_HIST, loss_phot_hist))
            LOSS_HALPHA_HIST = np.concatenate((LOSS_HALPHA_HIST, loss_halpha_hist))

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
        LOSS_PHOT_HIST,
        c="#0a7a80",
        linestyle="--",
        alpha=0.7,
        label="phot",
    )
    ax_loss.plot(
        STEPS,
        LOSS_HALPHA_HIST,
        c="#c87820",
        linestyle="--",
        alpha=0.7,
        label="halpha",
    )

    ax_loss.legend()
    ax_loss.set_ylabel("Poisson Negative Log-Likelihood")
    ax_loss.set_xlabel("steps")
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    plt.savefig(fit_diagnostics_save_drn + "/loss/loss_" + ts + ".png")
    plt.close()
