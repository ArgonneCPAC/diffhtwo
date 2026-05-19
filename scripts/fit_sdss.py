import argparse
import os
import time
from datetime import datetime

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
from diffhtwo.experimental.data_loaders import load_sdss
from diffhtwo.experimental.defaults import SDSS_Z_MAX, SDSS_Z_MIN
from diffhtwo.experimental.latin_hypercube import lh_utils as lhu
from diffhtwo.experimental.optimizers import Np_specphot_opt

DIFFSTARPOP_GALACTICUS_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash[
    "galacticus_in_plus_ex_situ"
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_sdss.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sdss_drn = cfg["base_path"] + "/sdss"
    ssp_filename = (
        cfg["base_path"]
        + "/ssp_data/ssp_w_emlines/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )

    # get ssp data
    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    emline_wave_aa = jnp.array(ssp_data.ssp_emline_wave[0])
    emline_wave_table = jnp.array([emline_wave_aa])

    # load sdss data
    ran_key = jran.key(0)
    SDSS = load_sdss.get_sdss_data(sdss_drn, ran_key, ssp_data)

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

    sdss_z_min = [SDSS_Z_MIN]
    sdss_z_max = [SDSS_Z_MAX]

    initial_pts = []
    start = time.time()
    for epoch in range(0, cfg["epoch"]["n_it"]):
        print(f'Running Epoch {epoch+1}/{cfg["epoch"]["n_it"]}...')
        sdss = load_sdss.refresh_lh_centroids(SDSS)

        # SDSS
        sdss_meta_data, sdss_fitting_data = lhu.get_zbins_lh_lc(
            ran_key,
            SDSS,
            sdss_z_min,
            sdss_z_max,
            ssp_data,
            cfg["epoch"]["N_centroids"],
            lh_N_z_savedir=fit_diagnostics_save_drn + "/lh_N_z",
            num_halos=cfg["epoch"]["num_halos"],
        )

        loss_hist, u_theta_fit = Np_specphot_opt.fit_N_multi_z(
            u_theta_fit,
            trainable_params,
            ran_key,
            sdss_meta_data,
            sdss_fitting_data,
            n_steps=cfg["epoch"]["n_steps"],
            step_size=cfg["epoch"]["step_size"],
        )

        param_collection_fit = pu.get_param_collection_from_u_theta(u_theta_fit)
        lc_mock.write_diffsky_param_collection_merging(
            fit_save_drn,
            cfg["fit_runid"] + "_" + cfg["fit_type"],
            param_collection_fit,
        )

        if epoch == 0:
            STEPS = np.arange(1, cfg["epoch"]["n_steps"] + 1, 1)

            LOSS_HIST = loss_hist

            initial_pts.append((STEPS[0], LOSS_HIST[0]))
        else:
            steps = np.arange(STEPS[-1] + 1, STEPS[-1] + cfg["epoch"]["n_steps"] + 1, 1)
            initial_pts.append((steps[0], loss_hist[0]))

            STEPS = np.concatenate((STEPS, steps))
            LOSS_HIST = np.concatenate((LOSS_HIST, loss_hist))

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
    ax_loss.scatter(start_step, start_loss, s=50, c="deepskyblue")

    ax_loss.plot(STEPS, LOSS_HIST, c="deepskyblue")
    ax_loss.set_ylabel("Poisson Loss")
    ax_loss.set_xlabel("steps")
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    plt.savefig(fit_diagnostics_save_drn + "/loss/sdss_loss_" + ts + ".png")
    plt.close()
