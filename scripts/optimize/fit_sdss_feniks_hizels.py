import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.merging.merging_model import DEFAULT_MERGE_PARAMS
from diffsky.param_utils.diffsky_param_wrapper_merging import (
    check_param_collection_is_ok,
)
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
from diffhtwo.experimental.optimizers import Np_photline_opt

DIFFSTARPOP_PARAMS = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]


def plot_loss(steps, loss, loss_sdss, loss_feniks, loss_hizels, initial_pts, path):
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 3), constrained_layout=True)
    ax[0].scatter(*zip(*initial_pts), s=50, c="k")
    ax[0].plot(steps, loss, c="k", label="total")
    for a, y, c, lab in zip(
        ax[1:],
        (loss_sdss, loss_feniks, loss_hizels),
        ("#0a7a80", "#c87820", "#DA291C"),
        ("sdss", "feniks", "hizels"),
    ):
        a.plot(steps, y, c=c, linestyle="--", alpha=0.7, label=lab)
    for a in ax:
        a.legend()
    fig.supylabel("Poisson Negative Log-Likelihood")
    fig.supxlabel("steps")
    fig.savefig(path)
    plt.close(fig)


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

    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    halpha_wave_aa = jnp.array(ssp_data.ssp_emline_wave[0])

    fit_save_drn = cfg["base_path"] + "/fits/" + cfg["fit_runid"] + "/"
    fit_diagnostics_save_drn = (
        cfg["base_path"]
        + "/fits/"
        + cfg["fit_runid"]
        + "/diagnostic_plots/"
        + cfg["fit_type"]
    )
    fit_name = f'{cfg["fit_runid"]}_{cfg["fit_type"]}'
    loss_file = f"{fit_diagnostics_save_drn}/loss/loss_history.npz"
    os.makedirs(fit_save_drn, exist_ok=True)
    os.makedirs(fit_diagnostics_save_drn + "/loss", exist_ok=True)

    if os.path.exists(loss_file):
        d = np.load(loss_file)
        start_epoch = int(d["epoch"])
        STEPS = d["steps"]
        LOSS_HIST = d["loss"]
        LOSS_SDSS_HIST = d["loss_sdss"]
        LOSS_FENIKS_HIST = d["loss_feniks"]
        LOSS_HIZELS_HIST = d["loss_hizels"]
        initial_pts = [tuple(x) for x in d["initial_pts"]]
        param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
            fit_save_drn, f"{fit_name}_ep{start_epoch}"
        )
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        STEPS = np.array([], dtype=int)
        LOSS_HIST = LOSS_SDSS_HIST = LOSS_FENIKS_HIST = LOSS_HIZELS_HIST = np.array([])
        initial_pts = []
        fit_start_drn = cfg["base_path"] + "/fits/" + cfg["start_runid"] + "/"
        param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
            fit_start_drn,
            cfg["start_runid"] + "_" + cfg["start_fit_type"],
        )
        if cfg["defaults"]["diffstarpop"]:
            param_collection_fit = param_collection_fit._replace(
                diffstarpop_params=DIFFSTARPOP_PARAMS
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
        lc_mock.write_diffsky_param_collection_merging(
            fit_save_drn, f"{fit_name}_ep0", param_collection_fit
        )

    u_theta_fit = pu.get_u_theta_from_param_collection(param_collection_fit)
    trainable_params = pu.get_trainable_params(fit_type=cfg["fit_type"])
    os.system(f"cp {args.config} {fit_diagnostics_save_drn}")

    with open(fit_diagnostics_save_drn + "/" + args.config) as f:
        cfg = yaml.safe_load(f)

    with open("../diagnose/config_diagnostics.yaml") as df:
        cfg_d = yaml.safe_load(df)
    cfg_d["model_drn"] = fit_save_drn
    cfg_d["model_nickname"] = fit_name
    cfg_d["fit_diagnostics_save_drn"] = fit_diagnostics_save_drn
    cfg_d["lgmp_min"] = cfg["lgmp_min"]
    cfg_d["lgmp_max"] = cfg["lgmp_max"]
    cfg_d["logHa_flux_limit"] = cfg["hizels"]["logHa_flux_limit"]

    new_config_diag_path = os.path.join(
        fit_diagnostics_save_drn, "config_diagnostics.yaml"
    )
    with open(new_config_diag_path, "w") as df:
        yaml.dump(cfg_d, df, default_flow_style=False, sort_keys=False)

    n_it = cfg["epoch"]["n_it"]
    n_steps = cfg["epoch"]["n_steps"]
    end_epoch = start_epoch + n_it

    start = time.time()
    for epoch in range(start_epoch, end_epoch):
        ran_key = jran.fold_in(jran.key(0), epoch)
        sdss_key, feniks_key, hizels_key, fit_key = jran.split(ran_key, 4)
        print(f"Running Epoch {epoch+1}/{end_epoch}...")

        sdss_fitting_data = load_sdss.get_sdss_fitting_data(
            sdss_drn,
            sdss_key,
            ssp_data,
            num_halos=cfg["sdss"]["num_halos"],
            lgmp_min=cfg["lgmp_min"],
            lgmp_max=cfg["lgmp_max"],
        )

        feniks_fitting_data = load_feniks.get_feniks_fitting_data(
            feniks_drn,
            feniks_key,
            ssp_data,
            num_halos=cfg["feniks"]["num_halos"],
            lgmp_min=cfg["lgmp_min"],
            lgmp_max=cfg["lgmp_max"],
        )

        hizels_fitting_data = load_hizels.get_hizels_data(
            hizels_drn,
            hizels_key,
            ssp_data,
            feniks_fitting_data.filter_info.tcurves,
            halpha_wave_aa,
            logflux_limit=cfg["hizels"]["logHa_flux_limit"],
            logLHa_halo_limit=cfg["hizels"]["logLHa_halo_limit"],
            num_halos=cfg["hizels"]["num_halos"],
            lgmp_min=cfg["lgmp_min"],
            lgmp_max=cfg["lgmp_max"],
        )

        (
            loss_hist,
            loss_sdss_hist,
            loss_feniks_hist,
            loss_hizels_hist,
            u_theta_fit,
        ) = Np_photline_opt.fit_sdss_feniks_hizels(
            u_theta_fit,
            trainable_params,
            fit_key,
            sdss_fitting_data,
            feniks_fitting_data,
            hizels_fitting_data,
            n_steps=n_steps,
            step_size=cfg["epoch"]["step_size"],
            w_sdss=cfg["w_sdss"],
            w_feniks=cfg["w_feniks"],
            w_hizels=cfg["w_hizels"],
        )

        jax.clear_caches()

        param_collection_fit = pu.get_param_collection_from_u_theta(u_theta_fit)
        assert check_param_collection_is_ok(param_collection_fit)
        lc_mock.write_diffsky_param_collection_merging(
            fit_save_drn, f"{fit_name}_ep{epoch+1}", param_collection_fit
        )
        lc_mock.write_diffsky_param_collection_merging(
            fit_save_drn, fit_name, param_collection_fit
        )

        steps = np.arange(len(STEPS) + 1, len(STEPS) + n_steps + 1)
        initial_pts.append((steps[0], loss_hist[0]))
        STEPS = np.concatenate((STEPS, steps))
        LOSS_HIST = np.concatenate((LOSS_HIST, loss_hist))
        LOSS_SDSS_HIST = np.concatenate((LOSS_SDSS_HIST, loss_sdss_hist))
        LOSS_FENIKS_HIST = np.concatenate((LOSS_FENIKS_HIST, loss_feniks_hist))
        LOSS_HIZELS_HIST = np.concatenate((LOSS_HIZELS_HIST, loss_hizels_hist))

        np.savez(
            loss_file,
            epoch=epoch + 1,
            steps=STEPS,
            loss=LOSS_HIST,
            loss_sdss=LOSS_SDSS_HIST,
            loss_feniks=LOSS_FENIKS_HIST,
            loss_hizels=LOSS_HIZELS_HIST,
            initial_pts=np.array(initial_pts),
        )

        ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plot_loss(
            STEPS,
            LOSS_HIST,
            LOSS_SDSS_HIST,
            LOSS_FENIKS_HIST,
            LOSS_HIZELS_HIST,
            initial_pts,
            f"{fit_diagnostics_save_drn}/loss/loss_epoch{epoch+1}_{ts}.png",
        )
    elapsed = time.time() - start
    print(f"Gradient descent took {elapsed/60:.3f} minutes for {n_steps*n_it} steps.")
    print(f"speed: {elapsed/(n_steps*n_it):.3f} s/it")

    os.system(
        f"python ../diagnose/generate_diagnostic_plots.py --config {new_config_diag_path}"
    )
