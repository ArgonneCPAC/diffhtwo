import os
import time
from datetime import datetime

import jax.numpy as jnp
import numpy as np
from diffsky.data_loaders.hacc_utils import lc_mock
from dsps import load_ssp_templates
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from diffhtwo.experimental import defaults as df
from diffhtwo.experimental import param_utils as pu
from diffhtwo.experimental.data_loaders import load_feniks, load_hizels, load_sdss
from diffhtwo.experimental.latin_hypercube import lh_utils as lhu
from diffhtwo.experimental.optimizers import Np_specphot_opt

if __name__ == "__main__":
    os.environ["BASE_DRN"] = "/home/kzaidi/diffdir"
    base_drn = os.environ["BASE_DRN"]

    # os.environ["SDSS_DRN"] = base_drn + "/sdss"
    os.environ["FENIKS_DRN"] = base_drn + "/feniks"
    # os.environ["HIZELS_DRN"] = base_drn + "/hizels"

    os.environ["SSP_DATA"] = (
        base_drn
        + "/ssp_data/ssp_w_emlines/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )

    feniks_drn = os.environ["FENIKS_DRN"]
    ssp_filename = os.environ["SSP_DATA"]

    # get ssp data
    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    emline_wave_aa = jnp.array(ssp_data.ssp_emline_wave[0])
    emline_wave_table = jnp.array([emline_wave_aa])

    # load feniks data
    ran_key = jran.key(0)
    FENIKS = load_feniks.get_feniks_data(
        feniks_drn,
        ran_key,
        ssp_data,
    )

    # start fit dirs
    start_runid = "run48"
    start_fit_type = "diffstarpop+spspop+merging"
    fit_start_drn = "/Users/kumail/diffdir/fits/" + start_runid + "/"
    param_collection_fit = lc_mock.load_diffsky_param_collection_merging(
        fit_start_drn,
        start_runid + "_" + start_fit_type,
    )
    u_theta_fit = pu.get_u_theta_from_param_collection(param_collection_fit)

    # fit dirs
    fit_runid = "run48"
    fit_type = "diffstarpop+spspop+merging"
    trainable_params = pu.get_trainable_params(fit_type=fit_type)
    fit_save_drn = "/Users/kumail/diffdir/fits/" + fit_runid + "/"
    fit_diagnostics_save_drn = (
        "/Users/kumail/diffdir/fits/" + fit_runid + "/diagnostic_plots/" + fit_type
    )
    os.makedirs(fit_diagnostics_save_drn + "/loss", exist_ok=True)

    feniks_z = np.linspace(df.FENIKS_Z_MIN, df.FENIKS_Z_MAX - 0.5, 4)
    feniks_z_min = feniks_z[:-1]
    feniks_z_max = feniks_z[1:]

    N_IT = 1
    N_STEPS = 2
    STEP_SIZE = 1e-1

    FENIKS_N_Z_BINS = 3
    FENIKS_N_CENTROIDS = 1000

    initial_pts = []
    start = time.time()
    for batch in range(0, N_IT):
        FENIKS = load_feniks.refresh_lh_centroids(FENIKS)

        feniks_z_idx = np.random.choice(
            len(feniks_z_min), FENIKS_N_Z_BINS, replace=False
        )
        print(feniks_z_idx)

        # FENIKS
        feniks_meta_data, feniks_fitting_data = lhu.get_zbins_lh_lc(
            ran_key,
            FENIKS,
            feniks_z_min[feniks_z_idx],
            feniks_z_max[feniks_z_idx],
            ssp_data,
            FENIKS_N_CENTROIDS,
            num_halos=3000,
        )

        loss_hist, u_theta_fit = Np_specphot_opt.fit_N_multi_z(
            u_theta_fit,
            trainable_params,
            ran_key,
            feniks_meta_data,
            feniks_fitting_data,
            n_steps=N_STEPS,
            step_size=STEP_SIZE,
        )

        param_collection_fit = pu.get_param_collection_from_u_theta(u_theta_fit)
        lc_mock.write_diffsky_param_collection_merging(
            fit_save_drn,
            fit_runid + "_" + fit_type,
            param_collection_fit,
        )

        if batch == 0:
            STEPS = np.arange(1, N_STEPS + 1, 1)

            LOSS_HIST = loss_hist

            initial_pts.append((STEPS[0], LOSS_HIST[0]))
        else:
            steps = np.arange(STEPS[-1] + 1, STEPS[-1] + N_STEPS + 1, 1)
            initial_pts.append((steps[0], loss_hist[0]))

            STEPS = np.concatenate((STEPS, steps))
            LOSS_HIST = np.concatenate((LOSS_HIST, loss_hist))

    end = time.time()
    elapsed = end - start
    print(f"Gradient descent took {elapsed/60:.3f} minutes for {N_STEPS*N_IT} steps.")
    print(f"speed: {elapsed/(N_STEPS*N_IT):.3f} s/it")
