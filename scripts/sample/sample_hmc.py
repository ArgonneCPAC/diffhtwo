#################################################
#  Script based on code from Natalia Rodriguez  #
#################################################
import argparse
import os
from pathlib import Path

import jax
import numpy as np
import yaml
from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.experimental.inference import hmc, utils
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from diffsky.param_utils.load_calib_params import load_param_collection
from dsps import load_ssp_templates
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from diffhtwo.experimental.data_loaders import load_feniks
from diffhtwo.experimental.samplers import hmc_post

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_hmc.yaml")
    args = p.parse_args()

    n_devices = jax.local_device_count()
    print(n_devices)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    feniks_drn = cfg["base_path"] + "/feniks"
    sdss_drn = cfg["base_path"] + "/sdss"
    hizels_drn = Path(cfg["base_path"]) / "hizels"
    ssp_filename = (
        cfg["base_path"]
        + "/ssp_data/ssp_w_emlines/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )
    out_drn = cfg["base_path"] + "/inference"

    imm_path = f"{out_drn}/covariance_matrix.npy"
    if os.path.exists(imm_path):
        imm = np.load(imm_path)
        print(f"loaded precomputed IMM from {imm_path}")
    else:
        # will compute IMM with warmup
        imm = None

    ran_key = jran.key(0)
    ssp_data = load_ssp_templates(fn=ssp_filename)
    ssp_data = lemi.get_subset_emline_data(ssp_data, ["Ba_alpha_6563"])
    halpha_wave_aa = ssp_data.ssp_emline_wave[0]

    feniks_fitting_data = load_feniks.get_feniks_fitting_data(
        feniks_drn,
        ran_key,
        ssp_data,
        num_halos=cfg["feniks"]["num_halos"],
    )

    diffsky_param_file = (
        cfg["base_path"]
        + "/fits/"
        + cfg["run_label"]
        + "/diffsky_"
        + cfg["run_label"]
        + "_"
        + cfg["run_type"]
        + "_param_collection.hdf5"
    )
    param_collection = load_param_collection(diffsky_param_file)

    var_uparams_list = cfg["var_uparams_list"]
    var_params_list = [utils.bounded_name(name) for name in var_uparams_list]
    uparam_collection = dpwm.get_u_param_collection_from_param_collection(
        *param_collection
    )
    uparam_flat = dpwm.unroll_u_param_collection_into_flat_array(*uparam_collection)

    var_uparam_flat = utils.get_var_param_flat_from_param_flat(
        uparam_flat, var_uparams_list
    )

    # List of indices of varied parameters in the flatten namedtuple
    var_flat_idx = utils.compute_varied_params_indices(var_uparam_flat, uparam_flat)

    hmc_settings = {
        "warmup_num_steps": cfg["warmup_num_steps"],
        "max_num_doublings": cfg["max_num_doublings"],
        "target_accept": cfg["target_accept"],
        "initial_step_size": cfg["initial_step_size"],
    }

    # keys
    ran_key = jran.key(cfg["ran_key"])
    warmup_key, sampler_key, init_key, lik_key = jax.random.split(ran_key, 4)
    warmup_keys = jran.split(warmup_key, cfg["num_chains"])
    sampler_keys = jran.split(sampler_key, cfg["num_chains"])

    # Run chain initialization
    chain_inits = hmc.make_chain_inits(
        var_uparam_flat,
        cfg["num_chains"],
        cfg["init_jitter"],
        imm,
        init_key,
    )

    # warmup
    warmup_states, step_sizes, warmup_info, imm_out = hmc.run_warmup(
        hmc_post.flat_logposterior_fn,
        warmup_keys,
        chain_inits,
        hmc_settings,
        inverse_mass_matrix=imm,
        diffsky_params=uparam_flat,
        loss_data=feniks_fitting_data,
        var_flat_idx=var_flat_idx,
        lik_key=lik_key,
    )

    # sampling
    positions, sample_info = hmc.run_sampling(
        hmc_post.flat_logposterior_fn,
        imm_out,
        cfg["max_num_doublings"],
        cfg["num_samples"],
        sampler_keys,
        warmup_states,
        step_sizes,
        diffsky_params=uparam_flat,
        loss_data=feniks_fitting_data,
        var_flat_idx=var_flat_idx,
        lik_key=lik_key,
    )
    positions_path = out_drn + "/" + cfg["run_label"] + "_" + cfg["run_type"]
    np.save(positions_path, positions)

    positions_concat = jax.tree_util.tree_map(lambda x: x.reshape(-1), positions)
    uparam_flat_samples = uparam_flat._replace(**positions_concat._asdict())

    # get uniform depth pytree
    uparam_flat_samples = utils.get_flat_params_all_same_shape(uparam_flat_samples)
    # convert into collection
    uparam_collection_samples = dpwm.get_u_param_collection_from_u_param_array(
        uparam_flat_samples
    )
    # convert into bounded
    param_collection_samples = dpwm.get_param_collection_from_u_param_collection(
        *uparam_collection_samples
    )
    lc_mock.write_diffsky_param_collection_merging(
        out_drn,
        cfg["run_label"] + "_" + cfg["run_type"],
        param_collection_samples,
    )
