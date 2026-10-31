from collections import namedtuple

import jax.numpy as jnp
from diffsky import diffndhist_lomem
from jax import jit as jjit

from .phot_kern import mag_kern


@jjit
def add_N_model_to_spaces(
    ran_key,
    param_collection,
    z_data,
    mag_thresh,
):
    gal_weight, mag_weight_ms, mag_weight_q, mag_weight_bursty, phot_data = mag_kern(
        ran_key,
        param_collection,
        z_data.lc_data,
        mag_thresh,
    )
    z_obs = z_data.lc_data.z_obs

    obs_mags_pops = (
        phot_data.obs_mags_ms,
        phot_data.obs_mags_q,
        phot_data.obs_mags_bursty,
    )
    weights_pops = (
        phot_data.weights_ms,
        phot_data.weights_q,
        phot_data.weights_bursty,
    )
    mag_weights_pops = (mag_weight_ms, mag_weight_q, mag_weight_bursty)

    fields = [f for f in z_data._fields if f not in {"z_min", "z_max", "lc_data"}]
    for f in range(len(fields)):
        space = getattr(z_data, fields[f])

        N_model = get_N_model(
            space,
            gal_weight,
            z_obs,
            obs_mags_pops,
            weights_pops,
            mag_weights_pops,
        )

        NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
        new = NewTuple(*space, N_model)
        z_data = z_data._replace(**{fields[f]: new})

    return z_data


@jjit
def get_obs_coords(space, obs_mags):
    col_idx = space.col_idx if "col_idx" in space._fields else None

    if "mag_idx" in space._fields:
        obs_mag = obs_mags[:, space.mag_idx]
        if col_idx is not None:
            # cmd space
            obs_color = obs_mags[:, col_idx[0]] - obs_mags[:, col_idx[1]]
            obs_coords = jnp.vstack((obs_mag, obs_color)).T
        else:
            # md space
            obs_coords = obs_mag.reshape(obs_mag.size, 1)
    else:
        # ccd space
        obs_colors = []
        for c in range(0, len(col_idx) - 1, 2):
            obs_color = obs_mags[:, col_idx[c]] - obs_mags[:, col_idx[c + 1]]
            obs_colors.append(obs_color)
        obs_coords = jnp.array(obs_colors).T

    return obs_coords


@jjit
def get_mag_weight_idx(space):
    parent_cut_idx = space.parent_cut_idx

    if "mag_idx" in space._fields:
        mag_idx = space.mag_idx
        if "col_idx" in space._fields:
            # cmd space
            col_idx = space.col_idx
            mag_weight_idx = [parent_cut_idx, mag_idx, col_idx[0], col_idx[1]]
        else:
            # md space
            mag_weight_idx = [parent_cut_idx, mag_idx]
    else:
        # ccd space
        col_idx = space.col_idx
        mag_weight_idx = [
            parent_cut_idx,
            col_idx[0],
            col_idx[1],
            col_idx[2],
            col_idx[3],
        ]

    return jnp.array(mag_weight_idx)


@jjit
def get_N_model(
    space,
    gal_weight,
    z_obs,
    obs_mags_pops,
    weights_pops,
    mag_weights_pops,
):
    z_sel = ((z_obs > space.zlo) & (z_obs <= space.zhi)).astype(gal_weight.dtype)
    mag_weight_idx = get_mag_weight_idx(space)

    N_model = 0.0
    for obs_mags_pop, weight_pop, mag_weight_pop in zip(
        obs_mags_pops, weights_pops, mag_weights_pops
    ):
        obs_coords = get_obs_coords(space, obs_mags_pop)
        mag_weight_space = jnp.prod(mag_weight_pop[:, mag_weight_idx], axis=1)

        N_model_pop = diffndhist_lomem.tw_ndhist_weighted(
            obs_coords,
            space.sig,
            gal_weight * z_sel * weight_pop * mag_weight_space,
            space.bin_lo,
            space.bin_hi,
        )
        N_model = N_model + N_model_pop

    return N_model
