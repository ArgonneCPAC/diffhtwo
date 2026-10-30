from collections import namedtuple

import jax.numpy as jnp
from diffsky import diffndhist_lomem
from jax import jit as jjit

from .phot_kern import mag_kern


@jjit
def N_colors_mags(
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

    obs_mags_ms = phot_data.obs_mags_ms
    obs_mags_q = phot_data.obs_mags_q
    obs_mags_bursty = phot_data.obs_mags_bursty

    weight_ms = phot_data.weights_ms
    weight_q = phot_data.weights_q
    weight_bursty = phot_data.weights_bursty

    fields = [f for f in z_data._fields if f not in {"z_min", "z_max", "lc_data"}]
    for f in range(len(fields)):
        space = getattr(z_data, fields[f])

        if "mag_idx" in space._fields:
            if "col_idx" in space._fields:
                # Magnitude-Color space
                parent_cut_idx = space.parent_cut_idx
                col_idx = space.col_idx
                mag_idx = space.mag_idx
                z_sel = ((z_obs > space.zlo) & (z_obs <= space.zhi)).astype(
                    gal_weight.dtype
                )

                # ms
                mag_ms = obs_mags_ms[:, mag_idx]
                obs_color_ms = obs_mags_ms[:, col_idx[0]] - obs_mags_ms[:, col_idx[1]]
                mag_weight_space_ms = (
                    mag_weight_ms[:, parent_cut_idx]
                    * mag_weight_ms[:, mag_idx]
                    * mag_weight_ms[:, col_idx[0]]
                    * mag_weight_ms[:, col_idx[1]]
                )
                obs_mag_color_ms = jnp.vstack((mag_ms, obs_color_ms)).T
                N_model_ms = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_color_ms,
                    space.sig,
                    gal_weight * z_sel * weight_ms * mag_weight_space_ms,
                    space.bin_lo,
                    space.bin_hi,
                )

                # q
                mag_q = obs_mags_q[:, mag_idx]
                obs_color_q = obs_mags_q[:, col_idx[0]] - obs_mags_q[:, col_idx[1]]
                mag_weight_space_q = (
                    mag_weight_q[:, parent_cut_idx]
                    * mag_weight_q[:, mag_idx]
                    * mag_weight_q[:, col_idx[0]]
                    * mag_weight_q[:, col_idx[1]]
                )
                obs_mag_color_q = jnp.vstack((mag_q, obs_color_q)).T
                N_model_q = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_color_q,
                    space.sig,
                    gal_weight * z_sel * weight_q * mag_weight_space_q,
                    space.bin_lo,
                    space.bin_hi,
                )

                # bursty
                mag_bursty = obs_mags_bursty[:, mag_idx]
                obs_color_bursty = (
                    obs_mags_bursty[:, col_idx[0]] - obs_mags_bursty[:, col_idx[1]]
                )
                mag_weight_space_bursty = (
                    mag_weight_bursty[:, parent_cut_idx]
                    * mag_weight_bursty[:, mag_idx]
                    * mag_weight_bursty[:, col_idx[0]]
                    * mag_weight_bursty[:, col_idx[1]]
                )
                obs_mag_color_bursty = jnp.vstack((mag_bursty, obs_color_bursty)).T
                N_model_bursty = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_color_bursty,
                    space.sig,
                    gal_weight * z_sel * weight_bursty * mag_weight_space_bursty,
                    space.bin_lo,
                    space.bin_hi,
                )

                N_model = N_model_ms + N_model_q + N_model_bursty

                NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
                new = NewTuple(*space, N_model)
                z_data = z_data._replace(**{fields[f]: new})
            else:
                # Apparent Magnitude space
                parent_cut_idx = space.parent_cut_idx
                mag_idx = space.mag_idx
                z_sel = ((z_obs > space.zlo) & (z_obs <= space.zhi)).astype(
                    gal_weight.dtype
                )

                # ms
                obs_mag_ms = obs_mags_ms[:, mag_idx]
                obs_mag_ms = obs_mag_ms.reshape(obs_mag_ms.size, 1)

                mag_weight_space_ms = (
                    mag_weight_ms[:, mag_idx] * mag_weight_ms[:, parent_cut_idx]
                )

                N_model_ms = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_ms,
                    space.sig,
                    gal_weight * z_sel * weight_ms * mag_weight_space_ms,
                    space.bin_lo,
                    space.bin_hi,
                )

                # q
                obs_mag_q = obs_mags_q[:, mag_idx]
                obs_mag_q = obs_mag_q.reshape(obs_mag_q.size, 1)

                mag_weight_space_q = (
                    mag_weight_q[:, mag_idx] * mag_weight_q[:, parent_cut_idx]
                )

                N_model_q = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_q,
                    space.sig,
                    gal_weight * z_sel * weight_q * mag_weight_space_q,
                    space.bin_lo,
                    space.bin_hi,
                )

                # bursty
                obs_mag_bursty = obs_mags_bursty[:, mag_idx]
                obs_mag_bursty = obs_mag_bursty.reshape(obs_mag_bursty.size, 1)

                mag_weight_space_bursty = (
                    mag_weight_bursty[:, mag_idx] * mag_weight_bursty[:, parent_cut_idx]
                )

                N_model_bursty = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_bursty,
                    space.sig,
                    gal_weight * z_sel * weight_bursty * mag_weight_space_bursty,
                    space.bin_lo,
                    space.bin_hi,
                )

                N_model = N_model_ms + N_model_q + N_model_bursty

                NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
                new = NewTuple(*space, N_model)
                z_data = z_data._replace(**{fields[f]: new})

        else:
            # Color-Color space
            parent_cut_idx = space.parent_cut_idx
            col_idx = space.col_idx
            z_sel = ((z_obs > space.zlo) & (z_obs <= space.zhi)).astype(
                gal_weight.dtype
            )

            # ms
            obs_colors_ms = []
            for c in range(0, len(col_idx) - 1, 2):
                obs_color_ms = (
                    obs_mags_ms[:, col_idx[c]] - obs_mags_ms[:, col_idx[c + 1]]
                )
                obs_colors_ms.append(obs_color_ms)
            obs_colors_ms = jnp.array(obs_colors_ms).T

            mag_weight_space_ms = (
                mag_weight_ms[:, parent_cut_idx]
                * mag_weight_ms[:, col_idx[0]]
                * mag_weight_ms[:, col_idx[1]]
                * mag_weight_ms[:, col_idx[2]]
                * mag_weight_ms[:, col_idx[3]]
            )

            N_model_ms = diffndhist_lomem.tw_ndhist_weighted(
                obs_colors_ms,
                space.sig,
                gal_weight * z_sel * weight_ms * mag_weight_space_ms,
                space.bin_lo,
                space.bin_hi,
            )

            # q
            obs_colors_q = []
            for c in range(0, len(col_idx) - 1, 2):
                obs_color_q = obs_mags_q[:, col_idx[c]] - obs_mags_q[:, col_idx[c + 1]]
                obs_colors_q.append(obs_color_q)
            obs_colors_q = jnp.array(obs_colors_q).T

            mag_weight_space_q = (
                mag_weight_q[:, parent_cut_idx]
                * mag_weight_q[:, col_idx[0]]
                * mag_weight_q[:, col_idx[1]]
                * mag_weight_q[:, col_idx[2]]
                * mag_weight_q[:, col_idx[3]]
            )

            N_model_q = diffndhist_lomem.tw_ndhist_weighted(
                obs_colors_q,
                space.sig,
                gal_weight * z_sel * weight_q * mag_weight_space_q,
                space.bin_lo,
                space.bin_hi,
            )

            # bursty
            obs_colors_bursty = []
            for c in range(0, len(col_idx) - 1, 2):
                obs_color_bursty = (
                    obs_mags_bursty[:, col_idx[c]] - obs_mags_bursty[:, col_idx[c + 1]]
                )
                obs_colors_bursty.append(obs_color_bursty)
            obs_colors_bursty = jnp.array(obs_colors_bursty).T

            mag_weight_space_bursty = (
                mag_weight_bursty[:, parent_cut_idx]
                * mag_weight_bursty[:, col_idx[0]]
                * mag_weight_bursty[:, col_idx[1]]
                * mag_weight_bursty[:, col_idx[2]]
                * mag_weight_bursty[:, col_idx[3]]
            )

            N_model_bursty = diffndhist_lomem.tw_ndhist_weighted(
                obs_colors_bursty,
                space.sig,
                gal_weight * z_sel * weight_bursty * mag_weight_space_bursty,
                space.bin_lo,
                space.bin_hi,
            )

            N_model = N_model_ms + N_model_q + N_model_bursty

            NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
            new = NewTuple(*space, N_model)
            z_data = z_data._replace(**{fields[f]: new})

    return z_data
