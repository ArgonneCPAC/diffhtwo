from collections import namedtuple
from functools import partial

import jax.numpy as jnp
from diffsky import diffndhist_lomem
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from .phot_kern import get_colors_mags, mag_kern


@jjit
def N_colors_mags(
    ran_key,
    param_collection,
    z_data,
    mag_thresh,
    frac_cat,
):
    obs_mags_weighted, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        param_collection,
        z_data.lc_data,
        mag_thresh,
        frac_cat,
    )
    fields = z_data._fields[3:]
    mag_thresh = jnp.array(mag_thresh)
    for f in range(0, len(fields)):
        space = getattr(z_data, fields[f])

        if isinstance(space, list):
            # Colors conditioned on mag space
            new_list = []
            for s in range(0, len(space)):
                space_n = space[s]
                col_idx = space_n.col_idx

                # get cond weight
                obs_mags_weighted_cond = obs_mags_weighted[:, space_n.cond_idx]
                cond = (obs_mags_weighted_cond > space_n.cond_min) & (
                    obs_mags_weighted_cond <= space_n.cond_max
                )
                weight = jnp.where(cond, gal_weight, 0.0)

                obs_color = (
                    obs_mags_weighted[:, col_idx[0]] - obs_mags_weighted[:, col_idx[1]]
                )
                obs_color = obs_color.reshape(obs_color.size, 1)

                N_model = diffndhist_lomem.tw_ndhist_weighted(
                    obs_color,
                    space_n.sig,
                    weight,
                    space_n.bin_lo,
                    space_n.bin_hi,
                )

                NewTuple = namedtuple(
                    type(space_n).__name__, [*space_n._fields, "N_model"]
                )
                new_list.append(NewTuple(*space_n, N_model))
            z_data = z_data._replace(**{fields[f]: new_list})

        elif "mag_idx" in space._fields:
            if "col_idx" in space._fields:
                # Magnitude-Color space
                col_idx = space.col_idx
                mag_idx = space.mag_idx

                mag = obs_mags_weighted[:, mag_idx]
                obs_color = (
                    obs_mags_weighted[:, col_idx[0]] - obs_mags_weighted[:, col_idx[1]]
                )
                obs_mag_color = jnp.vstack((mag, obs_color)).T

                N_model = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag_color,
                    space.sig,
                    gal_weight,
                    space.bin_lo,
                    space.bin_hi,
                )

                NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
                new = NewTuple(*space, N_model)
                z_data = z_data._replace(**{fields[f]: new})
            else:
                # Apparent Magnitude space
                mag_idx = space.mag_idx
                obs_mag = obs_mags_weighted[:, mag_idx]
                obs_mag = obs_mag.reshape(obs_mag.size, 1)

                N_model = diffndhist_lomem.tw_ndhist_weighted(
                    obs_mag,
                    space.sig,
                    gal_weight,
                    space.bin_lo,
                    space.bin_hi,
                )

                NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
                new = NewTuple(*space, N_model)
                z_data = z_data._replace(**{fields[f]: new})

        else:
            # Color-Color space
            col_idx = space.col_idx
            obs_colors = []
            for c in range(0, len(col_idx) - 1, 2):
                obs_color = (
                    obs_mags_weighted[:, col_idx[c]]
                    - obs_mags_weighted[:, col_idx[c + 1]]
                )
                obs_colors.append(obs_color)
            obs_colors = jnp.array(obs_colors).T

            N_model = diffndhist_lomem.tw_ndhist_weighted(
                obs_colors,
                space.sig,
                gal_weight,
                space.bin_lo,
                space.bin_hi,
            )

            NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
            new = NewTuple(*space, N_model)
            z_data = z_data._replace(**{fields[f]: new})

    return z_data


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def N_colors_mags_lh(
    ran_key,
    meta_data,
    fitting_data,
    param_collection,
    redshift_as_last_dimension_in_lh=True,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    obs_color_mag, weights, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        fitting_data.lc_data,
        meta_data.col_idx,
        meta_data.mag_idx,
        meta_data.mag_thresh,
        meta_data.frac_cat,
    )

    # calculate number density in LH bins
    sig = jnp.zeros(fitting_data.lh_centroids.shape) + (fitting_data.d_centroids / 2)
    lh_centroids_lo = fitting_data.lh_centroids - (fitting_data.d_centroids / 2)
    lh_centroids_hi = fitting_data.lh_centroids + (fitting_data.d_centroids / 2)

    if redshift_as_last_dimension_in_lh:
        z_obs = fitting_data.lc_data.z_obs.reshape(fitting_data.lc_data.z_obs.size, 1)
        obs_color_mag = jnp.hstack((obs_color_mag, z_obs))

        N = diffndhist_lomem.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            weights,
            lh_centroids_lo,
            lh_centroids_hi,
        )

    else:
        N = diffndhist_lomem.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            weights,
            lh_centroids_lo,
            lh_centroids_hi,
        )

    return N
