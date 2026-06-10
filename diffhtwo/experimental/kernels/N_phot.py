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
    obs_mags, gal_weight, phot_kern_results = mag_kern(
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
            new_list = []
            for s in range(0, len(space)):
                space_n = space[s]
                col_idx = space_n.col_idx

                # get cond weight
                obs_mags_cond = obs_mags[:, space_n.cond_idx]
                cond = (obs_mags_cond > space_n.cond_min) & (
                    obs_mags_cond <= space_n.cond_max
                )
                weight = jnp.where(cond, gal_weight, 0.0)

                # get mag_sel weight
                for c in range(0, len(col_idx)):
                    mag_sel = obs_mags[:, col_idx[c]] < mag_thresh[col_idx[c]]
                    weight *= jnp.where(mag_sel, 1.0, 0.0)

                obs_color = obs_mags[:, col_idx[0]] - obs_mags[:, col_idx[1]]
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
            mag_idx = space.mag_idx
            obs_mag = obs_mags[:, mag_idx]
            obs_mag = obs_mag.reshape(obs_mag.size, 1)

            # get mag_sel weight
            mag_sel = obs_mags[:, mag_idx] < mag_thresh[mag_idx]
            weight = jnp.where(mag_sel, gal_weight, 0.0)

            N_model = diffndhist_lomem.tw_ndhist_weighted(
                obs_mag,
                space.sig,
                weight,
                space.bin_lo,
                space.bin_hi,
            )

            NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
            new = NewTuple(*space, N_model)
            z_data = z_data._replace(**{fields[f]: new})

        else:
            col_idx = space.col_idx
            obs_colors = []
            for c in range(0, len(col_idx) - 1):
                obs_color = obs_mags[:, col_idx[c]] - obs_mags[:, col_idx[c + 1]]
                obs_colors.append(obs_color)
            obs_colors = jnp.array(obs_colors).T

            # get mag_sel weight
            weight = gal_weight.copy()
            for c in range(0, len(col_idx)):
                mag_sel = obs_mags[:, col_idx[c]] < mag_thresh[col_idx[c]]
                weight *= jnp.where(mag_sel, 1.0, 0.0)

            N_model = diffndhist_lomem.tw_ndhist_weighted(
                obs_colors,
                space.sig,
                weight,
                space.bin_lo,
                space.bin_hi,
            )

            NewTuple = namedtuple(type(space).__name__, [*space._fields, "N_model"])
            new = NewTuple(*space, N_model)
            z_data = z_data._replace(**{fields[f]: new})

    return z_data


@jjit
def N_mags_1d(
    ran_key,
    param_collection,
    magbin_bands,
    lc_data,
    mag_thresh,
    frac_cat,
    sig_scale=0.5,
):
    obs_mags, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        param_collection,
        lc_data,
        mag_thresh,
        frac_cat,
    )

    n_gals, n_bands = obs_mags.shape
    N_bands = []
    for band in range(0, n_bands):
        mags = obs_mags[:, band].reshape(obs_mags[:, band].size, 1)

        magbin_edges = magbin_bands[band]

        sig = jnp.diff(magbin_edges) * sig_scale
        sig = sig.reshape(sig.size, 1)

        mag_lo = magbin_edges[:-1].reshape(magbin_edges[:-1].size, 1)
        mag_hi = magbin_edges[1:].reshape(magbin_edges[1:].size, 1)

        N_mags = diffndhist_lomem.tw_ndhist_weighted(
            mags,
            sig,
            gal_weight,
            mag_lo,
            mag_hi,
        )
        N_bands.append(N_mags)

    return N_bands


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
        meta_data.mag_thresh,
        meta_data.in_lh_idx,
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
