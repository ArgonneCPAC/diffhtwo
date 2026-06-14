from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky import diffndhist_lomem

from ..defaults import AppMagFunc, ColorColor, ColorCondMag, FeniksFilters, MagColor


def get_N_1d(dim1, dim1_bin_edges=None, dmag=0.2, sig_scale=0.5):
    dataset = dim1.reshape(dim1.size, 1)
    if dim1_bin_edges is None:
        dim1_bin_edges = np.arange(dim1.min(), dim1.max(), dmag)

    bin_lo = dim1_bin_edges[:-1].reshape(dim1_bin_edges[:-1].size, 1)
    bin_hi = dim1_bin_edges[1:].reshape(dim1_bin_edges[1:].size, 1)

    sig = jnp.zeros_like(bin_lo) + (dmag * sig_scale)

    N_1d = diffndhist_lomem.tw_ndhist(
        dataset,
        sig,
        bin_lo,
        bin_hi,
    )

    return (
        N_1d,
        sig,
        bin_lo,
        bin_hi,
    )


def get_N_2d(dim1, dim2, sig_scale=0.5):
    dataset = np.vstack((dim1, dim2)).T

    dim1_bin_edges = np.linspace(dim1.min(), dim1.max(), 11)
    dim2_bin_edges = np.linspace(dim2.min(), dim2.max(), 11)

    dim1_lo = dim1_bin_edges[:-1]
    dim2_lo = dim2_bin_edges[:-1]
    bin_lo = np.meshgrid(dim1_lo, dim2_lo, indexing="ij")
    bin_lo = np.array(bin_lo).T.reshape(-1, 2)

    dim1_hi = dim1_bin_edges[1:]
    dim2_hi = dim2_bin_edges[1:]
    bin_hi = np.meshgrid(dim1_hi, dim2_hi, indexing="ij")
    bin_hi = np.array(bin_hi).T.reshape(-1, 2)

    sig1 = np.diff(dim1_bin_edges) * sig_scale
    sig2 = np.diff(dim2_bin_edges) * sig_scale
    sig = np.meshgrid(sig1, sig2, indexing="ij")
    sig = np.array(sig).T.reshape(-1, 2)

    N_2d = diffndhist_lomem.tw_ndhist(
        dataset,
        sig,
        bin_lo,
        bin_hi,
    )

    return N_2d, sig, bin_lo, bin_hi


def filter_name_to_idx(filter_name):
    return FeniksFilters._fields.index(filter_name)


def get_mag_space(namedtuple_name, mag, filter_name, z_sel, fit=True):
    AppMagFuncSpace = namedtuple(namedtuple_name, AppMagFunc._fields)
    mag_idx = filter_name_to_idx(filter_name)
    N_1d, sig, bin_lo, bin_hi = get_N_1d(mag[z_sel])
    return AppMagFuncSpace(mag_idx, sig, bin_lo, bin_hi, N_1d, fit)


def get_colorcolor_space(
    namedtuple_name, color1, color2, col_filter_names, z_sel, fit=True
):
    ColorColorSpace = namedtuple(namedtuple_name, ColorColor._fields)

    N_2d, sig, bin_lo, bin_hi = get_N_2d(color1[z_sel], color2[z_sel])

    col_idx = []
    for n in col_filter_names:
        col_idx.append(filter_name_to_idx(n))

    return ColorColorSpace(col_idx, sig, bin_lo, bin_hi, N_2d, fit)


def get_color_cond_space_list(
    namedtuple_name,
    color,
    cond_mag,
    col_filter_names,
    cond_filter_name,
    z_sel,
    cond_dmag=2,
    fit=True,
):
    ColorCondSpace = namedtuple(namedtuple_name, ColorCondMag._fields)

    col_idx = []
    for n in col_filter_names:
        col_idx.append(filter_name_to_idx(n))
    cond_idx = filter_name_to_idx(cond_filter_name)

    cond_mag_bins = np.arange(cond_mag[z_sel].min(), cond_mag[z_sel].max(), cond_dmag)

    color_cond_list = []
    for b in range(len(cond_mag_bins) - 1):
        cond_sel = (cond_mag[z_sel] > cond_mag_bins[b]) & (
            cond_mag[z_sel] <= cond_mag_bins[b + 1]
        )
        N_1d, sig, bin_lo, bin_hi = get_N_1d(color[z_sel][cond_sel])
        color_cond_list.append(
            ColorCondSpace(
                col_idx,
                cond_idx,
                cond_mag_bins[b],
                cond_mag_bins[b + 1],
                sig,
                bin_lo,
                bin_hi,
                N_1d,
                fit,
            )
        )

    return color_cond_list


def get_mag_color_space(
    namedtuple_name, mag, color, mag_filter_name, col_filter_names, z_sel, fit=True
):
    MagColorSpace = namedtuple(namedtuple_name, MagColor._fields)

    mag_idx = filter_name_to_idx(mag_filter_name)
    col_idx = []
    for n in col_filter_names:
        col_idx.append(filter_name_to_idx(n))

    N_2d, sig, bin_lo, bin_hi = get_N_2d(mag[z_sel], color[z_sel])

    return MagColorSpace(mag_idx, col_idx, sig, bin_lo, bin_hi, N_2d, fit)
