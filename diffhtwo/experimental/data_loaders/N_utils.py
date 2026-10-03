from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky import diffndhist_lomem

from ..defaults import AppMagFunc, ColorColor, MagColor


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


def get_N_2d(dim1, dim2, sig_scale=0.5, n_bins=11):
    dataset = np.vstack((dim1, dim2)).T

    dim1_bin_edges = np.linspace(dim1.min(), dim1.max(), n_bins)
    dim2_bin_edges = np.linspace(dim2.min(), dim2.max(), n_bins)

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


def filter_name_to_idx(filter_name, filters_namedtuple):
    return filters_namedtuple._fields.index(filter_name)


def get_mag_space(
    namedtuple_name,
    mag,
    filter_name,
    z_sel,
    filters_namedtuple,
    mag_sels,
    frac_cats,
    fit=True,
):
    AppMagFuncSpace = namedtuple(namedtuple_name, AppMagFunc._fields)

    mag_idx = filter_name_to_idx(filter_name, filters_namedtuple)
    sel = z_sel * mag_sels[:, mag_idx]
    N_1d, sig, bin_lo, bin_hi = get_N_1d(mag[sel])

    frac_cat = frac_cats[mag_idx]

    return AppMagFuncSpace(mag_idx, sig, bin_lo, bin_hi, N_1d, frac_cat, fit)


def get_colorcolor_space(
    namedtuple_name,
    color1,
    color2,
    col_filter_names,
    z_sel,
    filters_namedtuple,
    mag_sels,
    frac_cats,
    fit=True,
):
    ColorColorSpace = namedtuple(namedtuple_name, ColorColor._fields)

    sel = z_sel
    col_idx = []
    for n in col_filter_names:
        idx = filter_name_to_idx(n, filters_namedtuple)
        sel *= mag_sels[:, idx]
        col_idx.append(idx)

    frac_cat = 1.0
    for idx in set(col_idx):
        frac_cat *= frac_cats[idx]

    N_2d, sig, bin_lo, bin_hi = get_N_2d(color1[sel], color2[sel])

    return ColorColorSpace(col_idx, sig, bin_lo, bin_hi, N_2d, frac_cat, fit)


def get_mag_color_space(
    namedtuple_name,
    mag,
    color,
    mag_filter_name,
    col_filter_names,
    z_sel,
    filters_namedtuple,
    mag_sels,
    frac_cats,
    fit=True,
):
    MagColorSpace = namedtuple(namedtuple_name, MagColor._fields)

    mag_idx = filter_name_to_idx(mag_filter_name, filters_namedtuple)
    sel = z_sel * mag_sels[:, mag_idx]

    col_idx = []
    for n in col_filter_names:
        idx = filter_name_to_idx(n, filters_namedtuple)
        sel *= mag_sels[:, idx]
        col_idx.append(idx)

    frac_cat = 1.0
    for idx in {mag_idx, *col_idx}:
        frac_cat *= frac_cats[idx]

    N_2d, sig, bin_lo, bin_hi = get_N_2d(mag[sel], color[sel])

    return MagColorSpace(mag_idx, col_idx, sig, bin_lo, bin_hi, N_2d, frac_cat, fit)
