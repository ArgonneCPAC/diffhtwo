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


def get_N_2d(dim1, dim2, dmag=0.1, sig_scale=0.5, gauss_sig=3.0):
    H, xe, ye = np.histogram2d(
        dim1,
        dim2,
        [
            np.arange(dim1.min(), dim1.max() + dmag, dmag),
            np.arange(dim2.min(), dim2.max() + dmag, dmag),
        ],
    )

    h = np.sort(H.ravel())[::-1]
    k = np.searchsorted(np.cumsum(h) / h.sum(), 1 - np.exp(-(gauss_sig**2) / 2))
    i, j = np.where(H >= h[k])

    N_2d = H[i, j]
    bin_lo = np.stack([xe[i], ye[j]], 1)
    bin_hi = np.stack([xe[i + 1], ye[j + 1]], 1)
    bin_width = bin_hi - bin_lo
    sig = bin_width * sig_scale
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
    dmag=0.2,
    fit=True,
):
    AppMagFuncSpace = namedtuple(namedtuple_name, AppMagFunc._fields)

    mag_idx = filter_name_to_idx(filter_name, filters_namedtuple)
    sel = z_sel * mag_sels[:, mag_idx]
    N_1d, sig, bin_lo, bin_hi = get_N_1d(mag[sel], dmag=dmag)

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
    dmag=0.1,
    sig_scale=0.5,
    gauss_sig=3.0,
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

    N_2d, sig, bin_lo, bin_hi = get_N_2d(
        color1[sel], color2[sel], dmag=dmag, sig_scale=sig_scale, gauss_sig=gauss_sig
    )

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
    dmag=0.1,
    sig_scale=0.5,
    gauss_sig=3.0,
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

    N_2d, sig, bin_lo, bin_hi = get_N_2d(
        mag[sel], color[sel], dmag=dmag, sig_scale=sig_scale, gauss_sig=gauss_sig
    )

    return MagColorSpace(mag_idx, col_idx, sig, bin_lo, bin_hi, N_2d, frac_cat, fit)
