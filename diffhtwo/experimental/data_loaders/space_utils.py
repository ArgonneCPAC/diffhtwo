import os
import re
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky import diffndhist_lomem
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..kernels.line_emitter import compute_Sigma_excess
from ..lc_utils import zbin_vol
from .minerva_area import get_minerva_area_arcminsq

ColorColor = namedtuple(
    "ColorColor",
    [
        "parent_cut_idx",
        "col_idx",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "vol_mpc3_data",
        "fit",
        "zlo",
        "zhi",
        "vol_mpc3_model",
    ],
)

MagColor = namedtuple(
    "MagColor",
    [
        "parent_cut_idx",
        "mag_idx",
        "col_idx",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "vol_mpc3_data",
        "fit",
        "zlo",
        "zhi",
        "vol_mpc3_model",
    ],
)

AppMagFunc = namedtuple(
    "AppMagFunc",
    [
        "parent_cut_idx",
        "mag_idx",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "vol_mpc3_data",
        "fit",
        "zlo",
        "zhi",
        "vol_mpc3_model",
    ],
)

Lf = namedtuple(
    "Lf",
    ["sig", "bin_lo", "bin_hi", "N_data", "fit"],
)


def get_minerva_space_area_arcminsq(img_drn, space, area_file):
    areas = {}
    if os.path.exists(area_file):
        with open(area_file) as f:
            areas = {
                sp: float(area)
                for sp, area in (line.split() for line in f if line.strip())
            }

    if space not in areas:
        filters = sorted(set(re.findall(r"f\d{3}[wm]", space.lower())))
        areas[space] = sum(
            get_minerva_area_arcminsq(img_drn, space_filters=filters).values()
        )
        with open(area_file, "a") as f:
            f.write(f"{space} {areas[space]}\n")

    return areas[space]


def pick_line_emitters(BB, NB, BB_err, NB_err, Sigma_thresh=3.0, ZP=None):
    Sigma = compute_Sigma_excess(BB, NB, BB_err, NB_err, ZP)
    emitter_sel = Sigma > Sigma_thresh
    return emitter_sel


def split_space_entries(entries):
    names = [name for name, _, _ in entries]
    fits = {name: fit for name, fit, _ in entries}
    zs = {name: z for name, _, z in entries}
    return names, fits, zs


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


def get_filt_indx(space_name, filters_namedtuple):
    fields = filters_namedtuple._fields
    if re.search(r"F\d{3}", space_name, re.I):
        names = [n.lower() for n in re.findall(r"F\d{3}[a-z]", space_name, re.I)]
        return [fields.index(n) for n in names]
    bands = [f.split("_")[-1][0].lower() for f in fields]
    return [bands.index(b) for b in space_name.lower().replace("_", "")]


def get_mag_spaces_at_z(
    mag_namedtuples,
    filters_namedtuple,
    md_z,
    md_fit,
    redshift,
    mag_sels,
    mags,
    parent_idx,
    d_mag_1d,
    lc_sky_area_degsq,
    data_sky_area_degsq=None,
    minerva_drn=None,
    minerva_area_file=None,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    mag_z_tuples = []
    for space_name, space in mag_namedtuples.items():
        (mag_idx,) = get_filt_indx(space_name, filters_namedtuple)

        zlo, zhi = md_z[space_name]
        z_sel = (redshift > zlo) & (redshift <= zhi)

        if data_sky_area_degsq is None:
            space_area_arcminsq = get_minerva_space_area_arcminsq(
                minerva_drn, space_name, minerva_drn / minerva_area_file
            )
            space_area_degsq = space_area_arcminsq / 3600
            vol_mpc3_data = zbin_vol(space_area_degsq, zlo, zhi, cosmo_params)
        else:
            vol_mpc3_data = zbin_vol(data_sky_area_degsq, zlo, zhi, cosmo_params)

        vol_mpc3_model = zbin_vol(lc_sky_area_degsq, zlo, zhi, cosmo_params)

        sel = mag_sels[:, mag_idx] & z_sel
        mag_selected = mags[sel]

        N_1d, sig, bin_lo, bin_hi = get_N_1d(mag_selected[:, mag_idx], dmag=d_mag_1d)

        mag_z_tuples.append(
            space(
                parent_idx,
                mag_idx,
                sig,
                bin_lo,
                bin_hi,
                N_1d,
                vol_mpc3_data,
                md_fit[space_name],
                zlo,
                zhi,
                vol_mpc3_model,
            )
        )
    return mag_z_tuples


def get_ccd_spaces_at_z(
    ccd_namedtuples,
    filters_namedtuple,
    ccd_z,
    ccd_fit,
    redshift,
    mag_sels,
    mags,
    parent_idx,
    d_mag_2d,
    gauss_sig_2d,
    lc_sky_area_degsq,
    data_sky_area_degsq=None,
    minerva_drn=None,
    minerva_area_file=None,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    ccd_z_tuples = []
    for space_name, space in ccd_namedtuples.items():
        col_idx = get_filt_indx(space_name, filters_namedtuple)
        zlo, zhi = ccd_z[space_name]
        z_sel = (redshift > zlo) & (redshift <= zhi)

        if data_sky_area_degsq is None:
            space_area_arcminsq = get_minerva_space_area_arcminsq(
                minerva_drn, space_name, minerva_drn / minerva_area_file
            )
            space_area_degsq = space_area_arcminsq / 3600
            vol_mpc3_data = zbin_vol(space_area_degsq, zlo, zhi, cosmo_params)
        else:
            vol_mpc3_data = zbin_vol(data_sky_area_degsq, zlo, zhi, cosmo_params)

        vol_mpc3_model = zbin_vol(lc_sky_area_degsq, zlo, zhi, cosmo_params)

        a, b, c, d = col_idx
        sel = mag_sels[:, a] & mag_sels[:, b] & mag_sels[:, c] & mag_sels[:, d] & z_sel
        mag_selected = mags[sel]

        color1 = mag_selected[:, a] - mag_selected[:, b]
        color2 = mag_selected[:, c] - mag_selected[:, d]

        N_2d, sig, bin_lo, bin_hi = get_N_2d(
            color1, color2, dmag=d_mag_2d, gauss_sig=gauss_sig_2d
        )

        ccd_z_tuples.append(
            space(
                parent_idx,
                col_idx,
                sig,
                bin_lo,
                bin_hi,
                N_2d,
                vol_mpc3_data,
                ccd_fit[space_name],
                zlo,
                zhi,
                vol_mpc3_model,
            )
        )
    return ccd_z_tuples


def get_cmd_spaces_at_z(
    cmd_namedtuples,
    filters_namedtuple,
    cmd_z,
    cmd_fit,
    redshift,
    mag_sels,
    mags,
    parent_idx,
    d_mag_2d,
    gauss_sig_2d,
    lc_sky_area_degsq,
    data_sky_area_degsq,
    minerva_drn=None,
    minerva_area_file=None,
    mags_err=None,
    ZP=None,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    cmd_z_tuples = []
    for space_name, space in cmd_namedtuples.items():
        mag_idx, b, c = get_filt_indx(space_name, filters_namedtuple)
        zlo, zhi = cmd_z[space_name]
        z_sel = (redshift > zlo) & (redshift <= zhi)

        if data_sky_area_degsq is None:
            space_area_arcminsq = get_minerva_space_area_arcminsq(
                minerva_drn, space_name, minerva_drn / minerva_area_file
            )
            space_area_degsq = space_area_arcminsq / 3600
            vol_mpc3_data = zbin_vol(space_area_degsq, zlo, zhi, cosmo_params)
        else:
            vol_mpc3_data = zbin_vol(data_sky_area_degsq, zlo, zhi, cosmo_params)

        vol_mpc3_model = zbin_vol(lc_sky_area_degsq, zlo, zhi, cosmo_params)

        sel = mag_sels[:, mag_idx] & mag_sels[:, b] & mag_sels[:, c] & z_sel
        if "m" in space_name:
            emitter_sel = pick_line_emitters(
                mags[:, b],
                mags[:, mag_idx],
                mags_err[:, b],
                mags_err[:, mag_idx],
                ZP=ZP,
            )
            sel &= emitter_sel

        mag_selected = mags[sel]

        mag = mag_selected[:, mag_idx]
        color = mag_selected[:, b] - mag_selected[:, c]

        N_2d, sig, bin_lo, bin_hi = get_N_2d(
            mag, color, dmag=d_mag_2d, gauss_sig=gauss_sig_2d
        )

        col_idx = [b, c]

        cmd_z_tuples.append(
            space(
                parent_idx,
                mag_idx,
                col_idx,
                sig,
                bin_lo,
                bin_hi,
                N_2d,
                vol_mpc3_data,
                cmd_fit[space_name],
                zlo,
                zhi,
                vol_mpc3_model,
            )
        )
    return cmd_z_tuples
