from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..lightcone_generators import generate_lc_data

Hizels = namedtuple(
    "Hizels",
    [
        "line_wave_aa",
        "lg_Lbin_edges",
        "N_data",
        "vol_Mpc3_data",
        "lg_phi_data",
        "z",
        "dz",
        "lc_data",
        "nbins",
    ],
)
DELTA_L_HALPHA = -0.4  # uncorrect HiZELS h-alpha L for dust (A_halpha = 1 mag)


def get_hizels_data(
    drn,
    ran_key,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    num_halos=250,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=15,
):
    (
        hizels_lg_halpha_Lbin_edges_data,
        hizels_halpha_N_data,
        hizels_halpha_vol_Mpc3,
        hizels_halpha_lg_phi_data,
        hizels_halpha_z_data,
        hizels_halpha_delta_z_data,
        hizels_halpha_nbins,
    ) = get_hizels_halpha(drn)

    line_wave_aa = [halpha_wave_aa]
    lg_Lbin_edges = [hizels_lg_halpha_Lbin_edges_data]
    N_data = [hizels_halpha_N_data]
    vol_Mpc3_data = [hizels_halpha_vol_Mpc3]
    lg_phi_data = [hizels_halpha_lg_phi_data]
    z = [hizels_halpha_z_data]
    dz = [hizels_halpha_delta_z_data]

    lc_data = []
    for line in range(0, len(z)):
        line_z = z[line]
        line_dz = dz[line]
        line_lc_data = []
        for z_idx in range(0, len(line_z)):
            z_min = line_z[z_idx] - (line_dz[z_idx] / 2)
            z_max = line_z[z_idx] + (line_dz[z_idx] / 2)

            z_phot_table = 10 ** jnp.linspace(
                np.log10(z_min), np.log10(z_max), n_z_phot_table
            )

            lc_args = (
                ran_key,
                num_halos,
                z_min,
                z_max,
                lgmp_min,
                lgmp_max,
                lc_sky_area_degsq,
                ssp_data,
                tcurves,
                z_phot_table,
            )

            line_lc_data.append(generate_lc_data(*lc_args))
        lc_data.append(line_lc_data)

    return Hizels(
        line_wave_aa,
        lg_Lbin_edges,
        N_data,
        vol_Mpc3_data,
        lg_phi_data,
        z,
        dz,
        lc_data,
        hizels_halpha_nbins,
    )


def _get_lgL_bin_edges(
    table, L_colname, bin_width_full_colname, delta_L_halpha=DELTA_L_HALPHA
):
    edges = []
    for i in range(0, len(table)):
        edge = np.round(table[L_colname][i] - table[bin_width_full_colname][i] / 2, 2)
        edges.append(edge)
    last_edge = np.round(
        table[L_colname][-1] + table[bin_width_full_colname][-1] / 2, 2
    )
    edges.append(last_edge)
    edges = jnp.array(edges)
    edges = edges + delta_L_halpha
    return edges


def pad_dummy_lgL_bin_edges(lg_halpha_Lbin_edges, max_length=20, dummy_Lbin_width=0.2):
    N_dummy_Lbin_edges = max_length - lg_halpha_Lbin_edges.size

    dummy_Lbin_edges = jnp.linspace(
        lg_halpha_Lbin_edges[-1],
        lg_halpha_Lbin_edges[-1] + dummy_Lbin_width * N_dummy_Lbin_edges,
        endpoint=False,
        num=N_dummy_Lbin_edges,
    )
    return jnp.concatenate((lg_halpha_Lbin_edges, dummy_Lbin_edges[1:]))


def pad_dummy_lg_LF_data(lg_halpha_LF_data, lg_halpha_LF_dummy_err, max_length=18):
    pad_length = max_length - lg_halpha_LF_data.shape[1]

    lg_halpha_LF_data_padded = jnp.pad(
        lg_halpha_LF_data[0], (0, pad_length), constant_values=lg_halpha_LF_data[0][-1]
    )
    lg_halpha_LF_err_padded = jnp.pad(
        lg_halpha_LF_data[1], (0, pad_length), constant_values=lg_halpha_LF_dummy_err
    )
    return jnp.vstack((lg_halpha_LF_data_padded, lg_halpha_LF_err_padded))


def _lg_phi_h0p7_to_hdefault(lg_phi_h0p7):
    phi_h1p0 = (10**lg_phi_h0p7) / (0.7**3)
    return np.log10(phi_h1p0 * (DEFAULT_COSMOLOGY.h**3))


def _vol_h0p7_to_hdefault(vol_1e4Mpc3):
    vol_Mpc3_h1p0 = 1e4 * vol_1e4Mpc3 * (0.7**3)
    vol_Mpc3 = vol_Mpc3_h1p0 / (DEFAULT_COSMOLOGY.h**3)
    return vol_Mpc3


def _lg_phi_corr_to_N_corr(lg_phi_corr, vol_1e4Mpc3):
    phi_corr = 10**lg_phi_corr
    vol_Mpc3 = 1e4 * vol_1e4Mpc3
    N_corr = phi_corr * vol_Mpc3
    return N_corr


def get_hizels_halpha(drn):
    """z0p4"""
    HiZELS_halpha_z0p4 = ascii.read(drn / "halpha_LF_z0p4.dat")
    lg_halpha_Lbin_edges_z0p4 = _get_lgL_bin_edges(
        HiZELS_halpha_z0p4, "logLHa", "logLHa_binw_full"
    )
    halpha_N_data_z0p4 = _lg_phi_corr_to_N_corr(
        HiZELS_halpha_z0p4["logphi_corr"], HiZELS_halpha_z0p4["vol_1e4Mpc3"]
    )
    halpha_vol_Mpc3_z0p4 = _vol_h0p7_to_hdefault(HiZELS_halpha_z0p4["vol_1e4Mpc3"])

    halpha_lg_phi_data_z0p4 = jnp.vstack(
        (
            jnp.array(_lg_phi_h0p7_to_hdefault(HiZELS_halpha_z0p4["logphi_corr"])),
            jnp.array(HiZELS_halpha_z0p4["logphi_corr_err"]),
        )
    )

    """z0p84"""
    HiZELS_halpha_z0p84 = ascii.read(drn / "halpha_LF_z0p84.dat")
    lg_halpha_Lbin_edges_z0p84 = _get_lgL_bin_edges(
        HiZELS_halpha_z0p84, "logLHa", "logLHa_binw_full"
    )
    halpha_N_data_z0p84 = _lg_phi_corr_to_N_corr(
        HiZELS_halpha_z0p84["logphi_corr"], HiZELS_halpha_z0p84["vol_1e4Mpc3"]
    )
    halpha_vol_Mpc3_z0p84 = _vol_h0p7_to_hdefault(HiZELS_halpha_z0p84["vol_1e4Mpc3"])

    halpha_lg_phi_data_z0p84 = jnp.vstack(
        (
            jnp.array(_lg_phi_h0p7_to_hdefault(HiZELS_halpha_z0p84["logphi_corr"])),
            jnp.array(HiZELS_halpha_z0p84["logphi_corr_err"]),
        )
    )

    """z1p47"""
    HiZELS_halpha_z1p47 = ascii.read(drn / "halpha_LF_z1p47.dat")
    lg_halpha_Lbin_edges_z1p47 = _get_lgL_bin_edges(
        HiZELS_halpha_z1p47, "logLHa", "logLHa_binw_full"
    )
    halpha_N_data_z1p47 = _lg_phi_corr_to_N_corr(
        HiZELS_halpha_z1p47["logphi_corr"], HiZELS_halpha_z1p47["vol_1e4Mpc3"]
    )
    halpha_vol_Mpc3_z1p47 = _vol_h0p7_to_hdefault(HiZELS_halpha_z1p47["vol_1e4Mpc3"])

    halpha_lg_phi_data_z1p47 = jnp.vstack(
        (
            jnp.array(_lg_phi_h0p7_to_hdefault(HiZELS_halpha_z1p47["logphi_corr"])),
            jnp.array(HiZELS_halpha_z1p47["logphi_corr_err"]),
        )
    )

    """z2p23"""
    HiZELS_halpha_z2p23 = ascii.read(drn / "halpha_LF_z2p23.dat")
    lg_halpha_Lbin_edges_z2p23 = _get_lgL_bin_edges(
        HiZELS_halpha_z2p23, "logLHa", "logLHa_binw_full"
    )
    halpha_N_data_z2p23 = _lg_phi_corr_to_N_corr(
        HiZELS_halpha_z2p23["logphi_corr"], HiZELS_halpha_z2p23["vol_1e4Mpc3"]
    )
    halpha_vol_Mpc3_z2p23 = _vol_h0p7_to_hdefault(HiZELS_halpha_z2p23["vol_1e4Mpc3"])

    halpha_lg_phi_data_z2p23 = jnp.vstack(
        (
            jnp.array(_lg_phi_h0p7_to_hdefault(HiZELS_halpha_z2p23["logphi_corr"])),
            jnp.array(HiZELS_halpha_z2p23["logphi_corr_err"]),
        )
    )

    hizels_halpha_nbins = (
        (lg_halpha_Lbin_edges_z0p4.size - 1)
        + (lg_halpha_Lbin_edges_z0p84.size - 1)
        + (lg_halpha_Lbin_edges_z1p47.size - 1)
        + (lg_halpha_Lbin_edges_z2p23.size - 1)
    )

    hizels_lg_halpha_Lbin_edges_data = [
        lg_halpha_Lbin_edges_z0p4,
        lg_halpha_Lbin_edges_z0p84,
        lg_halpha_Lbin_edges_z1p47,
        lg_halpha_Lbin_edges_z2p23,
    ]

    hizels_halpha_N_data = [
        halpha_N_data_z0p4,
        halpha_N_data_z0p84,
        halpha_N_data_z1p47,
        halpha_N_data_z2p23,
    ]

    hizels_halpha_vol_Mpc3 = [
        halpha_vol_Mpc3_z0p4,
        halpha_vol_Mpc3_z0p84,
        halpha_vol_Mpc3_z1p47,
        halpha_vol_Mpc3_z2p23,
    ]

    hizels_halpha_lg_phi_data = [
        halpha_lg_phi_data_z0p4,
        halpha_lg_phi_data_z0p84,
        halpha_lg_phi_data_z1p47,
        halpha_lg_phi_data_z2p23,
    ]

    hizels_halpha_z_data = [
        jnp.float64(0.40),
        jnp.float64(0.84),
        jnp.float64(1.47),
        jnp.float64(2.23),
    ]

    hizels_halpha_delta_z_data = [
        0.02,
        0.03,
        0.032,
        0.046,
    ]

    return (
        hizels_lg_halpha_Lbin_edges_data,
        hizels_halpha_N_data,
        hizels_halpha_vol_Mpc3,
        hizels_halpha_lg_phi_data,
        hizels_halpha_z_data,
        hizels_halpha_delta_z_data,
        hizels_halpha_nbins,
    )
