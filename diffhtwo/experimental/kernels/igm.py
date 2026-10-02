import os

import h5py
from jax import numpy as jnp


def load_igm_attenuation(
    igm_drn,
    igm_bn,
    filters_namedtuple,
    filter_prefix,
):
    filters = []
    for filt in filters_namedtuple._fields:
        filters.append(filter_prefix + filt)

    file = os.path.join(igm_drn, igm_bn)
    with h5py.File(file, "r") as fobj:
        igm_attn_dict = {}
        for filt in filters:
            igm_attn_dict[filt + "/redshift"] = fobj[filt + "/redshift"][:]
            igm_attn_dict[filt + "/dmag"] = fobj[filt + "/igm_attenuation_inoue"][:]

    return igm_attn_dict, filters


def apply_igm_to_ssp_mag_table(
    precomputed_ssp_mag_table,
    z_phot_table,
    igm_drn,
    igm_bn,
    filters_namedtuple,
    filter_prefix,
):
    igm_attn_dict, filters = load_igm_attenuation(
        igm_drn,
        igm_bn,
        filters_namedtuple,
        filter_prefix,
    )

    igm_dmag = []
    for filt in filters:
        igm_dmag.append(
            jnp.interp(
                z_phot_table,
                igm_attn_dict[filt + "/redshift"],
                igm_attn_dict[filt + "/dmag"],
            )
        )
    igm_dmag = jnp.array(igm_dmag).T
    n_z, n_bands = igm_dmag.shape
    igm_dmag = igm_dmag.reshape(n_z, n_bands, 1, 1)
    precomputed_ssp_mag_table_w_igm = precomputed_ssp_mag_table + igm_dmag

    return precomputed_ssp_mag_table_w_igm
