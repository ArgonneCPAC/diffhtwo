from collections import namedtuple

from diffsky.experimental.lc_generators import lc_data_phot as lcdp
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from .kernels import igm
from .lc_utils import zbin_vol

N_SFH_TABLE = 100


def generate_lc_data(
    ran_key,
    n_host_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    apply_igm=False,
    igm_drn=None,
    igm_bn=None,
    igm_filters_namedtuple=None,
    igm_filter_prefix=None,
    logmp_cutoff=10.0,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    lc_args = (
        ran_key,
        n_host_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcdp.weighted_lc_data_phot(
        *lc_args, cosmo_params=cosmo_params, logmp_cutoff=logmp_cutoff
    )
    if apply_igm:
        precomputed_ssp_mag_table_w_igm = igm.apply_igm_to_ssp_mag_table(
            lc_data.precomputed_ssp_mag_table,
            z_phot_table,
            igm_drn,
            igm_bn,
            igm_filters_namedtuple,
            igm_filter_prefix,
        )
        lc_data = lc_data._replace(
            precomputed_ssp_mag_table=precomputed_ssp_mag_table_w_igm
        )

    lc_tot_vol_mpc3 = zbin_vol(sky_area_degsq, z_min, z_max, cosmo_params)

    return LCD(*lc_data, lc_tot_vol_mpc3, sky_area_degsq)


LCD = namedtuple(
    "LCD",
    lcdp.LCDataPhot._fields
    + (
        "precomputed_ssp_linelum_cgs_table",
        "line_wave_table",
        "lc_tot_vol_mpc3",
        "sky_area_degsq",
    ),
)
