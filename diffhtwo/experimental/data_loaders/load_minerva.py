import re
from collections import namedtuple
from difflib import get_close_matches
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from astropy.table import Table, vstack
from dsps.data_loaders import load_transmission_curve

from ..defaults import (
    MINERVA_AREA_DEG2,
    MINERVA_UDS_AREA_DEG2,
    AppMagFunc,
    ColorColor,
    FilterInfo,
    Lf,
    MagColor,
)
from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data
from . import N_utils
from .N_utils import get_N_1d, get_N_2d

BASE_PATH = Path(__file__).resolve().parent.parent

MINERVA_FILTERS_PATH = BASE_PATH / "data" / "minerva_filters"

IGM_DRN = BASE_PATH / "data" / "igm"
IGM_BN = "igm_attenuation_minerva.h5"

UDS_PHOT_CAT = (
    "uds/MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits"
)
UDS_EAZY_CAT = (
    "uds/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_zpiter_CATALOG_larson.zout.fits"
)

COSMOS_PHOT_CAT = (
    "cosmos/MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits"
)

COSMOS_EAZY_CAT = "cosmos/MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.larson.ZPiter.eazy.zout.fits"

EGS_PHOT_CAT = (
    "egs/MINERVA-EGS_n2.0_m2.1_v1.3.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits"
)
EGS_EAZY_CAT = (
    "egs/MINERVA-EGS_n2.0_v1.3_ACS+WEBB_Kf444w_SUPER_zpiter_CATALOG_larson.zout.fits"
)


TRANSLATE = "MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_zpiter_CATALOG.larson.eazypy.zphot.translate"
INFO = "FILTER.RES.latest.info"
TCURVES = "FILTER.RES.latest"

MinervaPhot = namedtuple(
    "MinervaPhot",
    [
        "redshift",
        "mags",
        "sels",
        "frac_cats",
        "mags_labels",
        "spaces",
        "zbins",
        "filter_info",
        "data_sky_area_degsq",
    ],
)


def _get_mag_ab(phot_table, col_name, ZP=28.9):
    with np.errstate(invalid="ignore"):
        flux = phot_table[col_name].data.data
        mag_ab = -2.5 * np.log10(flux) + ZP

    return mag_ab


def _get_tcurve(filter_number, filter_info_filename, tcurves_filename):
    with open(filter_info_filename) as INFO:
        info = INFO.readlines()
    with open(tcurves_filename) as TCURVES:
        tcurves = TCURVES.readlines()

    f_idx = filter_number - 1
    t_idx = tcurves.index(get_close_matches(info[f_idx], tcurves)[0])

    i = 0
    wave_aa = []
    trans = []
    while (len(tcurves[t_idx + 1 :][i].split()) <= 3) & (
        (t_idx + 2 + i) < len(tcurves)
    ):
        wave_aa.append(float(tcurves[t_idx + 1 :][i].split()[-2]))
        trans.append(float(tcurves[t_idx + 1 :][i].split()[-1]))
        i += 1

    return jnp.array(wave_aa), jnp.array(trans)


def write_eazy_filters_to_h5(
    drn,
    tcurve_savedir,
    translate_fn=TRANSLATE,
    info_fn=INFO,
    tcurves_fn=TCURVES,
):
    drn = Path(drn)
    translate_fn = drn / translate_fn
    info_fn = drn / info_fn
    tcurves_fn = drn / tcurves_fn

    translate = dict(line.split() for line in open(translate_fn))
    for minerva_filter in PhotFilters._fields:
        col_name = "f_" + minerva_filter

        # get tcurve
        filter_number = int(translate[col_name][1:])
        wave_aa, trans = _get_tcurve(filter_number, info_fn, tcurves_fn)

        with h5py.File(
            tcurve_savedir + "/" + minerva_filter + ".h5", "w"
        ) as tcurve_h5py:
            tcurve_h5py.create_dataset("wave", data=wave_aa)
            tcurve_h5py.create_dataset("transmission", data=trans)
    print("Saved tcurves successfully.")


def _merge_minerva_fields(
    uds_phot, uds_zout, cosmos_phot, cosmos_zout, egs_phot, egs_zout
):
    common = set(uds_phot.colnames) & set(cosmos_phot.colnames) & set(egs_phot.colnames)
    common = [c for c in uds_phot.colnames if c in common]

    uds_phot = uds_phot[common]
    cosmos_phot = cosmos_phot[common]
    egs_phot = egs_phot[common]

    cosmos_phot["flag_lowsnr"] = cosmos_phot["flag_lowsnr"].astype("int64")
    cosmos_phot["flag_star"] = cosmos_phot["flag_star"].astype("int64")
    cosmos_phot["use_phot"] = cosmos_phot["use_phot"].astype("int64")
    cosmos_phot["flag_nophot"] = cosmos_phot["flag_nophot"].astype("int32")
    cosmos_phot["flag_acs_coverage"] = cosmos_phot["flag_acs_coverage"].astype("int64")
    cosmos_phot["flag_singleband"] = cosmos_phot["flag_singleband"].astype("int64")
    cosmos_phot["flag_clean"] = cosmos_phot["flag_clean"].astype("int64")
    cosmos_phot["flag_kron"] = cosmos_phot["flag_kron"].astype("int64")
    cosmos_phot["use_circle"] = cosmos_phot["use_circle"].astype("int64")

    phot = vstack([uds_phot, cosmos_phot, egs_phot], metadata_conflicts="silent")

    common = set(uds_zout.colnames) & set(cosmos_zout.colnames) & set(egs_zout.colnames)
    common = [c for c in uds_zout.colnames if c in common]

    uds_zout = uds_zout[common]
    cosmos_zout = cosmos_zout[common]
    egs_zout = egs_zout[common]

    zout = vstack([uds_zout, cosmos_zout, egs_zout], metadata_conflicts="silent")

    return phot, zout


def get_minerva_phot(
    drn,
    ran_key,
    ssp_data,
    d_mag_1d=0.2,
    d_mag_2d=0.1,
    gauss_sig_2d=3.0,
    num_halos=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
    apply_igm=True,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
    uds_phot_cat=UDS_PHOT_CAT,
    uds_eazy_cat=UDS_EAZY_CAT,
    cosmos_phot_cat=COSMOS_PHOT_CAT,
    cosmos_eazy_cat=COSMOS_EAZY_CAT,
    egs_phot_cat=EGS_PHOT_CAT,
    egs_eazy_cat=EGS_EAZY_CAT,
):
    drn = Path(drn)
    uds_phot = Table.read(drn / uds_phot_cat)
    uds_zout = Table.read(drn / uds_eazy_cat)
    cosmos_phot = Table.read(drn / cosmos_phot_cat)
    cosmos_zout = Table.read(drn / cosmos_eazy_cat)
    egs_phot = Table.read(drn / egs_phot_cat)
    egs_zout = Table.read(drn / egs_eazy_cat)

    phot, zout = _merge_minerva_fields(
        uds_phot, uds_zout, cosmos_phot, cosmos_zout, egs_phot, egs_zout
    )

    spec_avail = zout["z_spec"] != -99.0  # goes in frac_cat?
    z_best = zout["z_ml"].copy()
    z_best[spec_avail] = zout["z_spec"][spec_avail]
    z_best = np.float32(z_best)

    use_phot = phot["use_phot"] == 1
    phot = phot[use_phot]
    zout = zout[use_phot]
    z_best = z_best[use_phot].data

    default_limits = (19.0, 27.0)
    minerva_mag_thresh = PhotFilters(
        f435w=default_limits,
        f606w=default_limits,
        f814w=default_limits,
        f105w=default_limits,
        f125w=default_limits,
        f160w=default_limits,
        f090w=default_limits,
        f115w=default_limits,
        f140m=default_limits,
        f150w=default_limits,
        f162m=default_limits,
        f182m=default_limits,
        f200w=default_limits,
        f210m=default_limits,
        f250m=default_limits,
        f277w=default_limits,
        f300m=default_limits,
        f356w=default_limits,
        f360m=default_limits,
        f410m=default_limits,
        f444w=default_limits,
        f460m=default_limits,
    )
    mag_f444w = _get_mag_ab(phot, "f_f444w")
    (f444w_idx,) = get_filt_indx("F444w", PhotFilters)
    mag_limit_f444w = getattr(minerva_mag_thresh, "f444w")
    sel_f444w = (mag_f444w > mag_limit_f444w[0]) & (mag_f444w < mag_limit_f444w[1])

    tcurves = []
    mag_per_band = []
    mag_labels = []
    sel_per_band = []
    frac_cat_per_band = []
    tcurves = []
    for minerva_filter in PhotFilters._fields:
        tcurve = load_transmission_curve(
            bn_pat=minerva_filter + "*", drn=MINERVA_FILTERS_PATH
        )
        tcurves.append(tcurve)

        # get magnitudes
        col_name = "f_" + minerva_filter
        mag = _get_mag_ab(phot, col_name)

        n_gals = phot[col_name].mask.size

        # originally masked in the phot cat due to missing coverage, for instance
        sel = ~phot[col_name].mask

        # based on removing masked gals (no coverage, etc.)
        frac_cat = sel.sum() / n_gals
        frac_cat_per_band.append(frac_cat)

        # masked due to converting flux to mag for -ve flux of droputs, for instance
        sel *= np.isfinite(mag)

        # mag thresh selection
        mag_limit = getattr(minerva_mag_thresh, minerva_filter)
        sel *= (mag > mag_limit[0]) & (mag < mag_limit[1])
        sel *= sel_f444w

        mag_per_band.append(mag)
        sel_per_band.append(sel)

        mag_labels.append(minerva_filter)

    mags = np.vstack(mag_per_band).T
    sels = np.vstack(sel_per_band).T
    frac_cats = np.array(frac_cat_per_band)

    filter_info = FilterInfo(minerva_mag_thresh, tcurves)

    md = [
        "F435w",
        "F606w",
        "F814w",
        "F125w",
        "F160w",
        "F090w",
        "F115w",
        "F150w",
        "F200w",
        "F277w",
        "F356w",
        "F444w",
    ]
    mag_namedtuples = {i: namedtuple(i, AppMagFunc._fields) for i in md}

    cc_cmd_spaces_at_z = [
        {
            "z": (1.0, 2.0),
            "ccd": ["F090wF150w_F150wF356w"],
            "cmd": ["F356w_F150wF356w"],
        },
        {
            "z": (2.0, 3.0),
            "ccd": ["F115wF200w_F200wF356w", "F150wF200w_F200wF277w"],
            "cmd": ["F356w_F115wF356w"],
        },
        {
            "z": (3.0, 4.0),
            "ccd": ["F150wF277w_F277wF444w", "F200wF277w_F277wF356w"],
            "cmd": ["F356w_F150wF356w"],
        },
        {
            "z": (4.0, 5.0),
            "ccd": [
                "F814wF150w_F150wF277w",
                "F200wF277w_F277wF444w",
                "F356wF410m_F410mF444w",
            ],
            "cmd": ["F444w_F150wF444w"],
        },
        {
            "z": (5.0, 6.0),
            "ccd": ["F115wF200w_F200wF444w", "F277wF356w_F356wF444w"],
            "cmd": ["F444w_F115wF444w"],
        },
    ]
    z_bins = np.array([sp["z"] for sp in cc_cmd_spaces_at_z])

    # cmd = ["F162m_F160wF162m"]#H-alpha emitted at z~1.5 figure

    spaces = []
    for zbin in range(len(z_bins)):
        z_min = z_bins[zbin][0]
        z_max = z_bins[zbin][1]

        ccd = cc_cmd_spaces_at_z[zbin]["ccd"]
        cmd = cc_cmd_spaces_at_z[zbin]["cmd"]

        ccd_namedtuples = {i: namedtuple(i, ColorColor._fields) for i in ccd}
        cmd_namedtuples = {i: namedtuple(i, MagColor._fields) for i in cmd}

        Spaces = namedtuple(
            "Spaces",
            [
                "z_min",
                "z_max",
                "data_vol_mpc3",
                "lc_data",
                *mag_namedtuples,
                *ccd,
                *cmd,
            ],
        )

        data_vol_mpc3 = zbin_volume(MINERVA_AREA_DEG2, zlow=z_min, zhigh=z_max).value

        z_phot_table = 10 ** jnp.linspace(
            jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
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

        lc_data = generate_lc_data(
            *lc_args,
            apply_igm=apply_igm,
            igm_drn=IGM_DRN,
            igm_bn=IGM_BN,
            igm_filters_namedtuple=PhotFilters,
            igm_filter_prefix="minerva_",
            logmp_cutoff=lgmp_min,
        )

        z_sel = (z_best > z_min) & (z_best <= z_max)

        mag_z_tuples = []
        for space_name, space in mag_namedtuples.items():
            (mag_idx,) = get_filt_indx(space_name, PhotFilters)

            sel = sels[:, mag_idx] * z_sel
            mag_selected = mags[sel]

            N_1d, sig, bin_lo, bin_hi = get_N_1d(
                mag_selected[:, mag_idx], dmag=d_mag_1d
            )

            frac_cat = frac_cats[mag_idx]

            mag_z_tuples.append(
                space(mag_idx, sig, bin_lo, bin_hi, N_1d, frac_cat, True)
            )

        ccd_z_tuples = []
        for space_name, space in ccd_namedtuples.items():
            col_idx = get_filt_indx(space_name, PhotFilters)

            a, b, c, d = col_idx
            sel = sels[:, a] * sels[:, b] * sels[:, c] * sels[:, d] * z_sel
            mag_selected = mags[sel]

            color1 = mag_selected[:, a] - mag_selected[:, b]
            color2 = mag_selected[:, c] - mag_selected[:, d]

            N_2d, sig, bin_lo, bin_hi = get_N_2d(
                color1, color2, dmag=d_mag_2d, gauss_sig=gauss_sig_2d
            )

            frac_cat = np.min((frac_cats[a], frac_cats[b], frac_cats[c], frac_cats[d]))

            ccd_z_tuples.append(
                space(col_idx, sig, bin_lo, bin_hi, N_2d, frac_cat, True)
            )

        cmd_z_tuples = []
        for space_name, space in cmd_namedtuples.items():
            mag_idx, b, c = get_filt_indx(space_name, PhotFilters)

            sel = sels[:, mag_idx] * sels[:, b] * sels[:, c] * z_sel
            mag_selected = mags[sel]

            mag = mag_selected[:, mag_idx]
            color = mag_selected[:, b] - mag_selected[:, c]

            N_2d, sig, bin_lo, bin_hi = get_N_2d(
                mag, color, dmag=d_mag_2d, gauss_sig=gauss_sig_2d
            )

            col_idx = [b, c]

            frac_cat = np.min((frac_cats[mag_idx], frac_cats[b], frac_cats[c]))

            cmd_z_tuples.append(
                space(mag_idx, col_idx, sig, bin_lo, bin_hi, N_2d, frac_cat, True)
            )

        spaces.append(
            Spaces(
                z_min,
                z_max,
                data_vol_mpc3,
                lc_data,
                *mag_z_tuples,
                *ccd_z_tuples,
                *cmd_z_tuples,
            )
        )

    return MinervaPhot(
        z_best,
        mags,
        sels,
        frac_cats,
        mag_labels,
        spaces,
        z_bins,
        filter_info,
        MINERVA_AREA_DEG2,
    )


def get_filt_indx(space_name, filters_namedtuple):
    """
    space_name: str
        e.g. "F105wF125w_F125wF162m"
    """

    filters = re.findall(r"F\d{3}[a-z]", space_name)
    filters = [f.lower() for f in filters]

    col_idx = []
    for filter in filters:
        col_idx.append(N_utils.filter_name_to_idx(filter, filters_namedtuple))
    return col_idx


def get_minerva_phot_fitting_data(
    drn,
    ran_key,
    ssp_data,
    d_mag_1d=0.1,
    d_mag_2d=0.05,
    gauss_sig_2d=3.0,
    num_halos=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
    apply_igm=True,
):
    minerva_phot = get_minerva_phot(
        drn,
        ran_key,
        ssp_data,
        d_mag_1d=d_mag_1d,
        d_mag_2d=d_mag_2d,
        gauss_sig_2d=gauss_sig_2d,
        num_halos=num_halos,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        apply_igm=apply_igm,
    )
    fields = [f for f in minerva_phot._fields if f != "mags_labels"]
    MinervaPhotFit = namedtuple("MinervaPhotFit", fields)
    return MinervaPhotFit(*(getattr(minerva_phot, f) for f in fields))


def get_minerva_halpha(
    halpha_drn,
    drn,
    ran_key,
    ssp_data,
    num_halos=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=15,
):
    halpha_drn = Path(halpha_drn)
    drn = Path(drn)

    # Transmission curves
    tcurves = []
    for halpha_filter in HalphaFilters._fields:
        tcurve = load_transmission_curve(
            bn_pat=halpha_filter + "*", drn=MINERVA_FILTERS_PATH
        )
        tcurves.append(tcurve)

    LumFunc = namedtuple(
        "LumFunc",
        ["z_min", "z_max", "data_vol_mpc3", "lc_data", "lf_data"],
    )
    lfs = []
    for f in HalphaFilters._fields:
        halpha = Table.read(halpha_drn / f"Ha_table_{f}_minerva-uds_power.fits")
        # cosmos_halpha = Table.read(
        #     halpha_drn / f"Ha_table_{f}_minerva-cosmos_power.fits"
        # )
        # egs_halpha = Table.read(halpha_drn / f"Ha_table_{f}_minerva-egs_power.fits")
        # halpha = vstack([uds_halpha, cosmos_halpha, egs_halpha])

        z_min = halpha["z_phot"].min()
        z_max = halpha["z_phot"].max()
        data_vol_mpc3 = zbin_volume(
            MINERVA_UDS_AREA_DEG2, zlow=z_min, zhigh=z_max
        ).value

        lgLHa = jnp.log10(
            abs(halpha["L_Ha"].data)
        )  # one of the Halpha L was negative, probably an error

        HalphaLf = namedtuple(f, Lf._fields)
        N_1d, sig, bin_lo, bin_hi = get_N_1d(lgLHa)
        lf_data = HalphaLf(sig, bin_lo, bin_hi, N_1d, True)

        z_phot_table = 10 ** jnp.linspace(
            jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
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

        lc_data = generate_lc_data(*lc_args)

        lfs.append(LumFunc(z_min, z_max, data_vol_mpc3, lc_data, lf_data))
    return lfs


HalphaFilters = namedtuple(
    "HalphaFilters",
    [
        "f162m",
        "f182m",
        "f210m",
        "f250m",
        "f300m",
        "f360m",
        "f410m",
        "f460m",
    ],
)


PhotFilters = namedtuple(
    "PhotFilters",
    [
        "f435w",
        "f606w",
        "f814w",
        "f105w",
        "f125w",
        "f160w",
        "f090w",
        "f115w",
        "f140m",
        "f150w",
        "f162m",
        "f182m",
        "f200w",
        "f210m",
        "f250m",
        "f277w",
        "f300m",
        "f356w",
        "f360m",
        "f410m",
        "f444w",
        "f460m",
    ],
)
