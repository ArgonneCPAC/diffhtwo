import os
import re
from collections import namedtuple
from difflib import get_close_matches
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from astropy.table import Table, vstack
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_transmission_curve

from ..defaults import (
    MINERVA_AREA_DEG2,
    FilterInfo,
)
from ..lc_utils import zbin_vol
from ..lightcone_generators import generate_lc_data
from .minerva_area import get_minerva_area_arcminsq
from .space_utils import (
    AppMagFunc,
    ColorColor,
    CondColor,
    Lf,
    MagColor,
    get_ccd_spaces_at_z,
    get_cmd_spaces_at_z,
    get_condc_spaces_at_z,
    get_mag_spaces_at_z,
    get_N_1d,
    split_space_entries,
)

BASE_PATH = Path(__file__).resolve().parent.parent

MINERVA_FILTERS_PATH = BASE_PATH / "data" / "minerva_filters"

IGM_DRN = BASE_PATH / "data" / "igm"
IGM_BN = "igm_attenuation_minerva.h5"

PHOT = "minerva_uds_cosmos_phot.fits"
ZOUT = "minerva_uds_cosmos_zout.fits"

TRANSLATE = "MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_zpiter_CATALOG.larson.eazypy.zphot.translate"
INFO = "FILTER.RES.latest.info"
TCURVES = "FILTER.RES.latest"

AREA_FILE = "minerva_space_areas_w_f444w_arcminsq.dat"

# space binning defaults
D_MAG_1d = 0.2
D_MAG_2d = 0.2
GAUSS_SIG_2d = 3.0
D_MAG_COND = 1.0

# Halo lightcone defaults
NUM_HALOS = 150
LGMP_MIN = 10.0
LGMP_MAX = 15.0
LC_SKY_AREA_DEGSQ = 100
N_Z_PHOT_TABLE = 30

ZP = 28.9

MinervaPhot = namedtuple(
    "MinervaPhot",
    [
        "redshift",
        "mags",
        "mags_err",
        "sels",
        "parent_cut_idx",
        "mags_labels",
        "spaces",
        "zbins",
        "filter_info",
        "data_sky_area_degsq",
    ],
)

SPACES_AT_Z = [
    {
        "z": (1.0, 2.0),
        "md": [
            ("F435w", True, (1.0, 2.0)),
            ("F606w", True, (1.0, 2.0)),
            ("F814w", True, (1.0, 2.0)),
            ("F090w", True, (1.0, 2.0)),
            ("F115w", True, (1.0, 2.0)),
            ("F150w", True, (1.0, 2.0)),
            ("F200w", True, (1.0, 2.0)),
            ("F277w", True, (1.0, 2.0)),
            ("F356w", True, (1.0, 2.0)),
            ("F444w", True, (1.0, 2.0)),
        ],
        "ccd": [("F090wF150w_F150wF356w", True, (1.0, 2.0))],
        "cmd": [
            ("F356w_F150wF356w", False, (1.0, 2.0)),
            ("F140m_F150wF140m", False, (1.03, 1.25)),
            ("F162m_F150wF162m", False, (1.36, 1.59)),
            ("F182m_F200wF182m", False, (1.67, 1.95)),
        ],
        "condc": [("F606wF150w_F444w", True, (1.0, 2.0))],
    },
    {
        "z": (2.0, 3.0),
        "md": [
            ("F435w", False, (2.0, 3.0)),
            ("F606w", True, (2.0, 3.0)),
            ("F814w", True, (2.0, 3.0)),
            ("F090w", True, (2.0, 3.0)),
            ("F115w", True, (2.0, 3.0)),
            ("F150w", True, (2.0, 3.0)),
            ("F200w", True, (2.0, 3.0)),
            ("F277w", True, (2.0, 3.0)),
            ("F356w", True, (2.0, 3.0)),
            ("F444w", True, (2.0, 3.0)),
        ],
        "ccd": [
            ("F115wF200w_F200wF356w", True, (2.0, 3.0)),
            ("F150wF200w_F200wF277w", True, (2.0, 3.0)),
        ],
        "cmd": [
            ("F356w_F115wF356w", False, (2.0, 3.0)),
            ("F210m_F200wF210m", False, (2.04, 2.35)),
            ("F250m_F277wF250m", False, (2.68, 2.95)),
        ],
        "condc": [("F814wF200w_F444w", True, (2.0, 3.0))],
    },
    {
        "z": (3.0, 4.0),
        "md": [
            ("F435w", False, (3.0, 4.0)),
            ("F606w", False, (3.0, 4.0)),
            ("F814w", True, (3.0, 4.0)),
            ("F090w", True, (3.0, 4.0)),
            ("F115w", True, (3.0, 4.0)),
            ("F150w", True, (3.0, 4.0)),
            ("F200w", True, (3.0, 4.0)),
            ("F277w", True, (3.0, 4.0)),
            ("F356w", True, (3.0, 4.0)),
            ("F444w", True, (3.0, 4.0)),
        ],
        "ccd": [
            ("F150wF277w_F277wF444w", True, (3.0, 4.0)),
            ("F200wF277w_F277wF356w", True, (3.0, 4.0)),
        ],
        "cmd": [("F356w_F150wF356w", False, (3.0, 4.0))],
        "condc": [("F115wF277w_F444w", True, (3.0, 4.0))],
    },
    {
        "z": (4.0, 5.0),
        "md": [
            ("F435w", False, (4.0, 5.0)),
            ("F606w", False, (4.0, 5.0)),
            ("F814w", True, (4.0, 5.0)),
            ("F090w", True, (4.0, 5.0)),
            ("F115w", True, (4.0, 5.0)),
            ("F150w", True, (4.0, 5.0)),
            ("F200w", True, (4.0, 5.0)),
            ("F277w", True, (4.0, 5.0)),
            ("F356w", True, (4.0, 5.0)),
            ("F444w", True, (4.0, 5.0)),
        ],
        "ccd": [
            ("F814wF150w_F150wF277w", True, (4.0, 5.0)),
            ("F200wF277w_F277wF444w", True, (4.0, 5.0)),
            ("F356wF410m_F410mF444w", True, (4.0, 5.0)),
        ],
        "cmd": [
            ("F444w_F150wF444w", False, (4.0, 5.0)),
            ("F360m_F356wF360m", False, (4.22, 4.81)),
        ],
        "condc": [("F115wF356w_F444w", True, (4.0, 5.0))],
    },
    {
        "z": (5.0, 6.5),
        "md": [
            ("F435w", False, (5.0, 6.5)),
            ("F606w", False, (5.0, 6.5)),
            ("F814w", False, (5.0, 6.5)),
            ("F090w", True, (5.0, 6.5)),
            ("F115w", True, (5.0, 6.5)),
            ("F150w", True, (5.0, 6.5)),
            ("F200w", True, (5.0, 6.5)),
            ("F277w", True, (5.0, 6.5)),
            ("F356w", True, (5.0, 6.5)),
            ("F444w", True, (5.0, 6.5)),
        ],
        "ccd": [
            ("F115wF200w_F200wF444w", True, (5.0, 6.5)),
            ("F277wF356w_F356wF444w", True, (5.0, 6.5)),
        ],
        "cmd": [
            ("F444w_F115wF444w", False, (5.0, 6.5)),
            ("F410m_F444wF410m", False, (5.0, 5.52)),
            ("F460m_F444wF460m", False, (5.88, 6.23)),
        ],
        "condc": [("F150wF444w_F444w", True, (5.0, 6.5))],
    },
]


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


def _get_mag_ab(phot_table, col_name, ZP=ZP):
    with np.errstate(invalid="ignore"):
        flux = phot_table["f_" + col_name].data.data
        mag_ab = -2.5 * np.log10(flux) + ZP

        error = phot_table["e_" + col_name].data.data
        error_ab = (2.5 / np.log(10)) * error / flux

    return mag_ab, error_ab


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
    for minerva_filter in MinervaFilters._fields:
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

    phot = vstack([uds_phot, cosmos_phot], metadata_conflicts="silent")

    common = set(uds_zout.colnames) & set(cosmos_zout.colnames) & set(egs_zout.colnames)
    common = [c for c in uds_zout.colnames if c in common]

    uds_zout = uds_zout[common]
    cosmos_zout = cosmos_zout[common]
    # egs_zout = egs_zout[common]

    zout = vstack([uds_zout, cosmos_zout], metadata_conflicts="silent")

    return phot, zout


def get_minerva_phot(
    drn,
    ran_key,
    ssp_data,
    spaces_at_z=SPACES_AT_Z,
    d_mag_1d=D_MAG_1d,
    d_mag_2d=D_MAG_2d,
    gauss_sig_2d=GAUSS_SIG_2d,
    d_mag_cond=D_MAG_COND,
    num_halos=NUM_HALOS,
    lgmp_min=LGMP_MIN,
    lgmp_max=LGMP_MAX,
    lc_sky_area_degsq=LC_SKY_AREA_DEGSQ,
    n_z_phot_table=N_Z_PHOT_TABLE,
    apply_igm=True,
    phot=PHOT,
    zout=ZOUT,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    drn = Path(drn)
    phot = Table.read(drn / phot)
    zout = Table.read(drn / zout)

    spec_avail = zout["z_spec"] != -99.0
    z_best = zout["z_ml"].copy()
    z_best[spec_avail] = zout["z_spec"][spec_avail]
    z_best = np.float32(z_best)

    use_phot = phot["use_phot"] == 1
    phot = phot[use_phot]
    zout = zout[use_phot]
    z_best = z_best[use_phot].data

    minerva_mag_thresh = MinervaFilters(
        f435w=(20.0, 27.5),
        f606w=(20.0, 27.5),
        f814w=(19.0, 27.5),
        f090w=(19.0, 27.5),
        f115w=(19.0, 27.5),
        f140m=(19.0, 27.5),
        f150w=(19.0, 27.5),
        f162m=(19.0, 27.5),
        f182m=(19.0, 27.5),
        f200w=(19.0, 27.5),
        f210m=(19.0, 27.5),
        f250m=(19.0, 27.5),
        f277w=(19.0, 27.5),
        f300m=(19.0, 27.5),
        f356w=(19.0, 27.5),
        f360m=(19.0, 27.5),
        f410m=(19.0, 27.5),
        f444w=(19.0, 27.5),
        f460m=(19.0, 27.0),
    )

    f444w_sel = ~phot["f_f444w"].mask

    f444w_mag, f444w_mag_err = _get_mag_ab(phot, "f444w")
    f444w_sel &= np.isfinite(f444w_mag)

    f444w_mag_thresh = getattr(minerva_mag_thresh, "f444w")
    f444w_sel &= (f444w_mag > f444w_mag_thresh[0]) & (f444w_mag < f444w_mag_thresh[1])

    f444w_idx = MinervaFilters._fields.index("f444w")

    tcurves = []
    mag_per_band = []
    mag_err_per_band = []
    mag_labels = []
    sel_per_band = []
    tcurves = []
    for minerva_filter in MinervaFilters._fields:
        tcurve = load_transmission_curve(
            bn_pat=minerva_filter + "*", drn=MINERVA_FILTERS_PATH
        )
        tcurves.append(tcurve)

        # get magnitudes and thresh
        mag, mag_err = _get_mag_ab(phot, minerva_filter)
        mag_limit = getattr(minerva_mag_thresh, minerva_filter)

        # originally masked in the phot cat due to missing coverage, for instance
        sel = ~phot["f_" + minerva_filter].mask

        # masked due to converting flux to mag for -ve flux of droputs, for instance
        sel &= np.isfinite(mag)

        # mag thresh selection
        sel &= (mag > mag_limit[0]) & (mag < mag_limit[1])

        # merge with parent f444w_sel
        sel &= f444w_sel

        mag_per_band.append(mag)
        mag_err_per_band.append(mag_err)
        sel_per_band.append(sel)

        mag_labels.append(minerva_filter)

    mags = np.vstack(mag_per_band).T
    mags_err = np.vstack(mag_err_per_band).T
    sels = np.vstack(sel_per_band).T

    filter_info = FilterInfo(minerva_mag_thresh, tcurves)

    z_bins = np.array([sp["z"] for sp in spaces_at_z])

    spaces = []
    for zbin in range(len(z_bins)):
        z_min = z_bins[zbin][0]
        z_max = z_bins[zbin][1]

        md, md_fit, md_z = split_space_entries(spaces_at_z[zbin].get("md", []))
        ccd, ccd_fit, ccd_z = split_space_entries(spaces_at_z[zbin].get("ccd", []))
        cmd, cmd_fit, cmd_z = split_space_entries(spaces_at_z[zbin].get("cmd", []))
        condc, condc_fit, condc_z = split_space_entries(
            spaces_at_z[zbin].get("condc", [])
        )

        mag_namedtuples = {i: namedtuple(i, AppMagFunc._fields) for i in md}
        ccd_namedtuples = {i: namedtuple(i, ColorColor._fields) for i in ccd}
        cmd_namedtuples = {i: namedtuple(i, MagColor._fields) for i in cmd}
        condc_namedtuples = {i: namedtuple(i, CondColor._fields) for i in condc}

        Spaces = namedtuple(
            "Spaces",
            [
                "z_min",
                "z_max",
                "lc_data",
                *[f"md_{n}" for n in mag_namedtuples],
                *[f"ccd_{n}" for n in ccd_namedtuples],
                *[f"cmd_{n}" for n in cmd_namedtuples],
                *[f"condc_{n}" for n in condc_namedtuples],
            ],
        )

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
            igm_filters_namedtuple=MinervaFilters,
            igm_filter_prefix="minerva_",
            logmp_cutoff=lgmp_min,
            cosmo_params=cosmo_params,
        )

        mag_z_tuples = get_mag_spaces_at_z(
            mag_namedtuples,
            MinervaFilters,
            md_z,
            md_fit,
            z_best,
            sels,
            mags,
            f444w_idx,
            d_mag_1d,
            lc_sky_area_degsq,
            data_sky_area_degsq=None,
            minerva_drn=drn,
            minerva_area_file=AREA_FILE,
        )

        ccd_z_tuples = get_ccd_spaces_at_z(
            ccd_namedtuples,
            MinervaFilters,
            ccd_z,
            ccd_fit,
            z_best,
            sels,
            mags,
            f444w_idx,
            d_mag_2d,
            gauss_sig_2d,
            lc_sky_area_degsq,
            data_sky_area_degsq=None,
            minerva_drn=drn,
            minerva_area_file=AREA_FILE,
        )

        cmd_z_tuples = get_cmd_spaces_at_z(
            cmd_namedtuples,
            MinervaFilters,
            cmd_z,
            cmd_fit,
            z_best,
            sels,
            mags,
            f444w_idx,
            d_mag_2d,
            gauss_sig_2d,
            lc_sky_area_degsq,
            data_sky_area_degsq=None,
            minerva_drn=drn,
            minerva_area_file=AREA_FILE,
            mags_err=mags_err,
            ZP=ZP,
        )

        condc_z_lists = get_condc_spaces_at_z(
            condc_namedtuples,
            MinervaFilters,
            condc_z,
            condc_fit,
            z_best,
            sels,
            mags,
            f444w_idx,
            d_mag_1d,
            d_mag_cond,
            lc_sky_area_degsq,
            data_sky_area_degsq=None,
            minerva_drn=drn,
            minerva_area_file=AREA_FILE,
        )

        spaces.append(
            Spaces(
                z_min,
                z_max,
                lc_data,
                *mag_z_tuples,
                *ccd_z_tuples,
                *cmd_z_tuples,
                *condc_z_lists,
            )
        )

    return MinervaPhot(
        z_best,
        mags,
        mags_err,
        sels,
        f444w_idx,
        mag_labels,
        spaces,
        z_bins,
        filter_info,
        MINERVA_AREA_DEG2,
    )


def get_minerva_phot_fitting_data(
    drn,
    ran_key,
    ssp_data,
    spaces_at_z=SPACES_AT_Z,
    d_mag_1d=D_MAG_1d,
    d_mag_2d=D_MAG_2d,
    gauss_sig_2d=GAUSS_SIG_2d,
    d_mag_cond=D_MAG_COND,
    num_halos=NUM_HALOS,
    lgmp_min=LGMP_MIN,
    lgmp_max=LGMP_MAX,
    apply_igm=True,
):
    minerva_phot = get_minerva_phot(
        drn,
        ran_key,
        ssp_data,
        spaces_at_z=spaces_at_z,
        d_mag_1d=d_mag_1d,
        d_mag_2d=d_mag_2d,
        gauss_sig_2d=gauss_sig_2d,
        d_mag_cond=d_mag_cond,
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
    ran_key,
    ssp_data,
    num_halos=NUM_HALOS,
    lgmp_min=LGMP_MIN,
    lgmp_max=LGMP_MAX,
    lc_sky_area_degsq=LC_SKY_AREA_DEGSQ,
    n_z_phot_table=N_Z_PHOT_TABLE,
    estimated_area_arcminsq=390,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    halpha_drn = Path(halpha_drn)
    estimated_area_degsq = estimated_area_arcminsq / 3600

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
        uds_halpha = Table.read(halpha_drn / f"Ha_table_{f}_minerva-uds_power.fits")
        cosmos_halpha = Table.read(
            halpha_drn / f"Ha_table_{f}_minerva-cosmos_power_err.fits"
        )
        halpha = vstack([uds_halpha, cosmos_halpha])

        z_min = halpha["z_phot"].min()
        z_max = halpha["z_phot"].max()

        data_vol_mpc3 = zbin_vol(estimated_area_degsq, z_min, z_max, cosmo_params)

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


MinervaFilters = namedtuple(
    "MinervaFilters",
    [
        "f435w",
        "f606w",
        "f814w",
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
