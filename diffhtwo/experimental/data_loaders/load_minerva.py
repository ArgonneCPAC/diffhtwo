from collections import namedtuple
from difflib import get_close_matches
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from dsps.data_loaders import load_transmission_curve
from dsps.data_loaders.defaults import TransmissionCurve

from ..defaults import (
    MINERVA_AREA_DEG2,
    AppMagFunc,
    ColorColor,
    ColorCondMag,
    FilterInfo,
    Lf,
    MagColor,
)
from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data
from .N_utils import get_N_1d, get_N_2d

PHOT_CAT = "MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits"
EAZY_CAT = "MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_zpiter_CATALOG_larson.zout.fits"
BASE_PATH = Path(__file__).resolve().parent.parent
MINERVA_FILTERS_PATH = BASE_PATH / "data" / "minerva_filters"

TRANSLATE = "MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_zpiter_CATALOG.larson.eazypy.zphot.translate"
INFO = "FILTER.RES.latest.info"
TCURVES = "FILTER.RES.latest"

MinervaPhot = namedtuple(
    "MinervaPhot",
    [
        "redshift",
        "mags",
        "sels",
        "mags_labels",
        "app_mag_funcs",
        "zbins",
        "filter_info",
        "frac_cat",
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


def get_minerva_phot(
    drn,
    ran_key,
    ssp_data,
    num_halos=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
    phot_cat=PHOT_CAT,
    eazy_cat=EAZY_CAT,
):
    drn = Path(drn)
    phot = Table.read(drn / phot_cat)
    zout = Table.read(drn / eazy_cat)

    spec_avail = zout["z_spec"] != -99.0
    z_best = zout["z_ml"].copy()
    z_best[spec_avail] = zout["z_spec"][spec_avail]
    z_best = np.float32(z_best)

    use_phot = phot["use_phot"] == 1
    phot = phot[use_phot]
    zout = zout[use_phot]
    z_best = z_best[use_phot].data

    minerva_mag_thresh = PhotFilters(
        f435w=(19.0, 28.0),
        f606w=(19.0, 28.0),
        f814w=(19.0, 28.0),
        f125w=(19.0, 28.0),
        f140w=(19.0, 28.0),
        f160w=(19.0, 28.0),
        f090w=(19.0, 28.0),
        f115w=(19.0, 28.0),
        f150w=(19.0, 28.0),
        f200w=(19.0, 28.0),
        f277w=(19.0, 28.0),
        f356w=(19.0, 28.0),
        f444w=(19.0, 28.0),
    )

    tcurves = []
    mag_per_band = []
    mag_labels = []
    sel_per_band = []
    tcurves = []
    for minerva_filter in PhotFilters._fields:
        tcurve = load_transmission_curve(
            bn_pat=minerva_filter + "*", drn=MINERVA_FILTERS_PATH
        )
        tcurves.append(tcurve)

        # get magnitudes
        col_name = "f_" + minerva_filter
        mag = _get_mag_ab(phot, col_name)

        # originally masked in the phot cat
        sel = ~phot[col_name].mask

        # masked due to converting flux to mag for negative flux for instance (like drop outs)
        sel *= np.isfinite(mag)

        # mag thresh selection
        mag_limit = getattr(minerva_mag_thresh, minerva_filter)
        sel *= (mag > mag_limit[0]) & (mag < mag_limit[1])

        mag_per_band.append(mag)
        sel_per_band.append(sel)

        mag_labels.append(minerva_filter)

    mags = np.vstack(mag_per_band).T
    sels = np.vstack(sel_per_band).T

    filter_info = FilterInfo(minerva_mag_thresh, tcurves)

    z_bins = np.array(
        [
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]
    )
    AppMagFuncs = namedtuple(
        "AppMagFuncs",
        ["z_min", "z_max", "data_vol_mpc3", "lc_data", *PhotFilters._fields],
    )

    filter_namedtuples = {
        f: namedtuple(f, AppMagFunc._fields) for f in PhotFilters._fields
    }

    app_mag_funcs = []
    for zbin in range(len(z_bins)):
        z_min = z_bins[zbin][0]
        z_max = z_bins[zbin][1]
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

        lc_data = generate_lc_data(*lc_args)

        z_sel = (z_best > z_min) & (z_best <= z_max)

        band_z_tuples = []
        for i, fname in enumerate(PhotFilters._fields):
            sel = sels[:, i] * z_sel
            N_1d, sig, bin_lo, bin_hi = get_N_1d(mags[:, i][sel])
            band_z_tuples.append(
                filter_namedtuples[fname](i, sig, bin_lo, bin_hi, N_1d, True)
            )

        app_mag_funcs.append(
            AppMagFuncs(z_min, z_max, data_vol_mpc3, lc_data, *band_z_tuples)
        )

    frac_cat = 0.9
    return MinervaPhot(
        z_best,
        mags,
        sels,
        mag_labels,
        app_mag_funcs,
        z_bins,
        filter_info,
        frac_cat,
        MINERVA_AREA_DEG2,
    )


def get_minerva_phot_fitting_data(
    drn,
    ran_key,
    ssp_data,
    num_halos=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
):
    minerva_phot = get_minerva_phot(
        drn,
        ran_key,
        ssp_data,
        num_halos=num_halos,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
    )
    fields = [f for f in minerva_phot._fields if f != "mags_labels"]
    MinervaPhotFit = namedtuple("MinervaPhotFit", fields)
    return MinervaPhotFit(*(getattr(minerva_phot, f) for f in fields))


def get_minerva_halpha(
    halpha_drn,
    drn,
    ran_key,
    ssp_data,
    translate_fn=TRANSLATE,
    info_fn=INFO,
    tcurves_fn=TCURVES,
    num_halos=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=15,
):
    halpha_drn = Path(halpha_drn)
    drn = Path(drn)

    # Transmission curves and filter mag thresholds
    translate_fn = drn / translate_fn
    info_fn = drn / info_fn
    tcurves_fn = drn / tcurves_fn

    translate = dict(line.split() for line in open(translate_fn))
    tcurves = []
    for halpha_filter in HalphaFilters._fields:
        col_name = "f_" + halpha_filter

        # get tcurve
        filter_number = int(translate[col_name][1:])
        wave_aa, trans = _get_tcurve(filter_number, info_fn, tcurves_fn)
        tcurves.append(TransmissionCurve(wave_aa, trans))

    LumFunc = namedtuple(
        "LumFunc",
        ["z_min", "z_max", "data_vol_mpc3", "lc_data", "lf_data"],
    )
    lfs = []
    for f in HalphaFilters._fields:
        halpha = Table.read(halpha_drn / f"Ha_table_{f}_minerva-uds_power.fits")
        z_min = halpha["z_phot"].min()
        z_max = halpha["z_phot"].max()
        data_vol_mpc3 = zbin_volume(MINERVA_AREA_DEG2, zlow=z_min, zhigh=z_max).value

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


PhotFilters = namedtuple(
    "PhotFilters",
    [
        "f435w",
        "f606w",
        "f814w",
        "f125w",
        "f140w",
        "f160w",
        "f090w",
        "f115w",
        "f150w",
        "f200w",
        "f277w",
        "f356w",
        "f444w",
    ],
)

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


# PhotFilters = namedtuple(
#     "PhotFilters",
#     [
#         "f435w",
#         "f606w",
#         "f775w",
#         "f814w",
#         "f098m",
#         "f105w",
#         "f125w",
#         "f140w",
#         "f160w",
#         "f090w",
#         "f115w",
#         "f140m",
#         "f150w",
#         "f162m",
#         "f182m",
#         "f200w",
#         "f210m",
#         "f250m",
#         "f277w",
#         "f300m",
#         "f335m",
#         "f356w",
#         "f360m",
#         "f410m",
#         "f430m",
#         "f444w",
#         "f460m",
#         "f480m",
#     ],
# )
