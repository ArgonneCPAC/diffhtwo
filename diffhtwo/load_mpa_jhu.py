"""
"""

import os

import numpy as np
from astropy.table import Table

SDSS_DRN = "/Users/aphearin/work/DATA/SDSS/DR7_MPA_JHU"
MIN_FLUX_RATIO = 1e-6

SDSS_KEYS = list(("FIBERID", "PHOTOID", "RA", "DEC", "PLUG_MAG", "Z", "KCOR_MAG"))


def load_sdss_main(drn=SDSS_DRN, magr_cut=18.0, zmin=0.02, zmax=0.2):
    sdss = Table.read(os.path.join(drn, "gal_info_dr7_v5_2.fit"))
    sdss_lines = Table.read(os.path.join(drn, "gal_line_dr7_v5_2.fit"))

    cont_keys = [key for key in sdss_lines.keys() if "CONT" == key[-4:]]

    msk_galaxy = sdss["SPECTROTYPE"] == "GALAXY"
    msk_redshift = sdss["Z_WARNING"] == 0
    msk_main = sdss["PRIMTARGET"] == 64

    msk_good = np.array(msk_galaxy & msk_redshift & msk_main)

    # kcor_mag = Synthesized gri magnitudes of the spectrum
    # after foreground dereddening and de-redshifting
    msk_magr = np.array(sdss["KCOR_MAG"][:, 1] < magr_cut)

    msk_zcut = np.array(sdss["Z"] > zmin)
    msk_zcut &= np.array(sdss["Z"] < zmax)

    msk_list = [sdss_lines[key] > 0 for key in cont_keys]
    msk_has_cont = np.ones(len(sdss_lines)).astype(bool)
    for msk in msk_list:
        msk_has_cont &= msk

    msk_main = np.array(msk_good & msk_magr & msk_zcut & msk_has_cont)

    # npts_nocuts = np.sum(msk_good & msk_magr & msk_zcut)
    # npts_has_cont = msk_main.sum()
    # frac_emline = npts_has_cont / npts_nocuts
    # print("{0:.2f} of galaxies passes emission line cut".format(frac_emline))

    sdss_out = Table()
    sdss_out["FIBERID"] = np.array(sdss["FIBERID"][msk_main])
    sdss_out["PHOTOID"] = np.array(sdss["PHOTOID"][msk_main])
    sdss_out["RA"] = np.array(sdss["RA"][msk_main])
    sdss_out["DEC"] = np.array(sdss["DEC"][msk_main])
    sdss_out["PLUG_MAG"] = np.array(sdss["PLUG_MAG"][msk_main])
    sdss_out["Z"] = np.array(sdss["Z"][msk_main])
    gri = np.array(sdss["KCOR_MAG"][msk_main])
    sdss_out["magi"] = gri[:, 2]
    sdss_out["gr"] = gri[:, 0] - gri[:, 1]
    sdss_out["ri"] = gri[:, 1] - gri[:, 2]

    for key in cont_keys:
        num = np.array(sdss_lines[key.replace("_CONT", "_FLUX")])[msk_main]
        denom = np.array(sdss_lines[key])[msk_main]
        ratio = num / denom
        ratio = np.where(ratio < MIN_FLUX_RATIO, MIN_FLUX_RATIO, ratio)
        lgratio = np.log10(ratio)
        assert np.all(np.isfinite(lgratio))
        sdss_out[key.replace("_CONT", "_lgratio")] = lgratio

    return sdss_out

    # for outkey in SDSS_KEYS:
    #     sdss_out[outkey] = sdss_main[outkey]
    # sdss_out.rename_column("KCOR_MAG", "gri")

    # sdss = sdss[msk_main]
    # sdss_lines = sdss_lines[msk_main]

    # sdss_main = vstack((sdss, sdss_lines))
    # return sdss_main, sdss, sdss_lines

    # return sdss_out
