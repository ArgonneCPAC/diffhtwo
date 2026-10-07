"""
Adapted from code by Ghassan Sarrouh
"""
import os

from astropy.io import fits

from .load_minerva import PhotFilters

PIX_SCALE = 0.04  # arcsec per pixel
FN_MASK_STARS = "MINERVA-UDS_40mas_mask_stars_n3.0_v1.2.fits"


def _get_filter_coverage_from_wht(fn_wht):
    with fits.open(fn_wht, memmap=True) as hdul:
        wht = hdul[0].data
        return wht > 0.0  # mask selecting area with data


def get_minerva_area(img_drn, pix_scale=PIX_SCALE):
    with fits.open(img_drn + "/" + FN_MASK_STARS, memmap=True) as hdul:
        mask_stars = hdul[0].data.astype(bool)
        no_stars = ~mask_stars

    listdir = os.listdir(img_drn)
    fns_wht = [
        img_drn + "/" + fn
        for fn in listdir
        if any(f in fn for f in PhotFilters._fields)
    ]
    print(fns_wht)
    fov = _get_filter_coverage_from_wht(fns_wht[0])
    for fn_wht in fns_wht[1:]:
        fov &= _get_filter_coverage_from_wht(fn_wht)

    n_pix_coverage = (fov & no_stars).sum()
    area_arcsecsq = n_pix_coverage * pix_scale**2
    area_arcminsq = area_arcsecsq / 60**2

    return area_arcminsq
