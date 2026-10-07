"""
Adapted from code by Ghassan Sarrouh
"""
import os

from astropy.io import fits

from .load_minerva import PhotFilters

PIX_SCALE = 0.04  # arcsec per pixel
FN_MASK_STARS = {
    "uds": "MINERVA-UDS_40mas_mask_stars_n3.0_v1.2.fits",
    "cosmos": "MINERVA-COSMOS_40mas_mask_stars_n3.0_v1.0.fits",
    "egs": "MINERVA-EGS_40mas_mask_stars_n2.0_v1.3.fits.gz",
}


def _get_filter_coverage_from_wht(fn_wht):
    with fits.open(fn_wht, memmap=True) as hdul:
        wht = hdul[0].data
        return wht > 0.0  # mask selecting area with data


def _get_no_star_boolean(fn_mask_stars):
    with fits.open(fn_mask_stars, memmap=True) as hdul:
        mask_stars = hdul[0].data.astype(bool)
        no_stars = ~mask_stars
    return no_stars


def get_minerva_area(drn, fields=["uds"], pix_scale=PIX_SCALE):
    field_areas = {}
    for field in fields:
        field_img_drn = f"{drn}/{field}/images"

        fn_mask_stars = f"{field_img_drn}/{FN_MASK_STARS[field]}"
        no_stars = _get_no_star_boolean(fn_mask_stars)

        listdir = os.listdir(field_img_drn)
        fns_wht = [
            field_img_drn + "/" + fn
            for fn in listdir
            if any(f in fn for f in PhotFilters._fields)
        ]
        fov = _get_filter_coverage_from_wht(fns_wht[0])
        for fn_wht in fns_wht[1:]:
            fov &= _get_filter_coverage_from_wht(fn_wht)

        n_pix_coverage = (fov & no_stars).sum()
        area_arcsecsq = n_pix_coverage * pix_scale**2
        area_arcminsq = area_arcsecsq / 60**2
        field_areas[field] = area_arcminsq

    return field_areas
