import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path

ARCSEC_PER_PIXEL = 0.2684

edge_verts = [
    (34.9044413, -4.6567358),
    (34.8942569, -4.6511511),
    (34.0121420, -4.6499159),
    (33.9989311, -4.6629939),
    (33.9967859, -5.5432032),
    (34.0027108, -5.5497307),
    (34.8897006, -5.5494500),
    (34.8994241, -5.5450941),
    (34.9029574, -5.5399144),
    (34.9038403, -5.5381018),
    (34.9061761, -5.5346894),
    (34.9044413, -4.6567358),
]

edge_codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]


def calc_multiband_area(
    mask_drn,
):
    bands = ["HSC_G", "HSC_R", "HSC_I", "HSC_Z", "UDS_J", "UDS_H"]

    uds_area_degsq, uds_mask = get_uds_area(mask_drn)
    uds_data = 1 - uds_mask

    for band in bands:
        mask = fits.getdata(mask_drn + "/" + band + "_Mask.fits")
        data = 1 - mask
        uds_data *= data

    uds_area_arcminsq = (uds_data.sum() * (ARCSEC_PER_PIXEL**2)) / 3600
    uds_area_degsq = uds_area_arcminsq / 3600

    return uds_area_degsq


def get_uds_area(mask_drn):
    uds_Mask_data = fits.getdata(mask_drn + "/UDS_K_Mask.fits")
    uds_Mask_wcs = WCS(fits.getheader(mask_drn + "/UDS_K_Mask.fits"))

    for v in range(0, len(edge_verts)):
        coord = SkyCoord(edge_verts[v][0], edge_verts[v][1], frame="icrs", unit="deg")
        edge_verts[v] = uds_Mask_wcs.world_to_pixel(coord)

    edge_path = Path(edge_verts, edge_codes)

    y = np.arange(0, np.shape(uds_Mask_data)[0], 1)
    x = np.arange(0, np.shape(uds_Mask_data)[1], 1)

    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    within_edge_path = edge_path.contains_points(points)

    within_edge_path = within_edge_path.reshape(np.shape(uds_Mask_data))

    outside_edge_path = within_edge_path != True

    uds_Mask_data[np.where(outside_edge_path)] = 1

    uds_arcmin2 = (np.sum(uds_Mask_data == 0) * (ARCSEC_PER_PIXEL**2)) / (3600)
    uds_degsq = uds_arcmin2 / 3600

    return uds_degsq, uds_Mask_data
