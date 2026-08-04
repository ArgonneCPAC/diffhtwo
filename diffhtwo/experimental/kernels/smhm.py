import numpy as np

from ..utils import weighted_median
from .lc_phot_kern import multiband_lc_phot_kern
from .sfh_rapid_q import get_logsfr_obs


def _get_logsm_obs_weighted_mean(logmp_bins, logmp_obs, logsm_obs, gal_weight):
    logsm_obs_weighted_mean = []
    for b in range(0, len(logmp_bins) - 1):
        in_bin = (logmp_obs > logmp_bins[b]) & (logmp_obs <= logmp_bins[b + 1])
        try:
            weighted_mean = np.average(logsm_obs[in_bin], weights=gal_weight[in_bin])
        except ZeroDivisionError:
            weighted_mean = np.nan
        logsm_obs_weighted_mean.append(weighted_mean)
    logsm_obs_weighted_mean = np.array(logsm_obs_weighted_mean)

    return logsm_obs_weighted_mean


def get_ex_situ_frac_median_v_hm(
    logmp_bins, logmp_obs, logsm_obs, logsm_obs_in_situ, gal_weight, is_central
):
    sm_obs_ex_situ = 10**logsm_obs - 10**logsm_obs_in_situ
    sm_obs = 10**logsm_obs
    ex_situ_frac = sm_obs_ex_situ / sm_obs

    ex_situ_frac_median = []
    for b in range(0, len(logmp_bins) - 1):
        cen_in_bin = (
            (logmp_obs > logmp_bins[b])
            & (logmp_obs <= logmp_bins[b + 1])
            & (is_central == 1)
        )

        ex_situ_frac_median.append(
            weighted_median(ex_situ_frac[cen_in_bin], gal_weight[cen_in_bin])
        )
    ex_situ_frac_median = np.array(ex_situ_frac_median)

    return ex_situ_frac_median


def get_ex_situ_frac_median_v_sm(
    logsm_bins, logsm_obs, logsm_obs_in_situ, gal_weight, is_central
):
    sm_obs_ex_situ = 10**logsm_obs - 10**logsm_obs_in_situ
    sm_obs = 10**logsm_obs
    ex_situ_frac = sm_obs_ex_situ / sm_obs

    ex_situ_frac_median = []
    for b in range(0, len(logsm_bins) - 1):
        cen_in_bin = (
            (logsm_obs > logsm_bins[b])
            & (logsm_obs <= logsm_bins[b + 1])
            & (is_central == 1)
        )

        ex_situ_frac_median.append(
            weighted_median(ex_situ_frac[cen_in_bin], gal_weight[cen_in_bin])
        )
    ex_situ_frac_median = np.array(ex_situ_frac_median)

    return ex_situ_frac_median


def _get_logsm_obs_weighted_median(logmp_bins, logmp_obs, logsm_obs, gal_weight):
    logsm_obs_weighted_median = []
    for b in range(0, len(logmp_bins) - 1):
        in_bin = (logmp_obs > logmp_bins[b]) & (logmp_obs <= logmp_bins[b + 1])

        if in_bin.sum() > 0:
            logsm_obs_weighted_median.append(
                weighted_median(logsm_obs[in_bin], gal_weight[in_bin])
            )
        else:
            logsm_obs_weighted_median.append(np.nan)

    logsm_obs_weighted_median = np.array(logsm_obs_weighted_median)

    return logsm_obs_weighted_median


def median_smhm_and_exsitu_frac(
    ran_key,
    param_collection,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    tcurves,
    logmp_obs_min=10.0,
    logmp_obs_max=15.0,
    mag_thresh=None,
    frac_cat=None,
    d_mh=0.15,
):
    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        mag_thresh=mag_thresh,
        frac_cat=frac_cat,
    )

    logmp_bins = np.arange(logmp_obs_min, logmp_obs_max + d_mh, d_mh)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2

    # cen+sat in+ex-situ
    logsm_obs_weighted_median = _get_logsm_obs_weighted_median(
        logmp_bins, lc_data.logmp_obs, phot_data.logsm_obs, gal_weight
    )

    # cen in-situ
    logsm_obs_weighted_median_cen_in_situ = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central == 1],
        phot_data.logsm_obs_in_situ[lc_data.is_central == 1],
        gal_weight[lc_data.is_central == 1],
    )

    # cen in+ex-situ
    logsm_obs_weighted_median_cen = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central == 1],
        phot_data.logsm_obs[lc_data.is_central == 1],
        gal_weight[lc_data.is_central == 1],
    )

    # sat in-situ
    logsm_obs_weighted_median_sat_in_situ = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central != 1],
        phot_data.logsm_obs_in_situ[lc_data.is_central != 1],
        gal_weight[lc_data.is_central != 1],
    )

    # sat post-merging (as sats don't accrete but only lose stellar mass, so no ex-situ)
    logsm_obs_weighted_median_sat = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central != 1],
        phot_data.logsm_obs[lc_data.is_central != 1],
        gal_weight[lc_data.is_central != 1],
    )

    ex_situ_frac_median = get_ex_situ_frac_median_v_hm(
        logmp_bins,
        lc_data.logmp_obs,
        phot_data.logsm_obs,
        phot_data.logsm_obs_in_situ,
        gal_weight,
        lc_data.is_central,
    )

    return (
        logmp_bin_centers,
        logsm_obs_weighted_median,
        logsm_obs_weighted_median_cen_in_situ,
        logsm_obs_weighted_median_cen,
        logsm_obs_weighted_median_sat_in_situ,
        logsm_obs_weighted_median_sat,
        ex_situ_frac_median,
    )


def median_smhm_q_sf(
    ran_key,
    param_collection,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    tcurves,
    logmp_obs_min=10.0,
    logmp_obs_max=15.0,
    mag_thresh=None,
    frac_cat=None,
    d_mh=0.15,
):
    (
        logsfr_obs,
        logsm_obs,
        logsfr_obs_in_situ,
        logsm_obs_in_situ,
        t_q,
        lc_data,
        phot_data,
        gal_weight,
    ) = get_logsfr_obs(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        mag_thresh=mag_thresh,
        frac_cat=frac_cat,
    )

    logmp_bins = np.arange(logmp_obs_min, logmp_obs_max + d_mh, d_mh)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2

    # all
    logsm_obs_weighted_median = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs,
        logsm_obs,
        gal_weight,
    )

    quenched = t_q < lc_data.t_obs

    # SF
    logsm_obs_weighted_median_sf = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[~quenched],
        logsm_obs_in_situ[~quenched],
        gal_weight[~quenched],
    )

    # quenched
    logsm_obs_weighted_median_q = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[quenched],
        logsm_obs[quenched],
        gal_weight[quenched],
    )

    return (
        logmp_bin_centers,
        logsm_obs_weighted_median,
        logsm_obs_weighted_median_sf,
        logsm_obs_weighted_median_q,
    )
