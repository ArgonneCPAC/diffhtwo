import numpy as np
from diffsky.experimental.mc_lightcone_generators import mc_lc_photdata
from diffsky.experimental.mc_phot import mc_lc_phot
from jax import random as jran
from scipy.stats import binned_statistic

from ..utils import weighted_percentiles
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
    for b in range(len(logmp_bins) - 1):
        cen_in_bin = (
            (logmp_obs > logmp_bins[b])
            & (logmp_obs <= logmp_bins[b + 1])
            & (is_central == 1)
        )

        if not np.any(cen_in_bin):
            ex_situ_frac_median.append(0.0)
            continue

        l16, median, u84 = weighted_percentiles(
            ex_situ_frac[cen_in_bin], gal_weight[cen_in_bin]
        )
        ex_situ_frac_median.append(median)

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

        if not np.any(cen_in_bin):
            ex_situ_frac_median.append(0.0)
            continue

        l16, median, u84 = weighted_percentiles(
            ex_situ_frac[cen_in_bin], gal_weight[cen_in_bin]
        )
        ex_situ_frac_median.append(median)

    ex_situ_frac_median = np.array(ex_situ_frac_median)

    return ex_situ_frac_median


def _get_logsm_obs_weighted_median(logmp_bins, logmp_obs, logsm_obs, gal_weight):
    logsm_obs_weighted_l16 = []
    logsm_obs_weighted_median = []
    logsm_obs_weighted_u84 = []
    for b in range(0, len(logmp_bins) - 1):
        in_bin = (logmp_obs > logmp_bins[b]) & (logmp_obs <= logmp_bins[b + 1])

        if in_bin.sum() > 0:
            l16, median, u84 = weighted_percentiles(
                logsm_obs[in_bin], gal_weight[in_bin]
            )
            logsm_obs_weighted_l16.append(l16)
            logsm_obs_weighted_median.append(median)
            logsm_obs_weighted_u84.append(u84)

        else:
            logsm_obs_weighted_l16.append(np.nan)
            logsm_obs_weighted_median.append(np.nan)
            logsm_obs_weighted_u84.append(np.nan)

    logsm_obs_weighted_l16 = np.array(logsm_obs_weighted_l16)
    logsm_obs_weighted_median = np.array(logsm_obs_weighted_median)
    logsm_obs_weighted_u84 = np.array(logsm_obs_weighted_u84)

    return logsm_obs_weighted_l16, logsm_obs_weighted_median, logsm_obs_weighted_u84


def mc_median_smhm(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    lgmp_min=10.5,
    lgmp_sub_min=10.5,
    lgmp_max=15.0,
    sky_area_degsq=0.1,
    d_mh=0.15,
    mc_merge=1,
):
    z_phot_table = np.linspace(z_min, z_max, 25)
    args = (
        ran_key,
        z_min,
        z_max,
        lgmp_min,
        lgmp_sub_min,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = mc_lc_photdata(*args)
    ran_key, sed_key = jran.split(ran_key, 2)

    phot_info, phot_randoms, merging_randoms = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection
    )

    lgmp_bins = np.arange(lgmp_min, lgmp_max + d_mh, d_mh)
    lgmp_bin_centers = (lgmp_bins[:-1] + lgmp_bins[1:]) / 2

    # cen+sat in+ex-situ
    median_logsm_obs, _, _ = binned_statistic(
        lc_data.logmp_obs, phot_info.logsm_obs, statistic="median", bins=lgmp_bins
    )

    # cen in-situ
    median_logsm_obs_cen_in_situ, _, _ = binned_statistic(
        lc_data.logmp_obs[lc_data.is_central == 1],
        phot_info.logsm_obs_in_situ[lc_data.is_central == 1],
        statistic="median",
        bins=lgmp_bins,
    )

    # cen in+ex-situ
    median_logsm_obs_cen, _, _ = binned_statistic(
        lc_data.logmp_obs[lc_data.is_central == 1],
        phot_info.logsm_obs[lc_data.is_central == 1],
        statistic="median",
        bins=lgmp_bins,
    )

    # sat in-situ
    median_logsm_obs_sat_in_situ, _, _ = binned_statistic(
        lc_data.logmp_obs[lc_data.is_central != 1],
        phot_info.logsm_obs_in_situ[lc_data.is_central != 1],
        statistic="median",
        bins=lgmp_bins,
    )

    # sat post-merging (as sats don't accrete but only lose stellar mass, so no ex-situ)
    median_logsm_obs_sat, _, _ = binned_statistic(
        lc_data.logmp_obs[lc_data.is_central != 1],
        phot_info.logsm_obs[lc_data.is_central != 1],
        statistic="median",
        bins=lgmp_bins,
    )

    return (
        lgmp_bin_centers,
        median_logsm_obs,
        median_logsm_obs_cen_in_situ,
        median_logsm_obs_cen,
        median_logsm_obs_sat_in_situ,
        median_logsm_obs_sat,
    )


def median_smhm_and_exsitu_frac(
    ran_key,
    param_collection,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    tcurves,
    lgmp_min=10.0,
    lgmp_max=15.0,
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
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
    )

    logmp_bins = np.arange(lgmp_min, lgmp_max + d_mh, d_mh)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2

    # cen+sat in+ex-situ
    _, logsm_obs_weighted_median, _ = _get_logsm_obs_weighted_median(
        logmp_bins, lc_data.logmp_obs, phot_data.logsm_obs, gal_weight
    )

    # cen in-situ
    _, logsm_obs_weighted_median_cen_in_situ, _ = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central == 1],
        phot_data.logsm_obs_in_situ[lc_data.is_central == 1],
        gal_weight[lc_data.is_central == 1],
    )

    # cen in+ex-situ
    _, logsm_obs_weighted_median_cen, _ = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central == 1],
        phot_data.logsm_obs[lc_data.is_central == 1],
        gal_weight[lc_data.is_central == 1],
    )

    # sat in-situ
    _, logsm_obs_weighted_median_sat_in_situ, _ = _get_logsm_obs_weighted_median(
        logmp_bins,
        lc_data.logmp_obs[lc_data.is_central != 1],
        phot_data.logsm_obs_in_situ[lc_data.is_central != 1],
        gal_weight[lc_data.is_central != 1],
    )

    # sat post-merging (as sats don't accrete but only lose stellar mass, so no ex-situ)
    _, logsm_obs_weighted_median_sat, _ = _get_logsm_obs_weighted_median(
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
