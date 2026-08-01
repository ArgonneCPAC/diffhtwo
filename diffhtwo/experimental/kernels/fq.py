import numpy as np


def get_fq_sm(
    logsm_obs,
    logsfr_obs,
    t_q,
    lc_data,
    phot_data,
    gal_weight,
    logsfms_func_at_z=None,
    type="all",
    quench_thresh="t_q",
    d_sm=0.15,
):
    if type == "all":
        sel = np.isfinite(lc_data.is_central)
    elif type == "cen":
        sel = lc_data.is_central == 1
    elif type == "sat":
        sel = lc_data.is_central != 1

    logsm_obs = logsm_obs[sel]
    logsfr_obs = logsfr_obs[sel]
    t_obs = lc_data.t_obs[sel]
    t_q = t_q[sel]
    gal_weight = gal_weight[sel]

    logssfr_obs = logsfr_obs - logsm_obs

    if quench_thresh == "t_q":
        quenched = t_q < t_obs

    elif quench_thresh == "lgssfr":
        quenched = logssfr_obs < -11

    elif quench_thresh == "MS-1dex":
        sfms = logsfms_func_at_z(logsm_obs)
        quenched = logsfr_obs < sfms - 1

    fq_list = []
    logsm_bins = np.arange(logsm_obs.min(), logsm_obs.max() + d_sm, d_sm)
    logsm_bin_centers = (logsm_bins[:-1] + logsm_bins[1:]) / 2
    for b in range(0, len(logsm_bins) - 1):
        in_bin = (logsm_obs >= logsm_bins[b]) & (logsm_obs < logsm_bins[b + 1])

        if gal_weight[in_bin].sum() > 0.0:
            f_q = gal_weight[quenched & in_bin].sum() / gal_weight[in_bin].sum()
        else:
            f_q = 0.0
        fq_list.append(f_q)

    f_q_arr = np.array(fq_list)

    return f_q_arr, logsm_bin_centers


def get_fq_hm(
    logsm_obs,
    logsfr_obs,
    t_q,
    lc_data,
    phot_data,
    gal_weight,
    logsfms_func_at_z,
    type="all",
    quench_thresh="t_q",
    d_hm=0.15,
):
    if type == "all":
        sel = np.isfinite(lc_data.is_central)
    elif type == "cen":
        sel = lc_data.is_central == 1
    elif type == "sat":
        sel = lc_data.is_central != 1

    logmp_obs = lc_data.logmp_obs[sel]
    logsm_obs = logsm_obs[sel]
    logsfr_obs = logsfr_obs[sel]
    t_obs = lc_data.t_obs[sel]
    t_q = t_q[sel]
    gal_weight = gal_weight[sel]

    logssfr_obs = logsfr_obs - logsm_obs

    if quench_thresh == "t_q":
        quenched = t_q < t_obs

    elif quench_thresh == "lgssfr":
        quenched = logssfr_obs < -11

    elif quench_thresh == "MS-1dex":
        sfms = logsfms_func_at_z(logsm_obs)
        quenched = logsfr_obs < sfms - 1

    fq_list = []
    logmp_bins = np.arange(logmp_obs.min(), logmp_obs.max() + d_hm, d_hm)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2
    for b in range(0, len(logmp_bins) - 1):
        in_bin = (logmp_obs >= logmp_bins[b]) & (logmp_obs < logmp_bins[b + 1])

        f_q = gal_weight[quenched & in_bin].sum() / gal_weight[in_bin].sum()
        fq_list.append(f_q)

    f_q_arr = np.array(fq_list)

    return f_q_arr, logmp_bin_centers
