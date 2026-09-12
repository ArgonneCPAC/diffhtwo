import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffmah import mah_halopop
from diffsky.experimental.mc_lightcone_generators import mc_lc_photdata
from diffsky.experimental.mc_phot import mc_lc_phot
from jax import random as jran
from jax import vmap

from ..kernels.sfr_tau import get_logsfr_100Myr
from ..tab_blue_orange_cmap import make_cmap

cmap = make_cmap()

plt.rc("font", family="serif", serif=["Times New Roman"])

interp_vmap = vmap(jnp.interp, in_axes=(0, None, 0))


def get_dmhdt_obs(lc_data):
    mah_params = lc_data.mah_params
    tarr = lc_data.t_table
    lgt0 = jnp.log10(tarr[-1])

    dmhdt, log_mah = mah_halopop(mah_params, tarr, lgt0)

    t_obs = lc_data.t_obs
    dmhdt_obs = interp_vmap(t_obs, tarr, dmhdt)

    return dmhdt_obs


def plot_sfr_har(
    ran_key,
    param_collection,
    z_min,
    z_max,
    tcurves,
    ssp_data,
    run_label,
    savedir,
    sky_area_degsq=0.1,
    lgmp_min=11,
    lgmp_sub_min=11,
    mc_merge=1,
    gridsize=100,
    plt_show=True,
):
    ran_key, lc_halo_key = jran.split(ran_key, 2)
    z_phot_table = np.linspace(z_min, z_max, 25)

    args = (
        lc_halo_key,
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
    logmp_obs = lc_data.logmp_obs
    logdmhdt_obs = np.log10(get_dmhdt_obs(lc_data))
    logsdmhdt_obs = logdmhdt_obs - logmp_obs
    # logsdmhdt_obs = logdmhdt_obs - lc_data.logmp_infall

    ran_key, sed_key = jran.split(ran_key, 2)
    phot_info, __, __ = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection
    )

    logsm_obs = phot_info.logsm_obs
    logsfr_100Myr = get_logsfr_100Myr(phot_info, lc_data, ssp_data)
    logssfr_100Myr = logsfr_100Myr - logsm_obs

    # logsdmhdt_obs = logsdmhdt_obs[phot_info.p_merge < 0.9]
    # logssfr_100Myr = logssfr_100Myr[phot_info.p_merge < 0.9]

    return (
        logdmhdt_obs,
        logsfr_100Myr,
        logsdmhdt_obs,
        logssfr_100Myr,
        lc_data,
        phot_info,
    )
