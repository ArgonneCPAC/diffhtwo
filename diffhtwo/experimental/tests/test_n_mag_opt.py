import jax.numpy as jnp
import numpy as np
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_PARAMS
from diffsky.ssp_err_model.ssp_err_model import ZERO_SSPERR_PARAMS
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity.umzr import DEFAULT_MZR_PARAMS
from jax import random as jran
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental import n_mag, n_mag_opt
from diffhtwo.experimental.data_loaders import retrieve_tcurves

from ..data_loaders import retrieve_fake_fsps_halpha

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_PARAMS
)

tcurves = []
tcurves.append(retrieve_tcurves.HSC_I)

ran_key = jran.key(0)
dmag = 0.2
mag_columns = [0]
mag_thresh_column = 0
mag_thresh = 24.5

"""Halo lightcone"""
ran_key, lc_key = jran.split(ran_key, 2)
zmin, zmax = 0.2, 0.5
lgmp_min = 10.0
sky_area_degsq = 10.0
lc_vol = jnp.array(6286141.795310545)  # copied from output of zbin_volume locally

"""weighted mc lightcone"""
num_halos = 5000
lgmp_max = 15.0
args = (lc_key, num_halos, zmin, zmax, lgmp_min, lgmp_max, sky_area_degsq)
lc_halopop = mclh.mc_weighted_halo_lightcone(*args)
lc_halopop["lc_vol_Mpc3"] = lc_vol


n_z_phot_table = 15

z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)
t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
lgt0 = jnp.log10(t_0)
t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
    tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
)

wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

bin_edges = np.arange(18.0 - dmag / 2, 26.0, dmag)
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_centers = bin_centers.reshape(bin_centers.size, 1)
dmag_centers = jnp.ones((bin_centers.shape[0], 1)) * dmag
lg_n_thresh = -8

ran_key, n_key = jran.split(ran_key, 2)
lg_n_true, lg_n_avg_err_true = n_mag.n_mag_kern(
    DIFFSTARPOP_UM_plus_exsitu,
    DEFAULT_SPSPOP_PARAMS,
    n_key,
    jnp.array(lc_halopop["z_obs"]),
    lc_halopop["t_obs"],
    lc_halopop["mah_params"],
    lc_halopop["logmp0"],
    lc_halopop["nhalos"],
    lc_halopop["lc_vol_Mpc3"],
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    DEFAULT_MZR_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    ZERO_SSPERR_PARAMS,
    bin_centers,
    dmag_centers,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    DEFAULT_COSMOLOGY,
    FB,
)
assert np.isfinite(lg_n_true).all()
assert np.isfinite(lg_n_avg_err_true).all()

ran_key, fit_n_key = jran.split(ran_key, 2)
loss_hist, grad_hist, u_theta_fit = n_mag_opt.fit_n(
    u_diffstarpop_theta_default,
    lg_n_true,
    lg_n_thresh,
    fit_n_key,
    jnp.array(lc_halopop["z_obs"]),
    lc_halopop["t_obs"],
    lc_halopop["mah_params"],
    lc_halopop["logmp0"],
    lc_halopop["nhalos"],
    lc_halopop["lc_vol_Mpc3"],
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    DEFAULT_MZR_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    bin_centers,
    dmag_centers,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    DEFAULT_COSMOLOGY,
    FB,
    n_steps=2,
    step_size=0.1,
)

assert np.isfinite(loss_hist).all()
assert np.isfinite(grad_hist).all()
assert np.isfinite(u_theta_fit).all()

ssp_halpha_luminosity = retrieve_fake_fsps_halpha.load_fake_ssp_halpha()

# Sobral+13 (HiZELS) z=0.4 halpha LF copied
lg_halpha_LF_target = jnp.array(
    [
        [
            -1.70275854,
            -1.74275854,
            -1.85275854,
            -1.97275854,
            -2.00275854,
            -2.07275854,
            -2.16275854,
            -2.31275854,
            -2.33275854,
            -2.46275854,
            -2.50275854,
            -2.61275854,
            -2.73275854,
            -2.77275854,
            -2.92275854,
            -3.07275854,
            -3.60275854,
            -3.75275854,
        ],
        [
            0.04,
            0.04,
            0.04,
            0.05,
            0.07,
            0.07,
            0.09,
            0.08,
            0.09,
            0.1,
            0.11,
            0.13,
            0.19,
            0.17,
            0.2,
            0.35,
            0.51,
            0.71,
        ],
    ]
)

lg_halpha_Lbin_edges = jnp.array(
    [
        40.05,
        40.15,
        40.25,
        40.35,
        40.45,
        40.55,
        40.65,
        40.75,
        40.85,
        40.95,
        41.05,
        41.15,
        41.25,
        41.35,
        41.45,
        41.55,
        41.7,
        41.95,
        42.25,
    ]
)
halpha_loss = n_mag_opt.get_halpha_loss(
    DIFFSTARPOP_UM_plus_exsitu,
    ran_key,
    lg_halpha_LF_target,
    lg_halpha_Lbin_edges,
    lg_n_thresh,
    lc_halopop["z_obs"],
    lc_halopop["t_obs"],
    lc_halopop["mah_params"],
    lc_halopop["nhalos"],
    lc_vol,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    DEFAULT_MZR_PARAMS,
    DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_COSMOLOGY,
    FB,
)
assert np.isfinite(halpha_loss)
