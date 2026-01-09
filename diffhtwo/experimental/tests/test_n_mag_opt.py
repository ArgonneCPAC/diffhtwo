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

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_PARAMS
)

tcurves = []
tcurves.append(retrieve_tcurves.HSC_I)

ran_key = jran.key(0)
dmag = 0.2
mag_column = 0

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
    dmag,
    mag_column,
    DEFAULT_COSMOLOGY,
    FB,
)
assert np.isfinite(lg_n_true).all()
assert np.isfinite(lg_n_avg_err_true).all()

ran_key, fit_n_key = jran.split(ran_key, 2)
loss_hist, grad_hist, u_theta_fit = n_mag_opt.fit_n(
    u_diffstarpop_theta_default,
    lg_n_true,
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
    dmag,
    mag_column,
    DEFAULT_COSMOLOGY,
    FB,
    n_steps=2,
    step_size=0.1,
)

assert np.isfinite(loss_hist).all()
assert np.isfinite(grad_hist).all()
assert np.isfinite(u_theta_fit).all()
