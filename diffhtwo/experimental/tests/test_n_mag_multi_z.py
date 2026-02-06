import os

import jax.numpy as jnp
import numpy as np
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils.spspop_param_utils import (
    DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SPSPOP_U_PARAMS,
)
from diffsky.ssp_err_model.defaults import ZERO_SSPERR_PARAMS, ZERO_SSPERR_U_PARAMS
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
)
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity.umzr import DEFAULT_MZR_PARAMS
from jax import random as jran
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental import n_mag
from diffhtwo.experimental.data_loaders import retrieve_tcurves
from diffhtwo.experimental.utils import zbin_volume

from .. import n_mag_opt

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(TEST_DIR, "..", "data_loaders")


ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_PARAMS
)


zbins = np.array(
    [
        [0.2, 0.5],
        [1.5, 1.75],
        [2.75, 3.5],
    ]
)

# Halo lightcone
ran_key = jran.key(0)
lc_halopop_multi_z = []
for zbin in range(0, len(zbins)):
    ran_key, lc_key = jran.split(ran_key, 2)
    lgmp_min = 10.0
    sky_area_degsq = 10.0
    lc_vol = zbin_volume(
        sky_area_degsq, zlow=zbins[zbin][0], zhigh=zbins[zbin][1]
    ).value
    lc_vol = jnp.array(lc_vol)

    """weighted mc lightcone"""
    num_halos = 5000
    lgmp_max = 15.0
    args = (
        lc_key,
        num_halos,
        zbins[zbin][0],
        zbins[zbin][1],
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
    )
    lc_halopop = mclh.mc_weighted_halo_lightcone(*args)
    lc_halopop["lc_vol_Mpc3"] = lc_vol
    lc_halopop_multi_z.append(lc_halopop)

# Transmission curves
tcurves = [
    retrieve_tcurves.MegaCam_uS,
    retrieve_tcurves.HSC_G,
    retrieve_tcurves.HSC_R,
    retrieve_tcurves.HSC_I,
    retrieve_tcurves.HSC_Z,
]

mag_column = 3
mag_thresh = 24.5
dmag = 0.2

n_z_phot_table = 15

lc_halopop_z_obs_multi_z = []
lc_halopop_t_obs_multi_z = []
lc_halopop_mah_params_multi_z = []
lc_halopop_nhalos_multi_z = []
lc_halopop_logmp0_multi_z = []
lc_halopop_vol_mpc3_multi_z = []

t_table_multi_z = []
precomputed_ssp_mag_table_multi_z = []
z_phot_table_multi_z = []
wave_eff_table_multi_z = []

lh_centroids_multi_z = []
for zbin in range(0, len(zbins)):
    zmin = zbins[zbin][0]
    zmax = zbins[zbin][1]

    z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = jnp.log10(t_0)
    t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
    )

    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    lc_halopop = lc_halopop_multi_z[zbin]

    lh_centroids = jnp.asarray(
        np.load(
            os.path.join(
                DATA_PATH,
                "lh_centroids_z_"
                + str(zbins[zbin][0])
                + "-"
                + str(zbins[zbin][1])
                + "_test.npy",
            )
        )
    )

    lc_halopop_z_obs_multi_z.append(lc_halopop["z_obs"])
    lc_halopop_t_obs_multi_z.append(lc_halopop["t_obs"])
    lc_halopop_mah_params_multi_z.append(lc_halopop["mah_params"])
    lc_halopop_logmp0_multi_z.append(lc_halopop["logmp0"])
    lc_halopop_nhalos_multi_z.append(lc_halopop["nhalos"])
    lc_halopop_vol_mpc3_multi_z.append(lc_halopop["lc_vol_Mpc3"])
    t_table_multi_z.append(t_table)
    precomputed_ssp_mag_table_multi_z.append(precomputed_ssp_mag_table)
    z_phot_table_multi_z.append(z_phot_table)
    wave_eff_table_multi_z.append(wave_eff_table)
    lh_centroids_multi_z.append(lh_centroids)

lc_halopop_z_obs_multi_z = jnp.asarray(lc_halopop_z_obs_multi_z)
lc_halopop_t_obs_multi_z = jnp.asarray(lc_halopop_t_obs_multi_z)
lc_halopop_mah_params_multi_z = jnp.asarray(lc_halopop_mah_params_multi_z)
lc_halopop_logmp0_multi_z = jnp.asarray(lc_halopop_logmp0_multi_z)
lc_halopop_nhalos_multi_z = jnp.asarray(lc_halopop_nhalos_multi_z)
lc_halopop_vol_mpc3_multi_z = jnp.asarray(lc_halopop_vol_mpc3_multi_z)
t_table_multi_z = jnp.asarray(t_table_multi_z)
precomputed_ssp_mag_table_multi_z = jnp.asarray(precomputed_ssp_mag_table_multi_z)
z_phot_table_multi_z = jnp.asarray(z_phot_table_multi_z)
wave_eff_table_multi_z = jnp.asarray(wave_eff_table_multi_z)
lh_centroids_multi_z = jnp.asarray(lh_centroids_multi_z)


ran_key, n_key = jran.split(ran_key, 2)
n_args_multi_z = (
    DEFAULT_SPSPOP_PARAMS,
    n_key,
    lc_halopop_z_obs_multi_z,
    lc_halopop_t_obs_multi_z,
    lc_halopop_mah_params_multi_z,
    lc_halopop_logmp0_multi_z,
    lc_halopop_nhalos_multi_z,
    lc_halopop_vol_mpc3_multi_z,
    t_table_multi_z,
    ssp_data,
    precomputed_ssp_mag_table_multi_z,
    z_phot_table_multi_z,
    wave_eff_table_multi_z,
    DEFAULT_MZR_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    ZERO_SSPERR_PARAMS,
    lh_centroids_multi_z,
    dmag,
    mag_column,
    mag_thresh,
    DEFAULT_COSMOLOGY,
    FB,
)

lg_n_multi_z, lg_n_avg_err_multi_z = n_mag.n_mag_kern_multi_z(
    DIFFSTARPOP_UM_plus_exsitu, *n_args_multi_z
)
lg_n_data_err_lh_multi_z = jnp.stack((lg_n_multi_z, lg_n_avg_err_multi_z), axis=1)

lg_n_multi_z2, lg_n_avg_err_multi_z2 = n_mag.n_mag_kern_multi_z(
    DEFAULT_DIFFSTARPOP_PARAMS, *n_args_multi_z
)

# loss w/ DEFAULT_DIFFSTARPOP when DIFFSTARPOP_UM_plus_exsitu is the target data
u_diffstarpop_theta2, u_diffstarpop_unravel = ravel_pytree(DEFAULT_DIFFSTARPOP_U_PARAMS)
u_spspop_theta2, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)
u_ssp_err_pop_theta2, u_ssp_err_pop_unravel = ravel_pytree(ZERO_SSPERR_U_PARAMS)
u_theta2 = (u_diffstarpop_theta2, u_spspop_theta2, u_ssp_err_pop_theta2)

lg_n_thresh = -8
loss_args_multi_z = (
    lg_n_thresh,
    n_key,
    lc_halopop_z_obs_multi_z,
    lc_halopop_t_obs_multi_z,
    lc_halopop_mah_params_multi_z,
    lc_halopop_logmp0_multi_z,
    lc_halopop_nhalos_multi_z,
    lc_halopop_vol_mpc3_multi_z,
    t_table_multi_z,
    ssp_data,
    precomputed_ssp_mag_table_multi_z,
    z_phot_table_multi_z,
    wave_eff_table_multi_z,
    DEFAULT_MZR_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    lh_centroids_multi_z,
    dmag,
    mag_column,
    mag_thresh,
    DEFAULT_COSMOLOGY,
    FB,
    None,
    None,
    None,
)
loss_multi_z = n_mag_opt._loss_kern_multi_z(
    u_theta2, lg_n_data_err_lh_multi_z, *loss_args_multi_z
)

for zbin in range(0, len(zbins)):
    zmin = zbins[zbin][0]
    zmax = zbins[zbin][1]

    z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = jnp.log10(t_0)
    t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
    )

    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    lc_halopop = lc_halopop_multi_z[zbin]

    lh_centroids = jnp.asarray(
        np.load(
            os.path.join(
                DATA_PATH,
                "lh_centroids_z_"
                + str(zbins[zbin][0])
                + "-"
                + str(zbins[zbin][1])
                + "_test.npy",
            )
        )
    )

    n_args_single_z = (
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
        lh_centroids,
        dmag,
        mag_column,
        mag_thresh,
        DEFAULT_COSMOLOGY,
        FB,
    )
    lg_n_single_z, lg_n_avg_err_single_z = n_mag.n_mag_kern(*n_args_single_z)
    assert np.allclose(lg_n_multi_z[zbin], lg_n_single_z)

    loss_args_single_z = (
        lg_n_thresh,
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
        lh_centroids,
        dmag,
        mag_column,
        mag_thresh,
        DEFAULT_COSMOLOGY,
        FB,
    )
    lg_n_data_err_lh_single_z = jnp.vstack((lg_n_single_z, lg_n_avg_err_single_z))

    loss_single_z = n_mag_opt._loss_kern(
        u_theta2, lg_n_data_err_lh_single_z, *loss_args_single_z
    )
    assert np.isclose(loss_multi_z[zbin], loss_single_z)
