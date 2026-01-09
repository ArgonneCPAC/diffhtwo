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
from dsps.data_loaders.defaults import TransmissionCurve
from dsps.metallicity.umzr import DEFAULT_MZR_PARAMS
from jax import random as jran
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental import n_mag, n_mag_opt
from diffhtwo.experimental.data_loaders import retrieve_tcurves
from diffhtwo.experimental.utils import zbin_volume

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
