import os
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from diffsky.experimental import lightcone_generators as lcg
from diffsky.mass_functions import mc_hosts
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import retrieve_fake_fsps_data
from jax import random as jran
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental.optimizers import n_specphot_opt

from .. import param_utils as pu
from ..data_loaders import retrieve_tcurves
from ..utils import zbin_vol

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DIFFSTARPOP_UM_plus_exsitu
)
BASE_PATH = Path(__file__).resolve().parent.parent
LH_CENTROIDS_PATH = BASE_PATH / "data_loaders/data"


@pytest.fixture(scope="module")
def fake_subset_ssp_data():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    emline_name = ssp_data.ssp_emline_wave._fields[0]
    emline_wave_aa = ssp_data.ssp_emline_wave[0]
    ssp_data = lemi.get_subset_emline_data(ssp_data, [emline_name])
    return ssp_data, emline_wave_aa


def test_n_specphot(ssp_data, emline_wave_aa):
    zbins = np.array(
        [
            [0.2, 0.5],
            [1.5, 1.75],
            [2.75, 3.5],
        ]
    )
    mag_columns = [3]
    mag_thresh_column = 3
    mag_thresh = 24.5
    dmag = 0.2
    lg_n_thresh = -8
    frac_cat = 1.0

    ran_key = jran.key(0)
    ran_key, n_key = jran.split(ran_key, 2)
    z_idx = 0
    lc_z_min = zbins[z_idx][0]
    lc_z_max = zbins[z_idx][1]

    lh_centroids = jnp.asarray(
        np.load(
            os.path.join(
                LH_CENTROIDS_PATH,
                "lh_centroids_z_" + str(lc_z_min) + "-" + str(lc_z_max) + "_test.npy",
            )
        )
    )
    dmag_centroids = jnp.ones((lh_centroids.shape[0], 1)) * dmag

    rng = np.random.default_rng(0)
    lg_n_data = rng.uniform(-17, -4, lh_centroids.shape[0])
    lg_n_err = rng.uniform(0.2, 12, lh_centroids.shape[0])
    lg_n_data_err_lh = np.vstack((lg_n_data, lg_n_err))

    num_halos = 100
    lc_sky_area_degsq = 0.1
    lgmp_min = 10.0
    lgmp_max = mc_hosts.LGMH_MAX

    tcurves = [
        retrieve_tcurves.MegaCam_uS,
        retrieve_tcurves.HSC_G,
        retrieve_tcurves.HSC_R,
        retrieve_tcurves.HSC_I,
        retrieve_tcurves.HSC_Z,
    ]

    n_z_phot_table = 15
    z_phot_table = 10 ** jnp.linspace(
        np.log10(lc_z_min), np.log10(lc_z_max), n_z_phot_table
    )

    lc_args = (
        ran_key,
        num_halos,
        lc_z_min,
        lc_z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcg.weighted_lc_photdata(*lc_args, cosmo_params=DEFAULT_COSMOLOGY)

    fields = (*lc_data._fields, "lc_vol_mpc3")
    lc_vol_mpc3 = zbin_vol(lc_sky_area_degsq, lc_z_min, lc_z_max, DEFAULT_COSMOLOGY)
    values = (*lc_data, lc_vol_mpc3)
    lc_data = namedtuple(lc_data.__class__.__name__, fields)(*values)

    phot_loss = n_specphot_opt.get_phot_loss(
        ran_key,
        lg_n_data_err_lh,
        lg_n_thresh,
        dpwm.DEFAULT_PARAM_COLLECTION,
        lc_data,
        emline_wave_aa,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        dmag_centroids,
        frac_cat,
    )

    assert np.isfinite(phot_loss)
    assert phot_loss >= 0

    u_theta_default = pu.get_u_theta_from_param_collection(
        dpwm.DEFAULT_PARAM_COLLECTION
    )

    loss_phot_kern = n_specphot_opt._loss_phot_kern(
        u_theta_default,
        ran_key,
        lg_n_data_err_lh,
        lg_n_thresh,
        lc_data,
        emline_wave_aa,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        dmag_centroids,
        frac_cat,
    )
    assert np.isfinite(loss_phot_kern)
    assert loss_phot_kern >= 0


	lg_n_data_err_lh_multi_z = jnp.array([lg_n_data_err_lh, lg_n_data_err_lh])

	lc_data_multi_z = [lc_data, lc_data]
	lc_data_multi_z = stack_lc_data(lc_data_multi_z)

	lh_centroids_multi_z = jnp.array([lh_centroids, lh_centroids])
	dmag_centroids_multi_z = jnp.array([dmag_centroids, dmag_centroids])

	n_specphot_opt._loss_phot_kern_multi_z(
	    u_theta_fit,
	    ran_key,
	    lg_n_data_err_lh_multi_z,
	    lg_n_thresh,
	    lc_data_multi_z,
	    emline_wave_aa,
	    mag_columns,
	    mag_thresh_column,
	    mag_thresh,
	    lh_centroids_multi_z,
	    dmag_centroids_multi_z,
	    frac_cat,
	)
