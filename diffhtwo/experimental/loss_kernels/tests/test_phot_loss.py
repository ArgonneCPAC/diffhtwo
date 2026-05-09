import jax.numpy as jnp
import numpy as np
import pytest
from diffsky.experimental import lightcone_generators as lcg
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import retrieve_fake_fsps_data
from jax import random as jran

from ... import param_utils as pu
from ...latin_hypercube import lh_utils as lhu
from ...lightcone_generators import generate_lc_data
from ..phot_loss import get_phot_loss


@pytest.fixture(scope="module")
def fake_subset_ssp_data():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    emline_name = ssp_data.ssp_emline_wave._fields[0]
    emline_wave_aa = ssp_data.ssp_emline_wave[0]
    ssp_data = lemi.get_subset_emline_data(ssp_data, [emline_name])
    return ssp_data, emline_wave_aa


def test_phot_loss(fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    # feniks like
    zbins = np.array(
        [
            [0.2, 0.5],
            [1.5, 1.75],
            [2.75, 3.5],
        ]
    )
    ran_key = jran.key(0)
    num_halos = 500
    N_centroids = 200
    # ran_key, n_key = jran.split(ran_key, 2)

    feniks_meta_data, feniks_fitting_data = lhu.get_zbins_lh_lc(
        ran_key,
        FENIKS,
        zbins[0][0],
        zbins[0][1],
        ssp_data,
        N_centroids,
        num_halos=num_halos,
    )

    phot_loss = get_phot_loss(
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
        DEFAULT_PARAM_COLLECTION,
    )
    assert np.isfinite(phot_loss)
