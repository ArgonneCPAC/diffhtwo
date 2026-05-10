import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ... import param_utils as pu
from ..phot_loss import _loss_phot_kern, get_phot_loss


def test_phot_loss(feniks_single_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_single_z_data

    ran_key = jran.key(0)

    phot_loss = get_phot_loss(
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
        DEFAULT_PARAM_COLLECTION,
    )

    assert np.isfinite(phot_loss)

    u_theta = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)
    phot_loss_kern = _loss_phot_kern(
        u_theta,
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
    )
    assert np.isfinite(phot_loss_kern)
