import numpy as np
import pytest
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ... import param_utils as pu
from ..phot_loss import (
    _loss_phot_kern,
    _loss_phot_kern_2d_multiz,
    get_phot_loss,
    get_phot_loss_2d_multiz,
)


@pytest.mark.skip(
    reason="LH dimensions need to be fixed in load_feniks before activating this test again"
)
def test_phot_loss(ran_key, feniks_single_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_single_z_data

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

    assert np.isclose(phot_loss, phot_loss_kern)


def test_phot_loss_2d(ran_key, feniks_fitting_data):
    phot_loss = get_phot_loss_2d_multiz(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_fitting_data.spaces,
        feniks_fitting_data.filter_info.mag_thresh,
        frac_cat=feniks_fitting_data.frac_cat,
    )
    assert np.isfinite(phot_loss)

    u_theta = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)
    phot_loss_kern = _loss_phot_kern_2d_multiz(u_theta, ran_key, feniks_fitting_data)
    assert np.isfinite(phot_loss_kern)

    assert np.isclose(phot_loss, phot_loss_kern)
