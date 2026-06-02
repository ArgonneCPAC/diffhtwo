import numpy as np
import pytest
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ... import param_utils as pu
from ..emline_loss import _loss_emline_kern, get_emline_loss


@pytest.mark.skip(reason="Currently mse based emline loss code is outdated")
def test_emline_loss(fake_subset_ssp_data, hizels):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    # pick first line, first zbin
    lg_emline_LF_target = hizels.lg_LF[0][0]
    lg_emline_Lbin_edges = hizels.lg_Lbin_edges[0][0]
    lc_data = hizels.lc_data[0][0]

    ran_key = jran.key(0)

    emline_loss = get_emline_loss(
        ran_key,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        DEFAULT_PARAM_COLLECTION,
        lc_data,
        emline_wave_aa,
    )

    assert np.isfinite(emline_loss)

    u_theta = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)
    emline_loss_kern = _loss_emline_kern(
        u_theta,
        ran_key,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        lc_data,
        emline_wave_aa,
    )
    assert np.isfinite(emline_loss_kern)
