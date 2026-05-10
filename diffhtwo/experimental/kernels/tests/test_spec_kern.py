import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..spec_kern import n_spec_kern


def test_spec_kern(fake_subset_ssp_data, hizels):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    emline_wave_table = jnp.array([emline_wave_aa])

    ran_key = jran.key(0)

    # pick first line, first zbin
    lg_emline_Lbin_edges = hizels.lg_Lbin_edges[0][0]
    lc_data = hizels.lc_data[0][0]

    lg_emline_LF_model = n_spec_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        lc_data,
        emline_wave_table,
        lg_emline_Lbin_edges,
    )
    assert np.isfinite(lg_emline_LF_model.all())
