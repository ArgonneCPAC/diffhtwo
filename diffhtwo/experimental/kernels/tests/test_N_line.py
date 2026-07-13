import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..N_line import N_linelum


def test_N_linelum(hizels_fitting_data):
    ran_key = jran.key(0)

    line_wave_table = jnp.array([hizels_fitting_data.line_wave_aa[0]])
    N = N_linelum(
        ran_key,
        line_wave_table,
        hizels_fitting_data.lg_Lbin_edges[0][0],
        hizels_fitting_data.lc_data[0][1],
        DEFAULT_PARAM_COLLECTION,
    )

    assert np.isfinite(N).all()
    assert (N >= 0.0).all()
