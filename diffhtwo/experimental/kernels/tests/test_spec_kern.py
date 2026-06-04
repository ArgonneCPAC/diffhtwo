import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..spec_kern import get_halpha_LF_q_ms_burst


def test_spec_kern(fake_subset_ssp_data, hizels, feniks):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    ran_key = jran.key(0)

    _res = get_halpha_LF_q_ms_burst(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        emline_wave_aa,
        hizels.lg_Lbin_edges[0][0],
        hizels.z[0][0],
        hizels.dz[0][0],
        ssp_data,
        feniks.filter_info.tcurves,
    )
    (
        lgL_bin_centers,
        lg_halpha_LF,
        lg_halpha_LF_q,
        lg_halpha_LF_ms,
        lg_halpha_LF_burst,
        lg_halpha_LF_in_situ,
        phot_kern_results,
        spec_kern_results,
        lg_halpha_Lbin_edges,
        lc_data,
    ) = _res
    assert np.isfinite(lg_halpha_LF.all())
    assert np.isfinite(lg_halpha_LF_q.all())
    assert np.isfinite(lg_halpha_LF_ms.all())
    assert np.isfinite(lg_halpha_LF_burst.all())
