import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ..N_phot import add_N_model_to_spaces


def test_add_N_model_to_spaces(ran_key, feniks):
    z_data = feniks.spaces[0]
    spaces = [f for f in z_data._fields if f not in {"z_min", "z_max", "lc_data"}]

    z_data_w_model = add_N_model_to_spaces(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_data,
        feniks.filter_info.mag_thresh,
    )

    for space_name in spaces:
        space = getattr(z_data_w_model, space_name)
        assert np.isfinite(space.N_model).all()
