import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ..N_phot import add_N_model_to_spaces


def test_add_N_model_to_spaces(ran_key, feniks):
    z_data = feniks.spaces[0]
    fields = [f for f in z_data._fields if f not in {"z_min", "z_max", "lc_data"}]

    z_data_w_model = add_N_model_to_spaces(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_data,
        feniks.filter_info.mag_thresh,
    )

    for space_name in fields:
        spaces = getattr(z_data, space_name)
        spaces_w_model = getattr(z_data_w_model, space_name)
        if isinstance(spaces_w_model, list):
            assert len(spaces_w_model) == len(spaces)
        else:
            spaces, spaces_w_model = [spaces], [spaces_w_model]
        for space, space_w_model in zip(spaces, spaces_w_model):
            assert "N_model" in space_w_model._fields
            assert space_w_model.N_model.shape == space.N_data.shape
            assert np.isfinite(space_w_model.N_model).all()
            assert (space_w_model.N_model >= 0).all()
