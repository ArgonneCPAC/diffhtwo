from ..load_feniks import FeniksFilters
from ..load_minerva import PhotFilters
from ..load_sdss import SdssFilters
from ..space_utils import get_filt_indx


def test_get_filt_indx():
    # sdss
    (u_idx,) = get_filt_indx("u", SdssFilters)
    (g_idx,) = get_filt_indx("g", SdssFilters)
    (r_idx,) = get_filt_indx("r", SdssFilters)
    (i_idx,) = get_filt_indx("i", SdssFilters)
    (z_idx,) = get_filt_indx("z", SdssFilters)

    assert (u_idx, g_idx, r_idx, i_idx, z_idx) == tuple(range(5))

    assert get_filt_indx("Ur_ri", SdssFilters) == [0, 2, 2, 3]
    assert get_filt_indx("Gr_ri", SdssFilters) == [1, 2, 2, 3]
    assert get_filt_indx("R_ur", SdssFilters) == [2, 0, 2]
    assert get_filt_indx("R_ri", SdssFilters) == [2, 2, 3]

    # feniks
    (u_idx,) = get_filt_indx("u", FeniksFilters)
    (g_idx,) = get_filt_indx("g", FeniksFilters)
    (r_idx,) = get_filt_indx("r", FeniksFilters)
    (i_idx,) = get_filt_indx("i", FeniksFilters)
    (z_idx,) = get_filt_indx("z", FeniksFilters)
    (J_idx,) = get_filt_indx("J", FeniksFilters)
    (H_idx,) = get_filt_indx("H", FeniksFilters)
    (K_idx,) = get_filt_indx("K", FeniksFilters)

    assert (u_idx, g_idx, r_idx, i_idx, z_idx, J_idx, H_idx, K_idx) == tuple(range(8))

    assert get_filt_indx("rz_zJ", FeniksFilters) == [2, 4, 4, 5]
    assert get_filt_indx("K_ug", FeniksFilters) == [7, 0, 1]

    # minerva
    (f435w_idx,) = get_filt_indx("f435w", PhotFilters)
    (f606w_idx,) = get_filt_indx("f606w", PhotFilters)
    (f814w_idx,) = get_filt_indx("f814w", PhotFilters)
    (f090w_idx,) = get_filt_indx("f090w", PhotFilters)
    (f115w_idx,) = get_filt_indx("f115w", PhotFilters)
    (f140m_idx,) = get_filt_indx("f140m", PhotFilters)
    (f150w_idx,) = get_filt_indx("f150w", PhotFilters)
    (f162m_idx,) = get_filt_indx("f162m", PhotFilters)
    (f182m_idx,) = get_filt_indx("f182m", PhotFilters)
    (f200w_idx,) = get_filt_indx("f200w", PhotFilters)
    (f210m_idx,) = get_filt_indx("f210m", PhotFilters)
    (f250m_idx,) = get_filt_indx("f250m", PhotFilters)
    (f277w_idx,) = get_filt_indx("f277w", PhotFilters)
    (f300m_idx,) = get_filt_indx("f300m", PhotFilters)
    (f356w_idx,) = get_filt_indx("f356w", PhotFilters)
    (f360m_idx,) = get_filt_indx("f360m", PhotFilters)
    (f410m_idx,) = get_filt_indx("f410m", PhotFilters)
    (f444w_idx,) = get_filt_indx("f444w", PhotFilters)
    (f460m_idx,) = get_filt_indx("f460m", PhotFilters)

    assert (
        f435w_idx,
        f606w_idx,
        f814w_idx,
        f090w_idx,
        f115w_idx,
        f140m_idx,
        f150w_idx,
        f162m_idx,
        f182m_idx,
        f200w_idx,
        f210m_idx,
        f250m_idx,
        f277w_idx,
        f300m_idx,
        f356w_idx,
        f360m_idx,
        f410m_idx,
        f444w_idx,
        f460m_idx,
    ) == tuple(range(19))

    assert get_filt_indx("F090wF150w_F150wF356w", PhotFilters) == [3, 6, 6, 14]
    assert get_filt_indx("F182m_F200wF182m", PhotFilters) == [8, 9, 8]
