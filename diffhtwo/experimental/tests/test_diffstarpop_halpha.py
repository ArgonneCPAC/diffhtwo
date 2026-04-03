import numpy as np
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils import spspop_param_utils as spspu
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity import umzr
from jax import random as jran

from ..emline_luminosity_pop import emline_luminosity_pop

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()


def test_emline_luminosity_pop():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 0.1

    ran_key, lc_key = jran.split(ran_key, 2)
    args = (lc_key, lgmp_min, z_min, z_max, sky_area_degsq)

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*args)

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    z_obs = lc_halopop["z_obs"]
    t_obs = lc_halopop["t_obs"]
    mah_params = lc_halopop["mah_params"]
    logmp0 = lc_halopop["logmp0"]
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = np.log10(t_0)

    t_table = np.linspace(T_TABLE_MIN, 10**lgt0, 100)

    mzr_params = umzr.DEFAULT_MZR_PARAMS

    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS
    scatter_params = DEFAULT_SCATTER_PARAMS

    emline_wave_aa = 6000

    ran_key, diffstarpop_key = jran.split(ran_key, 2)
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        diffstarpop_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        emline_wave_aa,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    halpha_L = emline_luminosity_pop(*args)

    for arr in halpha_L:
        assert np.all(np.isfinite(arr))
