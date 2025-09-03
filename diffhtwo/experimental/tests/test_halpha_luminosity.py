from .. import halpha_luminosity as halphaL
from dsps import load_ssp_templates
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
import h5py
import numpy as np
import jax.numpy as jnp
import os

z_obs = 0.5
t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs

halpha_file = os.path.join(os.path.dirname(__file__), "halpha_fsps_v3.2.h5")

ssp_file = os.path.join(os.path.dirname(__file__), "fsps_v3.2.h5")

with h5py.File(halpha_file, "r") as f:
    print("Keys in file:", list(f.keys()))  # list top-level groups/datasets
    ssp_halpha_line_luminosity = f["ssp_halpha_line_luminosity"][...]

ssp_data = load_ssp_templates(fn=ssp_file)
ssp_lgmet = ssp_data.ssp_lgmet
ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr


def test_get_Lhalpha_vmap():
    N = 10000
    lg_sfr_draws = np.random.uniform(-2, 2, N)

    gal_t_table = jnp.linspace(0.05, 13.8, 100)  # age of the universe in Gyr
    gal_sfr_tables = jnp.ones((gal_t_table.size, N)) * (
        10**lg_sfr_draws
    )  # SFR in Msun/yr
    gal_sfr_tables = gal_sfr_tables.T
    gal_lgmet = -1.0
    gal_lgmet_scatter = 0

    L_halpha_cgs, L_halpha_unit = halphaL.get_L_halpha_vmap(
        gal_sfr_tables,
        gal_lgmet,
        gal_lgmet_scatter,
        gal_t_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
    )
    assert jnp.all(jnp.isfinite(L_halpha_cgs))
    assert jnp.all(jnp.isfinite(L_halpha_unit))
