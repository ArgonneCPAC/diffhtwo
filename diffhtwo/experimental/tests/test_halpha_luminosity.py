from .. import halpha_luminosity as halphaL
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
import numpy as np
import jax.numpy as jnp

z_obs = 0.5
t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs


ssp_lgmet = np.linspace(-5.0, 1.0, 12)
ssp_lg_age_gyr = np.linspace(-4.0, 1.3, 107)
ssp_halpha_line_luminosity = np.random.uniform(
    0.0, 40.0, (ssp_lgmet.size, ssp_lg_age_gyr.size)
)


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
