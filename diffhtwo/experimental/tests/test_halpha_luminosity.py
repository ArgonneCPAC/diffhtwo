import jax.numpy as jnp
import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.data_loaders import retrieve_fake_fsps_data

from .. import halpha_luminosity as halphaL
from ..data_loaders import retrieve_fake_fsps_halpha

z_obs = 0.5
t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs


ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
ssp_lgmet = ssp_data.ssp_lgmet
ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr
ssp_halpha_line_luminosity = retrieve_fake_fsps_halpha.load_fake_ssp_halpha()


def test_halpha_luminosity():
    N = 10000
    lg_sfr_draws = np.random.uniform(-2, 2, N)

    gal_t_table = jnp.linspace(0.05, 13.8, 100)  # age of the universe in Gyr
    gal_sfr_tables = jnp.ones((gal_t_table.size, N)) * (
        10**lg_sfr_draws
    )  # SFR in Msun/yr
    gal_sfr_tables = gal_sfr_tables.T
    gal_lgmet = -1.0
    gal_lgmet_scatter = 0.1

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
    print(jnp.all(jnp.isfinite(L_halpha_cgs)), jnp.all(jnp.isfinite(L_halpha_unit)))
    assert jnp.all(jnp.isfinite(L_halpha_cgs))
    assert jnp.all(jnp.isfinite(L_halpha_unit))

    weights = jnp.ones_like(L_halpha_cgs)
    lgL_bin_edges, tw_hist_weighted = halphaL.get_halpha_luminosity_func(
        L_halpha_cgs, weights, sig=0.0
    )

    jnp_histogram = jnp.histogram(jnp.log10(L_halpha_cgs), bins=lgL_bin_edges)[0]

    assert jnp.allclose(tw_hist_weighted, jnp_histogram)
