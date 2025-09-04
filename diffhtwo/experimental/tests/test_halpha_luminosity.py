from .. import halpha_luminosity as halphaL
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
import numpy as np
import jax.numpy as jnp

z_obs = 0.5
t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs


ssp_lgmet = jnp.linspace(-5.0, 1.0, 12)
ssp_lg_age_gyr = jnp.linspace(-4.0, 1.3, 107)
arr = jnp.array(
    [
        18.99350973,
        15.18340321,
        18.59590888,
        14.51430117,
        15.50420145,
        18.89018259,
        14.6840321,
        15.44170473,
        18.62229038,
        18.23519039,
        15.46710684,
        18.09569229,
        18.72593381,
        18.74917029,
        18.84602614,
        18.90889461,
        20.18508488,
        19.64529091,
        20.39642718,
        19.94077289,
        20.36111634,
        20.77151474,
        21.80102465,
        22.86217362,
        24.00485424,
        25.53092415,
        26.12223314,
        29.38864892,
        30.45950998,
        34.16892198,
        39.6796686,
        39.59340563,
        33.72510107,
        29.60391916,
        23.58247219,
        16.62595202,
        13.70698426,
        8.85376548,
        2.83265497,
        2.0991581,
        1.48929108,
        0.96965843,
        0.58217645,
        0.35032972,
        0.22578957,
        0.14377021,
        0.09140591,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)
noise_std = 2
ssp_halpha_line_luminosity = [
    np.clip(arr + np.random.normal(0, noise_std, size=arr.shape), a_min=0, a_max=None)
    for _ in range(ssp_lgmet.size)
]
ssp_halpha_line_luminosity = jnp.array(ssp_halpha_line_luminosity)


def test_halpha_luminosity():
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

    weights = jnp.ones_like(L_halpha_cgs)
    lgL_bin_edges, tw_hist_weighted = halphaL.get_halpha_luminosity_func(
        L_halpha_cgs, weights, sig=0.0
    )

    jnp_histogram = jnp.histogram(jnp.log10(L_halpha_cgs), bins=lgL_bin_edges)[0]

    assert jnp.allclose(tw_hist_weighted, jnp_histogram)
