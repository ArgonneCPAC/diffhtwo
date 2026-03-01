# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
import jax.numpy as jnp
from diffsky.dustpop import tw_dustpop_mono_noise
from diffsky.experimental import mc_diffstarpop_wrappers as mcdw
from diffsky.experimental.kernels import mc_phot_kernels as mcpk
from diffsky.experimental.kernels import ssp_weight_kernels as sspwk
from jax import jit as jjit
from jax import vmap

# from jax.debug import print

LGMET_SCATTER = 0.2

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

# copied from astropy.constants.c.value in m/s
C = 299792458.0

# rest UV wavelength for continuum calculation in Angstroms
UV_WAVELENGTH_AA = 1500 + 1.713
UV_FREQUENCY_HZ = C / (UV_WAVELENGTH_AA * 1e-10)

_D = (None, None, 0, 0, 0, None, 0, 0, 0, None)
_calc_dust_ftrans_vmap = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)


@jjit
def _calc_singlegal_rest_uv_luminosity(
    ssp_wave, ssp_flux, ssp_weights, ftrans, dust=True
):
    """
    ssp_flux: ssp flux from ssp_data in default units of Lsun/Hz/Msun
    weights: combined age metallicity weights with shape (n_met, n_age)
    """

    n_met, n_age = ssp_weights.shape

    if dust is True:
        # broadcast ftrans due to dust across metallicity
        ssp_weights = ssp_weights * ftrans.reshape(1, n_age)

    # get weighted sed
    sed_weighted = jnp.sum(
        ssp_flux * ssp_weights.reshape((n_met, n_age, 1)), axis=(0, 1)
    )

    # uv_luminosity in units of Lsun/Msun/Hz
    uv_luminosity_per_hz = jnp.interp(UV_WAVELENGTH_AA, ssp_wave, sed_weighted)

    # get uv_luminosity in units of Lsun/Msun
    uv_luminosity = UV_FREQUENCY_HZ * uv_luminosity_per_hz

    return uv_luminosity


_S = (None, None, 0, 0, None)
_calc_galpop_rest_uv_luminosity = vmap(_calc_singlegal_rest_uv_luminosity, in_axes=_S)


@jjit
def compute_uv_luminosity(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    diffstarpop_params,
    spspop_params,
    mzr_params,
    scatter_params,
    t_table,
    ssp_data,
    cosmo_params,
    fb,
    lgmet_scatter=LGMET_SCATTER,
    n_t_table=mcdw.N_T_TABLE,
    dust=True,
):
    phot_randoms, sfh_params = mcpk.get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )

    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table
    )

    age_weights_smooth, lgmet_weights = sspwk.get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    _res = sspwk.compute_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        logsm_obs,
        logssfr_obs,
        age_weights_smooth,
        lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )

    ssp_weights, burst_params, mc_sfh_type = _res

    ftrans_args = (
        spspop_params.dustpop_params,
        UV_WAVELENGTH_AA,
        logsm_obs,
        logssfr_obs,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        scatter_params,
    )

    # _res_dust = ftrans, noisy_ftrans, dust_params, noisy_dust_params
    _res_dust = _calc_dust_ftrans_vmap(*ftrans_args)

    # frac_trans.shape = (n_gals, n_age)
    # dust_params = _res_dust[3]  # fields = ('av', 'delta', 'funo')
    frac_trans = _res_dust[1]

    L_UV_unit = _calc_galpop_rest_uv_luminosity(
        ssp_data.ssp_wave, ssp_data.ssp_flux, ssp_weights, frac_trans, dust=dust
    )  # [Lsun/Msun]

    _mstar = 10**logsm_obs

    L_UV_cgs = L_UV_unit * L_SUN_CGS * _mstar  # [erg/s]

    return L_UV_cgs, frac_trans
