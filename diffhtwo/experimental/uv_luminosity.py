import jax.numpy as jnp
from diffsky.dustpop import tw_dustpop_mono_noise
from diffsky.experimental import mc_diffstarpop_wrappers as mcdw
from diffsky.experimental.kernels import mc_phot_kernels as mcpk
from diffsky.experimental.kernels import ssp_weight_kernels as sspwk
from jax import jit as jjit
from jax import vmap
from jax.debug import print

LGMET_SCATTER = 0.2

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

# copied from astropy.constants.c.value in m/s
C = 299792458.0

# rest UV wavelength for continuum calculation in Angstroms
UV_WAVELENGTH_AA = 1500

_D = (None, None, 0, 0, 0, None, 0, 0, 0, None)
calc_dust_ftrans_vmap = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)


@jjit
def _get_integrated_luminosity(wave, sed):
    """
    Parameters:
            wave - wavelength array in units of Angstrom
            sed - [Lsun/Hz/Msun]

    Returns:
            integrated_luminosity - integrated sed in units of [Lsun/Msun]

    """
    freq = C / (wave * 1e-10)
    freq = jnp.flip(freq)

    integrated_luminosity = jnp.trapezoid(sed, freq)  # [Lsun/Msun]

    return integrated_luminosity


@jjit
def calc_singlegal_rest_uv_luminosity(ssp_data, weights):
    """
    ssp_flux: ssp flux from ssp_data in default units of Lsun/Hz/Msun
    weights: combined age metallicity weights with shape (n_met, n_age)
    """
    # get weighted sed
    n_met, n_ages = weights.shape
    sed_weighted = jnp.sum(
        ssp_data.ssp_flux * weights.reshape((n_met, n_ages, 1)), axis=(0, 1)
    )

    # get integrated uv luminosity within UV tophat window
    uv_tophat = (ssp_data.ssp_wave > UV_WAVELENGTH_AA - 50) & (
        ssp_data.ssp_wave < UV_WAVELENGTH_AA - 50
    )
    clipped_wave = jnp.where(uv_tophat, ssp_data.ssp_wave, 0.0)
    clipped_sed_weighted = jnp.where(uv_tophat, sed_weighted, 0.0)
    integrated_uv_luminosity = _get_integrated_luminosity(
        clipped_wave, clipped_sed_weighted
    )
    return integrated_uv_luminosity  # [Lsun/Msun]


_S = (None, 0)
calc_rest_uv_luminosity = vmap(calc_singlegal_rest_uv_luminosity, in_axes=_S)


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

    L_UV_unit = calc_rest_uv_luminosity(ssp_data, ssp_weights)  # [Lsun/Msun]

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
    _res_dust = calc_dust_ftrans_vmap(
        *ftrans_args
    )  # _res_dust = ftrans, noisy_ftrans, dust_params, noisy_dust_params
    frac_trans = _res_dust[1]  # ftrans.shape = (n_gals, n_bands, n_age)
    # dust_params = _res_dust[3]  # fields = ('av', 'delta', 'funo')

    _mstar = 10**logsm_obs

    print("frac_trans.shape:{}", frac_trans.shape)
    print("L_UV_unit.shape:{}", L_UV_unit.shape)
    print("_mstar.shape:{}", _mstar.shape)
    # L_UV_cgs = L_UV_unit * frac_trans * L_SUN_CGS * _mstar  # [erg/s]

    # return L_UV_cgs
