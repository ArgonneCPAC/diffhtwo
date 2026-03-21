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
from jax.debug import print

from . import diffndhist as diffndhist2

# from jax.debug import print

LGMET_SCATTER = 0.2

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

_D = (None, None, 0, 0, 0, None, 0, 0, 0, None)
_calc_dust_ftrans_vmap = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)


@jjit
def compute_emline_luminosity(
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
    emline_wave_aa,
    ssp_emline_luminosity,
    cosmo_params,
    fb,
    lgmet_scatter=LGMET_SCATTER,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_randoms, sfh_params = mcpk.get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    print("phot_randoms:{}", phot_randoms)

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
        emline_wave_aa,
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

    # dust_params = _res_dust[3]  # fields = ('av', 'delta', 'funo')
    frac_trans = _res_dust[1]
    n_gals, n_age = frac_trans.shape
    frac_trans = frac_trans.reshape(n_gals, 1, n_age)

    _mstar = 10**logsm_obs
    integrand = ssp_emline_luminosity * ssp_weights * frac_trans
    L_emline_cgs = jnp.sum(integrand, axis=(1, 2)) * (L_SUN_CGS * _mstar)

    print("L_emline_cgs:{}", L_emline_cgs)

    # no dust
    integrand_nodust = ssp_emline_luminosity * ssp_weights
    L_emline_cgs_nodust = jnp.sum(integrand_nodust, axis=(1, 2)) * (L_SUN_CGS * _mstar)

    print("L_emline_cgs_nodust:{}", L_emline_cgs_nodust)

    return L_emline_cgs, L_emline_cgs_nodust


@jjit
def get_emline_luminosity_func(
    L_emline_cgs,
    weights,
    dlgL_bin=0.2,
    lgL_min=38.0,
    lgL_max=45.0,
    sig=None,
    lgL_bin_edges=None,
):
    """
    Calculates the emline LF

    Parameters
    ----------
    L_emline_cgs : array of shape (n,) or (n, 1)
        h-alpha Luminosities in [erg/s]

    weights : array of shape (n,)
        weights to multiply with L_emline_cgs

    sig : array of shape (nbins, 1)
        bin dependent sigma for diffndhist

    Returns
    -------
    lgL_bin_edges : array of luminosity function bin edges in log10-space
        defined using default arguments of this function.
    tw_hist_weighted:
        luminosity function - weighted histogram counts using diffsky.diffndhist
    """

    n_L = L_emline_cgs.size
    L_emline = L_emline_cgs.reshape(n_L, 1)

    # mask: valid (strictly positive & finite)
    valid = jnp.isfinite(L_emline) & (L_emline > 0)
    L_emline = jnp.where(valid, L_emline, 10)
    lgL_emline = jnp.log10(L_emline)

    # weights: zero-out invalids
    w = jnp.where(
        valid.reshape(
            n_L,
        ),
        weights.reshape(
            n_L,
        ),
        0.0,
    )

    if lgL_bin_edges is None:
        lgL_bin_edges = jnp.arange(lgL_min, lgL_max, dlgL_bin)

    lgL_bin_lo = lgL_bin_edges[:-1].reshape(lgL_bin_edges[:-1].size, 1)
    lgL_bin_hi = lgL_bin_edges[1:].reshape(lgL_bin_edges[1:].size, 1)

    if sig is None:
        sig = jnp.zeros_like(lgL_bin_lo) + (dlgL_bin / 2)

    tw_hist_weighted = diffndhist2.tw_ndhist_weighted(
        lgL_emline, sig, w, lgL_bin_lo, lgL_bin_hi
    )

    return lgL_bin_edges, tw_hist_weighted
