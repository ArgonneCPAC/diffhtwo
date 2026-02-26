# import jax.numpy as jnp
from diffsky.dustpop import tw_dustpop_mono_noise
from diffsky.experimental import mc_diffstarpop_wrappers as mcdw
from diffsky.experimental.kernels import mc_phot_kernels as mcpk
from diffsky.experimental.kernels import ssp_weight_kernels as sspwk
from jax import jit as jjit
from jax import vmap
from jax.debug import print

LGMET_SCATTER = 0.2

# rest UV wavelength for continuum calculation in Angstroms
UV_WAVELENGTH_AA = 1500

_D = (None, None, 0, 0, 0, None, 0, 0, 0, None)
calc_dust_ftrans_vmap = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)


# @jjit
# def calc_singlegal_rest_uv_luminosity(ssp_flux, weights):
#     n_met, n_ages = weights.shape
#     sed_unit_mstar = jnp.sum(
#         ssp_flux * weights.reshape((n_met, n_ages, 1)), axis=(0, 1)
#     )
#     return sed_unit_mstar


# _S = (None, 0)
# calc_rest_uv_luminosity = vmap(calc_singlegal_rest_uv_luminosity, in_axes=_S)


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
    lgmet_scatter,
    cosmo_params,
    fb,
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

    print("ssp_weights.shape:{}", ssp_weights.shape)

    # L_UV = calc_rest_uv_luminosity(ssp_data.ssp_flux, ssp_weights)

    # ftrans_args = (
    #     spspop_params.dustpop_params,
    #     UV_WAVELENGTH_AA,
    #     logsm_obs,
    #     logssfr_obs,
    #     z_obs,
    #     ssp_data.ssp_lg_age_gyr,
    #     phot_randoms.uran_av,
    #     phot_randoms.uran_delta,
    #     phot_randoms.uran_funo,
    #     scatter_params,
    # )
    # _res_dust = calc_dust_ftrans_vmap(
    #     *ftrans_args
    # )  # _res_dust = ftrans, noisy_ftrans, dust_params, noisy_dust_params
    # frac_trans = _res_dust[1]  # ftrans.shape = (n_gals, n_bands, n_age)
    # dust_params = _res_dust[3]  # fields = ('av', 'delta', 'funo')
