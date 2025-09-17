import numpy as np
import jax.numpy as jnp
from jax import random as jran
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from jax.flatten_util import ravel_pytree
from numbers import Real
from diffstar.defaults import T_TABLE_MIN
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import flat_wcdm
from dsps.metallicity import umzr
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.param_utils import spspop_param_utils as spspu
from diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha_L
from diffstarpop_halpha import diffstarpop_halpha_lf_weighted as dpop_halpha_lf_weighted
from diffstarpop_halpha_opt import fit_diffstarpop

# from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
# from diffsky.ssp_err_model import ssp_err_model

idx = jnp.arange(65)
theta_default, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)


def _jitter_value(v, s, rng):
    if isinstance(v, Real):
        eps = s * (abs(v) if v != 0 else 1.0)
        return v + rng.normal(0.0, eps)
    return v


def perturb_namedtuple(nt, scale=0.01, rng=None, bounds=None, per_field_scale=None):
    """
    Perturb a namedtuple's numeric fields with small Gaussian noise.

    - scale: default fractional std for all numeric fields
    - per_field_scale: dict(field -> scale) to override per field
    - bounds: dict(field -> (lo, hi)) to clamp after jitter
    """
    if rng is None:
        rng = np.random.default_rng()

    updates = {}
    for f in nt._fields:
        v = getattr(nt, f)

        # Recurse if nested namedtuple
        if hasattr(v, "_fields"):
            updates[f] = perturb_namedtuple(
                v, scale=scale, rng=rng, bounds=bounds, per_field_scale=per_field_scale
            )
            continue

        s = per_field_scale.get(f, scale) if per_field_scale else scale
        v2 = _jitter_value(v, s, rng)

        if bounds and f in bounds:
            lo, hi = bounds[f]
            v2 = float(np.clip(v2, lo, hi))

        updates[f] = v2

    return nt._replace(**updates)


def perturb_diffstarpop(params, scale=0.01, rng=None):
    """
    Convenience wrapper for your DiffstarPopParams-like namedtuple,
    with sensible default bounds for probability-like params and stds.
    """
    # Heuristic bounds for common fields (edit to taste)
    bnds = {}
    # Anything that looks like a fraction or ylo/yhi: clamp to [0, 1]
    for f in getattr(params.sfh_pdf_cens_params, "_fields", []):
        if f.startswith("frac_") or f.endswith("_ylo") or f.endswith("_yhi"):
            bnds[f] = (0.0, 1.0)
        if f.startswith("std_"):
            bnds[f] = (-0.5, 4)  # stds positive

    for f in getattr(params.satquench_params, "_fields", []):
        if f.startswith("lgmu_") or f.endswith("_crit"):
            # leave unbounded by default; adjust if you have prior knowledge
            pass

    # Example per-field tweaks (optional): smaller noise on thresholds/slopes
    per_scale = {
        # e.g., keep critical points tighter than others
        "qp_lgmh_crit": 0.005,
        "td_lgmhc": 0.005,
        "td_mlo": 0.01,
        "td_mhi": 0.01,
    }

    # Apply to sub-nt’s and stitch back
    sq = perturb_namedtuple(params.sfh_pdf_cens_params, scale, rng, bnds, per_scale)
    sat = perturb_namedtuple(params.satquench_params, scale, rng, bnds, per_scale)

    # Outer container is also a namedtuple → use _replace
    return params._replace(sfh_pdf_cens_params=sq, satquench_params=sat)


def test_bimodal_sfh_opt():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    ssp_lgmet = ssp_data.ssp_lgmet
    ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr
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
        ]
    )
    noise_std = 2
    ssp_halpha_line_luminosity = [
        np.clip(
            arr + np.random.normal(0, noise_std, size=arr.shape), a_min=0, a_max=None
        )
        for _ in range(ssp_lgmet.size)
    ]
    zeros = np.zeros(ssp_lg_age_gyr.size - len(arr))
    ssp_halpha_line_luminosity = [
        np.append(_, zeros) for _ in ssp_halpha_line_luminosity
    ]
    ssp_halpha_line_luminosity = jnp.array(ssp_halpha_line_luminosity)

    # generate lightcone
    lc_ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 1.0

    args = (lc_ran_key, lgmp_min, z_min, z_max, sky_area_degsq)

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*args)

    # z_obs = lc_halopop["z_obs"]
    t_obs = lc_halopop["t_obs"]
    mah_params = lc_halopop["mah_params"]
    logmp0 = lc_halopop["logmp0"]
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = np.log10(t_0)

    t_table = np.linspace(T_TABLE_MIN, 10**lgt0, 100)

    mzr_params = umzr.DEFAULT_MZR_PARAMS

    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS
    # scatter_params = DEFAULT_SCATTER_PARAMS
    # ssp_err_pop_params = ssp_err_model.DEFAULT_SSPERR_PARAMS

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        jran.key(1),
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        mzr_params,
        spspop_params,
    )
    halpha_L_true = dpop_halpha_L(*args)

    (
        lgL_bin_edges,
        halpha_lf_weighted_smooth_ms_true,
        halpha_lf_weighted_q_true,
    ) = dpop_halpha_lf_weighted(halpha_L_true)

    halpha_lf_weighted_composite_true = (
        halpha_lf_weighted_smooth_ms_true + halpha_lf_weighted_q_true
    )

    rng = np.random.default_rng(123)
    # one perturbed draw (~1% fractional jitter)
    perturbed_params = perturb_diffstarpop(
        DEFAULT_DIFFSTARPOP_PARAMS, scale=0.1, rng=rng
    )

    theta_perturbed, _ = ravel_pytree(perturbed_params)

    fit_args = (
        theta_perturbed[idx],
        halpha_lf_weighted_composite_true,
        jran.key(1),
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        mzr_params,
        spspop_params,
    )

    loss_hist, theta_best_fit = fit_diffstarpop(*fit_args, n_steps=1000, step_size=1e-3)

    theta_full_best = theta_default.at[idx].set(theta_best_fit)
    fit_params = unravel_fn(theta_full_best)

    args = (
        fit_params,
        jran.key(1),
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        mzr_params,
        spspop_params,
    )
    halpha_L_fit = dpop_halpha_L(*args)
    (
        _,
        halpha_lf_weighted_smooth_ms_fit,
        halpha_lf_weighted_q_fit,
    ) = dpop_halpha_lf_weighted(halpha_L_fit)

    halpha_lf_weighted_composite_fit = (
        halpha_lf_weighted_smooth_ms_fit + halpha_lf_weighted_q_fit
    )

    assert np.allclose(
        halpha_lf_weighted_composite_true, halpha_lf_weighted_composite_fit, atol=1
    )
