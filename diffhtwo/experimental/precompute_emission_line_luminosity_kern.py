import jax.numpy as jnp
import numpy as np
import numpy.ma as ma
from jax import grad

# copied from astropy.constants.c.value
C = 299792458.0


def _get_clipped_sed(wave, sed, wave_lo, wave_hi):
    sel = (wave > wave_lo) & (wave < wave_hi)
    wave_clipped = wave[sel]
    sed_clipped = sed[sel]

    return wave_clipped, sed_clipped


def _get_masked_sed(continuum_wave, continuum_rest_sed, lo, hi):
    # mask to fit continuum
    mask = (continuum_wave >= lo) & (continuum_wave <= hi)
    continuum_wave_masked = ma.array(continuum_wave, mask=mask).compressed()
    continuum_rest_sed_masked = ma.array(continuum_rest_sed, mask=mask).compressed()

    return mask, continuum_wave_masked, continuum_rest_sed_masked


def _quad_continuum_model(theta, wave):
    c0 = theta["c0"]
    c1 = theta["c1"]
    c2 = theta["c2"]

    return (c2 * wave * wave) + (c1 * wave) + c0


def _mse(L_true: jnp.ndarray, L_pred: jnp.ndarray) -> jnp.float64:
    """Mean squared error function."""
    return jnp.mean(jnp.power(L_true - L_pred, 2))


def _mseloss(theta, model, wave, L_true):
    L_pred = model(theta, wave)
    return _mse(L_true, L_pred)


def _model_optimization_loop(
    theta, model, loss, wave, L_true, n_steps=1000, step_size=1e-18
):
    dloss = grad(loss)

    # initial continuum_rest_sed
    continuum_rest_sed_initial = model(
        dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), wave
    )

    losses = []
    for i in range(n_steps):
        grads = dloss(
            dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), model, wave, L_true
        )

        theta["c0"] = theta["c0"] - step_size * grads["c0"]
        theta["c1"] = theta["c1"] - step_size * grads["c1"]
        theta["c2"] = theta["c2"] - step_size * grads["c2"]

        losses.append(
            loss(
                dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]),
                model,
                wave,
                L_true,
            )
        )

    # fitted continuum_rest_sed
    continuum_rest_sed_fit = model(
        dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), wave
    )

    return losses, theta, continuum_rest_sed_initial, continuum_rest_sed_fit


def _get_line_sed(wave, sed, continuum_sed, line_mask):
    """
    Parameters:
            wave: wavelength in Angstroms
            sed: SED in units log10 of [Lsun/Hz/Msun]
            continuum_sed: rest-frame continuum fitted sed in units of Lsun/Hz/Msun
            line_mask: boolean mask array to select wave spanning line

    Returns:
            line_wave: wavelength in Angstroms spanning line
            line_sed: sed [Lsun/Hz/Msun] - continuum_sed [Lsun/Hz/Msun]
    """

    line_wave = wave[line_mask]
    line_sed = 10 ** sed[line_mask] - 10 ** continuum_sed[line_mask]

    # Protects against ANY sed wiggle within line wavelengths being below the
    # continuum sed fit --> Indication of no line.
    # Previously the code was written in a way that ALL sed wiggles had to be below
    # the continuum fit, for the line_sed to be set to zero array.
    if (line_sed <= 0).sum() >= 1:
        all_true = jnp.ones(line_sed.shape, dtype=bool)
        line_sed = jnp.where(
            all_true, np.zeros(line_sed.shape), np.zeros(line_sed.shape)
        )

    return line_wave, line_sed


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

    integrated_luminosity = np.trapezoid(sed, freq)  # [Lsun/Msun]

    return integrated_luminosity


def get_emission_line_luminosity(
    wave,
    sed,
    continuum_fit_lo_lo,
    continuum_fit_lo_hi,
    continuum_fit_hi_lo,
    continuum_fit_hi_hi,
    line_center,
    line_delta,
):
    """
    Notes:
            "clipped" means sed clipped between continuum_fit_lo_lo and
            continuum_fit_hi_hi
            "masked", in additon to wavelengths dropped in "clipped",
            masks line wavelengths for continuum fitting
    """
    wave_clipped, sed_clipped = _get_clipped_sed(
        wave, sed, continuum_fit_lo_lo, continuum_fit_hi_hi
    )

    # mask line wavelengths for continuum fitting
    mask, wave_masked, sed_masked = _get_masked_sed(
        wave_clipped, sed_clipped, continuum_fit_lo_hi, continuum_fit_hi_lo
    )

    # initialize with qudratic coefficients of a flat line at the mean of
    # continuum_rest_sed_masked
    c0_initial = jnp.mean(sed_masked)
    c1_initial = 0.0
    c2_initial = 0.0

    (
        losses,
        theta,
        continuum_sed_initial_masked,
        continuum_sed_fit_masked,
    ) = _model_optimization_loop(
        dict(c0=c0_initial, c1=c1_initial, c2=c2_initial),
        _quad_continuum_model,
        _mseloss,
        wave_masked,
        sed_masked,
    )

    # interpolate continuum rest sed in the line masked region
    continuum_sed_fit_clipped = jnp.interp(
        wave_clipped, wave_masked, continuum_sed_fit_masked
    )

    # limit to line wavelengths
    line_lo = line_center - line_delta
    line_hi = line_center + line_delta
    line_mask = (wave_clipped >= line_lo) & (wave_clipped <= line_hi)
    line_wave, line_sed = _get_line_sed(
        wave_clipped, sed_clipped, continuum_sed_fit_clipped, line_mask
    )

    line_integrated_luminosity = _get_integrated_luminosity(line_wave, line_sed)

    sed_dict = {
        "wave_masked": wave_masked,
        "sed_masked": sed_masked,
        "continuum_sed_initial_masked": continuum_sed_initial_masked,
        "continuum_sed_fit_masked": continuum_sed_fit_masked,
        "line_wave": line_wave,
        "line_sed": line_sed,
        "line_integrated_luminosity": line_integrated_luminosity,
    }

    return losses, theta, sed_dict
