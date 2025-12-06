import jax.numpy as jnp
import numpy as np
from diffhtwo.experimental import n_mag
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_PARAMS
from diffsky.ssp_err_model import ssp_err_model
from diffstar.defaults import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.metallicity import umzr
from jax import jacfwd


def n_mag_kern_wrapper(diffstarpop_fiducial_params, *args):
    lg_n, _ = n_mag.n_mag_kern(diffstarpop_fiducial_params, *args)

    return lg_n


def get_fisher_matrix(
    diffstarpop_params,
    lc_halopop,
    lh_centroids,
    dmag,
    mag_column,
    tcurves,
    ssp_data,
    n_key,
    zmin=0.2,
    zmax=0.5,
):
    n_z_phot_table = 15

    z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = jnp.log10(t_0)
    t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

    # params
    mzr_params = umzr.DEFAULT_MZR_PARAMS
    scatter_params = DEFAULT_SCATTER_PARAMS
    ssp_err_pop_params = ssp_err_model.ZERO_SSPERR_PARAMS

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
    )

    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    args = (
        DEFAULT_SPSPOP_PARAMS,
        n_key,
        lc_halopop,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        lh_centroids,
        dmag,
        mag_column,
    )

    # Get the Jacobian
    Jacobian = jacfwd(n_mag_kern_wrapper)(diffstarpop_params, *args)

    # Get error
    _, lg_n_avg_err = n_mag.n_mag_kern(diffstarpop_params, *args)
    w = 1.0 / lg_n_avg_err**2  # weights per number density bin

    # Compute the Fisher matrix
    Fisher = Jacobian.T @ (w[:, None] * Jacobian)

    return Fisher


def sample_fisher_gaussian(
    Fisher, diffstarpop_theta, labels=None, nsamp=20000, subset=None
):
    """
    Fisher : Fisher matrix (n_params, n_params)
    theta0 : fiducial parameter vector (n_params,)
    labels : list of parameter names (optional)
    nsamp  : number of Gaussian samples to draw
    subset : optional list of parameter indices to keep for plotting
             (e.g., range(20) or [0,1,2,10,11,...])
    """

    Sigma = np.linalg.pinv(Fisher)

    npar = len(diffstarpop_theta)

    if subset is not None:
        idx = np.array(subset)
        Sigma = Sigma[np.ix_(idx, idx)]
        diffstarpop_theta = diffstarpop_theta[idx]
        if labels is not None:
            labels = [labels[i] for i in idx]
    else:
        idx = np.arange(npar)

    # draw Gaussian samples
    return np.random.multivariate_normal(diffstarpop_theta, Sigma, size=nsamp)
