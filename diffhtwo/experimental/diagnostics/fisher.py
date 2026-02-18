import jax
import jax.numpy as jnp
import numpy as np
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_PARAMS
from diffsky.ssp_err_model import ssp_err_model
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop import get_unbounded_diffstarpop_params
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
    DiffstarPop_UParams_Diffstarpopfits_mgash,
)
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.metallicity import umzr
from jax import jacfwd
from jax import jit as jjit
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental import n_mag, n_mag_opt

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]
DIFFSTARPOP_U_UM_plus_exsitu = DiffstarPop_UParams_Diffstarpopfits_mgash["smdpl_dr1"]

DEFAULT_DIFFSTARPOP_THETA, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)


IDX = jnp.arange(16, 18, 1)


@jjit
def log_likelihood(diffstarpop_theta_sub, *args):
    diffstarpop_theta_full = DIFFSTARPOP_UM_plus_exsitu.at[IDX].set(
        diffstarpop_theta_sub
    )
    diffstarpop_params = unravel_fn(diffstarpop_theta_full)
    u_diffstarpop_params = get_unbounded_diffstarpop_params(diffstarpop_params)
    u_diffstarpop_theta, u_unravel_fn = ravel_pytree(u_diffstarpop_params)
    return -0.5 * n_mag_opt._loss_kern(u_diffstarpop_theta, *args)


def get_fisher(log_likelihood, diffstarpop_theta_sub, *args):
    return -jax.hessian(log_likelihood)(diffstarpop_theta_sub, *args)


def sample_fisher_gaussian(Fisher, diffstarpop_theta, nsamp=20000, subset=None):
    """
    Fisher : Fisher matrix (n_params, n_params)
    diffstarpop_theta : flat diffstarpop array (n_params,)
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
    else:
        idx = np.arange(npar)

    # draw Gaussian samples
    return np.random.multivariate_normal(diffstarpop_theta, Sigma, size=nsamp)
