# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


import jax.numpy as jnp
from jax import jit as jjit

N_FLOOR = 0.5
N_0 = 0.005


"""
Gehrels Poisson error 
"""


@jjit
def Gehrels_upp_eq9(Ngal):
    """
    upper limit approximation - Eq. 9 Gehrels (1986); 1-sigma
    """
    Ngal = jnp.asarray(Ngal, dtype=float)

    return (Ngal + 1) * (
        1 - (1 / (9 * (Ngal + 1))) + (1 / (3 * jnp.sqrt(Ngal + 1)))
    ) ** 3


@jjit
def Gehrels_low_eq12(Ngal):
    """
    lower limit approximation - Eq. 12 Gehrels (1986); 1-sigma
    """
    Ngal = jnp.asarray(Ngal, dtype=float)

    # use a safe placeholder for N=0 to avoid div/0 in the formula
    N_safe = jnp.where(Ngal > 0.0, Ngal, 1.0)

    low_raw = (
        N_safe * (1.0 - 1.0 / (9.0 * N_safe) - 1.0 / (3.0 * jnp.sqrt(N_safe))) ** 3
    )

    # now overwrite N=0 with 0.0
    return jnp.where(Ngal > 0.0, low_raw, 0.0)


@jjit
def get_n_data_err(N, vol, N_floor=N_FLOOR, N_o=N_0):
    """
    When `N <~ 0.5`, Gehrels_low_eq12(N) returns -ve values.
    `non_zero` boolean mask based on `N_floor` guards against that.
    See line --> `N_low = jnp.where(non_zero, N_low, N_o)` below.
    But it will fail for `N_floor` <~ 0.5.
    Therefore, keep `N_floor` > ~ 0.5.
    """
    non_zero = N > N_floor

    N = jnp.where(non_zero, N, N_o)
    lg_n = jnp.log10(N / vol)

    # upper limit approximation - Eq. 9 Gehrels (1986); 1-sigma
    N_upp = Gehrels_upp_eq9(N)
    lg_n_upp = jnp.log10(N_upp / vol)
    lg_n_upp_err = lg_n_upp - lg_n

    # lower limit approximation - Eq. 12 Gehrels (1986); 1-sigma
    N_low = Gehrels_low_eq12(N)
    N_low = jnp.where(non_zero, N_low, N_o)
    lg_n_low = jnp.log10(N_low / vol)

    lg_n_low_err = lg_n - lg_n_low

    lg_n_avg_err = (lg_n_low_err + lg_n_upp_err) / 2

    # just the upper limit for N ~ 0
    lg_n_upp_err_zero = jnp.log10(1.84 / vol) - lg_n
    lg_n_avg_err = jnp.where(non_zero, lg_n_avg_err, lg_n_upp_err_zero)

    return lg_n, lg_n_avg_err
