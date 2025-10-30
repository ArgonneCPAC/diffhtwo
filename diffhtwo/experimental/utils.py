import jax.numpy as jnp
from jax import jit as jjit


@jjit
def safe_log10(x):
    EPS = 1e-12
    return jnp.log(jnp.clip(x, EPS, jnp.inf)) / jnp.log(10.0)
