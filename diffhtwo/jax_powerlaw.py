"""pdf propto to x^{g-1} with g < 0
on an interval a<b which does not contain zero
"""

from functools import partial

from jax import jit as jjit
from jax import random as jran


@jjit
def powerlaw_pdf(x, a, b, g):
    """pdf(x) propto x^{g-1} for a<=x<=b. Assumes a<b and g!=0."""
    ag, bg = a**g, b**g
    return g * x ** (g - 1) / (bg - ag)


@partial(jjit, static_argnames=["npts"])
def powerlaw_rvs(ran_key, a, b, g, npts):
    """Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b. Assumes a<b and g!=0"""
    r = jran.uniform(ran_key, (npts,))
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r) ** (1.0 / g)
