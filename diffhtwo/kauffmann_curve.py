"""
"""

from collections import OrderedDict, namedtuple

from dsps.utils import _inverse_sigmoid, _sig_slope, _sigmoid
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

XTP = -0.5

BPTCUT_PDICT = OrderedDict(ytp=0.23, x0=0.18, slope_k=3.82, lo=-0.70, hi=-21.43)
BPTCutParams = namedtuple("BPTCutParams", BPTCUT_PDICT.keys())
DEFAULT_BPTCUT_PARAMS = BPTCutParams(**BPTCUT_PDICT)

_BPTCUT_UPNAMES = ["u_" + key for key in BPTCUT_PDICT.keys()]
BPTCutUParams = namedtuple("BPTCutUParams", _BPTCUT_UPNAMES)

BPTCUT_BOUNDS_PDICT = OrderedDict(
    ytp=(-2.0, 2.0),
    x0=(-0.75, 0.25),
    slope_k=(0.25, 15.0),
    lo=(-2.0, 0.5),
    hi=(-25.0, -1.0),
)

BPTCUT_PBOUNDS = BPTCutParams(**BPTCUT_BOUNDS_PDICT)


@jjit
def kauffmann_bpt_division(lgx):
    num = 0.61
    denom = lgx - 0.05
    return num / denom + 1.3


@jjit
def _get_bounded_bptcut_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_bptcut_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_bptcut_params_kern = jjit(vmap(_get_bounded_bptcut_param, in_axes=_C))
_get_bptcut_u_params_kern = jjit(vmap(_get_unbounded_bptcut_param, in_axes=_C))


@jjit
def get_bounded_bptcut_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _BPTCUT_UPNAMES])
    avpop_params = _get_bptcut_params_kern(
        jnp.array(u_params), jnp.array(BPTCUT_PBOUNDS)
    )
    return BPTCutParams(*avpop_params)


@jjit
def get_unbounded_bptcut_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_BPTCUT_PARAMS._fields]
    )
    u_params = _get_bptcut_u_params_kern(jnp.array(params), jnp.array(BPTCUT_PBOUNDS))
    return BPTCutUParams(*u_params)


@jjit
def model_pred(params, x):
    ytp, x0, slope_k, lo, hi = params
    pred = _sig_slope(x, XTP, ytp, x0, slope_k, lo, hi)
    return pred


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def _loss_kern(u_params, loss_data):
    params = get_bounded_bptcut_params(u_params)
    x_target, y_target = loss_data[0:2]
    pred = model_pred(params, x_target)
    loss = _mse(pred, y_target)
    return loss


@jjit
def bptcut_sigmoid_model(x):
    return model_pred(DEFAULT_BPTCUT_PARAMS, x)


@jjit
def probabilistic_bpt_class(lgx, lgy):
    lgy_cut = kauffmann_bpt_division(lgx)
    weight = _sigmoid(lgy, lgy_cut, 2.0, 1.0, 0.0)
    return weight
