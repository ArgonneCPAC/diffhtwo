from difflib import get_close_matches

import jax.numpy as jnp
from jax import jit as jjit
from jax.tree_util import tree_flatten_with_path


@jjit
def lupton_log10(t, log10_clip, t0=0.0, M0=0.0, alpha=1 / jnp.log(10.0)):
    """Clipped base-10 log function taken from
    https://github.com/ArgonneCPAC/shamnet/blob/d47c842bfc5ad751ad63d0b21100db709de52e58/shamnet/utils.py#L217C5-L217C17

    Parameters
    ----------
    t : ndarray of shape (n, )

    log10_clip : float
        Returned values of t larger than log10_clip will agree with log10(t).
        Values smaller than log10(t) will converge to 10**log10_clip.

    Returns
    -------
    lup : ndarray of shape (n, )

    """
    k = 10.0**log10_clip
    return M0 + alpha * (jnp.arcsinh((t - t0) / (2 * k)) + jnp.log(k))


@jjit
def safe_log10(x, EPS=1e-12):
    return jnp.log(jnp.clip(x, EPS, jnp.inf)) / jnp.log(10.0)


def get_tcurve(get_filter_number, filter_info_filename, tcurves_filename):
    with open(filter_info_filename) as INFO:
        info = INFO.readlines()
    with open(tcurves_filename) as TCURVES:
        tcurves = TCURVES.readlines()

    f_idx = get_filter_number - 1
    t_idx = tcurves.index(get_close_matches(info[f_idx], tcurves)[0])

    i = 0
    wave_aa = []
    trans = []
    while (len(tcurves[t_idx + 1 :][i].split()) <= 3) & (
        (t_idx + 2 + i) < len(tcurves)
    ):
        wave_aa.append(float(tcurves[t_idx + 1 :][i].split()[-2]))
        trans.append(float(tcurves[t_idx + 1 :][i].split()[-1]))
        i += 1

    return jnp.array(wave_aa), jnp.array(trans)


def get_feniks_filter_number_from_translate_file(translate_file, filter_name):
    idx = translate_file["col1"] == "fcol_" + filter_name
    filter_number = int(translate_file[idx]["col2"].data[0][1:])
    return filter_number


def get_param_names(params):
    paths, leaves = tree_flatten_with_path(params)
    names = [p[0][0].name for p in paths]  # each p is (GetAttrKey(...),)
    return names
