import numpy as np
from matplotlib.colors import LogNorm


def make_thresholded_reduce_C_function(weights, threshold=1e-5):
    total = np.nansum(weights)

    def _reduce(C_bin):
        frac = np.nansum(C_bin) / total
        return frac if frac >= threshold else np.nan

    return _reduce


def percentile_norm(all_c, lo=0.01, hi=98):
    concat = np.concatenate([np.asarray(c).ravel() for c in all_c])
    pos = concat[concat > 0]
    return LogNorm(vmin=np.percentile(pos, lo), vmax=np.percentile(pos, hi))
