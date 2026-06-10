import jax.numpy as jnp
import numpy as np
from diffsky import diffndhist_lomem


def get_N_1d(dim1, dim1_bin_edges=None, dmag=0.2, sig_scale=0.5):
    dataset = dim1.reshape(dim1.size, 1)
    if dim1_bin_edges is None:
        dim1_bin_edges = np.arange(dim1.min(), dim1.max(), dmag)

    bin_lo = dim1_bin_edges[:-1].reshape(dim1_bin_edges[:-1].size, 1)
    bin_hi = dim1_bin_edges[1:].reshape(dim1_bin_edges[1:].size, 1)

    sig = jnp.zeros_like(bin_lo) + (dmag * sig_scale)

    N_1d = diffndhist_lomem.tw_ndhist(
        dataset,
        sig,
        bin_lo,
        bin_hi,
    )

    return (
        N_1d,
        sig,
        bin_lo,
        bin_hi,
    )


def get_N_2d(dim1, dim2, sig_scale=0.5, dim1_is_mag=False):
    dataset = np.vstack((dim1, dim2)).T

    if dim1_is_mag:
        dim1_bin_edges = np.linspace(dim1.min(), dim1.max(), 4)
    else:
        dim1_bin_edges = np.linspace(dim1.min(), dim1.max(), 11)

    dim2_bin_edges = np.linspace(dim2.min(), dim2.max(), 11)

    dim1_lo = dim1_bin_edges[:-1]
    dim2_lo = dim2_bin_edges[:-1]
    bin_lo = np.meshgrid(dim1_lo, dim2_lo, indexing="ij")
    bin_lo = np.array(bin_lo).T.reshape(-1, 2)

    dim1_hi = dim1_bin_edges[1:]
    dim2_hi = dim2_bin_edges[1:]
    bin_hi = np.meshgrid(dim1_hi, dim2_hi, indexing="ij")
    bin_hi = np.array(bin_hi).T.reshape(-1, 2)

    sig1 = np.diff(dim1_bin_edges) * sig_scale
    sig2 = np.diff(dim2_bin_edges) * sig_scale
    sig = np.meshgrid(sig1, sig2, indexing="ij")
    sig = np.array(sig).T.reshape(-1, 2)

    N_2d = diffndhist_lomem.tw_ndhist(
        dataset,
        sig,
        bin_lo,
        bin_hi,
    )

    return N_2d, sig, bin_lo, bin_hi
