"""
Tabulate HII region emission line templates averaged over φ(Q_H)

Starting from L_ijk, an input HII line template of shape (n_lines, n_qh, n_met, n_age),
and from q_ij, a normalization table:
    * calculate L_ij, the Q_H-luminosity-function-weighted average of L_ijk,
    * calculate ⟨Q_H⟩ = ∑_k φ_k Q_H_k / ∑_k φ_k, the φ(Q_H)-averaged value of Q_H
    * X_ij = L_ij q_ij / ⟨Q_H⟩, the integrand in galaxy line luminosity calculations

X_ij is defined as follows:

    L_ij = ∑_k φ_k L_ijk / ∑_k φ_k

    ⟨Q_H⟩ = ∑_k φ_k Q_H_k / ∑_k φ_k

    Now define:
    X_ij := L_ij q_ij / ⟨Q_H⟩

    If M_ij is the SFH, then we compute the galaxy luminosity L as:
    L = ∑_ij X_ij*M_ij

The output hdf5 file created by this script has the following columns:

    * qh_weights : array, shape (n_lines, n_qh)
        values of the PDF weights φ_k used to compute L_ij and X_ij

    * avg_Q_H : ⟨Q_H⟩, float

    * integrand/{LINE_NAME} : array, shape (n_met, n_age)
        matrix X_ij for the line

    * pdf_weighted_lines/{LINE_NAME} : array, shape (n_met, n_age)
        matrix L_ij for the line

"""

import argparse
import os

import h5py
import numpy as np

from diffhtwo.line_luminosity_kernels import _get_pdf_weighted_htwo_grid
from diffhtwo.load_cloudy_grid import load_cloudy

TASSO_DATA_DRN = "/Users/aphearin/work/DATA/DSPS_data"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-drn", help="Name of the input grid", default=TASSO_DATA_DRN)
    parser.add_argument("-drnout", help="Name of the input grid", default="")
    parser.add_argument(
        "-basename", help="Basename of the input grid", default="cloudyTableBC2003.hdf5"
    )
    parser.add_argument("-alpha", help="Power law slope", default=-1.73, type=float)
    args = parser.parse_args()
    drn = args.drn
    drnout = args.drnout
    alpha = args.alpha
    basename = args.basename

    fname = os.path.join(drn, basename)
    grid_dict, lines_dict = load_cloudy(fname)

    grid_norms = grid_dict["ionizingLuminosityHydrogenNormalized"]

    Q_H_grid = grid_dict["ionizingLuminosityHydrogen"]

    n_lines = len(lines_dict)
    n_qh = grid_dict["ionizingLuminosityHydrogen"].size
    n_met = grid_dict["metallicity"].size
    n_age = grid_dict["age"].size

    htwo_grid = np.zeros((n_lines, n_qh, n_met, n_age))
    for i, key in enumerate(lines_dict.keys()):
        htwo_grid[i, :, :, :] = lines_dict[key]

    args = (Q_H_grid, alpha, htwo_grid, grid_norms)
    integrand, data = _get_pdf_weighted_htwo_grid(
        Q_H_grid, alpha, htwo_grid, grid_norms
    )
    qh_weights, pdf_weighted_lines, avg_Q_H = data

    outbase = basename.replace(".hdf5", ".avg.alpha={0:.2f}.hdf5".format(alpha))
    outname = os.path.join(drnout, outbase)

    with h5py.File(outname, "w") as hdfout:
        hdfout["qh_weights"] = qh_weights[:, :, 0, 0]
        hdfout["avg_Q_H"] = avg_Q_H[0]
        for i, line_name in enumerate(lines_dict.keys()):
            hdfout["integrand/" + line_name] = integrand[i, :, :]
            hdfout["pdf_weighted_lines/" + line_name] = pdf_weighted_lines[i, :, :]
