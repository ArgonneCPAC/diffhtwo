import numpy as np


def compute_Sigma_excess(BB, NB, BB_err, NB_err, ZP):
    """
    Notes:
        - "The parameter (see Bunker et al. 1995) quantifies the excess compared to the random
        scatter expected for a source with zero colour, as a function of narrow-band magnitude
        (see e.g. S09)" - Sobral+ 2013
        - NB can also be a medium-band as in Minerva dataset.
        - Compared to Sobral+13, the area term is dropped as we are considering total error (not per pixel)
    """
    numerator = 1 - 10 ** (-0.4 * (BB - NB))

    delta = np.sqrt(BB_err**2 + NB_err**2)
    denominator = delta * 10 ** (-0.4 * (ZP - NB))

    return numerator / denominator
