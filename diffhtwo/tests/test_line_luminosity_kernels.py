"""
"""

import numpy as np
from jax import random as jran

from ..line_luminosity_kernels import (
    _avg_Q_H_kern,
    _get_pdf_weighted_htwo_grid,
    _get_pdf_weighted_htwo_singleline_kern,
    _get_qh_weights,
)


def test_get_qh_weights():
    Q_H_grid = np.array((1e48, 1e49, 1e50, 1e51, 1e52))
    x_grid = Q_H_grid / 1e48
    alpha = -1.73

    weights = _get_qh_weights(x_grid, alpha)
    assert np.all(np.isfinite(weights))
    assert np.allclose(weights.sum(), 1.0, rtol=1e-3)

    assert np.all(np.diff(weights) < 0)

    n_weights = weights.size
    for i in range(1, n_weights):
        ratio_i = weights[0] / weights[i]
        correct_ratio = (x_grid[0] / x_grid[i]) ** alpha
        assert np.allclose(ratio_i, correct_ratio, rtol=0.01)


def test_avg_Q_H_kern_behaves_as_expected():
    Q_H_grid = np.array((1e48, 1e49, 1e50, 1e51, 1e52))
    x_grid = Q_H_grid / 1e48
    alpha = -1.73
    Q_H_avg = _avg_Q_H_kern(x_grid, alpha)
    Q_H_avg2 = _avg_Q_H_kern(x_grid, alpha + 0.5)
    Q_H_avg3 = _avg_Q_H_kern(x_grid, alpha + 1.0)
    assert Q_H_avg < Q_H_avg2 < Q_H_avg3


def test_get_pdf_weighted_htwo_line_templates():
    Q_H_grid = np.array((1e48, 1e49, 1e50, 1e51, 1e52))
    alpha = -1.73
    n_met, n_age = 6, 50
    ran_key = jran.PRNGKey(0)
    htwo_line_grid = jran.uniform(ran_key, shape=(Q_H_grid.size, n_met, n_age))
    grid_normalization_table = np.ones((n_met, n_age))
    integrand, data = _get_pdf_weighted_htwo_singleline_kern(
        Q_H_grid, alpha, htwo_line_grid, grid_normalization_table
    )
    assert np.all(np.isfinite(integrand))
    for x in data:
        assert np.all(np.isfinite(x))

    qh_weights, pdf_weighted_line, avg_Q_H = data
    assert Q_H_grid.min() < avg_Q_H < Q_H_grid.max()
    assert integrand.shape == (n_met, n_age)
    assert pdf_weighted_line.shape == (n_met, n_age)
    assert np.allclose(qh_weights.sum(), 1.0, rtol=0.001)


def test_get_pdf_weighted_htwo_grid():
    Q_H_grid = np.array((1e48, 1e49, 1e50, 1e51, 1e52))
    n_qh = Q_H_grid.size
    alpha = -1.73
    n_met, n_age = 6, 50
    n_lines = 4
    ran_key = jran.PRNGKey(0)
    htwo_line_grid = jran.uniform(ran_key, shape=(n_lines, Q_H_grid.size, n_met, n_age))
    grid_normalization_table = np.ones((n_met, n_age))
    integrand, data = _get_pdf_weighted_htwo_grid(
        Q_H_grid, alpha, htwo_line_grid, grid_normalization_table
    )
    assert np.all(np.isfinite(integrand))
    for x in data:
        assert np.all(np.isfinite(x))

    qh_weights, pdf_weighted_line, avg_Q_H = data
    assert avg_Q_H.shape == (n_lines,)
    assert np.all(Q_H_grid.min() < avg_Q_H)
    assert np.all(avg_Q_H < Q_H_grid.max())

    assert integrand.shape == (n_lines, n_met, n_age)
    assert pdf_weighted_line.shape == (n_lines, n_met, n_age)

    qh_weights = qh_weights.reshape((n_lines, n_qh))
    assert np.allclose(np.sum(qh_weights, axis=1), 1.0, rtol=0.001)
