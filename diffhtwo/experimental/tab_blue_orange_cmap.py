import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb


def _srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d**3, xyz ** (1 / 3), xyz / (3 * d**2) + 4 / 29)
    L, a, b = (
        116 * f[..., 1] - 16,
        500 * (f[..., 0] - f[..., 1]),
        200 * (f[..., 1] - f[..., 2]),
    )
    return np.stack([L, a, b], axis=-1)


def _lab_to_srgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29
    finv = lambda t: np.where(t > d, t**3, 3 * d**2 * (t - 4 / 29))
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], -1) * np.array(
        [0.95047, 1.0, 1.08883]
    )
    Minv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    lin = np.clip(xyz @ Minv.T, 0, None)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def make_cmap(c1="tab:blue", c2="tab:orange", mid="#4B2E83", n=256, name="blue_orange"):
    """Viridis-style perceptually-uniform colormap between two endpoint colors,
    bent through a `mid` anchor in Lab space to avoid a muddy gray midpoint,
    with a monotonic lightness ramp enforced."""
    lab1, labm, lab2 = (_srgb_to_lab(to_rgb(c)) for c in (c1, mid, c2))
    if not (lab1[0] < labm[0] < lab2[0]):
        labm[0] = (lab1[0] + lab2[0]) / 2  # keep lightness monotonic
    half = n // 2
    path = np.zeros((n, 3))
    for i in range(3):
        path[:half, i] = np.linspace(lab1[i], labm[i], half)
        path[half:, i] = np.linspace(labm[i], lab2[i], n - half)
    return LinearSegmentedColormap.from_list(name, _lab_to_srgb(path), N=n)


# usage:
# cmap = make_cmap()                      # tab:blue -> tab:orange (default)
# plt.imshow(data, cmap=cmap)
