import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from ..kernels.N_phot import N_colors_mags

plt.rc("font", family="serif", serif=["Times New Roman"])

# Pantone: Dress Blues → Classic Blue → Aqua Sky → Minty Green → Illuminating
density_cmap = LinearSegmentedColormap.from_list(
    "pantone_density",
    [
        "#1B2A4A",  # Dress Blues      — empty/low
        "#0F4C81",  # Classic Blue
        "#00A591",  # Arcadia
        "#84BD00",  # Greenery
        "#FEDF00",  # Illuminating     — peak density
    ],
)
dusk = LinearSegmentedColormap.from_list(
    "dusk",
    [
        "#1B1F3B",  # Evening Blue
        "#7B4F9E",  # Amethyst Orchid
        "#E8A598",  # Peach Pink
        "#F5E6C8",  # Almond Milk
    ],
)


def plot_density(
    bin_lo, bin_hi, N, ax, xlabel, ylabel, cmap, N_model=None, sigma=0.55, n_levels=8
):
    x_edges = np.unique(np.append(bin_lo[:, 0], bin_hi[-1, 0]))
    y_edges = np.unique(np.append(bin_lo[:, 1], bin_hi[-1, 1]))
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    Z = np.log10(
        gaussian_filter(
            (N / N.sum()).reshape(len(y_edges) - 1, len(x_edges) - 1).astype(float),
            sigma=sigma,
        ).clip(min=np.finfo(float).tiny)
    )
    levels = np.linspace(Z.min(), Z.max(), n_levels)
    qm = ax.contourf(xc, yc, Z, levels=levels, cmap=cmap, alpha=0.5)
    ax.get_figure().colorbar(qm, ax=ax, label=r"$\log_{10}(N / N_{\rm tot})$")
    if N_model is not None:
        Z_model = np.log10(
            gaussian_filter(
                (N_model / N_model.sum())
                .reshape(len(y_edges) - 1, len(x_edges) - 1)
                .astype(float),
                sigma=sigma,
            ).clip(min=np.finfo(float).tiny)
        )
        ax.contour(
            xc,
            yc,
            Z_model,
            levels=levels,
            cmap=cmap,
            linewidths=1.5,
            alpha=0.9,
            linestyles="dashed",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_density_raw(bin_lo, bin_hi, N, ax, xlabel, ylabel, cmap, N_model=None):
    x_edges = np.unique(np.append(bin_lo[:, 0], bin_hi[-1, 0]))
    y_edges = np.unique(np.append(bin_lo[:, 1], bin_hi[-1, 1]))
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    Z = np.log10(
        (N / N.sum())
        .reshape(len(y_edges) - 1, len(x_edges) - 1)
        .astype(float)
        .clip(min=np.finfo(float).tiny)
    )
    qm = ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap)
    ax.get_figure().colorbar(qm, ax=ax, label=r"$\log_{10}(N / N_{\rm tot})$")
    if N_model is not None:
        Z_model = np.log10(
            (N_model / N_model.sum())
            .reshape(len(y_edges) - 1, len(x_edges) - 1)
            .astype(float)
            .clip(min=np.finfo(float).tiny)
        )
        levels = np.linspace(Z.min(), Z.max(), 8)
        ax.contour(xc, yc, Z_model, levels=levels, cmap=cmap, linewidths=0.8, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_color_contours(
    ran_key,
    param_collection,
    data,
    mag_thresh,
    frac_cat,
    data_label,
    savedir,
):
    for z in range(0, len(data)):
        z_data = data[z]

        z_data_model = N_colors_mags(
            ran_key,
            param_collection,
            z_data,
            mag_thresh,
            frac_cat,
        )
        fields = z_data_model._fields[3:]
        z_min = z_data_model.z_min
        z_max = z_data_model.z_max

        for f in range(0, len(fields)):
            space = getattr(z_data_model, fields[f])

            if isinstance(space, list):
                pass

            else:
                fig, ax = plt.subplots(constrained_layout=True)
                fig.suptitle(str(z_min) + " < z < " + str(z_max))
                name = type(space).__name__
                xlabel, ylabel = parse_color_labels(name)
                plot_density(
                    space.bin_lo,
                    space.bin_hi,
                    space.N_data,
                    ax,
                    xlabel,
                    ylabel,
                    dusk,
                    N_model=space.N_model,
                )
                fig.savefig(
                    savedir
                    + "/"
                    + data_label
                    + "_"
                    + name
                    + "_"
                    + str(z_min)
                    + "-"
                    + str(z_max)
                    + ".png",
                    dpi=300,
                )
    plt.close()


def parse_color_labels(name):
    # "Ur_ri" → ["u-r", "r-i"]
    x_str, y_str = name.lower().split("_")

    def to_label(s):
        return f"${s[0]} - {s[1]}$"

    return to_label(x_str), to_label(y_str)
