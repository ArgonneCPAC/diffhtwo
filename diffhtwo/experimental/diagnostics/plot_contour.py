import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from ..kernels.N_phot import N_colors_mags

plt.rc("font", family="serif", serif=["Times New Roman"])
plt.rc(
    "mathtext",
    fontset="custom",
    rm="Times New Roman",
    it="Times New Roman:italic",
    bf="Times New Roman:bold",
)
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
    bin_lo,
    bin_hi,
    N,
    ax,
    xlabel,
    ylabel,
    cmap,
    data_label,
    fontsize=18,
    N_model=None,
    sigma=0.5,
    n_levels=10,
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
    Z_min = np.max((-10, Z.min()))
    Z_max = Z.max()
    # Z_min = -7
    # Z_max = -1
    levels = np.linspace(Z_min, Z_max, n_levels)
    qm = ax.contourf(
        xc, yc, Z, levels=levels, cmap=cmap, alpha=0.5, vmin=Z_min, vmax=Z_max
    )

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
            linewidths=0.6,
            alpha=1,
            linestyles="dashed",
            vmin=Z_min,
            vmax=Z_max,
        )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    return qm


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
    ax.set_xlabel(xlabel, labelpad=0.8)
    ax.set_ylabel(ylabel, labelpad=0.8)


def plot_color_contour_grid(
    ran_key,
    param_collection,
    data,
    mag_thresh,
    frac_cat,
    data_label,
    savedir,
    fields,
    sigma=0.5,
    n_levels=10,
):
    labelsize = 9
    fontsize = 10
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.8), constrained_layout=True)
    fig.get_layout_engine().set(
        h_pad=0.0, wspace=0.05, hspace=0.05, rect=(0, 0, 1, 0.925)
    )
    for z in range(0, len(data)):
        z_data = data[z]

        z_data_model = N_colors_mags(
            ran_key,
            param_collection,
            z_data,
            mag_thresh,
            frac_cat,
        )
        # fields = z_data_model._fields[4:]
        fields_at_z = fields[z]
        z_min = z_data_model.z_min
        z_max = z_data_model.z_max
        ax[0][z].set_title(
            str(z_min) + " < z < " + str(z_max), fontsize=fontsize, y=0.99
        )

        for f in range(0, len(fields_at_z)):
            space = getattr(z_data_model, fields_at_z[f])

            name = type(space).__name__
            xlabel, ylabel = parse_color_labels(name)
            qm = plot_density(
                space.bin_lo,
                space.bin_hi,
                space.N_data,
                ax[f][z],
                xlabel,
                ylabel,
                dusk,
                data_label,
                fontsize=fontsize,
                N_model=space.N_model,
                sigma=sigma,
                n_levels=n_levels,
            )
            ax[f][z].minorticks_on()
            ax[f][z].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[f][z].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )

    cbar = fig.colorbar(
        qm,
        ax=ax.ravel().tolist(),
        location="right",
        shrink=1,
        aspect=40,
        pad=0.01,
    )
    cbar.ax.tick_params(
        labelsize=labelsize, labelleft=False, labelright=False, direction="in", length=5
    )
    cbar.set_label(
        r"$\log_{10}(N / N_{\rm tot})$ [arbitrary levels]", fontsize=labelsize
    )

    legend_handles = [mpatches.Patch(color=dusk(0.7), alpha=0.5, label=data_label)]
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color=dusk(0.7),
            linewidth=1.5,
            linestyle="dashed",
            alpha=0.9,
            label="diffsky",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=fontsize,
        borderaxespad=0.0,
    )

    fig.savefig(
        savedir + "/cc_cm_grid.png",
        dpi=600,
    )
    plt.close()


def plot_color_contours(
    ran_key,
    param_collection,
    data,
    mag_thresh,
    frac_cat,
    data_label,
    savedir,
    sigma=0.5,
    n_levels=10,
):
    labelsize = 14
    for z in range(0, len(data)):
        z_data = data[z]

        z_data_model = N_colors_mags(
            ran_key,
            param_collection,
            z_data,
            mag_thresh,
            frac_cat,
        )
        fields = z_data_model._fields[4:]
        z_min = z_data_model.z_min
        z_max = z_data_model.z_max

        for f in range(0, len(fields)):
            space = getattr(z_data_model, fields[f])

            if isinstance(space, list):
                pass

            else:
                fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
                fig.suptitle(str(z_min) + " < z < " + str(z_max), fontsize=18, y=0.99)
                fig.get_layout_engine().set(h_pad=0.0, hspace=0.0, rect=(0, 0, 1, 0.95))

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
                    data_label,
                    N_model=space.N_model,
                    sigma=sigma,
                    n_levels=n_levels,
                )
                ax.minorticks_on()
                ax.tick_params(
                    which="major",
                    direction="in",
                    top=True,
                    right=True,
                    length=6,
                    width=1,
                    labelsize=labelsize,
                )
                ax.tick_params(
                    which="minor",
                    direction="in",
                    top=True,
                    right=True,
                    length=3,
                    width=0.8,
                    labelsize=labelsize,
                )

                legend_handles = [
                    mpatches.Patch(color=dusk(0.7), alpha=0.5, label=data_label)
                ]
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=dusk(0.7),
                        linewidth=1.5,
                        linestyle="dashed",
                        alpha=0.9,
                        label="diffsky",
                    )
                )

                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.1),
                    ncol=len(legend_handles),
                    frameon=False,
                    fontsize=16,
                    borderaxespad=0.0,
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


def parse_axis_label(s):
    nir_bands = {"j", "h", "k"}

    def fmt(b):
        return b.upper() if b in nir_bands else b

    if len(s) == 2:
        return f"${fmt(s[0])}-{fmt(s[1])}$"
    return f"${fmt(s)}$"


def parse_color_labels(name):
    x_str, y_str = name.lower().split("_")
    return parse_axis_label(x_str), parse_axis_label(y_str)
