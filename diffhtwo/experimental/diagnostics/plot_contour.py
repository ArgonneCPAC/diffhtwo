import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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

dusk = ListedColormap(
    [
        "#3E4577",  # Evening Blue   -> outermost (>3sigma)
        "#7B4F9E",  # Amethyst Orchid -> 3-2 sigma
        "#E8A598",  # Peach Pink      -> 2-1 sigma
        "#F5E6C8",  # Almond Milk     -> 1sigma-peak
    ],
    name="dusk",
)


def sigma_levels(Z_lin, sigmas=(1, 2, 3)):
    flat = np.sort(Z_lin.ravel())[::-1]
    cumsum = np.cumsum(flat)
    cumsum /= cumsum[-1]
    fracs = 1 - np.exp(-0.5 * np.array(sigmas) ** 2)
    idx = np.searchsorted(cumsum, fracs)
    return np.sort(flat[idx])


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
    sigmas=(1, 2, 3),
    model_own_levels=True,
):
    x_edges = np.unique(np.append(bin_lo[:, 0], bin_hi[-1, 0]))
    y_edges = np.unique(np.append(bin_lo[:, 1], bin_hi[-1, 1]))
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])

    Z_lin = gaussian_filter(
        (N / N.sum()).reshape(len(y_edges) - 1, len(x_edges) - 1).astype(float),
        sigma=sigma,
    ).clip(min=np.finfo(float).tiny)
    Z = np.log10(Z_lin)

    levels_lin = sigma_levels(Z_lin, sigmas=sigmas)
    levels = np.log10(levels_lin)  # ascending: outer sigma -> inner sigma
    levels = np.concatenate([[Z.min()], levels, [Z.max()]])

    qm = ax.contourf(xc, yc, Z, levels=levels, colors=cmap.colors, alpha=0.5)

    if N_model is not None:
        Z_model_lin = gaussian_filter(
            (N_model / N_model.sum())
            .reshape(len(y_edges) - 1, len(x_edges) - 1)
            .astype(float),
            sigma=sigma,
        ).clip(min=np.finfo(float).tiny)
        Z_model = np.log10(Z_model_lin)

        if model_own_levels:
            model_levels_lin = sigma_levels(Z_model_lin, sigmas=sigmas)
            model_levels = np.concatenate(
                [[Z_model.min()], np.log10(model_levels_lin), [Z_model.max()]]
            )
        else:
            model_levels = levels  # compare against data's thresholds

        ax.contour(
            xc,
            yc,
            Z_model,
            levels=model_levels,
            colors=cmap.colors,
            linewidths=0.6,
            alpha=1,
            linestyles="dashed",
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
    feniks_data,
    feniks_fields,
    feniks_mag_thresh,
    feniks_frac_cat,
    sdss_data,
    sdss_fields,
    sdss_mag_thresh,
    sdss_frac_cat,
    data_label,
    savedir,
    sigma=0.5,
    plt_show=True,
):
    labelsize = 9
    fontsize = 10
    n_cols = len(feniks_data) + 1  # +1 for SDSS column
    fig, ax = plt.subplots(2, n_cols, figsize=(7.1, 3.4), constrained_layout=True)
    fig.get_layout_engine().set(
        h_pad=0.0, wspace=0.05, hspace=0.05, rect=(0, 0, 1, 0.925)
    )

    """ SDSS """
    sdss_data_model = N_colors_mags(
        ran_key,
        param_collection,
        sdss_data[0],
        sdss_mag_thresh,
        sdss_frac_cat,
    )
    sdss_fields_at_z = sdss_fields[0]
    z_min = sdss_data_model.z_min
    z_max = sdss_data_model.z_max
    ax[0][0].set_title(str(z_min) + " < z < " + str(z_max), fontsize=fontsize, y=0.99)
    for f in range(0, len(sdss_fields_at_z)):
        space = getattr(sdss_data_model, sdss_fields_at_z[f])
        name = type(space).__name__
        xlabel, ylabel = parse_color_labels(name)
        qm = plot_density(
            space.bin_lo,
            space.bin_hi,
            space.N_data,
            ax[f][0],
            xlabel,
            ylabel,
            dusk,
            data_label,
            fontsize=fontsize,
            N_model=space.N_model,
            sigma=sigma,
        )
        ax[f][0].minorticks_on()
        ax[f][0].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[f][0].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    """ FENIKS """
    for z in range(0, len(feniks_data)):
        col = z + 1
        z_data = feniks_data[z]
        z_data_model = N_colors_mags(
            ran_key,
            param_collection,
            z_data,
            feniks_mag_thresh,
            feniks_frac_cat,
        )
        fields_at_z = feniks_fields[z]
        z_min = z_data_model.z_min
        z_max = z_data_model.z_max
        ax[0][col].set_title(
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
                ax[f][col],
                xlabel,
                ylabel,
                dusk,
                data_label,
                fontsize=fontsize,
                N_model=space.N_model,
                sigma=sigma,
            )
            ax[f][col].minorticks_on()
            ax[f][col].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[f][col].tick_params(
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
    # place ticks at bin centers and label with sigma bands
    level_edges = np.asarray(qm.levels)
    tick_locs = 0.5 * (level_edges[:-1] + level_edges[1:])
    sigma_labels = [r"$>3\sigma$", r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(sigma_labels)
    cbar.ax.tick_params(
        labelsize=labelsize, labelleft=False, labelright=True, direction="in", length=0
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
    if plt_show:
        plt.show()
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
                    dpi=600,
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
