import re

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
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


def plot_density_raw(bin_lo, bin_hi, n, ax, xlabel, ylabel, cmap=dusk, norm=None):
    w = (bin_hi - bin_lo)[0]
    ix, iy = np.round((bin_lo - bin_lo.min(0)) / w).astype(int).T
    x = bin_lo[:, 0].min() + w[0] * np.arange(ix.max() + 2)
    y = bin_lo[:, 1].min() + w[1] * np.arange(iy.max() + 2)

    Z = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    Z[iy, ix] = n

    ax.set_facecolor("0.9")
    qm = ax.pcolormesh(x, y, Z, cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel, labelpad=0.8)
    ax.set_ylabel(ylabel, labelpad=0.8)
    return qm


def plot_cc_cm_grid_raw(
    ran_key,
    param_collection,
    feniks_data,
    feniks_fields,
    feniks_mag_thresh,
    sdss_data,
    sdss_fields,
    sdss_mag_thresh,
    run_label,
    savedir,
    plt_show=True,
    percentile=(2, 98),
):
    labelsize = 9
    fontsize = 10

    sources = [(sdss_data[0], sdss_fields[0], sdss_mag_thresh)]
    sources += [
        (data, fields, feniks_mag_thresh)
        for data, fields in zip(feniks_data, feniks_fields)
    ]

    models = []
    panels = []
    n_data = []
    n_model = []
    for col, (data, fields, mag_thresh) in enumerate(sources):
        model = N_colors_mags(ran_key, param_collection, data, mag_thresh)
        models.append(model)
        for row, field in enumerate(fields):
            space = getattr(model, field)
            panels.append((row, col, space))
            n_data.append(space.N_data / model.data_vol_mpc3)
            n_model.append(space.N_model / model.lc_data.lc_tot_vol_mpc3)

    data_vals = np.concatenate([np.ravel(n) for n in n_data])
    vmin, vmax = np.percentile(data_vals[data_vals > 0], percentile)
    norm = LogNorm(vmin, vmax)

    figures = [
        ("SDSS or FENIKS", "data", n_data),
        ("diffsky", "model", n_model),
    ]
    for label, suffix, densities in figures:
        fig, ax = plt.subplots(
            2, len(models), figsize=(7.1, 3.4), constrained_layout=True
        )
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.05, hspace=0.05)

        for col, model in enumerate(models):
            ax[0][col].set_title(
                f"{model.z_min} < z < {model.z_max}", fontsize=fontsize, y=1.0, pad=3
            )

        for (row, col, space), n in zip(panels, densities):
            a = ax[row][col]
            xlabel, ylabel = parse_color_labels(type(space).__name__)
            qm = plot_density_raw(
                space.bin_lo, space.bin_hi, n, a, xlabel, ylabel, dusk, norm=norm
            )
            a.minorticks_on()
            for which, length, width in (("major", 6, 1), ("minor", 3, 0.8)):
                a.tick_params(
                    which=which,
                    direction="in",
                    top=True,
                    right=True,
                    length=length,
                    width=width,
                    labelsize=labelsize,
                )

        cbar = fig.colorbar(
            qm,
            ax=ax.ravel().tolist(),
            location="right",
            shrink=1,
            aspect=40,
            pad=0.01,
            extend="both",
        )
        cbar.set_label(r"$n\ [\mathrm{Mpc}^{-3}]$", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=labelsize)

        fig.suptitle(label, fontsize=fontsize)
        fig.savefig(f"{savedir}/{run_label}_cc_cm_grid_raw_{suffix}.png", dpi=600)
        if plt_show:
            plt.show()
        plt.close()


def plot_cc_cm_grid_raw_minerva(
    ran_key,
    param_collection,
    spaces,
    fields,
    mag_thresh,
    run_label,
    savedir,
    plt_show=True,
    percentile=(2, 98),
):
    labelsize = 9
    fontsize = 10

    models = []
    panels = []
    n_data = []
    n_model = []
    for col, (z_data, z_fields) in enumerate(zip(spaces, fields)):
        model = N_colors_mags(ran_key, param_collection, z_data, mag_thresh)
        models.append(model)
        for row, field in enumerate(z_fields):
            space = getattr(model, field)
            panels.append((row, col, space))
            n_data.append(space.N_data / model.data_vol_mpc3)
            n_model.append(space.N_model / model.lc_data.lc_tot_vol_mpc3)

    data_vals = np.concatenate([np.ravel(n) for n in n_data])
    vmin, vmax = np.percentile(data_vals[data_vals > 0], percentile)
    norm = LogNorm(vmin, vmax)

    figures = [
        ("MINERVA", "data", n_data),
        ("diffsky", "model", n_model),
    ]
    for label, suffix, densities in figures:
        fig, ax = plt.subplots(
            2, len(models), figsize=(7.1, 3.4), constrained_layout=True
        )
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.05, hspace=0.05)

        for col, model in enumerate(models):
            ax[0][col].set_title(
                f"{model.z_min} < z < {model.z_max}", fontsize=fontsize, y=1.0, pad=3
            )

        for (row, col, space), n in zip(panels, densities):
            a = ax[row][col]
            xlabel, ylabel = parse_color_labels(type(space).__name__)
            qm = plot_density_raw(
                space.bin_lo, space.bin_hi, n, a, xlabel, ylabel, dusk, norm=norm
            )
            a.minorticks_on()
            for which, length, width in (("major", 6, 1), ("minor", 3, 0.8)):
                a.tick_params(
                    which=which,
                    direction="in",
                    top=True,
                    right=True,
                    length=length,
                    width=width,
                    labelsize=labelsize,
                )

        cbar = fig.colorbar(
            qm,
            ax=ax.ravel().tolist(),
            location="right",
            shrink=1,
            aspect=40,
            pad=0.01,
            extend="both",
        )
        cbar.set_label(r"$n\ [\mathrm{Mpc}^{-3}]$", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=labelsize)

        fig.suptitle(label, fontsize=fontsize)
        fig.savefig(f"{savedir}/{run_label}_cc_cm_grid_raw_{suffix}.png", dpi=600)
        if plt_show:
            plt.show()
        plt.close()


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
    n,
    ax,
    xlabel,
    ylabel,
    cmap,
    fontsize=18,
    n_model=None,
    sigma=1.0,
    sigmas=(1, 2, 3),
    model_own_levels=True,
):
    w = (bin_hi - bin_lo)[0]
    ix, iy = np.round((bin_lo - bin_lo.min(0)) / w).astype(int).T
    shape = iy.max() + 1, ix.max() + 1
    xc = bin_lo[:, 0].min() + w[0] * (np.arange(shape[1]) + 0.5)
    yc = bin_lo[:, 1].min() + w[1] * (np.arange(shape[0]) + 0.5)

    def log_grid(dens):
        Z = np.zeros(shape)
        Z[iy, ix] = dens
        Z = gaussian_filter(Z, sigma).clip(np.finfo(float).tiny)
        logZ = np.log10(Z)
        levels = [logZ.min(), *np.log10(sigma_levels(Z, sigmas)), logZ.max()]
        return logZ, levels

    Z, levels = log_grid(n)
    qm = ax.contourf(xc, yc, Z, levels=levels, colors=cmap.colors, alpha=0.5)

    if n_model is not None:
        Zm, levels_m = log_grid(n_model)
        ax.contour(
            xc,
            yc,
            Zm,
            levels=levels_m if model_own_levels else levels,
            colors=cmap.colors,
            linewidths=1.5,
            linestyles="dashed",
        )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    return qm


def plot_cc_cm_grid(
    ran_key,
    param_collection,
    feniks_data,
    feniks_fields,
    feniks_mag_thresh,
    sdss_data,
    sdss_fields,
    sdss_mag_thresh,
    run_label,
    savedir,
    plt_show=True,
):
    labelsize = 9
    fontsize = 10

    sources = [(sdss_data[0], sdss_fields[0], sdss_mag_thresh)]
    sources += [
        (data, fields, feniks_mag_thresh)
        for data, fields in zip(feniks_data, feniks_fields)
    ]

    fig, ax = plt.subplots(2, len(sources), figsize=(7.1, 3.4), constrained_layout=True)
    fig.get_layout_engine().set(
        h_pad=0.0, wspace=0.05, hspace=0.05, rect=(0, 0, 1, 0.925)
    )

    for col, (data, fields, mag_thresh) in enumerate(sources):
        model = N_colors_mags(ran_key, param_collection, data, mag_thresh)
        ax[0][col].set_title(
            f"{model.z_min} < z < {model.z_max}", fontsize=fontsize, y=0.99
        )
        for row, field in enumerate(fields):
            space = getattr(model, field)
            xlabel, ylabel = parse_color_labels(type(space).__name__)
            qm = plot_density(
                space.bin_lo,
                space.bin_hi,
                space.N_data / model.data_vol_mpc3,
                ax[row][col],
                xlabel,
                ylabel,
                dusk,
                fontsize=fontsize,
                n_model=space.N_model / model.lc_data.lc_tot_vol_mpc3,
            )
            ax[row][col].minorticks_on()
            for which, length, width in (("major", 6, 1), ("minor", 3, 0.8)):
                ax[row][col].tick_params(
                    which=which,
                    direction="in",
                    top=True,
                    right=True,
                    length=length,
                    width=width,
                    labelsize=labelsize,
                )

    cbar = fig.colorbar(
        qm, ax=ax.ravel().tolist(), location="right", shrink=1, aspect=40, pad=0.01
    )
    edges = np.asarray(qm.levels)
    cbar.set_ticks(0.5 * (edges[:-1] + edges[1:]))
    cbar.set_ticklabels([r"$>3\sigma$", r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"])
    cbar.ax.tick_params(
        labelsize=labelsize, labelleft=False, labelright=True, direction="in", length=0
    )

    fig.legend(
        handles=[
            mpatches.Patch(color=dusk(0.7), alpha=0.5, label="SDSS or FENIKS"),
            mlines.Line2D(
                [],
                [],
                color=dusk(0.7),
                linewidth=1.5,
                linestyle="dashed",
                alpha=0.9,
                label="diffsky",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=fontsize,
        borderaxespad=0.0,
    )
    fig.savefig(f"{savedir}/{run_label}_cc_cm_grid.png", dpi=600)
    if plt_show:
        plt.show()
    plt.close()


def plot_color_contours(
    ran_key,
    param_collection,
    data,
    mag_thresh,
    data_label,
    run_label,
    savedir,
    plt_show=True,
):
    labelsize = 9
    fontsize = 10

    for z_data in data:
        model = N_colors_mags(ran_key, param_collection, z_data, mag_thresh)
        fields = [f for f in model._fields[4:] if "_" in f]

        for field in fields:
            space = getattr(model, field)
            if isinstance(space, list):
                continue

            name = type(space).__name__
            xlabel, ylabel = parse_color_labels(name)

            fig, ax = plt.subplots(figsize=(3.55, 3.5), constrained_layout=True)
            fig.get_layout_engine().set(
                h_pad=0.0, wspace=0.05, hspace=0.05, rect=(0, 0, 1, 0.92)
            )
            ax.set_title(f"{model.z_min} < z < {model.z_max}", fontsize=fontsize)

            plot_density(
                space.bin_lo,
                space.bin_hi,
                space.N_data / model.data_vol_mpc3,
                ax,
                xlabel,
                ylabel,
                dusk,
                fontsize=fontsize,
                n_model=space.N_model / model.lc_data.lc_tot_vol_mpc3,
            )

            ax.minorticks_on()
            for which, length, width in (("major", 6, 1), ("minor", 3, 0.8)):
                ax.tick_params(
                    which=which,
                    direction="in",
                    top=True,
                    right=True,
                    length=length,
                    width=width,
                    labelsize=labelsize,
                )

            fig.legend(
                handles=[
                    mpatches.Patch(color=dusk(0.7), alpha=0.5, label=data_label),
                    mlines.Line2D(
                        [],
                        [],
                        color=dusk(0.7),
                        linewidth=1.5,
                        linestyle="dashed",
                        alpha=0.9,
                        label="diffsky",
                    ),
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=2,
                frameon=False,
                fontsize=fontsize,
                borderaxespad=0.0,
            )

            fig.savefig(
                f"{savedir}/{run_label}_{name}_{model.z_min}-{model.z_max}.png",
                dpi=600,
            )
            if plt_show:
                plt.show()
            plt.close()


def plot_cc_cm_grid_minerva(
    ran_key,
    param_collection,
    spaces,
    fields,
    mag_thresh,
    run_label,
    savedir,
    plt_show=True,
):
    labelsize = 9
    fontsize = 10

    fig, ax = plt.subplots(2, len(spaces), figsize=(7.1, 3.4), constrained_layout=True)
    fig.get_layout_engine().set(
        h_pad=0.0, wspace=0.05, hspace=0.05, rect=(0, 0, 1, 0.925)
    )

    for col, (z_data, z_fields) in enumerate(zip(spaces, fields)):
        model = N_colors_mags(ran_key, param_collection, z_data, mag_thresh)
        ax[0][col].set_title(
            f"{model.z_min} < z < {model.z_max}", fontsize=fontsize, y=0.99
        )

        for row, field in enumerate(z_fields):
            space = getattr(model, field)
            a = ax[row][col]
            xlabel, ylabel = parse_color_labels(type(space).__name__)
            qm = plot_density(
                space.bin_lo,
                space.bin_hi,
                space.N_data / model.data_vol_mpc3,
                a,
                xlabel,
                ylabel,
                dusk,
                fontsize=fontsize,
                n_model=space.N_model / model.lc_data.lc_tot_vol_mpc3,
            )

            a.minorticks_on()
            for which, length, width in (("major", 6, 1), ("minor", 3, 0.8)):
                a.tick_params(
                    which=which,
                    direction="in",
                    top=True,
                    right=True,
                    length=length,
                    width=width,
                    labelsize=labelsize,
                )

    cbar = fig.colorbar(
        qm, ax=ax.ravel().tolist(), location="right", shrink=1, aspect=40, pad=0.01
    )
    edges = np.asarray(qm.levels)
    cbar.set_ticks(0.5 * (edges[:-1] + edges[1:]))
    cbar.set_ticklabels([r"$>3\sigma$", r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"])
    cbar.ax.tick_params(
        labelsize=labelsize, labelleft=False, labelright=True, direction="in", length=0
    )

    fig.legend(
        handles=[
            mpatches.Patch(color=dusk(0.7), alpha=0.5, label="MINERVA"),
            mlines.Line2D(
                [],
                [],
                color=dusk(0.7),
                linewidth=1.5,
                linestyle="dashed",
                alpha=0.9,
                label="diffsky",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=fontsize,
        borderaxespad=0.0,
    )
    fig.savefig(f"{savedir}/{run_label}_cc_cm_grid.png", dpi=600)
    if plt_show:
        plt.show()
    plt.close()


def parse_axis_label(s):
    nir_bands = {"j", "h", "k"}

    def fmt(b):
        return b.upper() if b in nir_bands else b

    bands = re.findall(r"f\d+[a-z]", s) or list(s)

    if len(bands) == 2:
        return f"${fmt(bands[0])}-{fmt(bands[1])}$"
    return f"${fmt(bands[0])}$"


def parse_color_labels(name):
    x_str, y_str = name.lower().split("_")
    return parse_axis_label(x_str), parse_axis_label(y_str)


# def parse_axis_label(s):
#     nir_bands = {"j", "h", "k"}

#     def fmt(b):
#         return b.upper() if b in nir_bands else b

#     if len(s) == 2:
#         return f"${fmt(s[0])}-{fmt(s[1])}$"
#     return f"${fmt(s)}$"


# def parse_color_labels(name):
#     x_str, y_str = name.lower().split("_")
#     return parse_axis_label(x_str), parse_axis_label(y_str)
