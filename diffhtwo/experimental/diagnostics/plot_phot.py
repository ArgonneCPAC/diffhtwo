import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from ..data_loaders.load_minerva import PhotFilters, get_filt_indx
from ..kernels.N_phot import N_colors_mags
from ..kernels.phot_kern import get_colors_mags, mag_kern
from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data

blue = "#1E90FF"  # DodgerBlue
orange = "#FF8C00"  # DarkOrange

mblue = "tab:blue"
morange = "tab:orange"
mred = "tab:red"

color1 = orange
color2 = "k"
color_data = blue

alpha1 = 1.0
alpha2 = 0.7
alpha_data = 0.5


lw = 1.5
fontsize = 40
labelsize = 40
legend_fontsize = 30

BASE_PATH = Path(__file__).resolve().parent.parent
IGM_DRN = BASE_PATH / "data" / "igm"
IGM_BN = "igm_attenuation_minerva.h5"

try:
    import matplotlib.lines as mlines
    from matplotlib import pyplot as plt

    plt.rc("font", family="serif", serif=["Times New Roman"])

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

MINERVA_ANCHORS = [
    "#7C93D6",
    "#8DC8D3",
    "#A3DBB4",
    "#BCDC9F",
    "#DAD9AB",
    "#EACFA2",
    "#E9C19D",
    "#D99B9B",
]


def minerva_colors(n):
    cmap = LinearSegmentedColormap.from_list("minerva", MINERVA_ANCHORS)
    return [cmap(x) for x in np.linspace(0, 1, n)]


def plot_app_mag_funcs(
    ran_key,
    param_collection,
    feniks_data,
    feniks_mag_thresh,
    sdss_data,
    sdss_mag_thresh,
    run_label,
    savedir,
    plt_show=True,
):
    band_colors = [
        "#001219",
        "#064D54",
        "#1B8686",
        "#5EB59D",
        "#95B481",
        "#BE8433",
        "#B55120",
        "#9B1D20",
    ]
    bands = [r"$u$", r"$g$", r"$r$", r"$i$", r"$z$", r"$J$", r"$H$", r"$K$"]
    fields = ["u", "g", "r", "i", "z", "J", "H", "K"]
    sdss_fields = fields[:5]
    feniks_fields = fields

    fig_width = 7.1
    fig_height = 2.6

    fontsize = 10
    labelsize = 10
    legendsize = 8
    alpha = 0.95
    lw = 0.75
    s = 2.5
    ypad = 0.2
    d_shift_dex = 0.2

    sources = [("SDSS", sdss_data[0], sdss_fields, sdss_mag_thresh)]
    for data in feniks_data:
        sources.append(("FENIKS", data, feniks_fields, feniks_mag_thresh))

    fig, ax = plt.subplots(
        1, len(sources), figsize=(fig_width, fig_height), constrained_layout=True
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))

    for col, (survey, data, band_fields, mag_thresh) in enumerate(sources):
        model = N_colors_mags(ran_key, param_collection, data, mag_thresh)
        a = ax[col]

        z_min, z_max = float(model.z_min), float(model.z_max)
        if survey == "SDSS":
            z_min_label, z_max_label = f"{z_min:.2f}", f"{z_max:.1f}"
        else:
            z_min_label, z_max_label = f"{z_min:.1f}", f"{z_max:.1f}"
        a.set_title(f"{z_min_label} < z < {z_max_label}\n{survey}", fontsize=fontsize)

        xs, ys = [], []
        for i, field in enumerate(band_fields):
            space = getattr(model, field)
            shift_dex = i * d_shift_dex
            bin_lo = space.bin_lo.ravel()
            bin_hi = space.bin_hi.ravel()
            bin_centers = (bin_lo + bin_hi) / 2
            dmag = bin_hi[0] - bin_lo[0]

            n_data = space.N_data / space.vol_mpc3_data
            n_model = space.N_model / space.vol_mpc3_model

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                y_data = np.log10(n_data) + shift_dex
                y_model = np.log10(n_model) + shift_dex

            a.scatter(bin_centers, y_data, c=band_colors[i], alpha=alpha, s=s)
            a.plot(bin_centers, y_model, c=band_colors[i], alpha=alpha, lw=lw)

            finite = np.isfinite(y_data)
            xs.append(bin_centers[finite])
            ys.append(y_data[finite])

        a.set_xticks(np.arange(10, 30, 2))
        a.minorticks_on()
        a.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        a.tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        x, y = np.concatenate(xs), np.concatenate(ys)
        a.set_xlim(x.min() - dmag, x.max() + dmag)
        a.set_ylim(y.min() - ypad, y.max() + ypad)

    ax[0].set_ylabel("log$_{10}$ (n [Mpc$^{-3}$])", fontsize=fontsize)

    ax[0].annotate(
        "",
        xy=(0.9, 0.3),
        xytext=(0.9, 0.06),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.75, color="k"),
    )
    ax[0].text(
        0.86,
        0.18,
        "shift by\n +0.2 dex",
        transform=ax[0].transAxes,
        ha="right",
        va="center",
        fontsize=legendsize - 2,
    )

    handles = [
        mlines.Line2D([], [], color=c, linewidth=6, solid_capstyle="butt", label=label)
        for c, label in zip(band_colors, bands)
    ]
    handles += [
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=3,
            label="data",
        ),
        mlines.Line2D([], [], linestyle="-", lw=1, color="gray", label="diffsky"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0, 1.0, 1, 0),
        mode="expand",
        frameon=False,
        fontsize=legendsize,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    fig.supxlabel("apparent magnitude [AB]")
    fig.savefig(savedir + "/" + run_label + "_app_mag_funcs.png", dpi=400)
    if plt_show:
        plt.show()
    plt.close()


def _add_marker_legend(ax, label, **kwargs):
    handle = Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markerfacecolor="gray",
        markeredgecolor="none",
        markersize=3,
        label=label,
    )
    ax.legend(
        handles=[handle],
        loc="upper center",
        frameon=False,
        handletextpad=0.0,
        **kwargs,
    )


def plot_app_mag_funcs_minerva(
    minerva_phot,
    run_label,
    param_collection,
    ran_key,
    savedir,
    plt_show=True,
):
    fig_width = 7.1
    fig_height = 2.5

    fontsize = 10
    labelsize = 10
    legendsize = 8
    alpha = 0.95
    lw = 0.75
    s = 2.5
    d_shift_dex = 0.3
    ypad = 0.2

    mags_labels = minerva_phot.mags_labels
    mag_thresh = minerva_phot.filter_info.mag_thresh

    sources = []
    for z_data in minerva_phot.spaces:
        fields = [
            f
            for f in z_data._fields
            if f not in {"z_min", "z_max", "lc_data"} and "_" not in f
        ]
        if fields:
            sources.append((z_data, fields))

    fig, ax = plt.subplots(
        1, len(sources), figsize=(fig_width, fig_height), constrained_layout=True
    )
    ax = np.atleast_1d(ax)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.85))

    for col, (z_data, fields) in enumerate(sources):
        model = N_colors_mags(ran_key, param_collection, z_data, mag_thresh)
        a = ax[col]

        z_min, z_max = float(model.z_min), float(model.z_max)
        a.set_title(f"{z_min:.1f} < z < {z_max:.1f}", y=1)

        colors = minerva_colors(len(fields))
        band_colors_fitted = []
        mags_labels_fitted = []

        xs, ys = [], []
        for i, field in enumerate(fields):
            space = getattr(model, field)
            shift_dex = i * d_shift_dex
            bin_lo = space.bin_lo.ravel()
            bin_hi = space.bin_hi.ravel()
            bin_centers = (bin_lo + bin_hi) / 2
            dmag = bin_hi[0] - bin_lo[0]

            n_data = space.N_data / space.vol_mpc3_data
            n_model = space.N_model / space.vol_mpc3_model

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                y_data = np.log10(n_data) + shift_dex
                y_model = np.log10(n_model) + shift_dex

            a.scatter(bin_centers, y_data, color=colors[i], alpha=alpha, s=s)
            a.plot(bin_centers, y_model, color=colors[i], alpha=alpha, lw=lw)

            finite = np.isfinite(y_data)
            xs.append(bin_centers[finite])
            ys.append(y_data[finite])

            (mag_idx,) = get_filt_indx(field, PhotFilters)
            band_colors_fitted.append(colors[i])
            mags_labels_fitted.append(mags_labels[mag_idx])

        a.set_xticks(np.arange(10, 30, 2))
        a.minorticks_on()
        a.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        a.tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        x, y = np.concatenate(xs), np.concatenate(ys)
        a.set_xlim(x.min() - dmag, x.max() + dmag)
        a.set_ylim(y.min() - ypad, y.max() + ypad)

    ax[0].set_ylabel("log$_{10}$ (n [Mpc$^{-3}$])", fontsize=fontsize)

    ax[0].annotate(
        "",
        xy=(0.9, 0.3),
        xytext=(0.9, 0.06),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.75, color="k"),
    )
    ax[0].text(
        0.86,
        0.18,
        f"shift by\n +{d_shift_dex} dex",
        transform=ax[0].transAxes,
        ha="right",
        va="center",
        fontsize=legendsize - 2,
    )

    leg = fig.legend(
        handles=[
            Line2D([], [], linestyle="none", label=l)
            for l in mags_labels_fitted  # noqa: E741
        ],
        loc="center left",
        ncol=len(mags_labels_fitted),
        frameon=False,
        fontsize=legendsize,
        handlelength=0,
        handletextpad=0,
        columnspacing=0.8,
    )
    for t, c in zip(leg.get_texts(), band_colors_fitted):
        t.set_color("w")
        t.set_bbox(dict(facecolor=c, edgecolor="none", pad=2))

    leg2 = fig.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="gray",
                markeredgecolor="none",
                markersize=3,
                label="MINERVA",
            ),
            Line2D([], [], linestyle="-", lw=1, color="gray", label="diffsky"),
        ],
        loc="center left",
        ncol=2,
        frameon=False,
        fontsize=legendsize,
    )

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    w1 = leg.get_window_extent().transformed(inv).width
    w2 = leg2.get_window_extent().transformed(inv).width
    gap = 0.03
    y = 0.93
    x0 = 0.5 - (w1 + gap + w2) / 2
    leg.set_bbox_to_anchor((x0, y), transform=fig.transFigure)
    leg2.set_bbox_to_anchor((x0 + w1 + gap, y), transform=fig.transFigure)

    fig.supxlabel("apparent magnitude [AB]")
    fig.savefig(savedir + "/" + run_label + "_minerva_app_mag_funcs.png", dpi=400)
    if plt_show:
        plt.show()
    plt.close()
