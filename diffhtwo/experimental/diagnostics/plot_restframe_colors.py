import matplotlib.pyplot as plt
import numpy as np
from dsps.data_loaders.defaults import TransmissionCurve

from ..rest_phot.rest_uvj import uvj_q_ms_burst
from ..utils import get_tcurve

FILTER_INFO = "kz_FILTER.RES.latest.info"
TCURVES_FILE = "kz_FILTER.RES.latest"


def uvj_diag(x):
    return 0.8 * x + 0.7


def plot_uvj(ran_key, param_collection, ssp_data, drn, savedir, num_halos=100):
    filter_info = drn + "/" + FILTER_INFO
    tcurves_file = drn + "/" + TCURVES_FILE

    # U: Johnsons Morgan; V: Johnsons Morgan; J: 2MASS
    filter_numbers = [150, 152, 161]
    uvj_tcurves = []
    for filter_number in filter_numbers:
        filter_wave_aa, filter_trans = get_tcurve(
            filter_number, filter_info, tcurves_file
        )
        uvj_tcurves.append(TransmissionCurve(filter_wave_aa, filter_trans))

    diag_x = np.arange(0.75, 3, 0.01)
    diag_y = uvj_diag(diag_x)

    fig, ax = plt.subplots(1, 4, figsize=(8.5, 3))
    fig.subplots_adjust(wspace=0.0, bottom=0.2, right=0.99, left=0.075, top=0.88)

    s = 0.5
    a = 0.5

    z_min = [0.2, 1, 2, 3]
    z_max = [1, 2, 3, 4]
    for z in range(0, len(z_min)):
        UVJ = uvj_q_ms_burst(
            ran_key,
            param_collection,
            z_min[z],
            z_max[z],
            ssp_data,
            uvj_tcurves,
            num_halos=num_halos,
        )

        ax[z].scatter(
            UVJ.vj[UVJ.mc_is_q],
            UVJ.uv[UVJ.mc_is_q],
            c="darkred",
            s=s,
            alpha=a,
            rasterized=True,
            label="q",
        )
        ax[z].scatter(
            UVJ.vj[UVJ.mc_is_ms],
            UVJ.uv[UVJ.mc_is_ms],
            c="deepskyblue",
            s=s,
            alpha=a,
            rasterized=True,
            label="ms",
        )
        ax[z].scatter(
            UVJ.vj[UVJ.mc_is_bursty],
            UVJ.uv[UVJ.mc_is_bursty],
            c="darkorange",
            s=s,
            alpha=a,
            rasterized=True,
            label="bursty",
        )

        ax[z].plot(diag_x, diag_y, c="k", lw=1)
        ax[z].axhline(1.3, xmax=0.377, c="k", lw=1)
        ax[z].set_xlim(-0.2, 2.3)
        ax[z].set_ylim(0, 2.8)
        ax[z].set_xlabel(r"$(V-J)_0$ [AB]")
        ax[z].set_title(str(z_min[z]) + " < z < " + str(z_max[z]))
        if z != 0:
            ax[z].set_yticks([])
    ax[0].set_ylabel(r"$(U-V)_0$ [AB]")
    plt.legend()
    plt.savefig(savedir + "_uvj.pdf", dpi=200)
    plt.show()
