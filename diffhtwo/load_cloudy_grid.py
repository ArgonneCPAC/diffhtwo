"""
"""

import h5py


def load_cloudy(fn):
    grid_dict = dict()
    lines_dict = dict()

    with h5py.File(fn, "r") as hdf:
        grid_keys = list(hdf.keys())
        grid_keys.pop(grid_keys.index("lines"))

        for key in grid_keys:
            grid_dict[key] = hdf[key][...]

        for line_key in hdf["lines"].keys():
            if line_key != "status":
                lines_dict[line_key] = hdf["lines/{0}".format(line_key)][...]

    return grid_dict, lines_dict
