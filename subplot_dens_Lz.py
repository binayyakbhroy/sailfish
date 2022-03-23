import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import argparse


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as file:
        chkpt = pickle.load(file)

        if require_solver is not None and chkpt["solver"] != require_solver:
            raise ValueError(
                f"checkpoint is from a run with solver {chkpt['solver']}, "
                f"expected {require_solver}"
            )
        return chkpt


fields = {
    "sigma": lambda p: p[:, :, 0],
    "vx": lambda p: p[:, :, 1],
    "vy": lambda p: p[:, :, 2],
}

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", type=str, nargs="+")
parser.add_argument(
    "--field",
    "-f",
    type=str,
    default="sigma",
    choices=fields.keys(),
    help="which field to plot",
)
parser.add_argument(
    "--log",
    "-l",
    default=False,
    action="store_true",
    help="use log scaling",
)
parser.add_argument(
    "--vmin",
    default=None,
    type=float,
    help="minimum value for colormap",
)
parser.add_argument(
    "--vmax",
    default=None,
    type=float,
    help="maximum value for colormap",
)

args = parser.parse_args()

for filename in args.checkpoints:
    fig, ax1 = plt.subplots(2, 1)
    fig.set_size_inches(20.0, 20.0)
    chkpt = load_checkpoint(filename, require_solver="cbdiso_2d")
    time_series_data = np.array(chkpt["timeseries"])
    Lz1 = -1.0 * time_series_data[:, 4]
    Lz2 = -1.0 * time_series_data[:, 8]
    time = time_series_data[:, 0]
    mesh = chkpt["mesh"]
    prim = chkpt["primitive"]
    print(chkpt["cfl_number"])
    f = fields[args.field](prim).T
    cm = ax1[0].imshow(f, origin="lower", vmin=0.0, vmax=10.0, cmap="magma")
    cbar = fig.colorbar(cm, ax=ax1[0])
    cbar.ax.tick_params(labelsize=20)
    ax1[0].tick_params(axis="y", which="major", labelsize=20)
    ax1[0].tick_params(axis="x", which="major", labelsize=20)
    ax1[1].plot(time, Lz1, label="Lz1_dot")
    ax1[1].plot(time, Lz2, label="Lz2_dot")
    ax1[1].set_xlabel("Time")
    ax1[1].set_ylabel("Lz_dot")
    fig.savefig(filename + ".png")
    fig.clf()
    # plt.close("all")
