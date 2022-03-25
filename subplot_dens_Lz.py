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

args = parser.parse_args()
list_of_files = sorted(args.checkpoints)
latest_file = np.array(list_of_files)[len(list_of_files) - 1]
chkpt = load_checkpoint(latest_file, require_solver="cbdiso_2d")
time_series_data = np.array(chkpt["timeseries"])
Lz1_lim = -1.0 * time_series_data[:, 4]
Lz2_lim = -1.0 * time_series_data[:, 8]
mdot1_lim = -1.0 * time_series_data[:, 1]
mdot2_lim = -1.0 * time_series_data[:, 5]
time_lim = time_series_data[:, 0]
prim = chkpt["primitive"]
f = fields[args.field](prim).T
vmax = np.amax(f)
vmin = np.amin(f)
vlog_max = np.amax(np.log10(f))
vlog_min = np.amin(np.log10(f))

for filename in args.checkpoints:
    fig, ax1 = plt.subplots(2, 2)
    fig.set_size_inches(40.0, 40.0)
    chkpt = load_checkpoint(filename, require_solver="cbdiso_2d")
    time_series_data = np.array(chkpt["timeseries"])
    Lz1 = -1.0 * time_series_data[:, 4]
    Lz2 = -1.0 * time_series_data[:, 8]
    mdot1 = -1.0 * time_series_data[:, 1]
    mdot2 = -1.0 * time_series_data[:, 5]
    time = time_series_data[:, 0]
    prim = chkpt["primitive"]
    f = fields[args.field](prim).T
    cm1 = ax1[0][0].imshow(f, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    cm2 = ax1[0][1].imshow(
        np.log10(f),
        origin="lower",
        vmin=vlog_min,
        vmax=vlog_max,
        cmap="magma",
    )
    cbar1 = fig.colorbar(cm1, ax=ax1[0][0])
    cbar1.ax.tick_params(labelsize=30)
    cbar2 = fig.colorbar(cm2, ax=ax1[0][1])
    cbar2.ax.tick_params(labelsize=30)
    ax1[0][0].tick_params(axis="y", which="major", labelsize=30)
    ax1[0][0].tick_params(axis="x", which="major", labelsize=30)
    ax1[0][1].tick_params(axis="y", which="major", labelsize=30)
    ax1[0][1].tick_params(axis="x", which="major", labelsize=30)
    ax1[1][0].plot(time, Lz1, label="Lz1_dot")
    ax1[1][0].plot(time, Lz2, label="Lz2_dot")
    ax1[1][0].set_xlabel("Time")
    ax1[1][0].set_ylabel("Lz_dot")
    ax1[1][0].set_xlim(0.0, max(time_lim))
    ax1[1][0].set_ylim(min(min(Lz1_lim), min(Lz2_lim)), max(max(Lz1_lim), max(Lz2_lim)))
    ax1[1][1].plot(time, mdot1, label="M1_dot")
    ax1[1][1].plot(time, mdot2, label="M2_dot")
    ax1[1][1].set_xlabel("Time")
    ax1[1][1].set_ylabel("M_dot")
    ax1[1][1].set_xlim(0.0, max(time_lim))
    ax1[1][1].set_ylim(
        min(min(mdot1_lim), min(mdot2_lim)), max(max(mdot1_lim), max(mdot2_lim))
    )
    fig.savefig(filename + ".png")
    fig.clf()
    plt.close()
