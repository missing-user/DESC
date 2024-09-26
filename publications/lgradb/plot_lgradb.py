import desc.io
import desc.plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import simsopt

import json
from scipy.spatial.distance import cdist


def coil_surf_distance(curves, lcfs) -> np.ndarray:
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    distances = [np.min(cdist(pointcloud1, c.gamma()), axis=0) for c in curves]

    return np.array(distances).T


def compute_coil_surf_dist(simsopt_filename):
    surfaces, coils = simsopt.load(simsopt_filename)
    lcfs = surfaces[-1].to_RZFourier()
    curves = [c.curve for c in coils]
    return coil_surf_distance(curves, lcfs)


with open("all_results.json") as f:
    j = json.load(f)
    coil_surf_dist = {}
    for simsopt_name in j.keys():
        coil_surf_dist[simsopt_name] = compute_coil_surf_dist(
            f"../../../single-stage-opt/replicate_lgradb/db/{simsopt_name}"
        )


desc_outputs = list(filter(lambda x: x.endswith("_output.h5"), os.listdir()))
for filename in desc_outputs[:1]:
    eq_fam = desc.io.load(filename)
    eq = eq_fam[-1]

    fig, ax = plt.subplots(2, 3)
    desc.plotting.plot_boundary(eq, ax=ax[0, 0])
    # desc.plotting.plot_2d(eq, "|J|", ax=ax[0, 1])
    desc.plotting.plot_2d(eq, "|B|", ax=ax[0, 1])
    desc.plotting.plot_1d(eq, "iota", ax=ax[0, 2])

    # Compute LgradB
    computed = eq.compute(["|grad(B)|", "|B|"])
    LgradB = np.sqrt(2) * computed["|B|"] / computed["|grad(B)|"]
    ax[1, 0].plot(LgradB)
    ax[1, 0].hlines(min(LgradB), 0, LgradB.shape[0], linestyles="dashed")
    ax[1, 0].set_title("LgradB metric")

    # REGCOIL distance
    simsopt_name = filename.replace("input.", "serial").replace("_output.h5", ".json")
    simsopt_path = f"../../../single-stage-opt/replicate_lgradb/db/{simsopt_name}"

    ax[1, 1].hlines(
        min(LgradB),
        0,
        LgradB.shape[0],
        linestyles="dashed",
    )
    ax[1, 1].hlines(
        j[simsopt_name],
        0,
        LgradB.shape[0],
    )
    ax[1, 1].legend(["LgradB", "Distance from REGCOIL iteration"])

    # QUASR Filament coil distance
    ax[1, 2].plot(coil_surf_dist[simsopt_name], label="coil")
    ax[1, 2].hlines(
        min(LgradB), 0, coil_surf_dist[simsopt_name].shape[0], linestyles="dashed"
    )
    ax[1, 2].set_title("Filament coil distance")
    ax[1, 2].legend()

    fig.suptitle(filename)
    fig.show()

plt.show()
