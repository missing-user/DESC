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
    regcoil_distances = json.load(f)
    coil_surf_dist = {}
    # The distances here were verified with the QUASR database GUI and are correct.
    for simsopt_name in regcoil_distances.keys():
        coil_surf_dist[simsopt_name] = compute_coil_surf_dist(
            f"../../../single-stage-opt/replicate_lgradb/db/{simsopt_name}"
        )
        print(
            simsopt_name,
            "has minimum filament coil distance",
            np.min(coil_surf_dist[simsopt_name]),
        )

LgradB_keyed = {}
desc_outputs = list(filter(lambda x: x.endswith("_output.h5"), os.listdir()))
for filename in desc_outputs:
    eq_fam = desc.io.load(filename)
    eq = eq_fam[-1]

    fig, ax = plt.subplots(2, 3)
    desc.plotting.plot_boundary(eq, ax=ax[0, 0])
    # desc.plotting.plot_2d(eq, "|J|", ax=ax[0, 1])
    desc.plotting.plot_2d(eq, "|B|", ax=ax[0, 1])
    desc.plotting.plot_1d(eq, "iota", ax=ax[0, 2])

    # Compute LgradB
    computed = eq.compute(["|grad(B)|", "|B|"])
    LgradBs = np.sqrt(2) * computed["|B|"] / computed["|grad(B)|"]
    LgradB = min(LgradBs)
    ax[1, 0].plot(LgradBs)
    ax[1, 0].hlines(LgradB, 0, LgradBs.shape[0], linestyles="dashed")
    ax[1, 0].set_title("LgradB metric")

    # REGCOIL distance
    simsopt_name = filename.replace("input.", "serial").replace("_output.h5", ".json")
    simsopt_path = f"../../../single-stage-opt/replicate_lgradb/db/{simsopt_name}"
    LgradB_keyed[simsopt_name] = LgradB

    ax[1, 1].hlines(
        LgradB,
        0,
        LgradBs.shape[0],
        linestyles="dashed",
    )
    ax[1, 1].hlines(
        regcoil_distances[simsopt_name],
        0,
        LgradBs.shape[0],
    )
    ax[1, 1].set_ylim(bottom=0)
    ax[1, 1].legend(["LgradB", "Distance from REGCOIL iteration"])

    # QUASR Filament coil distance
    ax[1, 2].plot(coil_surf_dist[simsopt_name], label="coil")
    ax[1, 2].hlines(
        LgradB, 0, coil_surf_dist[simsopt_name].shape[0], linestyles="dashed"
    )
    ax[1, 2].set_title("Filament coil distance")
    ax[1, 2].legend()
    ax[1, 2].set_ylim(bottom=0)

    fig.suptitle(filename)
    fig.show()

plt.show()

#########################

# Extract filenames and corresponding values for plotting
filenames = list(regcoil_distances.keys())
regcoil_vals = [regcoil_distances[f] for f in filenames]
LgradB_vals = [LgradB_keyed[f] for f in filenames]
coil_min_vals = [np.min(coil_surf_dist[f]) for f in filenames]

# Plot 1: regcoil_distances vs LgradB_keyed
plt.figure(figsize=(10, 6))
plt.scatter(regcoil_vals, LgradB_vals, color="blue", label="regcoil vs LgradB")
plt.xlabel("regcoil_distances")
plt.ylabel("LgradB_keyed")
plt.title("Scatter Plot: regcoil_distances vs LgradB_keyed")
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.grid(True)
plt.legend()

# Plot 2: regcoil_distances vs min(coil_surf_dist)
plt.figure(figsize=(10, 6))
plt.scatter(
    regcoil_vals, coil_min_vals, color="green", label="regcoil vs min(coil_surf_dist)"
)
plt.xlabel("regcoil_distances")
plt.ylabel("min(coil_surf_dist)")
plt.title("Scatter Plot: regcoil_distances vs min(coil_surf_dist)")
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.grid(True)
plt.legend()

# Plot 3: LgradB_keyed vs min(coil_surf_dist)
plt.figure(figsize=(10, 6))
plt.scatter(
    LgradB_vals, coil_min_vals, color="red", label="LgradB vs min(coil_surf_dist)"
)
plt.xlabel("LgradB_keyed")
plt.ylabel("min(coil_surf_dist)")
plt.title("Scatter Plot: LgradB_keyed vs min(coil_surf_dist)")
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.grid(True)
plt.legend()

# Plot over filenames: regcoil_distances, min(coil_surf_dist), LgradB_keyed
plt.figure(figsize=(12, 8))
plt.plot(filenames, regcoil_vals, label="regcoil_distances", marker="o")
plt.plot(filenames, coil_min_vals, label="min(coil_surf_dist)", marker="s")
plt.plot(filenames, LgradB_vals, label="LgradB_keyed", marker="^")
plt.xlabel("Filenames")
plt.ylabel("Values")
plt.title("Values across filenames")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#########################
