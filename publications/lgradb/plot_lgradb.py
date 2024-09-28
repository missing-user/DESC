import desc.io
import desc.plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import simsopt

import json
from scipy.spatial.distance import cdist
from scipy.stats import linregress


def coil_surf_distance(curves, lcfs) -> np.ndarray:
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    distances = [np.min(cdist(pointcloud1, c.gamma()), axis=0) for c in curves]

    return np.array(distances).T


def compute_coil_surf_dist(simsopt_filename):
    surfaces, coils = simsopt.load(simsopt_filename)
    lcfs = surfaces[-1].to_RZFourier()
    curves = [c.curve for c in coils]
    return coil_surf_distance(curves, lcfs)


def compare_bdistrib(simsopt_name):
    base_filepath = f"../../../single-stage-opt/replicate_lgradb/tmp/dist05_mpol=12/bdistrib_out{simsopt_name.replace('.json','.nc')}"
    # Some fits of the exponential decay, take a look at compareSingularValuesPlot.py and bdistrib_util.py
    return decay


with open("all_results.json") as f:
    regcoil_distances = json.load(f)
    for key in regcoil_distances.keys():
        regcoil_distances[key] /= 5

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
    ax[1, 0].set_title("$L^*_{\\nabla B}$")

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
    ax[1, 1].legend(["$L^*_{\\nabla B}$", "Coil winding surf. dist."])

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


#########################

# Extract filenames and corresponding values for plotting
filenames = list(LgradB_keyed.keys())
regcoil_vals = ([regcoil_distances[f] for f in filenames], "REGCOIL distance")
LgradB_vals = ([LgradB_keyed[f] for f in filenames], "$L^*_{\\nabla B}$")
coil_min_vals = (
    [np.min(coil_surf_dist[f]) for f in filenames],
    "QUASR coil distance",
)


def vs_plot(x_data, y_data):
    x_vals, x_label = x_data
    y_vals, y_label = y_data
    title = x_label + " vs " + y_label

    # Linear fit
    # TODO this Fails because some values are inf!!
    reg = linregress(x_vals, y_vals)
    plt.axline(
        xy1=(0, reg.intercept),
        slope=reg.slope,
        color="k",
        label=f"Linear fit: $R^2$ = {reg.rvalue**2:.2f}",
    )

    plt.scatter(x_vals, y_vals, label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.grid(True)
    plt.legend()


plt.figure()
vs_plot(regcoil_vals, LgradB_vals)
plt.figure()
vs_plot(regcoil_vals, coil_min_vals)
plt.figure()
vs_plot(LgradB_vals, coil_min_vals)

# Plot over filenames: regcoil_distances, filament coil distance, L*_{\nabla B}
plt.figure(figsize=(12, 8))
plt.plot(filenames, regcoil_vals[0], marker="o", label=regcoil_vals[1])
plt.plot(filenames, coil_min_vals[0], marker="s", label=coil_min_vals[1])
plt.plot(filenames, LgradB_vals[0], marker="^", label=LgradB_vals[1])
plt.xlabel("Filenames")
plt.ylabel("Values")
plt.title("Comparison of different metrics")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#########################
