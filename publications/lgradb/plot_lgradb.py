import desc.io
import desc.plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import simsopt

import json
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from scipy.io import netcdf


def coil_surf_distance(curves, lcfs) -> np.ndarray:
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    distances = [np.min(cdist(pointcloud1, c.gamma()), axis=0) for c in curves]

    return np.array(distances).T


def compute_coil_surf_dist(simsopt_filename):
    surfaces, coils = simsopt.load(simsopt_filename)
    lcfs = surfaces[-1].to_RZFourier()
    curves = [c.curve for c in coils]
    return coil_surf_distance(curves, lcfs)


def fit_exponential_rate(sequence):
    x = np.linspace(0, 1, len(sequence))
    fit = np.polyfit(x, np.log(sequence), 1)
    return fit


def compare_bdistrib(simsopt_name):
    filepath = f"../../../single-stage-opt/replicate_lgradb/tmp/dist05_mpol=12/bdistrib_out{simsopt_name.replace('.json','.nc').replace('serial','')}"
    # Some fits of the exponential decay, take a look at compareSingularValuesPlot.py and bdistrib_util.py

    bdistrib_variables = [
        "svd_s_transferMatrix",
        "svd_s_inductance_plasma_middle",
        "Bnormal_from_1_over_R_field_inductance",
        "Bnormal_from_1_over_R_field_transfer",
    ]
    fit_types = ["log_linear", "value"]  # "windowed_upper_bound",
    fits = {}
    with netcdf.netcdf_file(filepath, "r", mmap=False) as f:
        for key in bdistrib_variables:
            vararray = f.variables[key][()]
            for fit_type in fit_types:
                if fit_type == "value":
                    fit_val = np.max(vararray)
                elif fit_type == "log_linear":
                    fit_val = fit_exponential_rate(vararray)
                else:
                    windowed_array = vararray
                    fit_val = fit_exponential_rate(windowed_array)
                fits[key + fit_type] = fit_val
    return fits


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

    # Compute LgradB
    computed = eq.compute(["grad(B)", "|grad(B)|", "|B|", "L_grad(B)"])
    LgradBs = computed["L_grad(B)"]
    LgradBnucs = (
        np.sqrt(2)
        * computed["|B|"]
        / np.linalg.norm(computed["grad(B)"], ord="nuc", axis=(1, 2))
    )
    LgradB2s = (
        np.sqrt(2)
        * computed["|B|"]
        / np.linalg.norm(computed["grad(B)"], ord=2, axis=(1, 2))
    )
    LgradB = np.min(LgradBs)

    # REGCOIL distance
    simsopt_name = filename.replace("input.", "serial").replace("_output.h5", ".json")
    simsopt_path = f"../../../single-stage-opt/replicate_lgradb/db/{simsopt_name}"
    LgradB_keyed[simsopt_name] = np.array(
        [LgradB, np.min(LgradB2s), np.min(LgradBnucs)]
    )

    # Only plot all individual equilibria for small datasets
    if len(desc_outputs) <= 10:
        fig, ax = plt.subplots(2, 3)
        desc.plotting.plot_boundary(eq, ax=ax[0, 0])
        desc.plotting.plot_2d(eq, "|B|", ax=ax[0, 1])
        desc.plotting.plot_1d(eq, "iota", ax=ax[0, 2])

        desc.plotting.plot_1d(eq, "L_grad(B)", ax=ax[1, 0])
        ax[1, 0].hlines(LgradB, 0, 1, linestyles="dashed")

        ax[1, 1].hlines(
            LgradB,
            0,
            1,
            linestyles="dashed",
        )
        ax[1, 1].hlines(
            regcoil_distances[simsopt_name],
            0,
            1,
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
Bdistrib_vals = (
    [list(compare_bdistrib(f).values()) for f in filenames],
    "efficient fields seqence",
)
coil_min_vals = (
    [np.min(coil_surf_dist[f]) for f in filenames],
    "QUASR coil distance",
)


def vs_plot(x_data, y_data):
    x_vals, x_label = x_data
    y_vals, y_label = y_data
    title = x_label + " vs " + y_label

    print(np.shape(x_vals), np.shape(y_vals))
    if len(np.shape(y_vals)) >= 2:
        y_vals = np.array(y_vals).T
    elif len(np.shape(y_vals)) == 1:
        y_vals = np.reshape(y_vals, (1,) + np.shape(y_vals))
    print(np.shape(x_vals), np.shape(y_vals))

    # Linear fit
    # TODO this Fails because some values are inf!!
    for i, y in enumerate(y_vals):
        print(np.shape(x_vals), np.shape(y))
        plt.scatter(x_vals, y, label=title)
        reg = linregress(x_vals, y)
        plt.axline(
            xy1=(0, reg.intercept),
            slope=reg.slope,
            color="k",
            label=f"Linear fit {i}: $R^2$ = {reg.rvalue:.2f}",
        )

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
vs_plot(coil_min_vals, LgradB_vals)
plt.figure()
vs_plot(coil_min_vals, Bdistrib_vals)

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
