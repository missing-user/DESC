import desc.io
import desc.plotting
import os
import numpy as np
import matplotlib.pyplot as plt

def (simsopt_filename)

desc_outputs = list(filter(lambda x: x.endswith("_output.h5"), os.listdir()))
for filename in desc_outputs[:1]:
    eq_fam = desc.io.load(filename)
    eq = eq_fam[-1]

    fig, ax = plt.subplots(2, 2)
    desc.plotting.plot_boundary(eq, ax=ax[0, 0])
    # desc.plotting.plot_2d(eq, "|J|", ax=ax[0, 1])
    desc.plotting.plot_2d(eq, "|B|", ax=ax[0, 1])
    desc.plotting.plot_1d(eq, "iota", ax=ax[1, 1])

    # Compute LgradB
    computed = eq.compute(["|grad(B)|", "|B|"])
    LgradB = np.sqrt(2) * computed["|B|"] / computed["|grad(B)|"]
    ax[1, 0].plot(LgradB)
    ax[1, 0].hlines(min(LgradB), 0, LgradB.shape[0])

    fig.suptitle(filename)
    fig.show()

plt.show()
