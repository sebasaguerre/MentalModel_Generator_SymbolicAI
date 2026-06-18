"""
Compare all four compressors on every environment, mirroring experiments.ipynb:
number of states vs epochs, each compressor against the raw ModelStructure.

Compressors:
  - Bisimulation            (BiSimMini.bisim)
  - k-Bisimulation          (BiSimMini.k_bisim) for k = 2, 3
  - Approximate bisimulation(bisim_approx.ApproxBisim) for epsilon = 0.15, 0.05
  - Simulation preordering  (bisim_approx.SimQuotient)

Environments: ToroidalGrid, RoomsGrid, DistractorGrid, GridWorld.

Run:  python experiments_compressors.py   ->  writes compare_<env>.png
Flip MULTI_EDGES / COMPLEX_LABELS at the top to explore other regimes.
"""
import matplotlib
matplotlib.use("Agg")                       # never block; just write PNGs
import matplotlib.pyplot as plt
import numpy as np

from mentalmodel import KripkeMM
from bisim_approx import ApproxBisim, SimQuotient
from envirs import ToroidalGrid, RoomsGrid, DistractorGrid
from playground import GridWorld

MULTI_EDGES = True            # labeled edges (planning-relevant representation)
COMPLEX_LABELS = False        # basic labels -> measure pure structural compression
EPOCHS = 120
MODEL_ITER = 4                # recompute compressed sizes every MODEL_ITER epochs
MAX_STEPS = 300              # cap random-walk episode length
SEED = 0


def env_labelling(self, s, action, next_s, reward, done):
    if s == next_s:
        self.labels[s].add("bounded")
    if done:
        self.labels[next_s].add("Goal" if reward > 0 else "TS")
    else:
        self.labels[s].add("NTS")
        self.labels[next_s].add("NTS")


# (key, label, color, linestyle)
SERIES = [
    ("struct",   "Original states", "black",     "-"),
    ("bisim",    "Bisim",           "blue",      "--"),
    ("kbisim2",  "k-Bisim k=2",     "red",       "-."),
    ("kbisim3",  "k-Bisim k=3",     "darkred",   ":"),
    ("approx15", "Approx ε=0.15", "green",   "--"),
    ("approx10", "Approx ε=0.10", "limegreen", "-."),
    ("sim",      "Sim-preorder",    "purple",    ":"),
]


def run_env(make_env):
    rng = np.random.default_rng(SEED)
    env = make_env()
    model = KripkeMM(n_action=len(env.actions), labelling_function=env_labelling,
                     complex_labels=COMPLEX_LABELS, multi_edges=MULTI_EDGES)

    xaxis = []
    data = {key: [] for key, *_ in SERIES}

    for i in range(1, EPOCHS + 1):
        env.reset()
        ep, steps = [], 0
        while steps < MAX_STEPS:
            a = int(rng.integers(len(env.actions)))
            xp = env.step(a)
            ep.append(xp)
            steps += 1
            if xp[-1]:
                break
        model.update_structure(ep)

        if i % MODEL_ITER == 0:
            struct, comp = model.struct, model.compressor
            xaxis.append(i)
            data["struct"].append(len(struct.states))
            data["bisim"].append(len(comp.bisim(maps=True)[0]))
            data["kbisim2"].append(len(comp.k_bisim(2, maps=True)[0]))
            data["kbisim3"].append(len(comp.k_bisim(3, maps=True)[0]))
            data["approx15"].append(len(ApproxBisim(struct, k=3, discount=0.5,
                                                    epsilon=0.15).compress(maps=True)[0]))
            data["approx10"].append(len(ApproxBisim(struct, k=3, discount=0.5,
                                                    epsilon=0.10).compress(maps=True)[0]))
            data["sim"].append(len(SimQuotient(struct).compress(maps=True)[0]))

    return xaxis, data


def plot_env(name, xaxis, data):
    plt.figure(figsize=(8, 5))
    for key, label, color, ls in SERIES:
        plt.plot(xaxis, data[key], label=label, color=color, linestyle=ls, linewidth=1.8)
    edge = "multi" if MULTI_EDGES else "simple"
    lbl = "complex" if COMPLEX_LABELS else "basic"
    plt.title(f"Compression vs epochs — {name}  ({edge} edges, {lbl} labels)")
    plt.xlabel("Epochs")
    plt.ylabel("Number of states")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fname = f"compare_{name}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname


ENVS = {
    "ToroidalGrid":  lambda: ToroidalGrid(8, 4),
    "RoomsGrid":     lambda: RoomsGrid(3),
    "DistractorGrid": lambda: DistractorGrid(4),
    "GridWorld":     lambda: GridWorld(10, 3),
}

if __name__ == "__main__":
    for name, make in ENVS.items():
        xaxis, data = run_env(make)
        fname = plot_env(name, xaxis, data)
        final = {k: data[k][-1] for k, *_ in SERIES}
        print(f"{name:<15} -> {fname} | final sizes: {final}")