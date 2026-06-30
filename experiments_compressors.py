"""
Compare all four compressors on every environment, mirroring experiments.ipynb:
number of states vs epochs, each compressor against the raw ModelStructure.

Data collection follows playground.py: the structure grows every epoch as the
RANDOM agent explores, and a compressed size is 0 until its compressor first
runs (epoch >= MODEL_ITER), then the actual size. All compressors compress the
SAME ModelStructure at the SAME epoch (model.compressor holds a live reference
to model.struct; ApproxBisim/SimQuotient read model.struct directly).

Compressors:
  - Bisimulation             (BiSimMini.bisim)
  - k-Bisimulation           (BiSimMini.k_bisim)         k = 2, 3
  - Approximate bisimulation (bisim_approx.ApproxBisim)  epsilon = 0.15, 0.10
  - Simulation preordering   (bisim_approx.SimQuotient)

Environments: ToroidalGrid, RoomsGrid, DistractorGrid, GridWorld (all with the
same number of death pits).

Run:  python experiments_compressors.py   ->  writes compare_<env>.png
"""
import matplotlib
matplotlib.use("Agg")                       # never block; just write PNGs
import matplotlib.pyplot as plt
import numpy as np

from mentalmodel import SymbolicMM
from bisim_approx import ApproxBisim, SimQuotient
from envirs import ToroidalGrid, RoomsGrid
from playground import GridWorld

MULTI_EDGES = True            # labeled edges (planning-relevant representation)
COMPLEX_LABELS = False        # basic labels -> measure pure structural compression
EPOCHS = 50
MODEL_ITER = 3                # compressors start running at this epoch (0 before)
MAX_STEPS = 200               # safety cap (episodes normally end at goal / pit)
NDEATHPITS = 3                # same for every environment
SEED = 0


def env_labelling(self, s, action, next_s, reward, done):
    if s == next_s:
        self.labels[s].add("bounded")
    if done:
        self.labels[next_s].add("Goal" if reward > 0 else "TS")
    else:
        self.labels[s].add("NTS")
        self.labels[next_s].add("NTS")


# (key, label, color, linestyle, marker) -- markers so lines that coincide
# (e.g. k-Bisim k=3 often equals Bisim) stay individually visible
SERIES = [
    ("struct",   "Original states", "black",     "-",  None),
    ("bisim",    "Bisim",           "blue",      "--", "o"),
    ("kbisim2",  "k-Bisim k=2",     "red",       "-",  "s"),
    ("kbisim3",  "k-Bisim k=3",     "darkorange", "-", "^"),
    ("approx15", "Approx ε=0.15", "green",     "--", "v"),
    ("approx10", "Approx ε=0.10", "limegreen", "--", "D"),
    ("sim",      "Sim-preorder",    "purple",    ":",  "x"),
]
_COMPRESSORS = [k for k, *_ in SERIES if k != "struct"]


def run_env(make_env):
    rng = np.random.default_rng(SEED)
    env = make_env()
    model = SymbolicMM(n_action=len(env.actions), labelling_function=env_labelling,
                     complex_labels=COMPLEX_LABELS, multi_edges=MULTI_EDGES)

    xaxis = []
    data = {key: [] for key, *_ in SERIES}

    for i in range(1, EPOCHS + 1):
        # ---- random agent rollout (terminates at goal or death pit) ----
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

        xaxis.append(i)
        data["struct"].append(len(model.struct.states))

        if i < MODEL_ITER:
            # compressors have not run yet -> size 0 (mirrors playground.py)
            for key in _COMPRESSORS:
                data[key].append(0)
        else:
            # SAME structure for every compressor, at the SAME epoch
            struct, comp = model.struct, model.compressor
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
    plt.figure(figsize=(9, 5.5))
    for key, label, color, ls, marker in SERIES:
        plt.plot(xaxis, data[key], label=label, color=color, linestyle=ls,
                 linewidth=1.6, marker=marker, markersize=5, markevery=3, alpha=0.9)
    edge = "multi" if MULTI_EDGES else "simple"
    lbl = "complex" if COMPLEX_LABELS else "basic"
    plt.title(f"Compression vs epochs — {name}  ({edge} edges, {lbl} labels)")
    plt.xlabel("Epochs")
    plt.ylabel("Number of states")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fname = f"compare_{name}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname


# all envs ~100 states, matching GridWorld(10, .) in experiments.ipynb
ENVS = {
    "ToroidalGrid": lambda: ToroidalGrid(10, 5, ndeathpits=NDEATHPITS),  # 100 cells
    "RoomsGrid":    lambda: RoomsGrid(5, ndeathpits=NDEATHPITS),         # ~104 free cells
    "GridWorld":    lambda: GridWorld(10, NDEATHPITS),                   # 100 cells
}

if __name__ == "__main__":
    for name, make in ENVS.items():
        xaxis, data = run_env(make)
        fname = plot_env(name, xaxis, data)
        final = {k: data[k][-1] for k, *_ in SERIES}
        print(f"{name:<15} -> {fname} | final: {final}")
