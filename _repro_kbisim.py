"""Validate redundancy envs (exact bisim) + the two batch approximations."""
from mentalmodel import KMMcompare
from bisim_approx import ApproxBisim, SimQuotient
from envirs import ToroidalGrid, RoomsGrid, DistractorGrid
import numpy as np


def env_labelling(self, s, action, next_s, reward, done):
    if s == next_s:
        self.labels[s].add("bounded")
    if done:
        self.labels[next_s].add("Goal" if reward > 0 else "TS")
    else:
        self.labels[s].add("NTS")
        self.labels[next_s].add("NTS")


def build(env, multi_edges, n_eps=120, seed=0):
    rng = np.random.default_rng(seed)
    model = KMMcompare(compare_models=True, compare_struct=False,
                       n_action=len(env.actions), labelling_function=env_labelling,
                       complex_labels=False, multi_edges=multi_edges)
    for _ in range(n_eps):
        env.reset()
        ep, steps = [], 0
        while steps < 300:
            a = int(rng.integers(len(env.actions)))
            xp = env.step(a)
            ep.append(xp)
            steps += 1
            if xp[-1]:
                break
        model.update_structure(ep)
    return model


for name, make in [("ToroidalGrid(8,4)", lambda: ToroidalGrid(8, 4)),
                   ("RoomsGrid(3)", lambda: RoomsGrid(3)),
                   ("DistractorGrid(4)", lambda: DistractorGrid(4))]:
    print(f"\n=== {name} ===")
    for multi in (False, True):
        model = build(make(), multi)
        comp = model.compressor
        n = len(model.struct.states)
        nb, *_ = comp.bisim(maps=True)
        na, *_ = ApproxBisim(model.struct, k=3, discount=0.5, epsilon=0.25).compress(maps=True)
        ns, *_ = SimQuotient(model.struct).compress(maps=True)
        tag = "multi " if multi else "simple"
        print(f"  {tag} edges: struct={n:>3}  exact-bisim={len(nb):>3}  "
              f"approx(eps.25)={len(na):>3}  sim-quotient={len(ns):>3}")