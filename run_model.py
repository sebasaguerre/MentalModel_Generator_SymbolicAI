import numpy as np
from playground import GridWorld
from mentalmodel import SymbolicMM


def env_labelling(self, s, action, next_s, reward, done):
    # if s == next_s:
    #     self.labels[s].add("bounded")
    if done:
        if reward > 0:
            self.labels[next_s].add("Goal")
        else:
            self.labels[next_s].add("TS")
    else:
        self.labels[s].add("NTS")
        self.labels[next_s].add("NTS")


def run_model(epochs=350, grid_size=10, n_pits=4, k=2, complex_labels=True,
              multi_edges=True, labelling_function=env_labelling, seed=8,
              model_iter=3):
    """
    Train a SymbolicMM on a GridWorld and return the model.

    Returns:
        model (SymbolicMM): trained model; call model.visualize(model.abst) to view it.
        env   (GridWorld):  the environment used.

    Note: complex_labels=True requires multi_edges=True — generate_labels computes
    action entropy from edge keys, which only exist in multi-edge mode.
    """
    env = GridWorld(grid_size, n_pits, seed=seed)

    model = SymbolicMM(
        num_act=len(env.actions),
        labelling_function=labelling_function,
        complex_labels=complex_labels,
        multi_edges=multi_edges,
        zone_radious=2
    )

    for i in range(1, epochs + 1):
        env.reset()
        episode_xp = []

        while True:
            action = np.random.choice(len(env.actions))
            xp = env.step(action)
            episode_xp.append(xp)
            if xp[-1]:  # done
                break

        model.update_structure(episode_xp)

        # if i % model_iter == 0:
        #     model.generate_model(k=k)

    # ensure model is generated after the final epoch
    model.generate_model(k=k)

    return model, env


if __name__ == "__main__":
    model, env = run_model()

    print(f"Structure states : {len(model.struct.states)}")
    print(f"Abstract states  : {len(model.abst.states)}")
    # print(f"Abstract labels  : {dict(model.abst.labels)}")

    model.visualize(model.abst, title="Symbolic Mental Model — 10x10 GridWorld")
