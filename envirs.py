"""
Test environments with *built-in redundancy*, designed to actually exhibit
bisimulation symmetry (unlike a single-goal navigation grid, where every cell
has a unique geodesic signature and almost nothing merges).

All envs are duck-type compatible with playground.GridWorld:
    .actions, .action_map, .start_pos, .reset(), .step(action_idx), .render()
step(action_idx) -> (state, action_idx, next_state, reward, done)

So you can swap them straight into experiments:  env = ToroidalGrid(8, period=4)

Every env has a GOAL (reward +10, terminal) and DEATH PITS (reward -10,
terminal) just like GridWorld, so episodes terminate quickly and the structure
grows gradually as the random agent explores. ndeathpits matches GridWorld's.

Each env also exposes a DIFFERENT kind of redundancy:
  - ToroidalGrid : translational symmetry from wrap-around + periodic goals.
                   NOTE: arbitrary death pits weaken this symmetry; set
                   ndeathpits=0 (or place them periodically) to see the clean
                   labeled-bisimulation compression.
  - RoomsGrid    : bottleneck/room structure (natural macro-regions; good for
                   approximate bisimulation and hierarchical abstraction).
  - DistractorGrid: an irrelevant per-episode "color" dimension that never
                   affects dynamics or reward => exact bisimulation collapses
                   it perfectly (~2x), independent of the death pits.
"""
import numpy as np

# shared action set: Up, Down, Right, Left  (matches GridWorld)
_ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_ACTION_MAP = {0: "Up", 1: "Down", 2: "Right", 3: "Left"}


def _sample_pits(rng, all_cells, n, forbidden):
    """Pick n distinct death-pit cells avoiding `forbidden` (start/goals/walls)."""
    pool = [c for c in all_cells if c not in forbidden]
    rng.shuffle(pool)
    return set(pool[:n])


class ToroidalGrid:
    """
    Wrap-around grid. No walls => every move is valid (no -1 bump states), so
    boundary cells are not special. Goals sit on a periodic sublattice (every
    `period` cells on each axis); translation by `period` maps goals->goals and
    preserves action labels, so cells in the same translation orbit are
    bisimilar -- UNLESS death pits break that symmetry (set ndeathpits=0 for the
    clean demo).
    """
    def __init__(self, size=8, period=4, ndeathpits=3, seed=8):
        assert size % period == 0, "period must divide size for clean symmetry"
        self.rng = np.random.default_rng(seed)
        self.grid_size = size
        self.period = period
        self.actions = _ACTIONS
        self.action_map = _ACTION_MAP
        self.goals = {(r, c) for r in range(0, size, period)
                      for c in range(0, size, period)}
        off = period // 2 if period > 1 else 0
        self.start_pos = (off, off)
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        self.death_pits = _sample_pits(self.rng, all_cells, ndeathpits,
                                       self.goals | {self.start_pos})
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos

    def step(self, action_idx):
        dx, dy = self.actions[action_idx]
        x, y = self.agent_pos
        nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size
        self.agent_pos = (nx, ny)

        if self.agent_pos in self.death_pits:
            return ((x, y), action_idx, self.agent_pos, -10, True)
        if self.agent_pos in self.goals:
            return ((x, y), action_idx, self.agent_pos, 10, True)
        return ((x, y), action_idx, self.agent_pos, 0, False)

    def render(self):
        for x in reversed(range(self.grid_size)):
            row = ""
            for y in range(self.grid_size):
                if (x, y) == self.agent_pos:
                    row += "A "
                elif (x, y) in self.goals:
                    row += "$ "
                elif (x, y) in self.death_pits:
                    row += "# "
                else:
                    row += ". "
            print(row)
        print()


class RoomsGrid:
    """
    Four rooms separated by a cross of walls, connected by single-cell doorways
    (bottlenecks). Walls are blocked cells: stepping into one keeps you in place
    with reward -1. One goal in the far room plus death pits scattered in the
    free cells.

    Rooms are natural macro-regions: deep-interior cells funnel through the same
    doorway -> similar (not identical) futures -> good testbed for APPROXIMATE
    bisimulation and a clean hierarchy (plan over rooms, then within a room).
    """
    def __init__(self, room=3, ndeathpits=3, seed=8):
        self.rng = np.random.default_rng(seed)
        self.room = room
        self.grid_size = 2 * room + 1
        self.mid = room
        self.actions = _ACTIONS
        self.action_map = _ACTION_MAP

        # wall = the central cross, minus 4 doorways (one per wall segment)
        self.walls = set()
        m, n = self.mid, self.grid_size
        for i in range(n):
            self.walls.add((m, i))
            self.walls.add((i, m))
        d = room // 2
        for door in [(m, d), (m, m + 1 + d), (d, m), (m + 1 + d, m)]:
            self.walls.discard(door)

        self.start_pos = (0, 0)
        self.goal_pos = (n - 1, n - 1)
        self.walls.discard(self.start_pos)
        self.walls.discard(self.goal_pos)

        all_cells = [(r, c) for r in range(n) for c in range(n)]
        self.death_pits = _sample_pits(
            self.rng, all_cells, ndeathpits,
            self.walls | {self.start_pos, self.goal_pos})
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos

    def step(self, action_idx):
        dx, dy = self.actions[action_idx]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        # blocked by edge or wall -> stay put, small penalty
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size) \
                or (nx, ny) in self.walls:
            return (self.agent_pos, action_idx, self.agent_pos, -1, False)

        self.agent_pos = (nx, ny)
        if self.agent_pos in self.death_pits:
            return ((x, y), action_idx, self.agent_pos, -10, True)
        if self.agent_pos == self.goal_pos:
            return ((x, y), action_idx, self.agent_pos, 10, True)
        return ((x, y), action_idx, self.agent_pos, 0, False)

    def render(self):
        for x in reversed(range(self.grid_size)):
            row = ""
            for y in range(self.grid_size):
                if (x, y) == self.agent_pos:
                    row += "A "
                elif (x, y) == self.goal_pos:
                    row += "$ "
                elif (x, y) in self.death_pits:
                    row += "X "
                elif (x, y) in self.walls:
                    row += "# "
                else:
                    row += ". "
            print(row)
        print()


class DistractorGrid:
    """
    A small bounded navigation grid (goal + death pits) PLUS an irrelevant
    per-episode "color" bit that is part of the state but never affects
    transitions or reward. Color is chosen at reset and held constant, so the
    state graph is two isomorphic disjoint copies (color 0 / color 1). Exact
    bisimulation merges the copies perfectly -> ~2x, *independent* of the pits.

    State = (x, y, color). 'color' is intentionally NOT exposed to the labeller.
    """
    def __init__(self, size=4, ndeathpits=3, seed=8):
        self.rng = np.random.default_rng(seed)
        self.grid_size = size
        self.actions = _ACTIONS
        self.action_map = _ACTION_MAP
        self.goal_cell = (size - 1, size - 1)
        self.start_cell = (0, 0)
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        self.death_pits = _sample_pits(self.rng, all_cells, ndeathpits,
                                       {self.goal_cell, self.start_cell})
        self.color = 0
        self.reset()

    def reset(self):
        # color is fixed for the whole episode and is pure noise
        self.color = int(self.rng.integers(2))
        self.agent_pos = (self.start_cell[0], self.start_cell[1], self.color)

    def step(self, action_idx):
        dx, dy = self.actions[action_idx]
        x, y, c = self.agent_pos
        nx, ny = x + dx, y + dy

        # bump on edge -> stay (color preserved, so s == next_s holds)
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return (self.agent_pos, action_idx, self.agent_pos, -1, False)

        nxt = (nx, ny, c)
        self.agent_pos = nxt
        if (nx, ny) in self.death_pits:
            return ((x, y, c), action_idx, nxt, -10, True)
        if (nx, ny) == self.goal_cell:
            return ((x, y, c), action_idx, nxt, 10, True)
        return ((x, y, c), action_idx, nxt, 0, False)

    def render(self):
        x, y, c = self.agent_pos
        print(f"[color={c}]")
        for r in reversed(range(self.grid_size)):
            row = ""
            for col in range(self.grid_size):
                if (r, col) == (x, y):
                    row += "A "
                elif (r, col) == self.goal_cell:
                    row += "$ "
                elif (r, col) in self.death_pits:
                    row += "# "
                else:
                    row += ". "
            print(row)
        print()
