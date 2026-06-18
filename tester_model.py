"""
tester_model.py

A hand-built labelled Kripke model for testing the bisimulation / context
generation pipeline.

Structure matches `ModelStructure` when `multi_edges=True` (see mentalmodel.py):

    relations : {state : {action_label : set(next_states)}}
        - several labelled relation functions (a1, a2, a3, a4), one entry per
          action that is enabled at a given state.

    labels    : {state : set(props)}
        - a simple dictionary mapping each state to its set of true
          propositions.

The model is intentionally "stretched out" so that the diameter (the longest
shortest path between two states) is >= 4, i.e. there exist states that are at
least 4 steps apart. (A literal "every pair is >= 4 apart" is impossible, since
a state and its direct successor are always distance 1.)
"""

from collections import deque

from mentalmodel import ModelStructure


class TesterModel:
    """Static labelled Kripke model exposed via .states/.relations/.labels."""

    # reuse the visualizer from ModelStructure (needs only relations/labels/multi_edges)
    visualize = ModelStructure.visualize

    def __init__(self):
        self.multi_edges = True

        # -------------------------------------------------------------------
        # States
        # -------------------------------------------------------------------
        self.states = {
            "s0", "s1", "s2", "s3", "s4", "s5",
            "s6", "s7", "s8", "s9", "s10", "s11",
            "s12", "s13", "s14", "s15",
        }

        # -------------------------------------------------------------------
        # Labelled relations: {state : {action : set(next_states)}}
        # Four labelled relation functions: a1, a2, a3, a4
        # -------------------------------------------------------------------
        # Forward edges drive progress toward the goal; backward edges model
        # "undo"/regress transitions so the flow is no longer strictly sequential.
        self.relations = {
            "s0":  {"a1": {"s1"},  "a2": {"s2"}},
            "s1":  {"a1": {"s3"},  "a3": {"s4"},  "a2": {"s0"}},   # back -> s0
            "s2":  {"a2": {"s4"},  "a4": {"s5"},  "a1": {"s0"}},   # back -> s0
            "s3":  {"a1": {"s6"},  "a2": {"s7"},  "a3": {"s1"}},   # back -> s1
            "s4":  {"a3": {"s7"},  "a4": {"s8"},  "a1": {"s2"}},   # back -> s2
            "s5":  {"a2": {"s8"},  "a1": {"s9"},  "a3": {"s2"}},   # back -> s2
            "s6":  {"a4": {"s10"}, "a1": {"s11"}, "a2": {"s3"}, "a3": {"s15"}},  # -> goal (distant route), back -> s3
            "s7":  {"a2": {"s11"}, "a3": {"s12"}, "a1": {"s4"}},   # back -> s4
            "s8":  {"a1": {"s12"}, "a4": {"s13"}, "a3": {"s4"}},   # back -> s4
            "s9":  {"a3": {"s13"}, "a2": {"s14"}, "a4": {"s5"}},   # back -> s5
            "s10": {"a1": {"s14"}, "a2": {"s6"}},                  # back -> s6
            "s11": {"a4": {"s14"}, "a1": {"s7"}},                  # back -> s7
            "s12": {"a1": {"s15"}, "a2": {"s8"},  "a3": {"s13"}},  # -> goal, -> s13 (adjacent route), back -> s8
            "s13": {"a3": {"s15"}, "a1": {"s9"}},                  # -> goal (near s12), back -> s9
            "s14": {"a2": {"s10"}, "a4": {"s11"}},                 # back -> s10, s11
            "s15": {},  # Goal: terminal state, no outgoing actions
        }

        # -------------------------------------------------------------------
        # Labels: {state : set(props)}
        # -------------------------------------------------------------------
        self.labels = {
            "s0":  {"start", "NTS"},
            "s1":  {"NTS", "p"},
            "s2":  {"NTS", "q"},
            "s3":  {"NTS", "p"},
            "s4":  {"NTS", "p", "q"},
            "s5":  {"NTS", "q"},
            "s6":  {"NTS", "p"},
            "s7":  {"NTS"},
            "s8":  {"NTS", "q"},
            "s9":  {"NTS", "p"},
            "s10": {"NTS", "p", "q"},
            "s11": {"NTS"},
            "s12": {"NTS", "q"},
            "s13": {"NTS", "p"},
            "s14": {"NTS", "q"},
            "s15": {"Goal"},
        }

    # -------------------------------------------------------------------
    # Sanity check: confirm the diameter is >= 4
    # -------------------------------------------------------------------
    def _bfs_distances(self, source):
        """Shortest-path distances from `source` over the labelled multi-edge graph."""
        dist = {source: 0}
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            actions = self.relations.get(cur) or {}
            successors = set().union(*actions.values()) if actions else set()
            for nxt in successors:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist

    def diameter_report(self):
        """Return (max_shortest_path, (src, dst)) over all reachable ordered pairs."""
        best = -1
        pair = (None, None)
        for s in self.states:
            for t, d in self._bfs_distances(s).items():
                if t != s and d > best:
                    best, pair = d, (s, t)
        return best, pair


if __name__ == "__main__":
    model = TesterModel()

    diam, (src, dst) = model.diameter_report()
    print(f"states            : {len(model.states)}")
    print(f"relations (states): {len(model.relations)}")
    print(f"diameter (max SP) : {diam}  e.g. {src} -> {dst}")
    assert diam >= 4, "diameter must be at least 4"
    print("OK: diameter >= 4")

    # visualize the model (opens an SVG)
    model.visualize(title="Tester Kripke Model")
