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

from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
states = {
    "s0", "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10", "s11",
    "s12", "s13", "s14", "s15",
}

# ---------------------------------------------------------------------------
# Labelled relations: {state : {action : set(next_states)}}
# Four labelled relation functions: a1, a2, a3, a4
# ---------------------------------------------------------------------------
relations = {
    "s0":  {"a1": {"s1"},  "a2": {"s2"}},
    "s1":  {"a1": {"s3"},  "a3": {"s4"}},
    "s2":  {"a2": {"s4"},  "a4": {"s5"}},
    "s3":  {"a1": {"s6"},  "a2": {"s7"}},
    "s4":  {"a3": {"s7"},  "a4": {"s8"}},
    "s5":  {"a2": {"s8"},  "a1": {"s9"}},
    "s6":  {"a4": {"s10"}, "a1": {"s11"}},
    "s7":  {"a2": {"s11"}, "a3": {"s12"}},
    "s8":  {"a1": {"s12"}, "a4": {"s13"}},
    "s9":  {"a3": {"s13"}, "a2": {"s14"}},
    "s10": {"a1": {"s14"}},
    "s11": {"a2": {"s15"}, "a4": {"s14"}},
    "s12": {"a1": {"s15"}},
    "s13": {"a3": {"s15"}},
    "s14": {"a4": {"s15"}},
    "s15": {"a1": {"s0"}},  # close the loop -> strongly connected
}

# ---------------------------------------------------------------------------
# Labels: {state : set(props)}
# ---------------------------------------------------------------------------
labels = {
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


# ---------------------------------------------------------------------------
# Sanity check: confirm the diameter is >= 4
# ---------------------------------------------------------------------------
def _bfs_distances(source):
    """Shortest-path distances from `source` over the labelled multi-edge graph."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        successors = set().union(*relations.get(cur, {}).values()) if relations.get(cur) else set()
        for nxt in successors:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def diameter_report():
    """Return (max_shortest_path, (src, dst)) over all reachable ordered pairs."""
    best = -1
    pair = (None, None)
    for s in states:
        for t, d in _bfs_distances(s).items():
            if t != s and d > best:
                best, pair = d, (s, t)
    return best, pair


if __name__ == "__main__":
    diam, (src, dst) = diameter_report()
    print(f"states            : {len(states)}")
    print(f"relations (states): {len(relations)}")
    print(f"diameter (max SP) : {diam}  e.g. {src} -> {dst}")
    assert diam >= 4, "diameter must be at least 4"
    print("OK: diameter >= 4")
