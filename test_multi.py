from mentalmodel import ModelStructure
from bisim import BiSimMini


def flat_label(self, s, action, next_s, reward, done):
    # constant atomic label so every state shares the initial block;
    # refinement is then driven purely by (labeled) transition structure
    self.labels[s].add("p")
    self.labels[next_s].add("p")


def run(name, data, n_action):
    m = ModelStructure(n_action=n_action, labelling_function=flat_label,
                       multi_edges=True, complex_labels=False)
    m.generate(data)

    print(f"\n===== {name} =====")
    print("relations   :", {s: dict(v) for s, v in m.relations.items()})

    engine = BiSimMini(m, multi_edges=True)
    macro, rel, labels, mapping, bisim_states = engine.k_bisim(k=10, maps=True)

    print("bisim groups :", bisim_states)
    print("quotient rel :", {s: dict(v) for s, v in rel.items()})
    return mapping


# --- Scenario A: separation. 0 and 1 reach 2 via DIFFERENT actions -> must split
mapping = run("A: separation", [
    (0, 0, 2, 0, False),   # 0 -a0-> 2
    (1, 1, 2, 0, False),   # 1 -a1-> 2   (same target, DIFFERENT action)
    (2, 0, 3, 0, True),    # 2 -a0-> 3,  3 terminal
], n_action=2)
assert mapping[0] != mapping[1], "FAIL: 0 and 1 merged despite different actions"
print("OK: 0 and 1 kept separate (action-awareness works)")


# --- Scenario B: merging. 0 and 1 reach 2 via the SAME action -> must merge
mapping = run("B: merging", [
    (0, 0, 2, 0, False),   # 0 -a0-> 2
    (1, 0, 2, 0, False),   # 1 -a0-> 2   (same target, SAME action)
    (2, 0, 3, 0, True),    # 2 -a0-> 3,  3 terminal
], n_action=1)
assert mapping[0] == mapping[1], "FAIL: bisimilar states 0 and 1 were NOT merged"
print("OK: 0 and 1 merged (does not over-split)")

print("\nALL GOOD")
