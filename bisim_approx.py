from collections import deque, defaultdict
import numpy as np

"""
All bellow methods are 'on-the-fly' for compressing a LTS, Kripke models and DFA/DFSM
"""

class kBisim:
    """
    k-Bisimulaiton: Bisimulation of depth k 
    """
    def __init__(self, k):
        self.k = k
        self.signature_cache = dict()

    def compute_k_signature(self, state, current_depth, max_k):
        """
        Compute the signature of all successors of the current state
        to a depth of k recursively
         
        """
        # temination condition 
        if current_depth == max_k:
            return tuple(self.labels[state])
        
        # collect childerens signatures 
        child_signatures = []
        for successor in self.relations[state]:
            # recurse on successors 
            sig = self.compute_k_signature(successor, current_depth + 1, max_k)
            child_signatures.append(sig)

        # sort signatures -> guarantees structural invariance 
        child_signatures.sort()

        return (self.labels[state], tuple(child_signatures))

    def dynamic_k_bisim(self, new_state, max_k):

        # compute signature for newly found state 
        new_sig = self.compute_k_signature(new_state, 0, max_k)

        # if isgnature exist return equivalent state
        # otherwise store signature and return state 
        if new_sig in self.signature_cache:
            return  self.signature_cache[new_sig]
        else:
            self.signature_cache[new_sig] = new_state 
            return  new_state
        


class BoundLVCompress:
    """
    Compression via Bounded Logic and Valuation distance:
    Instead of using exact proposition and relation alignemnt 
    we use the Harmming distance for proposition similariy,
    and th Huasdorff distance for successor state similarity.
    """

    def __init__(self, k, discount, epsilon):
        self.k = k 
        self.discount = discount 
        self.epsilon = epsilon
        self.canonical_states = []

    def hamming_dist(self, s, x):

        # extend proposition length 
        if len(self.labels[s]) != len(self.labels[x]):
            labels_s_extended = self.labels[s] | {"¬" + prop for prop in self.labels[x] if prop not in self.labels[s]}
            labels_x_extended = self.labels[x] | {"¬" + prop for prop in self.lables[s] if prop not in self.labels[x]}

            return len(labels_s_extended & labels_x_extended) / 2 * len(labels_s_extended)

        else:
            return len(self.labels[s] & self.labels[x]) / 2 * len(self.labels[s])
    
    def hausdorff_dist(self, x, s, current_depth):
        hauf_dist = 0
        # iterate over successors
        for next_x in self.relations[x]:
            min_match = np.inf
            #iterate over successors of successor 
            for next_s in self.relations[s]:
                dist = self.compute_distance(next_x, next_s, current_depth + 1)
                # selct minimal distance 
                if dist < min_match:
                    min_match = dist 
            # update distance 
            hauf_dist = max(hauf_dist, min_match)
        
        return hauf_dist 

    def compute_distance(self, x, s, current_depth, max_k):
        # caluclate label distance using hamming dist.
        label_dist = self.hamming_dist(x, s)

        if current_depth == max_k or (len(self.relations[s]) == 0 and len(self.relations[x]) == 0):
            return label_dist 
        
        # penalty for path asymmetri
        if not self.relations[x] or not self.relations[s]:
            return 1.0                 # maximum relative distance 

        
        # compute successor state difference via Hausdorff dist.
        # forward directed hasudorff dist. x -> s
        hauf_x_to_s = self.hausdorff_dist(x, current_depth)

        # backwards direction haudorff dist. s -> x 
        hauf_s_to_x = self.hamming_dist(s, current_depth)

        # final haufdorp distance 
        hauf_dist = max(hauf_x_to_s, hauf_s_to_x)

        return label_dist + (self.discount * hauf_dist)

    def process_state(self, state):

        # check if state matches other states 
        for canonical_id in self.canonical_states:
            dist = self.compute_distance(state, canonical_id, 0, self.k)
            if dist <= self.epsilon:
                return canonical_id
        
        # no match found the, the state is uniquely canonical
        self.canonical_states.append(state)
        return state


        # h_x_to_s = 0
        # # iterate over successors
        # for next_x in self.relations[x]:
        #     min_match = np.inf
        #     #iterate over successors of successor 
        #     for next_next_x in self.relations[next_x]:
        #         dist = self.compute_distance(next_x, next_next_x, current_depth + 1)
        #         # selct minimal distance 
        #         if dist < min_match:
        #             min_match = dist 
        #     # update distance 
        #     h_x_to_s = max(h_x_to_s, min_match)
            


class SimPreorder:
    def __init__(self, k):
        self.k = k 

    def check_simulation(self, x, s, current_depth, max_k):
        """
        Returns true if state s can simulate state x (s <= x {s preceds x }.
        We check if x is a safe substitute for s
        
        """
        
        # state s must match staes x's labels 
        if self.labels[s] != self.labels[x]:
            return False 
        
        # base case 
        if current_depth == max_k:
            return True 
        
        # every step x does must be simulated by some choice in s. Univeral Quantifier 
        for next_x in self.relations[x]:
            sim_match = False 

            # iterate over states of s. Existential Quantifier 
            for next_s in self.relations[s]:
                if self.check_simulation(next_x, next_s, current_depth + 1, max_k):
                    sim_match = True 
                    break
            
            # state s was not able to mimic a path that x contians 
            if not sim_match:
                return False 

        return True 
    

class DynamicBuilder:

    def __init__(self, compressor):
        self.k = compressor
        self.explored_structure = {}

    def dynam_expand_sys(self, initial_state):
        "BFS structure dyscovery tracking loop"
        
        # create root proxy
        root = ()
        root = self.compressor.process_state(root)

        queue = deque([(root, initial_state["pointer"])])
        self.explored_graph[root.name] = root 

        # iterate over queue 
        while queue:
            current_node, raw_sys = queue.popleft()

            # discover unrolled futur transitions
            for child in raw_sys.smt():

                # create a temporal node with sign rep
                temp_child = Node(child.id, child.labels)

                # look ahead to populate immediate child options for the signature
                for grandchild in child.smt():
                    temp_child.add_successor(Node((grandchild.id, grandchild.labels)))
                
                # filter through compressor 
                canonical_child = self.compressor.process_node(temp_child)

                # attach the edge dynamically
                current_node.add_successor(canonical_child)
            
            if canonical_child.name not in self.explored_graph:
                self.explored_graph[canonical_child.name] = canonical_child
                queue.append((canonical_child, child))


#####################################################################################################################################
# BATCH adaptations that operate on a whole ModelStructure (not on-the-fly).
# Both return the same 5-tuple shape as BiSimMini.bisim(maps=True) so they slot
# straight into the existing pipeline / quotient consumers:
#     macro_states, relations, labels, mapping, bisim_states
#####################################################################################################################################

class _ModelView:
    """
    Uniform, action-keyed read-only view over a ModelStructure, so the batch
    compressors below don't have to branch on multi_edges everywhere.

    succ(s) always returns {action: set(next_states)}:
      - multi edges  -> the real action map
      - simple edges -> a single pseudo-action None: {action None: successors}
    """
    def __init__(self, model):
        self.states = set(model.states)
        self.labels = model.labels
        self.edges = model.relations
        self.multi = model.multi_edges

    def lbl(self, s):
        return frozenset(self.labels.get(s, ()))

    def succ(self, s):
        if self.multi:
            amap = self.edges.get(s, {})
            return {a: set(t) for a, t in amap.items() if t}
        nxt = self.edges.get(s, set())
        return {None: set(nxt)} if nxt else {}


def _quotient_from_mapping(view, mapping, maps):
    """
    Build a quotient model from a state->representative mapping.
    Shared by both batch compressors.
    """
    # assign a macro id per representative and gather members
    rep_name, bisim_states = {}, {}
    for s, rep in mapping.items():
        if rep not in rep_name:
            name = f"x{len(rep_name) + 1}"
            rep_name[rep] = name
            bisim_states[name] = []
        bisim_states[rep_name[rep]].append(s)

    macro_mapping = {s: rep_name[rep] for s, rep in mapping.items()}
    macro_states = set(rep_name.values())
    quotient_labels = {name: set(view.lbl(rep)) for rep, name in rep_name.items()}

    # relations: representative's edges, lifted through the macro mapping
    if view.multi:
        quotient_relations = {m: defaultdict(set) for m in macro_states}
        for rep, name in rep_name.items():
            for a, targets in view.succ(rep).items():
                for t in targets:
                    quotient_relations[name][a].add(macro_mapping[t])
    else:
        quotient_relations = {m: set() for m in macro_states}
        for rep, name in rep_name.items():
            for t in view.succ(rep).get(None, set()):
                quotient_relations[name].add(macro_mapping[t])

    if maps:
        return macro_states, quotient_relations, quotient_labels, macro_mapping, bisim_states
    return macro_states, quotient_relations, quotient_labels


class ApproxBisim:
    """
    Approximate bisimulation by bounded behavioural distance, applied to a whole
    ModelStructure (batch version of BoundLVCompress).

    distance(x, s) in [0, 1] is a convex blend:
        d = (1 - discount) * label_dist  +  discount * successor_dist
      - label_dist     : Jaccard distance between the two states' label sets.
      - successor_dist : symmetric Hausdorff between successor sets, per action,
                         recursed to depth k. The worst action dominates, and an
                         action present on one side only counts as max distance.

    Compression is greedy epsilon-canonicalization: scan states, attach each to
    the first representative within epsilon, else open a new representative.

    NOTE: behavioural distance is NOT transitive, so the clustering is
    order-dependent (states are sorted for determinism). That is inherent to
    metric/approximate bisimulation, not a bug. Larger epsilon -> more merging.
    """
    def __init__(self, model, k=3, discount=0.5, epsilon=0.15):
        self.view = _ModelView(model)
        self.k = k
        self.discount = discount
        self.epsilon = epsilon
        self._memo = {}

    def _label_dist(self, a, b):
        la, lb = self.view.lbl(a), self.view.lbl(b)
        if not la and not lb:
            return 0.0
        return len(la ^ lb) / len(la | lb)

    def _dist(self, x, s, depth):
        if x == s:
            return 0.0
        # symmetric: canonicalize the key by repr ordering
        lo, hi = (x, s) if repr(x) <= repr(s) else (s, x)
        key = (lo, hi, depth)
        if key in self._memo:
            return self._memo[key]
        # guard against cycles in the recursion
        self._memo[key] = self._label_dist(x, s)

        ld = self._label_dist(x, s)
        sx, ss = self.view.succ(x), self.view.succ(s)

        # depth budget exhausted or both terminal -> labels only
        if depth >= self.k or (not sx and not ss):
            self._memo[key] = ld
            return ld

        worst_action = 0.0
        for act in set(sx) | set(ss):
            ax, bs = sx.get(act, set()), ss.get(act, set())
            if not ax or not bs:
                sd = 1.0                      # action available on one side only
            else:
                sd = max(self._directed(ax, bs, depth),
                         self._directed(bs, ax, depth))
            worst_action = max(worst_action, sd)

        d = (1 - self.discount) * ld + self.discount * worst_action
        self._memo[key] = d
        return d

    def _directed(self, A, B, depth):
        # directed Hausdorff: every a in A must find a close partner in B
        worst = 0.0
        for a in A:
            worst = max(worst, min(self._dist(a, b, depth + 1) for b in B))
        return worst

    def compress(self, maps=False):
        canon, mapping = [], {}
        for s in sorted(self.view.states, key=repr):
            match = next((rep for rep in canon
                          if self._dist(s, rep, 0) <= self.epsilon), None)
            if match is None:
                canon.append(s)
                mapping[s] = s
            else:
                mapping[s] = match
        return _quotient_from_mapping(self.view, mapping, maps)


class SimQuotient:
    """
    Simulation-equivalence quotient, applied to a whole ModelStructure (batch
    greatest-fixpoint version of SimPreorder).

    s simulates x   (write x <= s)   iff   labels match AND for every action a
    and every a-successor x' of x there is an a-successor s' of s with x' <= s'.
    Simulation EQUIVALENCE: x ~ s iff x <= s and s <= x. We quotient by ~.

    Caveat: on DETERMINISTIC systems simulation equivalence coincides with
    bisimulation, so it yields no extra compression there. Its real planning
    value is the PREORDER itself (available via .preorder()): if x <= s then s's
    options dominate x's, so a planner can prune the dominated state/action.
    """
    def __init__(self, model):
        self.view = _ModelView(model)

    def preorder(self):
        """Return the simulation preorder as a set of (x, s) pairs with x <= s."""
        states = list(self.view.states)
        succ = {s: self.view.succ(s) for s in states}
        lbl = {s: self.view.lbl(s) for s in states}

        # start optimistic: every label-matching pair is assumed related
        R = {(x, s) for x in states for s in states if lbl[x] == lbl[s]}

        changed = True
        while changed:
            changed = False
            for (x, s) in list(R):
                # x <= s requires: each of x's a-moves is matched by some a-move of s
                holds = True
                for a, x_targets in succ[x].items():
                    s_targets = succ[s].get(a, set())
                    for xt in x_targets:
                        if not any((xt, st) in R for st in s_targets):
                            holds = False
                            break
                    if not holds:
                        break
                if not holds:
                    R.discard((x, s))
                    changed = True
        return R

    def compress(self, maps=False):
        R = self.preorder()
        reps, mapping = {}, {}
        for s in sorted(self.view.states, key=repr):
            match = next((rep for rep in reps
                          if (s, rep) in R and (rep, s) in R), None)
            if match is None:
                reps[s] = True
                mapping[s] = s
            else:
                mapping[s] = match
        return _quotient_from_mapping(self.view, mapping, maps)



