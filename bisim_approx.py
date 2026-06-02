from collections import deque
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



