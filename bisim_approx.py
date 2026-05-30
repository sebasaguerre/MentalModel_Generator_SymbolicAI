
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

    def __init__(self):
        pass

    def hamming_dist(self, s, x):

        # extend proposition length 
        if len(self.labels[s]) != len(self.labels[x]):
            labels_s_extend = self.labels[s] + ["¬" + prop for prop in self.labels[x] if prop not in self.labels[s]]

    def compute_distance(self, x, s, current_depth, max_k, discount):
        # caluclate
        pass 
