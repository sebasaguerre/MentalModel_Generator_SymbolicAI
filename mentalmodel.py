import numpy as np
from graphviz import Digraph

class BiSimulatMini:
    """ 
    Doing BiSimulation Minimization through Paige Tarjan Algorithm
    followed by quotient construction
    """
    def __init__(self, model):
        self.states = model.states
        self.edges = model.relations
        self.premap = model.rev_relations
        self.labels = model.labels
        self.worklist = list()  # list of splitter candidates

    # NOTE: this function might not be useful anymore... premap computed at generation
    def get_premap(self):
        self.premap = dict()

        # loop over the entire relation structure 
        for world, successors in self.edges.items():
            for s in successors:
                self.preimage.setdefault(s, set()).add(world)
    
    def partition0(self):
        """
        Generate the initial partition
        NOTE: NOT yet defined, but probably the most logical start is:
            - Successful terminal states
            - Failure terminal states
            - Non terminal states 
        """

        # TODO: implement inital partition

        P = None

        # self intial worklist by using all blocks 
        self.worklist += [block for block in self.P]

        return P 

    def get_preimage(self, S):
        """
        This is the function:
            E^-1(S) = {x | some y in S st xEy}
        """
        self.preimageS = set()

        # iterate over elements of S
        for x in S:
            self.preimageS.update(self.preimage[x])

    def split(self, S, P):
        """
        Refinement of P obtained by replacing each block B in P st
        B \/ E^-1(S) != {}  and B - E^-1(S) != {}, with:
        B' = B /\ E^-1(S) and B'' = B - E^-1(S).

        S is the `splitter` of P if split(S, P) != P; making Q unstable 
        with respects to S
        """
        
        # refinement partition of P 
        Q = list()

        # compute pre image of splitter 
        self.get_preimage(S)

        # check for block split conditions for each block B in P 
        for B in P:

            # if union of B and preimageS and B - preimageS are non empty => split B
            # that is the elements of B both reach S and do not reach S => B is unstable under S
            if (B & self.preimageS) and (B - self.preimageS):
                # generate split blocks
                B_in = B & self.preimageS # elements of B that reach S
                B_out = B - self.preimageS # elements of B that do not reach S

                # split blocks to refined partition 
                Q.append(B_in)
                Q.append(B_out)

                # update worklist based on `smaller half` rule
                if B in self.worklist:
                    # if B on worklist, add both new blocks
                    self.worklist.append(B_in)
                    self.worklist.append(B_out)
                else:
                    # if B not on worklist, add the smaller block 
                    # smaller_half = B_in if len(B_in) <= len(B_out) else B_out
                    smaller_half = min(B_in, B_out, key=len)
                    self.worklist.append(smaller_half)
            
            # B is stable under S
            else:
                Q.append(B)

        # partition refinement finalized 
        return Q
    
    def PTalgorithm(self):

        # get intial partition and select splitter 
        P = self.partition0()
        S = self.worklist.pop()

        # iteratively refine partition via splitting
        while len(self.worklist) != 0:
            # refine partition 
            P = self.split(S, P) # 
            # select new splitter
            S = self.worklist.pop()

        return P



class ModelStructure:
    def __init__(self):
        self.states = set()
        self.relations = dict()
        self.rev_relations = dict() # pre image function for bisimulation
        self.labels = dict()
    
    def generate(self, data: list[tuple]):
        
        # extract structural values 
        for s, action, next_s, reward, done  in data:
            self.states.add(s)
            self.states.add(next_s)
            self.relations.setdefault(s, set()).add(next_s)
            self.rev_relaitons.setdefaul(next_s, set()).add(s)
            

            # labels at current state
            # TODO: add labels at current node 

            # only add plausable actions 
            if s != next_s:
                self.labels.setdefault(s, set()).add(f"a{action}")
            
            # labels at future state 
            self.labels.setdefault(next_s, set()).add(f"r_{reward}")
            if done:
                self.labels.setdefault(next_s, set()).add(f"TS")
    
    def visualize(self):
        # create directed graph object 
        dot = Digraph()

        # add nodes 
        for s, props in self.labels.items():
            label = f"{s}\n" + "\n".join([str(prop) for prop in props])
            dot.node(str(s), label= label)

        # add edges 
        for s, next_states in self.relations.items():
            for next_s in next_states:
                dot.edge(str(s), str(next_s))
        
        # add labels 
        for s, props in self.labels.items():
            pass
    
        # display graph
        dot.render(view=True, format="png")

    # abstraction / compression methods
    def quot_construction(self):
        # intial grouping
        pass

    def bisim_mini(self):
        pass        


class KripkeMM:
    def __init__(self):
        self.struct = ModelStructure()
        self.bisimulation = BiSimulatMini()
    
    def one_step_props(self, state):
        # get all the one step future proposition
        future_props = []

        for s_next in self.structure.relations[state]:
            future_props.append(self.struct.labels[s_next])
        
        return future_props
    


            

