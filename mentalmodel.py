import numpy as np
from graphviz import Digraph

class ModelStructure:
    def __init__(self):
        self.states = set()
        self.relations = dict()
        self.labels = dict()
    
    def generate(self, data: list[tuple]):
        
        # extract structural values 
        for s, action, next_s, reward, done  in data:
            self.states.add(s)
            self.states.add(next_s)
            self.relations.setdefault(s, set()).add(next_s)

            # labels at current state

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

    def bisim_mini(self):
        pass

        


class KripkeMM:
    def __init__(self):
        self.struct = ModelStructure()

    
    def one_step_props(self, state):
        # get all the one step future proposition
        future_props = []

        for s_next in self.structure.relations[state]:
            future_props.append(self.structure.labels[s_next])
        
        return future_props
    

"""




"""




    



            

