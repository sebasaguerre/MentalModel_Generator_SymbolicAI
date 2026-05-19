import numpy as np
from graphviz import Digraph
from bisim import BiSimulatMini

class ModelStructure:
    def __init__(self, maps=None):
        self.states = set()
        self.relations = dict()
        self.rev_relations = dict()             # pre image function for bisimulation
        self.labels = dict()
        self.maps = maps

    def generate_labels(self):
        pass
    
    def generate_structure(self, data: list[tuple]):
        
        # extract structural values 
        for s, action, next_s, reward, done  in data:
            self.states.add(s)
            self.states.add(next_s)
            self.relations.setdefault(s, set()).add(next_s)
            self.rev_relations.setdefault(next_s, set()).add(s)
            
            # labels at current state
            # TODO: add labels at current node 

            # add bounds variables 
            # this is only for GridWorld
            if s == next_s:
                if self.maps:
                    bound_label = "bound" + self.maps[action][0]
                    self.labels.setdefault(s, set()).add(bound_label)

            # add action at current state 
            self.labels.setdefault(s, set()).add(f"{action}")

            # labels at future state 
            self.labels.setdefault(next_s, set()).add(f"r_{reward}")
            
            if done:
                self.labels.setdefault(next_s, set()).add(f"TS")
    
    def generate(self, data: list[tuple]):
        # generate structure and basic labels
        self.genrate_structure(data)
        # labeling function: add complex plabels 
        self.generate_labels()
            
    
    # def visualize(self):
    #     # create directed graph object 
    #     dot = Digraph()

    #     # add nodes 
    #     for s, props in self.labels.items():
    #         label = f"{s}\n" + "\n".join([str(prop) for prop in props]) 
    #         dot.node(str(s), label= label)

    #     # add edges 
    #     for s, next_states in self.relations.items():
    #         for next_s in next_states:
    #             dot.edge(str(s), str(next_s))
        
    #     # add labels 
    #     for s, props in self.labels.items():
    #         pass
    
    #     # display graph
    #     dot.render(view=True, format="png")
    
    def visualize(self):

        # initialize with better layout attributes
        dot = Digraph(node_attr={
            'shape': 'padded_box', # Custom style via HTML or rounded
            'fontname': 'Helvetica,Arial,sans-serif',
            'fontsize': '12'
        })
        
        # increase spacing between nodes and layers to reduce clutter
        dot.attr(nodesep='0.6', ranksep='0.6')
        dot.attr(rankdir='LR') # Left-to-Right layout often reads better for models

        # add nodes with clean HTML formatting
        for s, props in self.labels.items():
            prop_str = ", ".join([str(prop) for prop in props]) if props else "Ø"
            
            # Using HTML formatting for a crisp, split-box look
            html_label = f'''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
            <TR><TD BGCOLOR="#EAEAEA"><B>{s}</B></TD></TR>
            <TR><TD BGCOLOR="#FFFFFF"><FONT POINT-SIZE="10">{prop_str}</FONT></TD></TR>
            </TABLE>
            >'''
            
            dot.node(str(s), label=html_label, shape='none') # 'none' lets the HTML table be the shape

        # add edges (with slight formatting)
        for s, next_states in self.relations.items():
            for next_s in next_states:
                dot.edge(str(s), str(next_s), color="#4a4a4a", arrowhead="vee")
        
        # display graph (SVG is often sharper than PNG)
        dot.render(view=True, format="svg")



class CompressedModel:
    def __init__(self, states, edges, labels, mapping, bisim_states):
        self.states = states            # macro states
        self.edges = edges              # quotient relations 
        self.labels = labels            # quotient labels 
        self.map = mapping              # original_state -> macro_state
        self.bisim_states               # macro_state -> [bisim original_states]


class KripkeMM:
    """
    Wrapper class that bring all of the components together 
    """
    def __init__(self):
        self.struct = ModelStructure()
        self.compressor = BiSimulatMini(self.struct)
        self.contex_generator = None
        self.abst = None
    
    def one_step_props(self, state):
        # get all the one step future proposition
        future_props = []

        for s_next in self.struct.relations[state]:
            future_props.append(self.struct.labels[s_next])
        
        return future_props
    
    def update_structure(self, data):
        self.struct.generate(data)
    
    def generate_model(self):
        # generate abstract state
        macro_states, relations, labels, mapping, bisim_states = self.compressor.bisim(maps=True)

        # assign abstract model 
        self.abst = CompressedModel(
            states=macro_states,
            edges=relations,
            labels=labels,
            mapping=mapping,
            bisim_states=bisim_states
        )
    
    