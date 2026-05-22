import numpy as np
import types
from graphviz import Digraph
from bisim import BiSimulatMini
from collections import defaultdict

class ModelStructure:
    def __init__(self, labelling_function=None, maps=None):
        self.states = set()
        self.relations = defaultdict(set)
        self.rev_relations = defaultdict(set)             # pre image function for bisimulation
        self.labels = defaultdict(set)
        self.maps = maps
        # if labelling function is give then use that instead of default
        if labelling_function:
            self.add_labels = types.MethodType(labelling_function, self)   # same as labellinf_function.__get__(self, ModelStructure)

    def generate_labels(self):
        """
        Generate higher order labels lables that change dynamically.
        Labels like: 
            - Level of entropy
            - Proximity to goal 
            - Proximity to terminal states 
        """
        pass

    def add_labels(self, s, next_s, action, reward, done):

        # add bound 
        if s == next_s:
                self.labels[s].add("bound")
                    
        # add action at current state 
        self.labels[s].add(f"a{action}")

        # labels at future state 
        self.labels[next_s].add(f"r_{reward}")
        
        if done:
            # create gaol label
            if reward > 0:
                self.labels[next_s].add("Goal")
            else:
                self.labels[next_s].add("TS")
            
        pass
    
    def generate_structure(self, data: list[tuple]):
        
        # extract structural values 
        for s, action, next_s, reward, done  in data:
            self.states.add(s)
            self.states.add(next_s)
            self.relations[s].add(next_s)
            self.rev_relations[next_s].add(s)
            
            # add labels 
            self.add_labels(s, next_s, action, reward, done) 

            # if reached terminal state add empty relations 
            if done:
                self.relations[next_s] = set()
    
    def generate(self, data: list[tuple]):
        # generate structure and basic labels
        self.generate_structure(data)
        # labeling function: add complex plabels 
        self.generate_labels()
            

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

        # add edges (with slight formatting)2
        for s, next_states in self.relations.items():
            for next_s in next_states:
                dot.edge(str(s), str(next_s), color="#4a4a4a", arrowhead="vee")
        
        # display graph (SVG is often sharper than PNG)
        dot.render(view=True, format="svg")


class CompressedModel:
    def __init__(self, states, relations, labels, mapping, bisim_states):
        self.states = states                           # macro states
        self.relations = relations                         # quotient relations 
        self.labels = labels                           # quotient labels 
        self.map = mapping                             # original_state -> macro_state
        self.bisim_states = bisim_states               # macro_state -> [bisim original_states]


class KripkeMM:
    """
    Wrapper class that bring all of the components together 
    """
    def __init__(self, **kwargs):
        self.struct = ModelStructure(**kwargs)                      # underlying structure 
        self.compressor = BiSimulatMini(self.struct)        # compresion engine 
        self.contex_generator = None                        # formula generator on basis of model
        self.abst = None                                    # learned compressed model 
    
    def one_step_props(self, state):
        # get all the one step future proposition
        future_props = []

        for s_next in self.struct.relations[state]:
            future_props.append(self.struct.labels[s_next])
        
        return future_props
    
    def visualize(self, model):

        # initialize with better layout attributes
        dot = Digraph(node_attr={
            'shape': 'padded_box', # Custom style via HTML or rounded
            'fontname': 'Helvetica,Arial,sans-serif',
            'fontsize': '12'
        })
        
        # increase spacing between nodes and layers to reduce clutter
        dot.attr(nodesep='0.6', ranksep='0.6', rankdir="LR")
        # dot.attr(rankdir='LR') # Left-to-Right layout often reads better for models

        # add nodes with clean HTML formatting
        for s, props in model.labels.items():
            node_id = str(s)
            prop_str = ", ".join(map(str, props)) if props else "Ø"
            
            # Stripped whitespace from HTML string to reduce string processing overhead
            html_label = (
                f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
                f'<TR><TD BGCOLOR="#EAEAEA"><B>{node_id}</B></TD></TR>'
                f'<TR><TD BGCOLOR="#FFFFFF"><FONT POINT-SIZE="10">{prop_str}</FONT></TD></TR>'
                f'</TABLE>>'
            )
            dot.node(node_id, label=html_label, shape='none')

        # optimize edge creation
        # using dot.edges() with a generator expression is much faster than nested loops
        edges = (
            (str(s), str(next_s)) 
            for s, next_states in model.relations.items() 
            for next_s in next_states
        )
        dot.edges(edges)

        # apply edge styling globally to the graph instead of per-edge to save memory
        dot.edge_attr.update(color="#4a4a4a", arrowhead="vee")

        # display graph (SVG is often sharper than PNG)
        dot.render(view=True, format="svg")
    
    def update_structure(self, data):
        self.struct.generate(data)
    
    def generate_model(self):
        # generate abstract state
        macro_states, relations, labels, mapping, bisim_states = self.compressor.bisim(maps=True)

        # assign abstract model 
        self.abst = CompressedModel(
            states=macro_states,
            relations=relations,
            labels=labels,
            mapping=mapping,
            bisim_states=bisim_states
        )
    
    