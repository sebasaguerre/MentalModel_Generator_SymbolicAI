import os
import re
import types
import tempfile
import numpy as np
import networkx as nx
from graphviz import Digraph
import matplotlib.pyplot as plt
from bisim import BiSimulatMini
from collections import defaultdict, deque


class ModelStructure:
    def __init__(self, n_action, labelling_function=None, maps=None, zones=None):
        self.n_action = n_action
        self.states = set()
        self.relations = defaultdict(set)
        self.rev_relations = defaultdict(set)             # pre image function for bisimulation
        self.labels = defaultdict(set)
        self.maps = maps

        # cset zones depending on instantiation
        if zones is not None:
            self.zones = zones 
        else:
            self.zones = {"GoalZone":"Goal", "DeathZone":"TN"}

        # if labelling function is give then use that instead of default
        if labelling_function:
            self.add_labels = types.MethodType(labelling_function, self)   # same as labellinf_function.__get__(self, ModelStructure)
        
      


    def within_radious_dfs(self, state, label, max_steps):
        # baseline success: label found at current state 
        if label in self.model.labels[state]:
            return True
        
        # safety check, if max_steps is non-positive => no more search 
        if max_steps <= 0:
            return False
        
        # queue with tuples (current_state, current_depth)
        queue = deque([(state, 0)])

        # track visited states to prevent inf loops
        visited = {state}

        while queue:
            current_state, current_depth = queue.popleft()

            # check if we can do further searching 
            if current_depth >= max_steps:
                continue

            # look at all successors of the current state 
            for next_s in self.model.relations[current_state]:
                if next_s not in visited:
                    # check if label is true at state 
                    if label in self.model.labels[next_s]:
                        return True 
                    
                    # label not found => Update visited and queue 
                    visited.add(next_s)
                    queue.append((next_s, current_depth + 1))
        
        # label not found within radious 
        return False
    

    def generate_labels(self, zone_radious=3):
        """
        Generate higher order labels lables that change dynamically.
        Labels like: 
            - Level of entropy
            - Proximity to goal 
            - Proximity to terminal states 
        """
        # simple label patterns 
        act_pattern = re.compile(r"^\d+a$")
        entroy_pattern =   re
            
        for s in self.states:

            # TODO: this needs to be adapted to multi-edges 
            s_labels = self.labels[s]

            bounds = any(label == "bound" for label in s_labels)
            terminal = any(label in ["TN", "Goal"] for label in s_labels)
            goal = any(label == "Goal" for label in s_labels )
            
            act_exp= sum(1 for elem in self.labels[s] if act_pattern.match(elem)) / (self.n_action - bounds) 

            # entropy level of state 
            entropy = f"E{'low' if act_exp <= 0.33 else 'mid' if act_exp <= 0.66 else 'high'}"

            # add dynamic entropy label and zones-labels 
            if not terminal:
                
                # check if state has an entopy level 
                current_entropy = [label for label in self.labels[s] if label.statrtswith("E")]
                
                # add or update entropy
                if not current_entropy:
                    self.labels[s].add(entropy)
                elif current_entropy[0] != entropy:
                    self.labels[s].remove(current_entropy[0])
                    self.labels[s].add(entropy)

                # add zone label if applicable 
                for zone, label in self.zoness.items():
                    if zone not in self.labels[s]:
                        if self.within_radious_dfs(s, label, zone_radious):
                            self.labels[s].append(zone)

                
    def add_labels(self, s, action, next_s, reward, done):

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
            
    
    def generate_structure(self, data: list[tuple]):
        
        # extract structural values 
        for s, action, next_s, reward, done  in data:
            self.states.add(s)
            self.states.add(next_s)
            self.relations[s].add(next_s)
            self.rev_relations[next_s].add(s)
            
            # add labels 
            self.add_labels(s, action, next_s, reward, done) 

            # if reached terminal state add empty relations 
            if done:
                self.relations[next_s] = set()
    
    def generate(self, data: list[tuple]):
        # generate structure and basic labels
        self.generate_structure(data)

        # labeling function: add complex labels 
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
    
    def visualize(self, model, title=None):

        # initialize with better layout attributes
        dot = Digraph(node_attr={
            'shape': 'padded_box', # Custom style via HTML or rounded
            'fontname': 'Helvetica,Arial,sans-serif',
            'fontsize': '12'
        })

        # add title if given 
        if title:
            dot.attr(label=title, labelloc="t", fontsize="16", fontname="Helvetica-Bold")
        
        # increase spacing between nodes and layers to reduce clutter
        dot.attr(nodesep='0.6', ranksep='0.6', rankdir="LR")
        dot.attr(size='10,6!', ratio='compress')                 # give a more zoomed out graph

        # add nodes with clean HTML formatting
        for s, props in model.labels.items():
            node_id = str(s)
            prop_str = ", ".join(map(str, props)) if props else "Ø"
            
            # Stripped whitespace from HTML string to reduce string processing overhead
            html_label = (
                f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" STYLE="ROUNDED">'
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


        # create temporary file path that diappears later
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gv") as temp_file:
            temp_base = temp_file.name

        # display graph and clean up immediately
        dot.render(temp_base, view=True, format="svg")
        try:
            os.remove(temp_base)
        except OSError:
            pass
    
    def update_structure(self, data):
        self.struct.generate(data)
    
    def generate_model(self, k=None):
        # generate abstract state with standard bisim of k-bisim
        if not k:
            macro_states, relations, labels, mapping, bisim_states = self.compressor.bisim(maps=True)
        else:
            macro_states, relations, labels, mapping, bisim_states = self.compressor.k_bisim(k, maps=True)

        # assign abstract model 
        self.abst = CompressedModel(
            states=macro_states,
            relations=relations,
            labels=labels,
            mapping=mapping,
            bisim_states=bisim_states
        )

# class wrapper to compare Bisim vs k-Bisim compresison 
class KMMcompare(KripkeMM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init abst_k
        self.abst_k = None

    def generate_model(self, k=None):

        # check if k is used and generate new compressde model while saving other model
        if k is not None:

            # store self.abst befroe parent overrite 
            model_backup = getattr(self, 'abst', None)

            # overwrite self.abst
            super().generate_model(k=k)

            # migrate k_model 
            self.abst_k = self.abst 

            # restore original model 
            self.abst = model_backup 

        else:
            super().generate_model(k=k)


    
    