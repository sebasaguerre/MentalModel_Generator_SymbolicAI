import numpy as np
import networkx as nx
from graphviz import Digraph
import matplotlib.pyplot as plt
from bisim import BiSimMini
from itertools import chain
import os, types, tempfile, textwrap
from collections import defaultdict, deque


class ModelStructure:
    def __init__(self, n_action, labelling_function=None, multi_edges=False):
        """
        Arguments:
            - n_action := number of actions possible
            - labelling_function := function that maps xp:(state, action, next_state, reward, done) into state labels 
            - zones := dict{"zone1":"label1", "zone2":"label2", ...}
        """
        # structure attributes
        self.states = set()
        self.labels = defaultdict(set)
        # set up depending on edges 
        self.init_setup_for_edges(multi_edges)

        # building attributes 
        self.n_action = n_action
        self.multi_edges = multi_edges

        # if labelling function is give then use that instead of default
        if labelling_function:
            self.add_labels = types.MethodType(labelling_function, self)   # same as labellinf_function.__get__(self, ModelStructure)

    def init_setup_for_edges(self, multi_edges):
        "Initialize class with prper structure according to edge type"

        if multi_edges:
            self.relations = defaultdict(lambda : defaultdict(set))       # {state : {action1: [next1,... ], act2: [next1, ...], ...} 
            self.rev_relations = defaultdict(lambda : defaultdict(set))
            self.generate_structure = self._generate_structure_multi_edge 
        else:
            self.relations = defaultdict(set)
            self.rev_relations = defaultdict(set)    # pre image function for bisimulation
            self.generate_structure = self._generate_structure_simple_edge
            
    def within_radious_dfs(self, state, label, max_steps):

        # baseline success: label found at current state 
        if label in self.labels[state]:
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
            
            # adaptation for dealing with edge type 
            if self.multi_edges:
                successors = set().union(*self.relations[current_state].values())
            else:
                successors = self.relations[current_state]

            # look at all successors of the current state 
            for next_s in successors:
                if next_s not in visited:
                    # check if label is true at state 
                    if label in self.labels[next_s]:
                        return True 
                    
                    # label not found => Update visited and queue 
                    visited.add(next_s)
                    queue.append((next_s, current_depth + 1))
        
        # label not found within radious 
        return False
                
    def add_labels(self, s, action, next_s, reward, done):
        "Envir. labels. Based on experiments, the labeller(tested) with highest compression rate"
         # add bounds
        if s == next_s:
            self.labels[s].add("bounded")

        # label terminal states 
        if done:
            # create gaol label
            if reward > 0:
                self.labels[next_s].add("Goal")
            else:
                self.labels[next_s].add("TS")
        # label non terminal states
        else:
            self.labels[s].add("NTS")
            self.labels[next_s].add("NTS")
            
    def _generate_structure_simple_edge(self, data: list[tuple]):
        
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
                self.relations[next_s]        # or assign the empty set 
    
    def _generate_structure_multi_edge(self, data:list[tuple]):

        # extract structural values 
        for s, action, next_s, reward, done in data:

            self.states.add(s)
            self.states.add(next_s)

            # normalizing action string
            action_label = f"a{action}"

            self.relations[s][action_label].add(next_s)      
            self.rev_relations[next_s][action_label].add(s)        

            # add labels 
            self.add_labels(s, action, next_s, reward, done)

            # if reached state is terminal assign empty dict => no actions
            if done:
                self.relations[next_s] = {}

    def _visualize(self):
        "Visualize Kripke model using boxes"

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
    
    def visualize(self):

        # Initialize graph with a clean font
        dot = Digraph(node_attr={
            'fontname': 'Helvetica,Arial,sans-serif',
            'fontsize': '10',
            'style': 'filled',
            'fillcolor': '#fcfcfc',
            'fixedsize': 'shape',
            'width': '1.2',
            'height': '1.2'
            
        })
        
        # Title and compact layout configuration
        # dot.attr(label="Kripke Model Semantics", labelloc="t", fontsize="14", fontname="Helvetica-Bold")
        dot.attr(nodesep='0.6', ranksep='0.8', rankdir="LR")
        dot.attr(size='10,6!', ratio='compress')

        # Add nodes based on their structural properties
        for s, props in self.labels.items():
            node_id = str(s)
            
            # Format the label nicely using standard newlines
            # Format: State ID on top, propositions inside brackets underneath
            prop_inline = ", ".join(props)
            prop_str = f"{{{textwrap.fill(prop_inline, width=10)}}}"
            standard_label = f"{node_id}\n{prop_str}"
            
            # Determine if the state is terminal (dead-end)
            # It's terminal if it has no outgoing relations or its relation list is empty
            
            is_terminal = s not in self.relations or not self.relations[s]
            
            if is_terminal:
                # Terminal states = Square (box)
                dot.node(node_id, label=standard_label, shape='box', color='#d9534f', penwidth='2')
            else:
                # Normal states = Circle
                dot.node(node_id, label=standard_label, shape='circle', color='#4a4a4a')

        # Optimize edge creation using a generator expression
        if not self.multi_edges:
            edges = (
                    (str(s), str(next_s)) 
                    for s, next_states in self.relations.items() 
                    for next_s in next_states
                )
            dot.edges(edges)
        else:
            for s, actions in self.relations.items():
                for action, next_states in actions.items():
                    for next_s in next_states:
                        # We must call this individually because 'label=str(action)' 
                        # changes dynamically for every single transition.
                        dot.edge(str(s), str(next_s), label=str(action))
        

        # Clean global edge styling
        dot.edge_attr.update(color="#4a4a4a", arrowhead="vee", arrowsize="1.0")

        # Render to a temporary file instead of the local directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gv") as temp_gv:
            temp_base = temp_gv.name

        # dot.render creates 'temp_base.svg' and opens it
        dot.render(temp_base, view=True, format="svg")
    
        # Cleanup the temporary source file immediately
        try:
            os.remove(temp_base)
        except OSError:
            pass

class CompressedModel:
    def __init__(self, states, relations, labels, mapping, bisim_states, multi_edges):
        self.states = states                           # macro states
        self.relations = relations                         # quotient relations 
        self.labels = labels                           # quotient labels 
        self.map = mapping                             # original_state -> macro_state
        self.bisim_states = bisim_states               # macro_state -> [bisim original_states]
        self.multi_edges = multi_edges


class KripkeMM:
    """
    Wrapper class that bring all of the components together 
    """
    def __init__(self, multi_edges=False, complex_labels=True,
                  zones=None, zone_radious=None, **kwargs):
        self.struct = ModelStructure(multi_edges=multi_edges, **kwargs)         # underlying structure 
        self.compressor = BiSimMini(self.struct, multi_edges=multi_edges)       # compresion engine 
        self.contex_generator = None                                            # formula generator on basis of model
        self.abst = None
        
        self.complex_labels = complex_labels 
        # set zones depending on instantiation
        if zones is not None:
            self.zones = zones 
        else:
            self.zones = {"GoalZone":"Goal", "DeathZone":"TS"}

        if zone_radious is not None:
            self.zone_radious = zone_radious
        else:
            self.zone_radious = 3

    def within_radious_dfs(self, state, label, max_steps):

        # baseline success: label found at current state 
        if label in self.labels[state]:
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
            
            # adaptation for dealing with edge type 
            if self.multi_edges:
                successors = chain.from_iterable(*self.relations[current_state].values())
            else:
                successors = self.relations[current_state]

            # look at all successors of the current state 
            for next_s in successors:
                if next_s not in visited:
                    # check if label is true at state 
                    if label in self.labels[next_s]:
                        return True 
                    
                    # label not found => Update visited and queue 
                    visited.add(next_s)
                    queue.append((next_s, current_depth + 1))
        
        # label not found within radious 
        return False
        
    def generate_labels(self):
        """
        Generate higher order labels lables that change dynamically.
        Labels like: 
            - Level of entropy
            - Proximity to goal 
            - Proximity to terminal states 
        """
        # optimize to avoid lookups
        relations = self.abst.labels
        labels = self.abst.labels 
        n_actions = self.struct.n_actions
        

        for s in relations.keys():

            s_labels = labels[s]

            bounds = any(label == "bound" for label in s_labels)
            terminal = any(label in ["TS", "Goal"] for label in s_labels)
            
           # compute proportion of actions explored 
            act_exp = len(s_labels.values()) / n_actions
            entropy = f"E_{'high' if act_exp <= 0.33 else 'mid' if act_exp <= 0.66 else 'low'}"

            # add dynamic entropy label and zones-labels 
            if not terminal:
                
                # check if state has an entopy level 
                current_entropy = [label for label in labels[s] if label.startswith("E")]
                
                # add or update entropy
                if not current_entropy:
                    self.abst.labels[s].add(entropy)
                elif current_entropy[0] != entropy:
                    self.abst.labels[s].remove(current_entropy[0])
                    self.abst.labels[s].add(entropy)

                # add zone label if applicable 
                for zone, label in self.zones.items():
                    if zone not in labels[s]:
                        if self.within_radious_dfs(s, label, self.zone_radious):
                            self.abst.labels[s].add(zone)                                                        # learned compressed model 
    
    def one_step_props(self, state):
        # get all the one step future proposition
        future_props = []

        for s_next in self.struct.relations[state]:
            future_props.append(self.struct.labels[s_next])
        
        return future_props
  
    def visualize(self, model, title=None):
        """
        Visualizer of model, adapted to work for single and multi edges
        """

        # Initialize graph with a clean font
        dot = Digraph(node_attr={
            'fontname': 'Helvetica,Arial,sans-serif',
            'fontsize': '10',
            'style': 'filled',
            'fillcolor': '#fcfcfc',
            'fixedsize': 'shape',
            'width': '1.2',
            'height': '1.2'
            
        })
        
        # add title if given 
        if title:
            dot.attr(label=title, labelloc="t", fontsize="14", fontname="Helvetica-Bold")
        dot.attr(nodesep='0.6', ranksep='0.8', rankdir="LR")
        dot.attr(size='10,6!', ratio='compress')

        # Add nodes based on their structural properties
        for s, props in model.labels.items():
            node_id = str(s)
            
            # Format the label nicely using standard newlines
            # Format: State ID on top, propositions inside brackets underneath
            prop_inline = ", ".join(props)
            prop_str = f"{{{textwrap.fill(prop_inline, width=10)}}}"
            standard_label = f"{node_id}\n{prop_str}"
            
            # Determine if the state is terminal (dead-end)
            # It's terminal if it has no outgoing relations or its relation list is empty
            
            is_terminal = s not in model.relations or not model.relations[s]
            
            if is_terminal:
                # Terminal states = Square (box)
                dot.node(node_id, label=standard_label, shape='box', color='#d9534f', penwidth='2')
            else:
                # Normal states = Circle
                dot.node(node_id, label=standard_label, shape='circle', color='#4a4a4a')

        # Optimize edge creation using a generator expression
        if not self.struct.multi_edges:
            edges = (
                    (str(s), str(next_s)) 
                    for s, next_states in model.relations.items() 
                    for next_s in next_states
                )
            dot.edges(edges)
        else:
            for s, actions in model.relations.items():
                for action, next_states in actions.items():
                    for next_s in next_states:
                        # We must call this individually because 'label=str(action)' 
                        # changes dynamically for every single transition.
                        dot.edge(str(s), str(next_s), label=str(action))
        

        # Clean global edge styling
        dot.edge_attr.update(color="#4a4a4a", arrowhead="vee", arrowsize="1.0")

        # Render to a temporary file instead of the local directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gv") as temp_gv:
            temp_base = temp_gv.name

        # dot.render creates 'temp_base.svg' and opens it
        dot.render(temp_base, view=True, format="svg")
    
        # Cleanup the temporary source file immediately
        try:
            os.remove(temp_base)
        except OSError:
            pass
    
    def update_structure(self, data):
        self.struct.generate_structure(data)
    
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
            bisim_states=bisim_states,
            multi_edges=self.compressor.multi_edges
        )

        # populate model with complext labels
        if self.complex_labels:
            self.generate_labels()



class KMMcompare(KripkeMM):
    """Current adjusted version to account for possible 4*4 comparion"""

    def __init__(self, compare_models, compare_struct, multi_edges=False, **kwargs):
        super().__init__(multi_edges=multi_edges, **kwargs)
        # init abst_k
        self.abst_k = None
        self.complex_labels = kwargs.get("complex_labels")
        self.simple_abst = None
        self.simple_abst_k = None
        self.compare_struct = compare_struct

        # set up model generator 
        self.generate_model = self._generate_model_compare if compare_models else self._generate_model_single_model

        # if comparing with complex lables 
        if self.compare_struct:
            # initialize simple structure and its compressor
            kwargs["complex_labels"] = False
            self.simple_struct = ModelStructure(multi_edges=multi_edges, **kwargs)
            self.simple_compressor = BiSimMini(self.simple_struct, multi_edges=multi_edges)

    def _compress(self, compressor, k=None):
        # generate abstract state with standard bisim of k-bisim
        if k is not None:
            macro_states, relations, labels, mapping, bisim_states = compressor.k_bisim(k, maps=True)
        else:
            macro_states, relations, labels, mapping, bisim_states = compressor.bisim(maps=True)

        # abstract model 
        return CompressedModel(
            states=macro_states,
            relations=relations,
            labels=labels,
            mapping=mapping,
            bisim_states=bisim_states
        )

    def _generate_model_compare(self, k=None):

        # compare structure difference alone or also compare compressors
        if self.complex_labels and self.compare_struct:
            self.simple_abst = self._compress(self.simple_compressor, k=None)
            self.abst = self._compress(self.compressor, k=None)
            # k-bisim 
            if k is not None:
                self.simple_abst_k = self._compress(self.simple_compressor, k=k)
                self.abst_k = self._compress(self.compressor, k=k)
        # compare compressors  with complex labels
        elif self.complex_labels:
            self.abst = self._compress(self.compressor, k=None)
            if k is not None:
                self.abst_k = self._compress(self.compressor, k=k)
        # compare compressors without complex labels 
        else:
            self.simple_abst = self._compress(self.compressor, k=None)
            if k is not None:
                self.simple_abst_k = self._compress(self.compressor, k=k)

    def _generate_model_single_model(self, k=None):
        
        if self.complex_labels:
            if k is not None:
                self.abst_k = self._compress(self.compressor, k=k)
            else:
                self.abst = self._compress(self.compressor, k=None)
        else:
            if k is not None:
                self.simple_abst_k = self._compress(self.compressor, k=k)
            else:
                self.simple_abst = self._compress(self.compressor, k=None)

    def update_structure(self, data):

        self.struct.generate(data)
        # update structures 
        if self.compare_struct:
            self.simple_struct.generate(data)