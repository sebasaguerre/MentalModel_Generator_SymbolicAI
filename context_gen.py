from itertools import chain, islice
from collections import deque, defaultdict

# structure used for extracitng and managing labels 
class LabelTree:
    def __init__(self, state, props, successors):
        self.state = state
        self.props = props                   # set of propositions true at 
        self.children = successors           # {action : [LabelTree]}
    
    def is_leaf(self):
        return not self.children

class Extractor():
    def __init__(self, model):
        self.model = model
        self.get_successors = self._get_successors_multi if model.multi_edges else self._get_successors_simple
        self._extract_cache = {}       # cache that persists across calls
    
    def update_model(self, new_model):
        "update model and clear out cache"
        self.model = new_model
        self._extract_cache = {} 

    def _get_successors_simple(self, state):
        return self.model.relations[state]
        
    def _get_successors_multi(self, state):
        return chain.from_iterable(self.model.relations[state].values()) 

    def within_radious_dfs(self, state, label, current_step, max_steps):
        # success base case: label found in current state 
        if label in self.model.labels[state]:
            return True 
        
        # failure base case 
        if current_step == max_steps:
            return False 
        
        # continue to seach label in successors until max depth 
        for target in self.get_successors(state):
            if self.traverse_graph(target, label, current_step + 1, max_steps):
                return True 
        
        # nothing found 
        return False
    
    def retrieve_neighborhood(self, states, max_steps):

        # state nighboorhood 
        neighbourhood = set()
        to_visit = deque()
        visited = set(states)

        # extract base level states
        for s in states:    
            to_visit.append((s, 0))

        # traverse level by level 
        while to_visit:
            current_state, current_depth = to_visit.popleft()
            
            if self.multi_edges:
                successors = chain.from_iterable(self.relations[current_state].values())
            else:
                successors = self.relations[current_state]
            
            for next_s in successors:
                # save neighbout and store for further search
                if next_s not in visited:
                    visited.add(next_s)
                    neighbourhood.add(next_s)
                    if current_depth + 1 < max_steps:
                        to_visit.append((next_s, current_depth + 1))
        
        return neighbourhood

    def within_radious_bfs(self, state, label, max_steps):
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
            for next_s in self.get_successors(current_state):
                if next_s not in visited:
                    # check if label is true at state 
                    if label in self.model.labels[next_s]:
                        return True 
                    
                    # label not found => Update visited and queue 
                    visited.add(next_s)
                    queue.append((next_s, current_depth + 1))
        
        # label not found within radious 
        return False 
    
    def extract_labels(self, state, max_depth):
        """
        For a given model extract all labels to a max depth of "max_steps" via DFS,
        and generate a tree.
        This function utilizes memoization to reduce computaitonal complexity and
        reuse previous extractions 
        """
        # optimize efficiency by avoiding lookups
        labels = self.model.labels
        relations = self.model.relations
        cache = self._extract_cache

        def execute(state, steps_avail):
            
            # execution key
            key = (state, steps_avail)

            # if extraction has been done before return precomputed value 
            if key in cache: 
                return cache[key]

            # get propositions of current state 
            props = labels[state]

            # base case: no more steps are possible 
            if steps_avail == 0:
                node =  LabelTree(state, props, {})
            else:

                # next states reached via the actions =>  action : [LabelTree]
                next_states = {}

                # iterate over action possible at current state 
                for action, successors in relations[state].items():
                    # store successors of next_states
                    children = []
                    
                    # get node for all successor states
                    for succ in successors:
                        children.append(execute(succ, steps_avail -1))

                    # link action to successor nodes
                    next_states[action] = children
                
                # create node in labeld tree 
                node = LabelTree(state, props, next_states)

            # store extraction in cache
            cache[key] = node 

            return node

        return  execute(state, max_depth)

    
    def print_label_tree(self, tree, indent=2, action_taken=None):
        prefix = "  " * indent
        action_str = f"--[{action_taken}]--> " if action_taken else ""
        print(f"{prefix}{action_str}props: {set(tree.props)}")
        for action, subtrees in tree.children.items():
            for subtree in subtrees:
                self.print_label_tree(subtree, indent + 2, action)
    
    # def parse_labels(self, label_dag, max_depth):

    #     order_paths = defaultdict(dict)

    #     def traverse(label_dag, steps_taken, max_steps):
            
    #         state_prop, action_dag = label_dag

    #         # base case 
    #         if steps_taken == max_depth or action_dag == {}:
    #             return (max_depth, state_prop, None)
            
    #         for action, successor_dag in action_dag.items():
    #             for successor in successor_dag:
    #                 order_paths[(steps_taken, state_prop, action)] = dict(traverse(successor, steps_taken + 1, max_depth))

    #     return order_paths

####### Support funcitons 
def subscript(text: str) -> str:
    # A complete map of available lowercase and numeric Unicode subscripts
    sub_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', 
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ', 
        'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 
        'v': 'ᵥ', 'x': 'ₓ'
    }
    # Convert character if it exists in the map, otherwise keep it as-is
    return "".join(sub_map.get(char, char) for char in text.lower())

####### AST node clases
class Atom:
    def __init__(self, prop):
        self.prop = prop

    def __repr__(self):
        return self.prop

class AND:
    def __init__(self, conjuncts):
        self.conjuncts = conjuncts

    def __repr__(self):
        return "(" + " ∧ ".join(repr(c) for c in self.conjuncts) + ")"

class OR:
    def __init__(self, disjuncts):
        self.disjuncts = disjuncts

    def __repr__(self):
        return "(" + " ∨ ".join(repr(c) for c in self.disjuncts) + ")"
    
class X:
    "Next"
    def __init__(self, f, action, quant=None):  # quant: 'E', 'A', or None
        self.f = f
        self.action = action
        self.quant = quant
    
    def __repr__(self):
        q = self.quant or '?'
        return f"{q}X{subscript(self.action)}({self.f})"

class U:
    "Until"
    def __init__(self, f, g, action, quant=None):
        self.f = f
        self.g = g
        self.action = action
        self.quant = quant
    
    def __repr__(self):
        q = self.quant or '?'
        return f"{q}({self.f} {subscript(self.action)}U {self.g})"

class F:
    "Eventually"
    def __init__(self, f, action, quant=None):
        self.f = f
        self.action = action
        self.quant = quant
    
    def __repr__(self):
        q = self.quant or '?'
        return f"{q} {subscript(self.action)}F{self.f})"

class G:
    "Global"
    def __init__(self, f, action, quant=None):
        self.f = f
        self.action = action 
        self.quant = quant

    def __repr__(self):
        q = self.quant or '?'
        return f"{q} {subscript(self.action)}G{self.f}"

class Generator:
    def __init__(self, button_up=True):
        # set the type of algorithm used for fomula generation 
        if not button_up:
            self.generate_formula = self._generate_top_down
        else:
            self.generate_formula = self._generate_button_up

        # formula cache: state -> StateFormula
        self._formula_cache = {}

    def _generate_top_down(self, label_tree):
        pass

    
    def _generate_button_up(self, label_tree, tree_depth):
        
        root = label_tree
        cache = self._formula_cache
        context = tree_depth

        def execute(node, depth):
            
            # key used for hashing & storing state formulas with a given depth 
            key = (node.state, context - depth)

            # check is syntactic node exists in cache 
            if key in cache:
                return cache[key]

            # check if leave node reached 
            if node.is_leaf():
                # create stateformula, store in cache and return val
                if len(node.props) > 1:
                    s_node = AND([Atom(p) for p in node.props])
                else:
                    s_node = Atom(next(iter(node.props)))
                
                cache[key] = s_node 
                
                return s_node 
            # 
            else:
                
                children = defaultdict(list)

                for action, successors in node.children.items():
                    for succ_node in successors:
                        children[action].append(execute(succ_node, depth + 1))
                    
                # bubble-up -> wrap kids in Next Wrapper
                state_s_nodes = deque()                            # all syntactic nodes that belong to this state
                for action, s_nodes in children.items():

                    if len(s_nodes) > 1:
                        s_node = X(OR(s_nodes), action, quant="A")
                    else:
                        s_node = X(s_nodes[0], action, quant="A")
                    
                    state_s_nodes.append(s_node)
                
                # now that we iterated over all the succesors and added nXt, we get state props 
                if len(node.props) > 1:
                    state_s_nodes.appendleft(AND([Atom(p) for p in node.props]))
                else:
                    state_s_nodes.appendleft(Atom(next(iter(node.props))))
                
                # now we create the parent node and return 
                parent_s_node = AND(state_s_nodes)

                # store parent node in cache 
                cache[key] = parent_s_node

                return parent_s_node

        return execute(root, 0)
    
    def action_formulas(self, AST):
        return islice(AST.conjuncts, 1, None)

    

# reusability of formulas generated at a state 
class StateFormula:

    def __inti__(self, state):
        self.state = state
        self._by_depth = {}

    def get(self, depth):
        return self._by_depth[depth]

    def store(self, depth, formula):
        self._by_depth[depth] = formula

class ContextGenerator:
    
    def __init__(self, model, KN_d):
        self.model = model
        self.extractor = Extractor()
        self.generator = Generator()
        self.state_formula_mapping = {}
        self.context_depth = KN_d
        # basic primary objectives 
        self.objectives = {"goal": ["Goal", "GoalZone"], "explore": ["E_high", "E_mid"]}
        
        # primary things to avoid 
        self.avoid = ["TS", "DangerZone"]

    def update_model(self, model):
        self.extractor._extract_cache = {}
        self.generator._formula_cache = {}
        self.

    def gen_formula(state, objective):

        if objective:
            pass