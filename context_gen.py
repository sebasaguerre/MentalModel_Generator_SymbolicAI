from itertools import chain, islice
from collections import deque, defaultdict

##### Support funtions
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

##### Structure used to Extract Labels 
class LabelTree:
    def __init__(self, state, props, successors):
        self.state = state
        self.props = props                   # set of propositions true at 
        self.children = successors           # {action : [LabelTree]}
    
    def is_leaf(self):
        return not self.children

###### AST node clases used formula generation
# propsition atoms 
class Atom:
    def __init__(self, prop):
        self.prop = prop

    def __repr__(self):
        return self.prop

# connectives 
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

class XOR:
    "XOR is defined as (¬p ∧ q) ∨ (p ∧ ¬q)"
    def __init__(self, exjuncts):
        self.exjuncts = exjuncts
    
    def __repr__(self):
        return "(" + " ⊕ ".join(repr(c) for c in self.disjuncts) + ")"

# temporal operators   
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

##### Label Extractor 
class Extractor():
    def __init__(self, model):
        self.model = model
        # self.get_successors = self._get_successors_multi if model.multi_edges else self._get_successors_simple
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

##### Generate formulas via AST construction using Labeld Tree
class Generator:
    def __init__(self, button_up=True):
        # set the type of algorithm used for fomula generation 
        if not button_up:
            self.generate_formula = self._generate_top_down
        else:
            self.generate_formula = self._generate_button_up

        # formula cache: (state, depth_k) -> state_formula_depth_k
        self._formula_cache = {}

        # self.sym_model = parent 

    def _generate_top_down(self, label_tree):
        pass

    def _generate_button_up(self, label_tree, tree_depth, objectives=None, ignore=set()):
        
        root = label_tree
        cache = self._formula_cache
        context = tree_depth

        def execute(node, depth):
            
            # key used for hashing & storing state formulas with a given depth 
            formula_depth = context - depth 
            key = (node.state, formula_depth)

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

                    # nondeterministic transitions 
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

                # # update parent node if depth is greater than 
                # if formula_depth >= 2:
                #     parent_s_node = self.update_formula(parent_s_node, objectives)


                # store parent node in cache 
                cache[key] = parent_s_node

                return parent_s_node

        return execute(root, 0)
    
    def action_formulas(self, AST):
        return islice(AST.conjuncts, 1, None)
    
    def get_props(self, node):
        "Currently only implemented for conjunction"
        # TODO: when considering non-determ8inisticv systems we need to include OR
        # and thus we also need to type of the connective, thus return that

        if isinstance(node, AND):
            return set(elem.prop for elem in node.conjuncts)
        else:
            return set(node.prop)
        
    def decouple_state_formula(self, formula):
        return self.get_props(formula.conjuncts[0]), self.action_formulas(formula)
    
    def update_formulas(self, formula, goals, ignore=set()):
        """
        Refine formulas for more expresivity and taylor to objective
        """
        # deconstruct parent fomrula: state prop + action formulas
        parent_props, action_formulas = self.decouple_state_formula(formula)
        updated_formula = deque()

        # update action formulas
        for i, af in enumerate(action_formulas):
            # get formula details
            temp_operat = type(af)
            action = af.action
            quant = af.quant
            child_props, nxt_formulas = self.decouple_state_formula(af.f)

            # track objective props across paths 
            obj_props = defaultdict(lambda: defaultdict(list))

            # # check if next prop is primary objective; if so nXT wrapper 
            # if child_props in goals[1]:
            #     updated_formula.append(X(child_props - ignore, action, quant))
            #     continue

            # check if next prop are in objective, if primary => nXt, else store
            for rank, objectives in goals.items():
                if child_props in objectives:
                    child_obj_props = set(prop for prop in child_props if prop in objectives)
                    # if rank 1 objectives then update immediately 
                    if rank == 1:
                        updated_formula.append(X(self.wrap_props(child_obj_props), action, quant))
                    else:
                        obj_props["child"][rank].append(child_obj_props)
                
            
            # check for prop equiavalence in trajectory and remove irrelevant props 
            match_child = set(prop for prop in child_props if prop in parent_props) - ignore 
            match_nxt = defaultdict(list)

            # iterate over function terms
            for nxt_elem in nxt_formulas:
                # nxt fromula relevant details
                nxt_operator = type(nxt_elem)
                nxt_quant = nxt_elem.quant
                
                # chekc dim of temporal fomula 
                if not (until := isinstance(nxt_elem, U)):
                    props = (self.get_props(nxt_elem.f), )
                else:
                    props = (self.get_props(nxt_elem.f), self.get(nxt_elem.g))

                # check if props in trajectory are part of objective (per rank)
                for rank, objectives in goals.items():
                    if props[1] in objectives:
                        nxt_obj_props = set(prop for prop in props[1] if prop in objectives)
                        obj_props["nxt"][rank].append(set(prop for prop in props[1] if prop in objectives))

                # if child to parent matching, check matching with next
                if match_child:
                    match_nxt["full"].append([prop for prop in props[1] if prop in match_child])
            
            #### action formula update ####

            # check match_child for Until or Global formula 
            if match_child:
                # check for match_child for Global formula
                if match_nxt["full"]:
                    
                    if len(global_path_props:= set.intersection(*match_nxt)) == 0:

                    # quantifier attribution
                    if len(match_nxt["full"]) == len(nxt_formulas):

                        updated_formula.append(G(self.wrap_props(match_child), action=action, quant="A"))
                    else:
                        updated_formula.append(G(self.wrap_props(match_child), action=action, quant="E"))
                    continue

            # check for other operators based on objectives 
            for rank, props in obj_props:
                # check if props of current rank have been found 
                if props:
                    unique_props = set.union(*props)
                    shared_props = set.intersection(*props)

                    # check for quantifiers 
                    if len(props) == len(nxt_formulas):
                        updated_formula.append(F(self.wrap_props(shared_props), action=action, quant="A"))
                    else:
                        updated_formula.append(F(self.wrap_props(shared_props), action=action, quant="E"))
                    continue




        # prepend state props to new formula then return ocnjunction
        updated_formula.appendleft(formula.conjuncts[0])

        return AND(updated_formula)
    
    def wrap_props(self, props, conjunct=True):
        if len(props) > 1:
            if conjunct:
                return AND(props)
            else:
                return OR(props)
        else:
            return Atom(props)
        


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
    
    def __init__(self, parent, model=None, objectives=None, avoid=None):
        self.sym_model = parent
        self.extractor = Extractor(model)
        self.generator = Generator()
        self.state_formula_mapping = {}

        # setup for formulas 
        self.objectives = {1:("Goal", "GoalZone"), 2:("E_high", "E_mid", "E_unknown")}  # basic primary objectives: by ranking
        self.avoid = ["TS", "DangerZone"]                                               # primary things to avoid 
        self.irrelevant = set(["NTS"])

    def update_and_clear_cache(self, new_model):
        """
        Update model and clear out the existing caches,
        preventing outdated information from being used
        """
        self.model = new_model
        self.extractor.model = new_model
        self.extractor._extract_cache = {}
        self.generator._formula_cache = {}

    def add_missing_action_formulas(self, state, AST):
        """
        Generate formulas for unexplored actions.
        Unexplored actions recieve a formula using 'E_unknown' reflecting
        the uncertainty of the agent when selecting that action. Since the 
        action has never been taken before it could lead to unvisited state,
        or a previouslly visited state with known E quotient value.
        """

        # get unexplored actions at current state 
        explored_actions = [int(action[1:]) for action in self.model.relations[state].keys()]
        unexplored_actions = [f"a{action}" for action in range(self.parent.num_act) if action not in explored_actions]

        # generate formulas 
        unexp_action_nodes = []

        for unexp_act in unexplored_actions:
            unexp_action_nodes.append(X(Atom("E_unknown"), unexp_act, quant="A"))
        
        # extend AST node with 
        AST.conjuncts.extend(unexp_action_nodes)

        return AST

    def action_prioritization(self, formula):
        pass

    def gen_formula(self, state, context):

        # extract labels, then convert into AST and generate contextual formulas
        label_tree = self.extractor.extract_labels(state, context)
