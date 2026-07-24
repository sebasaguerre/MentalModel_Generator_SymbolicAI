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

# support class for formula update
class UpdateContext:
    "collect all the information needed to perform a formula update"
    
    def __init__(self, action, goal_props):
        self.action = action
        self.goal_props
        self.goal_map = {elem: idx for idx, elem in enumerate(goal_props)}
        # both collect the objective props for the designated states
        self.child_props = list()           # list(set)   # NOTE:  this is done to also work with non-determinism 
        self.nxt_props = list()             # list(set)   # NOTE: only contains ist argument props
        self.nxt_ops = dict()               # (oper, quant, iter) = props 
        self.update_props = set()
        self.prev = False 

    def get_update_info(self):
        "Generate info needed for update"
        #TODO: update to working with child_prop as a list

        # compute existential values
        self.exi_paths = [
            self.child_props.intersection(nxt_prop)
            for nxt_prop in self.nxt_props ]
        self.exi_nxt = set.union(*self.nxt_props)

        # compute operator options and qunt
        self.op_options = set().intersection(
                {self.update_options(nxt[0]) for nxt in self.nxt_ops.keys()}
                )
        self.merge_quant = "E" if any("E" in tpl for tpl in self.nxt_ops.keys()) else "A"

        # compute univeral values only if universal formulas are possible
        if self.merge_quant == "A":
            self.uni_paths = set.intersection(*self.exi_paths) 
            self.uni_nxt = set.intersection(*self.nxt_props)
        else:
            self.uni_paths = set()
            self.uni_nxt = set()

    def set_new_formula(self, props, operator, quant):
        self.update_props.update(props)
        self.new_operator = operator
        self.new_quant = quant 
    
    def extend_previous_formula(self, props, operator, quant):
        "Store formulas that could be extende and then compare with new formula"

        if not self.prev:
            self.prev = True 
            self.extend_formulas = []
            self.append((props, operator, quant))
        
        else:
            self.append((props, operator, quant))

    def get_new_formula(self):
        pass

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
        self.op_imporance = (G, U, X, F)

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
                state_action_nodes = deque()                            # all syntactic nodes that belong to this state
                for action, s_nodes in children.items():

                    # nondeterministic transitions 
                    if len(s_nodes) > 1:
                        action_node = X(OR(s_nodes), action, quant="A")
                    else:
                        action_node = X(s_nodes[0], action, quant="A")
                    
                    state_action_nodes.append(action_node)
                
                # now that we iterated over all the succesors and added nXt, we get state props 
                if len(node.props) > 1:
                    state_action_nodes.appendleft(AND([Atom(p) for p in node.props]))
                else:
                    state_action_nodes.appendleft(Atom(next(iter(node.props))))
                
                # now we create the parent node and return 
                parent_s_node = AND(state_action_nodes)

                # # update parent node if depth is greater than 
                # if formula_depth >= 2:
                #     parent_s_node = self.update_formula(parent_s_node, objectives)


                # store parent node in cache 
                cache[key] = parent_s_node

                return parent_s_node

        return execute(root, 0)
    
    def action_formulas(self, AST):
        return islice(AST.conjuncts, 1, None)

    def _is_simple(self, node):
        "True if node is an atom, or a conjunction/disjunction of only atoms (no temporal ops)"
        if isinstance(node, Atom):
            return True
        if isinstance(node, (AND, OR)):
            items = node.conjuncts if isinstance(node, AND) else node.disjuncts
            return all(self._is_simple(item) for item in items)
        return False

    def _flat(self, node):
        "Single-line rendering of a simple (temporal-op-free) node"
        if isinstance(node, Atom):
            return repr(node)
        if isinstance(node, (AND, OR)):
            is_and = isinstance(node, AND)
            items = node.conjuncts if is_and else node.disjuncts
            sym = "∧" if is_and else "∨"
            return "( " + f" {sym} ".join(self._flat(item) for item in items) + " )"
        return repr(node)

    def _pretty_lines(self, node, indent, tab):
        pad = tab * indent

        if isinstance(node, Atom):
            return [pad + repr(node)]

        if isinstance(node, (AND, OR)):
            if self._is_simple(node):
                return [pad + self._flat(node)]

            is_and = isinstance(node, AND)
            items = node.conjuncts if is_and else node.disjuncts
            sym = "∧" if is_and else "∨"
            child_pad = tab * (indent + 1)

            lines = [pad + "("]
            for i, item in enumerate(items):
                item_lines = self._pretty_lines(item, indent + 1, tab)
                if i > 0:
                    first = item_lines[0]
                    stripped = first[len(child_pad):] if first.startswith(child_pad) else first.lstrip()
                    item_lines[0] = f"{child_pad}{sym} {stripped}"
                lines.extend(item_lines)
            lines.append(pad + ")")
            return lines

        if isinstance(node, X):
            q = node.quant or "?"
            header = f"{pad}{q}X{subscript(node.action)}("
            body = self._pretty_lines(node.f, indent + 1, tab)
            return [header] + body + [pad + ")"]

        if isinstance(node, U):
            q = node.quant or "?"
            header = f"{pad}{q}("
            f_lines = self._pretty_lines(node.f, indent + 1, tab)
            mid = f"{tab * (indent + 1)}{subscript(node.action)}U"
            g_lines = self._pretty_lines(node.g, indent + 1, tab)
            return [header] + f_lines + [mid] + g_lines + [pad + ")"]

        if isinstance(node, F):
            q = node.quant or "?"
            header = f"{pad}{q}{subscript(node.action)}F("
            body = self._pretty_lines(node.f, indent + 1, tab)
            return [header] + body + [pad + ")"]

        if isinstance(node, G):
            q = node.quant or "?"
            header = f"{pad}{q}{subscript(node.action)}G("
            body = self._pretty_lines(node.f, indent + 1, tab)
            return [header] + body + [pad + ")"]

        # fallback for unknown node types
        return [pad + repr(node)]

    def pretty_format(self, node, indent=0, tab="    "):
        "Render a formula AST indented level-by-level for readability"
        return "\n".join(self._pretty_lines(node, indent, tab))
    
    def get_props(self, node):
        "Currently only implemented for conjunction"
        # TODO: when considering non-determ8inisticv systems we need to include OR
        # and thus we also need to type of the connective, thus return that

        if isinstance(node, AND):
            return set(elem.prop for elem in node.conjuncts)
        
        elif isinstance(node, OR):

            if any(isinstance(disjunt, AND) for disjunt in node.disjuncts):
                disjoint_sets = set()
                for disjunct in node.disjuncts:
                    if isinstance(disjunct, AND):
                        disjoint_sets.add(set(elem.prop for elem in disjunct.conjuncts))
                    else:
                        disjoint_sets.add(disjunct.prop)

        else:
            return set(node.prop)
        
    def decouple_state_formula(self, formula):
        return self.get_props(formula.conjuncts[0]), self.action_formulas(formula)
    
    def wrap_elems(self, elements, conjunct=True):
        if len(elements) > 1:
            if conjunct:
                return AND(elements)
            else:
                return OR(elements)
        else:
            return Atom(next(iter(elements)))
    
    def update_options(self, operator):
        "Given a temporal operator which operators can we update to"

        match operator:
            case F(): 
                return {F, X}
            case G():
                return {F, X, U, G}
            case X():
                return {F, X, U, G}
            case U():
                return {U, F, X}
            
    def get_highest_rank_set(self, update_ctx, list_of_sets):
        """
        Compare the rank of all sets according to the agents objectives.
        Iteratively shrink the numbers of sets checkd by requiring them to be subsets
        of the previously collected high ranking propositions. Props are stored in update_ctx
        """
        
        while len(list_of_sets) > 0 :
            # rank eack existenial path by other 
            sets_by_rank = [
                min([update_ctx.goal_map(prop) for prop in exi_path]) for exi_path in list_of_sets
                ]
            
            # if more than one contain the same max gaol rank then store and iterate 
            max_rank = min(sets_by_rank)
            max_elem = update_ctx.goal_props[max_rank]
            update_ctx.update_props.add(max_elem)

            if sets_by_rank.count(max_elem) > 1:
                list_of_sets = [(prop_set - {max_elem}) for prop_set in list_of_sets if max_elem in prop_set]
            else:
                return
            
    def existential(self, update_ctx, list_of_sets, operator, g_prop):
        """
        Check for existential update for a given operator.
        Given a list of sets with path/state props, find the set containing g_prop that
        has the higher rank, by checking the rank of the other propos and select that one. 
        """

        if not any(g_prop in prop_set for prop_set in list_of_sets ):
            return False

        # updated list of sets to search 
        matched_sets = [prop_set - {g_prop} for prop_set in list_of_sets if g_prop in prop_set]

        if len(matched_sets) > 1:
            # break tie between matched sets 
            update_ctx.update_props.add(g_prop)
            self.get_hihgest_rank_set(update_ctx , matched_sets)
            highest_ranking_set = next((prop_set for prop_set in list_of_sets
                                            if update_ctx.update_props <= prop_set), None)
            update_ctx.set_new_formula(highest_ranking_set, operator, "E")

        else:
            update_ctx.set_new_formula(matched_sets[0], operator, "E")
        
        return True 
    
    def universal(self, update_ctx, list_of_sets, operator, g_prop):
        """
        Check is univerdal formula update is possible for a given operator
        """

        if (update_ctx.merge_quant == "A") and (g_prop in update_ctx.list_of_sets):
            update_ctx.set_new_formula(update_ctx.list_of_sets, operator, "A")
            return True 

    def check_update(self, operator, g_prop, update_ctx):
        """
        Given the update context, check if a goal propsition matches the
        operators conditions. If condition is met, the operator is set and 
        include the other goal propositions that the condition met.
        """
        
        g_prop_rank = update_ctx.goal_map(g_prop)

        match operator:
            case G():
                # Universal Global
                if update_ctx.merge_quant == "A" and g_prop in update_ctx.uni_paths:
                    update_ctx.set_new_formula(update_ctx.uni_paths, operator, "A")
                    return True 
                
                # Existential Global
                elif self.existential(update_ctx, update_ctx.exi_paths, operator, g_prop):
                    return True

                #NOTE: under revision after the function is fully implemented 
                # # preserve nxt Global   
                # elif G in update_ctx.nxt_ops.keys():
                #     # evaluate if its worth keeping global 
                #     global_props = [global_prop for key, global_prop in update_ctx.nxt_ops.items() if G in key]
                #     global_props_rank = [min([update_ctx.goal_map(prop) for prop in global_set]) for global_set in global_props]
                #     max_rank_global = min(global_props_rank)

                #     if (max_rank_global < g_prop_rank) or abs(g_prop_rank - max_rank_global) < 2:
                #         max_global_props = global_props[global_props_rank[max_rank_global]]
                        
                # No Global possible 
                else:
                    return False
                    
            case U():
                if update_ctx.merge_quant == "A":
                    pass
                else: 
                    pass
    
            case X():
                # Universal Next - this only depends on whether non-deter. and prop valuation
                if any(g_prop in child_prop for child_prop in update_ctx.child_props):

                    # check non-determinism
                    if len(update_ctx.child_prop) > 1:
                        if all(g_prop in child_prop for child_prop in update_ctx.child_props):
                            update_ctx.set_new_formula(set.intersection(*update_ctx.child_props), operator, "A")
                            return True
                        else:
                            matched_sets = [child_set - {g_prop} for child_set in update_ctx.child_props if g_prop in child_set]


                    else:
                    # determinism, add all props 
                        update_ctx.set_new_formula.update(update_ctx.child_props[0], operator, "A")
                        return True
                    
                else:
                    return False
                
            case F():
                # Universal Eventually 
                if self.universal(update_ctx, update_ctx.uni_nxt, operator, g_prop):
                    return True 
                
                # Existential Eventually 
                if self.existential(update_ctx, update_ctx.exi_nxt, operator, g_prop):
                    return True
                
                else: 
                    return False
                
    
    def update_action_formula(self, update_ctx, goal_props):
        "Iterative checking formula update for the most meaningul update"

        # goal proposition in order of relevance 
        for g_prop in goal_props:

            # skip g_props not present 
            if not ((g_prop in update_ctx.exi_nxt) or (g_prop in update_ctx.child_props)):
                continue

            # iterate over possible oprators in order of importance
            for operator in self.op_imporance:

                if not (operator in update_ctx.op_options):
                    continue 

                # check if operator fits 
                if self.check_update(operator, update_ctx, g_prop):
                    return 

    
    def update_formulas(self, formula, goals, ignore=set()):
        """
        Refine formulas for more expresivity and taylor to objective
        """
        #NOTE: for later: consider the possibility to also track rank 1 goals for U & G
        #NOTE: this does not consider non-determinisitic systems: take care later

        # deconstruct parent fomrula: state prop + action formulas
        parent_props, action_formulas = self.decouple_state_formula(formula)
        updated_formula = deque()
        goal_props = tuple(rank_prop for rank_props in goals.values() for rank_prop in rank_props)

        # update action formulas
        for _ , af in enumerate(action_formulas):

            # get formula details
            temp_operat = type(af)
            action = af.action
            quant = af.quant
            child_props, nxt_formulas = self.decouple_state_formula(af.f)

            # track all info needed for update 
            update_ctx = UpdateContext(action, goal_props)
            
            # storen props in objective. NOTE 
            update_ctx.child_props.append(self.get_props(child_props).intersection(goal_props))

            # collect the operators that the child successors have for formula update
            nxt_ops = dict()
    
            # iterate over temporal subformulas: successors of child
            for i, nxt_elem in enumerate(nxt_formulas):

                # nxt fromula relevant details
                nxt_quant = nxt_elem.quant

                # extract props if existing
                nxt_obj_props = self.get_props(nxt_elem.f).intersection(goal_props) 
                update_ctx.nxt_props.append(nxt_obj_props)  

                #TODO: incorporate a way to save U as a special case
                if isinstance(nxt_elem, U):
                    until_props = self.get_props(nxt_elem.g).intersection(goal_props)
                    nxt_obj_props = (nxt_obj_props, until_props)

                # store operators, props and quant
                update_ctx.nxt_ops[(nxt_elem, nxt_quant, i)] =  nxt_obj_props

            #### Action formula Update ####

            # compute all the information needed to procced with an update
            update_ctx.get_update_info()

            # update action formula
            updated_formula.append(
                self.update_action_formula(update_ctx, goal_props)
                )

        # prepend state props to new formula then return ocnjunction
        updated_formula.appendleft(formula.conjuncts[0])

        return AND(updated_formula)


    def old_update_formulas(self, formula, goals, ignore=set()):
        """
        Refine formulas for more expresivity and taylor to objective
        """
        # deconstruct parent fomrula: state prop + action formulas
        parent_props, action_formulas = self.decouple_state_formula(formula)
        updated_formula = deque()

        # update action formulas
        for _ , af in enumerate(action_formulas):

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

            #NOTE: for later: consider the possibility to also track rank 1 goals for U & G
            #NOTE: this does not consider non-determinisitic systems: take care later 



            #NOTE: this avoids completley the storage of objectives for a branch -> simplify + simpler
            # but reducess expressivity 

            # check if next prop are in objective, if primary => nXt, else store
            for rank, objectives in goals.items():
                child_obj_props = set(prop for prop in child_props if prop in objectives)
                # if rank 1 objectives then update immediately 
                if child_obj_props and rank == 1:
                    updated_formula.append(X(self.wrap_elems(child_obj_props), action, quant))
                    # if immediate update (?)
                    continue 
                else:
                    obj_props["child"][rank].append(child_obj_props)
            
            
            # check for prop equiavalence in trajectory and remove irrelevant props 
            match_child = set(prop for prop in child_props if prop in parent_props) - ignore 
            match_nxt = []

            # iterate over temporal subformulas: successors of child
            for nxt_elem in nxt_formulas:

                # nxt fromula relevant details
                nxt_operator = type(nxt_elem)
                nxt_quant = nxt_elem.quant

                # NOTE: this might not be useful, possibly eliminate 
                # chekc dim of temporal fomula 
                if not (until := isinstance(nxt_elem, U)):
                    props = (self.get_props(nxt_elem.f), )
                else:
                    props = (self.get_props(nxt_elem.f), self.get(nxt_elem.g))


                # check if props in trajectory are part of objective (per rank)
                for rank, objectives in goals.items():
                    nxt_obj_props = set(prop for prop in props[1] if prop in objectives)
                    obj_props["nxt"][rank].append(nxt_obj_props)
                    # if props[1] in objectives:
                    #     nxt_obj_props = set(prop for prop in props[1] if prop in objectives)    

                # if child to parent matching, check matching with next. This is for G
                if match_child:
                    match_nxt.append(set(prop for prop in props[1] if prop in match_child))
            
            #### Action formula Update ####
            # checking formulas in order of relevance

            # intersection of paths per rank for all nxt paths 
            exi_paths = {
                        rank: [
                            set.intersection(parent_props, *obj_props["child"][rank], path_set)
                            for path_set in obj_props["nxt"][rank]
                        ]
                        for rank in obj_props["nxt"].keys()
                        }
            
            # set of objc props that are true across all paths after action was taken 
            uni_paths = {rank: set.intersection(*exi_paths[rank]) for rank in exi_paths}

   
            # check for AG
            if uni_paths:
                updated_formula.append(G(self.wrap_elems(uni_paths), action, quant="A"))
                continue
            
            # parent to child path intersection and nxt goal props that hold in all nxt
            child_uni_paths = set.intersection(parent_props, *obj_props["child"])
            uni_nxt_props = set.intesection(*obj_props["nxt"].values())           # box phi 

            # checking for AU
            if child_uni_paths & uni_nxt_props: 
                updated_formula.append(U(self.wrap_elems(child_uni_paths),
                                         self.wrap_elems(uni_nxt_props), action, quant="A"))
                continue 
            
           # check for AX
            for rank_props in obj_props["child"].items():
                if rank_props:
                    updated_formula.append(X(self.wrap_elems(rank_props), action, quant="A"))

            # existing variables per rank in nxt successors 
            nxt_exi_props = {rank : set.union(*obj_props["nxt"][rank]) for rank in obj_props["nxt"].keys()}

            # check for EG and EU: heuristic based for ranks
            for rank, prop_list in exi_paths.items():
                if rank_exi_path := [props for props in prop_list if props]:
                    updated_formula.append(G(self.wrap_elems(rank_exi_path[0], conjunct=False),
                                            action, quant="A"))
                    continue 
                elif child_uni_paths & nxt_exi_props[rank]:
                    updated_formula.append(U(self.wrap_elems(child_uni_paths),
                                             self.wrap_elems(next(iter(nxt_exi_props[rank]))),
                                             action, quant="E"))

            # check for AF
            if uni_nxt_props:
                updated_formula.append(F(self.wrap_elems(uni_nxt_props), action, quant="A"))
                continue

            # check for EF 
            for rank, prop_list in obj_props["nxt"].items():
                    if exi_props := [props for props in prop_list if props]:
                        updated_formula.append(F(self.wrap_elems(exi_props[0]),
                                                action, quant="E"))
                        continue 


        # prepend state props to new formula then return ocnjunction
        updated_formula.appendleft(formula.conjuncts[0])

        return AND(updated_formula)
    

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
