from itertools import chain
from collections import deque, defaultdict

# structure used for extracitng and managing labels 
class LabelTree:
    def __init__(self, props, successors):
        self.props=props                   # set of propositions true at 
        self.children=successors           # {action : [LabelTree]}
    
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
                node =  LabelTree(props, {})
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
                node = LabelTree(props, next_states)

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


# reusability of formulas generated at a state 
class StateFormula:

    def __inti__(self, state):
        self.state = state
        self._by_depth = {}

    def get(self, depth):
        return self._by_depth[depth]

    def store(self, depth, formula):
        self._by_depth[depth] = formula

def ContextGenerator():
    
    def __init__(self, model, KN_d):
        self.model = model
        self.extractor = Extractor()
        self.context = KN_d
        self.operators = {[]}
        self.objectives = {"goal": ["Goal", "GoalZone"], "explore": ["E_high", "E_mid"]}

    

    def gen_formula(state, objective):

        if objective:
            pass