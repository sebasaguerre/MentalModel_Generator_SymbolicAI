from itertools import chain
from collections import deque, defaultdict

class Verify():
    def __init__(self, model):
        self.model = model
        self.get_successors = self._get_successors_multi if model.multi_edges else self._get_successors_simple


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
    
    def label_extractor_v1(self, state, max_steps):
        """
        For a given model extract all labels to a max depth of "max_steps" via BFS
        Return {depth:[(action, {prop}), ... ]}
        """
        # optimize efficiency by avoiding lookups
        relations = self.model.relations
        labels = self.model.labels
        
        # init and append initial state and props
        labels_per_layer = defaultdict(list)
        labels_per_layer[0].append((None, labels[state]))

        queue = deque([(state, 0)])
        visited = set()

        while queue:

            current_state, current_depth = queue.popleft()
            # append labels of of current state at current level 

            # iterate over successors of state 
            for action, next_states in relations[current_state].items() :
                for next_state in next_states:
                    # extract labels form successors and at to queue
                    labels_per_layer[current_depth + 1].append((action, labels[next_state]))
                    if current_depth + 1 <= max_steps:
                        queue.append((next_state, current_depth + 1))

        return labels_per_layer 
    
    def extract_labels(self, state, max_depth):
        """
        For a given model extract all labels to a max depth of "max_steps" via DFS.
        This return a tree for each action/labeld edge.
        return {action: [(props, {next_action : (prop, {next_next_as: ... })), next_action' : ... }), ... }
        
        """
        # optimize efficiency by avoiding lookups
        labels = self.model.labels
        relations = self.model.relations

        def execute(state, steps_avail):
            props = self.model.labels[state]

            # base case: no more steps are possible 
            if steps_avail == 0:
                return (props, {})
            
            next_states = {}

            for action in relations[state]:
                next_states[action] = [
                    self.extract_labels(succ, steps_avail - 1)
                    for succ in relations[state][action]
                ]
            
            return (props, next_states)

        return  execute(state, max_depth)


        while queue:

            current_state, current_depth = queue.popleft()

            # append labels of of current state at current level 
            labels_per_layer[current_depth].append(labels[current_state])
            
            successors = self.get_successors(current_state)

            # iterate over successors of state 
            for next_state in successors:
                # extract labels form successors and at to queue
                labels_per_layer[current_depth + 1].append(labels[next_state])
                if current_depth + 1 <= max_steps:
                    queue.append((next_state, current_depth + 1))

        return labels_per_layer 

def ContextGenerator():
    
    def __init__(self, model, KN_d):
        self.model = model
        self.verifier = Verify()
        self.context = KN_d
        self.operators = {[]}
        self.objectives = {"explore", "goal"}

    def gen_formula(state, objective=""):
        pass