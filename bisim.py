from collections import defaultdict

### Bisimulation component objects
"""
The set of objects bellow are used as 'records' for the implmentation of
Fast PTalgorithm, as lookup and assignment should stay at O(1)
"""
class Node:
    __slots__ = ["data", "next", "prev", "parent_dll"]
    
    def __init__(self, data, parent_dll):
        self.data = data
        self.next = None
        self.prev = None
        self.parent_dll = parent_dll

    def move_node(self, new_dll):
        # remove node from curren dll
        self.parent_dll.remove(self)
        
        # assign new dll and insert node
        self.parent_dll = new_dll 
        return self.parent_dll.insert(self, node=True)


class DLL:
    """
    Double link lists data structure
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def insert(self, data, node=False):

        if not node:
            # create node object 
            node = Node(data, self)
        else:
            # this is to accomodate easy movement of nodes between DLLs
            # in this case the data is the node itself  
            node = data 

        # check if list is empty 
        if not self.head:
            self.head, self.tail = node, node
            node.next, node.prev = None, None 
            
        else:
            self.head.prev = node 
            node.next = self.head
            self.head = node
            node.prev = None

        # increment size counter 
        self.size += 1

        return node 

    def remove(self, node):

        # check if list is empty
        if self.size == 0 or node is None:
            print("Empty DLL")
            return None

        # link adjustment 
        # if only one node exists
        if node is self.head and node is self.tail:
            self.head, self.tail = None, None

        # if node is head
        elif node is self.head:
            self.head = node.next 
            node.next.prev = None

        # if node is tail 
        elif node is self.tail:
            self.tail = node.prev
            node.prev.next = None
        
        # if node is intermediary
        else:
            node.prev.next = node.next 
            node.next.prev = node.prev 

        # delete reference to element and reduce size counter 
        node.next = None
        node.prev = None
        self.size -= 1


    def pop(self):

        # get tail node, remove from DLL and return node 
        node = self.tail
        self.remove(node)
        return node

    def search(self, data):
        """
        Search starting from the tail as older items are more likely 
        to be searched and deleted
        """
        current_node = self.tail 
        # iterate through the list from the back 
        while current_node is not None:
            # conditional checking 
            if current_node.data == data:
                return current_node 
            
            # update current node
            current_node = current_node.prev 

        print("Element not in list")
        return None 

class State:
    __slots__ = ["id", "block_q", "preimage", "position_q"] # memeory efficiency

    def __init__(self, id):
        self.id = id
        self.block_q = None   # pointer to Q block 
        self.preimage = []    # list of Edge records: all x xEy for the node y 
        self.node = None      # pointer to node in block of q 
        self.current_count = None

class Edge:
    __slots__ = ["source", "count"]

    def __init__(self, count):
        self.source = count
        # self.target = y 
        self.count = None
        # self.label = label    TODO: First without labeld edges then add 

class SmallB:
    "Blocks of Q (they are a subset of the blocks of X)"
    __slots__ = ["block_x", "elements", "node_x", "node"]

    def __init__(self):
        self.block_x = None
        self.elements = DLL() 
        self.node_x = None
        self.node = None
    
    @property 
    def size(self):
        return self.elements.size

class LargeB:
    "Blocks of X"

    __slots__ = ["sub_blocks", "in_c"]

    def __init__(self):
        self.sub_blocks = DLL()   # This sub blocks are blocks of Q
        self.in_c = None  
    
    @property
    def compound(self):
        return self.sub_blocks.size > 1 
    
class Counter:
    
    __slots__ = ["value"]

    def __init__(self, value):
        self.value = value               

#####################################################################################################################################

class BiSimulatMini:
    """ 
    Doing BiSimulation Minimization through Paige Tarjan Algorithm
    followed by quotient construction
    """
    def __init__(self, model):
        self.states = model.states
        self.edges = model.relations
        self.premap = model.rev_relations
        self.labels = model.labels
        self.worklist = DLL()                # list of splitter candidates
    
    def record_builder(self):

        self.records = {"states": [], "edges": [], "Qblocks": [], "Xblocks": []}

        # NOTE: # we can make this efficient to recicle states, by checking if the entry exists
        # state to record map:
        state_to_record = {s: State(s) for s in self.state}
        self.records["states"] = state_to_record

        # couter map: init count record count(x, U) for edge
        intcounter_map = {s : Counter(len(self.edges[s])) for s in self.states} 

        # generate edge records 
        # iterate over premap
        for target_state, pre_states in self.premap.items():
            # get target record 
            target_record = state_to_record[target_state]

            # loop over pre_states
            for source_state in pre_states:
                # create edge for target_record 
                source_record = state_to_record[source_state]
                edge = Edge(source_record)

                # assign initial counter for all edges
                edge.count = intcounter_map[source_state]
                
                # add edge to taget preimage
                target_record.preimage.append(edge)

                # add edges to record file 
                self.records["edges"].append(edge)


    # NOTE: this function might not be useful anymore... premap computed at generation
    def get_premap(self):
        self.premap = dict()

        # loop over the entire relation structure 
        for world, successors in self.edges.items():
            for s in successors:
                self.preimage.setdefault(s, set()).add(world)
    
    def partition0(self):
        """
        Generate the initial partition

        NOTE: Apparently we need the first partition to be already clustering states 
        with the same labels, so we need to sort them as such.
        """

        # inital partion init
        P = DLL()

        # grouping and  normalization
        groups = defaultdict(list)
        for state, labels in self.labels.items():
            # generate signature per state -> set of labels 
            signature = tuple(sorted(labels))
            # get state record 
            state_record = self.records["states"][state]
            # group state records
            groups[signature].append(state_record)

        # turn groups into Q blocks
        for group in groups.values():
            Q_block = SmallB()
            for state_record in group:
                Q_block.elements.insert(state_record)
                state_record.block_q = Q_block
                state_record.node = Q_block.elements.head

            # add block to P and set Q_node
            Q_block.node = P.insert(Q_block)

        return P
    
    def inti_X(self, Q):

        # initialize X partition
        X = DLL()
        X_block = LargeB()            # universe block
        
        Q_block = Q.head
        # iterate through initial partition block of Q and add them to the Xblock
        while Q_block is not None:
            X_block.sub_blocks.insert(Q_block)
            # point Qblock to parent block and add node 
            Q_block.block_x = X_block
            Q_block.node_x = X_block.sub_blocks.head   # insert = prepend 
            
            # get next block 
            Q_block = Q_block.next
        
        # add Xblock to X
        X.insert(X_block)

        return X 

    def get_preimage(self, S):
        """
        This is the function:
            E^-1(S) = {x | some y in S st xEy}
        """
        self.preimageS = set()

        # iterate over elements of S
        for x in S:
            self.preimageS.update(self.preimage[x])

    def split(self, S, P):
        """
        Refinement of P obtained by replacing each block B in P st
        B \/ E^-1(S) != {}  and B - E^-1(S) != {}, with:
        B' = B /\ E^-1(S) and B'' = B - E^-1(S).

        S is the `splitter` of P if split(S, P) != P; making Q unstable 
        with respects to S

        This splitting function makes the algorithm run in O(mn)
        """
        
        # refinement partition of P 
        Q = list()

        # compute pre image of splitter 
        self.get_preimage(S)

        # check for block split conditions for each block B in P 
        for B in P:

            # if union of B and preimageS and B - preimageS are non empty => split B
            # that is the elements of B both reach S and do not reach S => B is unstable under S
            if (B & self.preimageS) and (B - self.preimageS):
                # generate split blocks
                B_in = B & self.preimageS # elements of B that reach S
                B_out = B - self.preimageS # elements of B that do not reach S

                # split blocks to refined partition 
                Q.append(B_in)
                Q.append(B_out)

                # update worklist based on `smaller half` rule
                if B in self.worklist:
                    # if B on worklist, add both new blocks
                    self.worklist.append(B_in)
                    self.worklist.append(B_out)
                else:
                    # if B not on worklist, add the smaller block 
                    # smaller_half = B_in if len(B_in) <= len(B_out) else B_out
                    smaller_half = min(B_in, B_out, key=len)
                    self.worklist.append(smaller_half)
            
            # B is stable under S
            else:
                Q.append(B)

        # partition refinement finalized 
        return Q
    
    def PTalgo(self):

        # get intial partition and select splitter 
        P = self.partition0()
        S = self.worklist.pop()

        # iteratively refine partition via splitting
        while len(self.worklist) != 0:
            # refine partition 
            P = self.split(S, P) # 
            # select new splitter
            S = self.worklist.pop()

        return P
    
    def findsplitter(self, X, P):
        
        S = None # splitter block from X
        B = None # block from P that is a subset of S 

        # loop over blocks of X
        for s in X:

            if s not in P: 

                # loop over blocks of P
                for b in P:
                    
                    # check if b is subset of S and |b| <= |s|/2
                    if (b <= s) and (len(b) <= len(s)/2):
                        # assign blocks
                        S = s
                        B = b 

                        # update partition 
                        X.remove(s)
                        X.extend([B, S - B])

                        # return splitter 
                        return S, B
                    
     # Block form X
                    
    def fastsplit(self, preB, Q):
        """
        Fast splitting utilizing pointers
        """

        associated = {}                                 #  asosciate D with D' 
        splited_blocks = set()                          # list of D blocks that have been split 
                 
        # scan pre image for splitting 
        for x in preB:
            
            # split D block into the element going into 
            D = x.block_q
            
            # check if D' has been created or not 
            if D in associated:
                Dprime = associated[D]
            else:
                # create D prime and update data structure 
                Dprime = SmallB()                                 # D' = X^-1(B) intersect  D 
                Dprime.block_x = D.block_x
                Dprime.node_x = D.block_x.sub_blocks.insert(Dprime)
                Dprime.node = Q.insert(Dprime)

                # add D to splited blocks
                splited_blocks.add(D)

            # move node
            x.node = x.node.move_node(Dprime)
            x.block_q = Dprime


        # cleaning blocks after split 
        for D in splited_blocks:

            # eliminate block if empty
            if D.size == 0:
                # remove  from Q partition
                Q.remove(D.node)
                # remove form parent node in X partition
                D.block_x.sub_blocks.remove(D.node_x)
            
            # if the parent block of D in X is compound, add to working list 
            if D.block_x.compound and not D.block_x.in_c:
                self.worklist.insert(D.block_x)
                D.block_x.in_c = True 

        return Q

    def refinement(self, X, Q):

        # Step 1: selecting refinment block
        # get splitter and examin first two subblocks
        S = self.worklist.pop()
        S.in_c = False
        B1 = S.sub_blocks.head
        B2 = S.sub_blocks.head.next 

        # select the smaller one 
        B = B1 if B1.size <= B2.size else B2 

        # Step 2: update X 
        S.sub_blocks.remove(B.node_x)                 
        Sprime = LargeB()                 # create new block of X which is simple 
        Sprime.sub_blocks.insert(B)       # insert B into S'
        X.insert(Sprime)                  # insert S' into X

        # put S back into worklist if compund 
        if S.compound:
            self.worklist.insert(S)
            S.in_c = True

        # Step 3: compute E^-1(B)
        # copy all the elmenets of B into Bprime and find preB
        Bprime = list()
        preB = set()
        x_count_B = dict()
        
        # iterate through block
        node = B.elements.head 
        while node is not None:
            
            Bprime.append(node.data)                             # copy state record 

            # loop over the preimages of the current node y in xEy
            for edge in node.data.preimage:
                x = edge.source
                preB.add(x)                          # add preimages 

                # get count(x, B)
                if x not in x_count_B:
                    x_count_B[x] = []                 # generate counter 
                x_count_B[x].append(edge)
                    

    
            node = node.next

        # update count record and give x its shortcut 
        for x, edge_list in x_count_B.items():
            count = Counter(len(edge_list))
            x.current_count = count

            # # update edge counters 
            # for edge in edge_list:
            #     edge.count = count 

        # Step 4: refinining Q with respects to B
        Q = self.fastsplit(preB, Q)

        # Step 5: computing E^-1(B) - E^-(S - B)
        purepreB = set()

        for x, edges_to_B in x_count_B.items():

            # get old count(x, S)
            old_count = edges_to_B[0].count

            # add node to E^-1(B) - E^-(S - B) if counts are the same 
            if x.current_count.value == old_count.value:
                purepreB.add(x)

            # updtae edge count
            for edge in edges_to_B:
                # edge now points to count(x,B)
                edge.count = x.current_count

        # Step 6: refine Q with respect to S-B
        Q = self.fastsplit(purepreB, Q)

        # Step 7: update counts 


    def _fPTalgo(self):
        """
        The fast implementation of the PT algorithm.
        This requires to keep a second partion X which we use to aid refinement
        and the use or records for states, edges and blocks
        """

        # generate records
        self.record_builder()
        
        # initalize partition based on labels
        Q = self.partition0()          # "fine" partition 

        # this is the "coarse" partition and the compound blocks
        X = self.inti_X(Q)                                            # starts as the entire set of states 
        self.worklist.insert(X.head)                                  # U is compound so it goes to the worklist             

        S, B = self.findspliter(X, Q)  # the spliter is S - B st  S and B meet the conditions required 

        # iterative partition refinment 
        while self.worklist.size > 0:
            
            # refinment step 
            P = self.split(B, P)
            P = self.split(S - B, P)
            S, B = self.findsplitter(X, P)

        return P 
    
    def fastPTalgo(self):
        """
        The fast implementation of the PT algorithm.
        This requires to keep a second partion X which we use to aid refinement
        and the use or records for states, edges and blocks
        """

        # generate records
        self.record_builder()
        
        # initalize partition based on labels
        Q = self.partition0()          # "fine" partition 

        # this is the "coarse" partition and the compound blocks
        X = self.inti_X(Q)                                            # starts as the entire set of states 
        self.worklist.insert(X.head)                                  # U is compound so it goes to the worklist  
        X.head.in_c = True                                            # compound  block added to worklist            

        # iterative partition refinment 
        while self.worklist.size > 0:
            
            # get splitter
            self.refinement(X, Q)

        return Q
    


    