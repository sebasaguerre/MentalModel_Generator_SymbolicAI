from bisim import BiSimulatMini

# testing PT algorithm
class MockModel:
    def __init__(self):
        # initial states
        self.states = ["x1", "x2", "x3", "x4"]
        
        # transitions (unlabeled relations)
        self.relations = {
            "x1": ["x2", "x3"],  # Top p -> middle p, right q
            "x2": ["x2", "x4"],  # Middle p -> self, bottom q
            "x3": [],            # Right q -> dead end
            "x4": []             # Bottom q -> dead end
        }
        
        # reverse transitions for preimages
        self.rev_relations = {
            "x1": [],
            "x2": ["x1", "x2"],  # Came from top p and self
            "x3": ["x1"],        # Came from top p
            "x4": ["x2"]         # Came from middle p
        }
        
        # labels (atomic propositions)
        self.labels = {
            "x1": {"p"},
            "x2": {"p"},
            "x3": {"q"},
            "x4": {"q"}
        }
        
        # expected partition for bisimulation
        self.expected = set(frozenset(block) for block in [["x1", "x2"], ["x3", "x4"]])

class AdvancedGauntletModel:
    def __init__(self):
        # 7 States with 3 distinct initial labels
        self.states = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
        
        # Dense forward relations (highly overlapping futures)
        self.relations = {
            "x1": ["x3", "x5"],        # Splits evenly across distinct q-blocks
            "x2": ["x3", "x4"],        # Points to the SAME q-block x3, but a different second one
            "x3": ["x6", "x7"],        # Connects q-world to the r-world
            "x4": ["x6"],              # Connects to r-world differently than x3
            "x5": ["x1", "x2"],        # Cyclic feedback loop back to the p-world!
            "x6": ["x6", "x7"],        # r-state with a self-loop and a cross-edge
            "x7": ["x7"]               # Pure self-loop dead-end
        }
        
        # Rigorously mapped reverse relations (Preimages)
        self.rev_relations = {
            "x1": ["x5"],
            "x2": ["x5"],
            "x3": ["x1", "x2"],        # Critical shared pivot for p-states
            "x4": ["x2"],
            "x5": ["x1"],
            "x6": ["x3", "x4", "x6"],  # Dense preimage block
            "x7": ["x3", "x6", "x7"]
        }
        
        # Initial label grouping (Atomic Propositions)
        self.labels = {
            "x1": {"p"},
            "x2": {"p"},
            "x3": {"q"},
            "x4": {"q"},
            "x5": {"q"},
            "x6": {"r"},
            "x7": {"r"}
        }
        
        # EXPECTED STABLE PARTITION:
        self.expected = set(frozenset(block) for block in [
            ["x1"], 
            ["x2"], 
            ["x3", "x4"], 
            ["x5"], 
            ["x6", "x7"]
        ])

def run_reduction_test(test_model):
    # initialize mockmodel 
    model = test_model
    
    # feed mock model to bisimulation class
    minimizer = BiSimulatMini(model)
    
    print("--- Executing Paige-Tarjan Reduction Engine ---")
    try:
        # Run your entry method
        final_Q = minimizer.fastPTalgo()
        
        print("\n--- Final Merged Partition Blocks (Q) ---")
        block_node = final_Q.head
        block_idx = 0
        final_blocks = []
        
        while block_node is not None:
            q_block = block_node.data
            elements_in_block = []
            
            # Walk through the state records inside this block
            state_node = q_block.elements.head
            while state_node is not None:
                
                elements_in_block.append(state_node.data.id) 
                state_node = state_node.next


            print(f"Block {block_idx}: {elements_in_block}")
            block_idx += 1
            block_node = block_node.next

            # append block
            final_blocks.append(elements_in_block)

        algo_output = set(frozenset(block) for block in final_blocks)
            
        # Validation Logic
        if block_idx == len(model.expected):
            if algo_output == model.expected:
                print(f"\nSUCCESS! Successfully shrank to {len(model.expected)} stable blocks correctly! ")
            else:
                print(f"\nCorrect number partitions partitions but wrong classification...")
        else:
            print(f"\nFAILED: Expected exactly {len(model.expected)} collapsed blocks, but got {block_idx}.")
            print("Check if a block was split incorrectly or if the cleanup didn't reset counts.")

            
    except Exception as e:
        print(f"\n💥 CRASH: The engine threw an exception during refinement.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
   print("Testing with simple model\n")
   run_reduction_test(MockModel())

   print("Testing with complex model\n")
   run_reduction_test(AdvancedGauntletModel())
