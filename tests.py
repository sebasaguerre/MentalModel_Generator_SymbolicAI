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

def run_reduction_test():
    # initialize mockmodel 
    model = MockModel()
    
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
            
        # Validation Logic
        if block_idx == 2:
            print("\nSUCCESS! The 4-state system successfully shrank to 2 stable blocks.")
            print("Paige-Tarjan three-way split works perfectly!")
        else:
            print(f"\n FAILED: Expected exactly 2 collapsed blocks, but got {block_idx}.")
            print("Check if a block was split incorrectly or if the cleanup didn't reset counts.")
            print(f"The following blocks where found:\n {final_blocks}")

            
    except Exception as e:
        print(f"\n💥 CRASH: The engine threw an exception during refinement.")
        import traceback
        traceback.print_exc()

def main():
    
    # initialize bisimulation and mock model 
    mock_model = MockModel()
    bisim_mini = BiSimulatMini(mock_model)


if __name__ == "__main__":
    main()
