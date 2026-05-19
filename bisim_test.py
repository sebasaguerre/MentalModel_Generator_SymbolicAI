from bisim import BiSimulatMini

# testing PT algorithm
class MockModel:
    def __init__(self):
        # initial states
        self.states = ["s1", "s2", "s3", "s4"]
        
        # transitions (unlabeled relations)
        self.relations = {
            "s1": ["s2", "s3"],  # Top p -> middle p, right q
            "s2": ["s2", "s4"],  # Middle p -> self, bottom q
            "s3": [],            # Right q -> dead end
            "s4": []             # Bottom q -> dead end
        }
        
        # reverse transitions for preimages
        self.rev_relations = {
            "s1": [],
            "s2": ["s1", "s2"],  # Came from top p and self
            "s3": ["s1"],        # Came from top p
            "s4": ["s2"]         # Came from middle p
        }
        
        # labels (atomic propositions)
        self.labels = {
            "s1": {"p"},
            "s2": {"p"},
            "s3": {"q"},
            "s4": {"q"}
        }
        
        # expected partition for bisimulation
        self.expected = set(frozenset(block) for block in [["s1", "s2"], ["s3", "s4"]])

        # expected quotient model 
        # self.quotient_model = {
        #     "states": set(["x1", "x2"]), 
        #     "relations" : {"x1" : set(["x1", "x2"]), "x2" : set()}, 
        #     "labels" : {"x1": set(["p"]), "x2" : set(["q"])}
        # }
        self.expected_quotient = {
            "s1": {"label": {"p"}, "targets": ["s1", "s3"]},
            "s3": {"label": {"q"}, "targets": []}
        }

class AdvancedGauntletModel:
    def __init__(self):
        # 7 States with 3 distinct initial labels
        self.states = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        
        # Dense forward relations (highly overlapping futures)
        self.relations = {
            "s1": ["s3", "s5"],        # Splits evenly across distinct q-blocks
            "s2": ["s3", "s4"],        # Points to the SAME q-block s3, but a different second one
            "s3": ["s6", "s7"],        # Connects q-world to the r-world
            "s4": ["s6"],              # Connects to r-world differently than s3
            "s5": ["s1", "s2"],        # Cyclic feedback loop back to the p-world!
            "s6": ["s6", "s7"],        # r-state with a self-loop and a cross-edge
            "s7": ["s7"]               # Pure self-loop dead-end
        }
        
        # Rigorously mapped reverse relations (Preimages)
        self.rev_relations = {
            "s1": ["s5"],
            "s2": ["s5"],
            "s3": ["s1", "s2"],        # Critical shared pivot for p-states
            "s4": ["s2"],
            "s5": ["s1"],
            "s6": ["s3", "s4", "s6"],  # Dense preimage block
            "s7": ["s3", "s6", "s7"]
        }
        
        # Initial label grouping (Atomic Propositions)
        self.labels = {
            "s1": {"p"},
            "s2": {"p"},
            "s3": {"q"},
            "s4": {"q"},
            "s5": {"q"},
            "s6": {"r"},
            "s7": {"r"}
        }
        
        # EXPECTED STABLE PARTITION:
        self.expected = set(frozenset(block) for block in [
            ["s1"], 
            ["s2"], 
            ["s3", "s4"], 
            ["s5"], 
            ["s6", "s7"]
        ])

        # used for testing the quotient construction
        self.expected_quotient = {
            "s1": {"label": {"p"}, "targets": ["s3", "s5"]},
            "s2": {"label": {"p"}, "targets": ["s3"]},
            "s3": {"label": {"q"}, "targets": ["s6"]},
            "s5": {"label": {"q"}, "targets": ["s1", "s2"]},
            "s6": {"label": {"r"}, "targets": ["s6", "s7"]}
        }

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
            
        # validation Logic Partition
        if algo_output == model.expected:
            print(f"\nSUCCESS! Successfully shrank to {len(model.expected)} stable blocks correctly!")
        elif block_idx == len(model.expected):
            print(f"\nCorrect number partitions partitions but wrong classification...")
            print("Incorrect splitting: check if a block was split incorrectly or if the cleanup didn't reset counts.")
            return
        else: 
            print(f"\nFAILED: Expected exactly {len(model.expected)} collapsed blocks, but got {block_idx}.")  
            return 
        
        # testing Quotient Construction 
        print(f"\n---- Executing Quotient Construction ---")

        # test for Quotient Reconstruction
        macro_states, quo_relations, quo_labels, mapping, bisim_states = minimizer.quotient_construction(final_Q)

        # print quotient model
        for macro_state in macro_states:
            print(f"Macro state {macro_state}:\n\t Relations:{quo_relations[macro_state]} \n\tLabels: {quo_labels[macro_state]}")
        
        print(f"\nMapping Macro -> Original: \n{bisim_states}")

        for rep_state, expected_data in model.expected_quotient.items():

            # Find what 'xi' token your engine assigned to this original state
            my_macro_token = mapping[rep_state]
            
            # A. Check Labels
            if quo_labels[my_macro_token] != expected_data["label"]:
                print(f"FAILED: Label mismatch at {my_macro_token} (rep: {rep_state})")
                return
                
            # B. Check Relations
            # Translate your hardcoded original targets into whatever 'xi' tokens they became
            expected_tokens = set(mapping[target] for target in expected_data["targets"])
            generated_tokens = set(quo_relations[my_macro_token])
            
            if expected_tokens != generated_tokens:
                print(f"FAILED: Structural edge mismatch at {my_macro_token} (rep: {rep_state})")
                print(f"Expected: {expected_tokens}")
                print(f"Generated: {generated_tokens}")
                return
                
        print("\n\nSUCCESS! Quotient constructio is structurally sound.")

    except Exception as e:
        print(f"\n💥 CRASH: The engine threw an exception during refinement.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
   print("Testing with simple model\n")
   run_reduction_test(MockModel())

   print("\n\nTesting with complex model\n")
   run_reduction_test(AdvancedGauntletModel())
