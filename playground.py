import time 
import math
import numpy as np
from rl_agent import XP_Replay 
from mentalmodel import KripkeMM, KMMcompare

class GridWorld:
    """Simple implementation of GridWorld for testing"""
    def __init__(self, size, ndeathpits, seed=8):
        self.rng = np.random.default_rng(seed=seed)
        self.grid_size = size
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.action_map = {0 : "Up", 1 : "Down", 2: "Right", 3: "Left"}
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        # self.grid = np.zeros((height, width))
        self.death_pits = self.gen_terminal(n=ndeathpits)
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        # return self.agent_pos

    def gen_terminal(self, n, padding=1):
        """
        Generate n death pits spread across the grid using a Jittered Grid approach,
        guaranteeing they do not touch the outer edges.
        """
        death_pits = []

        # Define the inner usable area to completely avoid edges
        usable_start = padding
        usable_end = self.grid_size - padding 
        usable_size = usable_end - usable_start

        if usable_size <= 0:
            raise ValueError("Grid size is too small for the requested padding.")

        # Determine grid split for jittering
        bins_per_side = int(np.ceil(np.sqrt(n)))
        bin_size = usable_size / bins_per_side

        # Dynamic minimal distance calculation based on the usable zone
        min_dist = math.ceil(0.6 * (usable_size / np.sqrt(n)))

        # Navigate through the bins 
        for r_bin in range(bins_per_side):
            for c_bin in range(bins_per_side):
                if len(death_pits) >= n: 
                    break

                attempts = 0 
                while attempts < 100:
                    attempts += 1
                    
                    r_low = int(usable_start + r_bin * bin_size)
                    r_high = int(usable_start + (r_bin + 1) * bin_size)
                    c_low = int(usable_start + c_bin * bin_size)
                    c_high = int(usable_start + (c_bin + 1) * bin_size)

                    r_high = max(r_low + 1, min(r_high, usable_end))
                    c_high = max(c_low + 1, min(c_high, usable_end))

                    r = self.rng.integers(r_low, r_high)
                    c = self.rng.integers(c_low, c_high)

                    too_close = any(((r - dp[0])**2 + (c - dp[1])**2)**0.5 < min_dist for dp in death_pits)

                    if not too_close:
                        if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                            death_pits.append((r, c))
                            break

        self.rng.shuffle(death_pits)
        return death_pits[:n]
    
    # def gen_terminal(self, n, min_dist=3):
    #     """
    #     Generate n deathe that are evenly dist across the grid
    #     using Jittered Grid and min dist.
    #     -> Constrained Jittered Grid
    #     """
    #     death_pits = []

    #     # upper bound number of bins to account for odd number of TS
    #     bins_per_side = int(np.ceil(np.sqrt(n)))
    #     bin_size = self.grid_size // bins_per_side

    #     # compute minimal distance (accoring to theoretical values maximal spread of 50-75%)
    #     min_dist = 0.6 * (self.grid_size / np.sqrt(n))

    #     # navigate through the bins 
    #     for r_bin in range(bins_per_side):
    #         for c_bin in range(bins_per_side):

    #             while True:
    #                 # generte coordinate on bin
    #                 r = self.rng.integers(r_bin * bin_size, (r_bin + 1) * bin_size)
    #                 c = self.rng.integers(c_bin * bin_size, (c_bin + 1) * bin_size)

    #                 too_close = any(((r - dp[0])**2 + (c -dp[1])**2)**0.5 < min_dist for dp
    #                                 in death_pits)

    #                 # check if state matches requirements
    #                 if not too_close:
    #                     if (r, c) != self.start_pos and (r, c) != self.goal_pos:
    #                         # add state 
    #                         death_pits.append((r, c))
    #                         break
                
    #     # shuffle terminal state and then choose
    #     self.rng.shuffle(death_pits)
    #     return death_pits[:n]
    
    def step(self, action_idx):
        action = self.actions[action_idx]
        action_name = self.action_map[action_idx]

        #  coordinate change 
        x, y = self.agent_pos
        nx, ny = x + action[0], y + action[1]

        # default return values
        reward = 0 
        done = False


        # check if action is action is valid
        if  0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            self.agent_pos = (nx, ny)

            # check if agent reached a terminal state  
            if self.agent_pos in self.death_pits:
                # death pit
                reward = -10
                done = True 
            elif self.agent_pos == self.goal_pos:
                # goal state 
                reward = 10
                done = True 

            return ((x, y), action_idx, self.agent_pos, reward, done)
        else:
            # no location update + punishment for illegal action
            reward = -1 
            return (self.agent_pos, action_idx, self.agent_pos, reward, done)

    def render(self):
        """Simple text base rendering"""
        symbols = {
            "agent": "🤖",
            "goal": "🏁",
            "death": "☠️",
            "wall": "🧱",
            "empty": "·" }

        for x in reversed(range(self.grid_size)):
            row = ""
            for y in range(self.grid_size):
                position = (x, y)
                if position == self.agent_pos:
                    row += "A "
                elif position == self.goal_pos:
                    row += "$ "
                elif position in self.death_pits:
                    row += "# "
                else:
                    row += ". "
            print(row)
        print()

def sample_episode(envir, policy):
    pass


def experiment(env, model, epochs, visualize=False,
               render=False, view_env=False, compare=False, **kwargs):
    
    # memory = XP_Replay(1000)
    model = model

    # display GW env
    if view_env:
        env.render()
        input()

    # simple statistics
    e_lengths = []
    struct_size = []
    meta_size = {"bisim": []} if not compare else {"bisim": [], "k_bisim" : []}

    if render:
        print("\nExperiment starts")


    # collect data for n epoch
    for i in range(1, epochs + 1):

        # episode len counter
        episode_len = 0 
        env.reset()

        # act until reaching some terminal state 
        while True:
            # select random action 
            action = np.random.choice(len(env.actions))
            # get and save experience 
            # (s, a, s', r, done)
            xp = env.step(action)
            # memory.push(xp)

            # increase counter 
            episode_len += 1

            # render grid
            if render:
                env.render()
                time.sleep(1)

            # generate mental model incrementally
            model.update_structure([xp])

            # end epoch if agend reaches terminal state 
            if xp[-1] == True:
                break

        # display current kripke structure 
        if visualize:
            print(f"Kripke structure on episode {i}, episode with {episode_len} transitions")
            model.struct.visualize()
            input()

        # generate abstract model 
        if i % 3 == 0 and i != 0:
            # generate based on bisim
            if compare == True:
                model.generate_model()
                model.generate_model(**kwargs)
            else:
                model.generate_model(**kwargs)

            if visualize:
                print(f"Compressed model at iter {i}, total nodes in structure: {len(model.struct.states)}")
                if compare and getattr(model, "abst_k", None):
                    print(f"Standard Bisimulation ({len(model.abst.states)} states)")
                    model.visualize(model.abst, title="Bisim")
                    input()
                    print(f"Visualizing k-Bisimulation({len(model.abst_k.states)} states)")
                    model.visualize(model.abst_k, title="k-Bisim")
                else:
                    model.visualize(model.abst)
                
                # give time to view generated models
                input()
        
        # update statistics
        e_lengths.append(episode_len)
        struct_size.append(len(model.struct.states))

        # check if model has been generated yet 
        if not model.abst:
            
            meta_size["bisim"].append(0)

            if compare:
                meta_size["k_bisim"].append(0)        
        else:
            meta_size["bisim"].append(len(model.abst.states))

            if compare:
                meta_size["k_bisim"].append(len(model.abst_k.states))   

    return e_lengths, struct_size, meta_size 


def main():
    # inti env
    env = GridWorld(6, 3)

    # experiment info
    n = int(input("Number of epochs? "))
   
    if input("Exp. details? ").lower().strip() == "y":
        # render = input("Render experiment (y/n)? ").lower().strip() == "y"

        # display GW env
        view_env = input("Display GW env. (beggining state)? ").lower().strip() == "y"
        visualize = input("Visuaize model? ").lower().strip() == "y"
    
    else: 
        render, view_env, visualize = False, False, False
        visualize = {"structure" : False}
    
    #compare models 
    compare = input("Compare models? ").lower().strip() == "y"
    
    # select model details 
    if input("Select model parameters? ").lower().strip() == "y":
        k = input("Choose k-depth (yes=int, no=no): ").strip()
        k = int(k) if k.isnumeric() else None
        zone_radious = int(input("Zone radious: ").strip())
    else:
        k = None
        zone_radious = None
    

    # init model
    if compare:
        model = KMMcompare(n_action=len(env.actions), complex_lables=True, zone_radious=zone_radious)
    else:
        model = KripkeMM(n_action=len(env.actions), complex_labels=True, zone_radious=zone_radious)

    # run experiment 
    experiment(env, model, n, visualize=True, render=False, view_env=view_env, compare=compare, k=k)

# program execution 
if __name__ == "__main__":
    main()