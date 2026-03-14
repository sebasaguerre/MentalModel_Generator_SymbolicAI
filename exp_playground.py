import numpy as np
from rl_agent import XP_Replay 
from mentalmodel import ModelStructure

class GridWorld:
    """Simple implementation of GridWorld for testing"""
    def __init__(self, size, ndeathpits, seed=10):
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
        return self.agent_pos
    
    def gen_terminal(self, n, min_dist=3):
        """
        Generate n deathe that are evenly dist across the grid
        using Jittered Grid and min dist.
        -> Constrained Jittered Grid
        """
        death_pits = []

        # upper bound number of bins to account for odd number of TS
        bins_per_side = int(np.ceil(np.sqrt(n)))
        bin_size = self.grid_size // bins_per_side

        # compute minimal distance (accoring to theoretical values maximal spread of 50-75%)
        min_dist = 0.6 * (self.grid_size / np.sqrt(n))

        # navigate through the bins 
        for r_bin in range(bins_per_side):
            for c_bin in range(bins_per_side):

                while True:
                    # generte coordinate on bin
                    r = self.rng.integers(r_bin * bin_size, (r_bin + 1) * bin_size)
                    c = self.rng.integers(c_bin * bin_size, (c_bin + 1) * bin_size)

                    too_close = any(((r - dp[0])**2 + (c -dp[1])**2)**0.5 < min_dist for dp
                                    in death_pits)

                    # check if state matches requirements
                    if not too_close:
                        if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                            # add state 
                            death_pits.append((r, c))
                            break
                
        # shuffle terminal state and then choose
        self.rng.shuffle(death_pits)
        return death_pits[:n]
    
    def step(self, action_idx):
        action = self.actions[action_idx]

        #  coordinate change 
        x, y = self.agent_pos
        nx, ny = x + action[0], y + action[1]

        # default return values
        reward = 0 
        done = False


        # check if action is action is valid
        if  0<= nx < self.grid_size and 0 <= ny < self.grid_size:
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


def main():

    # experiment info
    n = int(input("Number of epochs? "))
    while True:
        
        render = input("Render experiment (y/n)? ").lower().strip()
        if render in ["n", "y"]:
            render = (render == "y")
            break
        print("Please enter 'n' or 'y'.")

    # init set-up
    env = GridWorld(6, 3)
    memory = XP_Replay(1000)
    kripke = ModelStructure()

    # display GW env
    if input("Display GW env? ").lower().strip() == "y":
        env.render()
        input()

    # simple statistics
    e_lengths = []

    print("\nExperiment starts")
    # collect data for n epoch
    for i in range(n):

        # episode len counter
        episode_len = 0 

        # act until reaching some terminal state 
        while True:
            # select random action 
            action = np.random.choice(len(env.actions))
            # get and save experience 
            xp = env.step(action)
            memory.push(xp)

            # increase counter 
            episode_len += 1

            # render grid
            if render:
                env.render()

            # generate mental model incrementally
            kripke.generate([xp])

            # end epoch if agend reaches terminal state 
            if xp[-1] == True:
                break

        # update statistics
        e_lengths.append(episode_len)

        # display current kripke structure 
        print(f"Kripke structure on episode {i}, episode with {episode_len} transitions")
        kripke.visualize()
        input()


# program execution 
main()