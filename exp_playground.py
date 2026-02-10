import numpy as np

class GridWorld:
    """Simple implementation of GridWorld for testing"""
    def __init__(self, height=10, width=10, seed=30):
        self.rng = np.random.default_rng(seed=seed)
        self.height = height
        self.width = width
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.start_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)
        # self.grid = np.zeros((height, width))
        self.death_pits = self.gen_death_pits(n=3)
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos
    
    def gen_death_pits(self, n):
        """Generate n death pits"""
        return [tuple(self.rng.integers(low=0, high=9, size=2)) for _ in range(n)]
    
    def step(self, action_idx):
        action = self.actions[action_idx]
        #  coordinate change 
        x, y = self.agent_pos
        nx, ny = x + action[0], y + action[1]

        # default return values
        reward = 0 
        done = False

        # check if action is action is valid
        if  0<= nx < self.width and 0 <= ny < self.height:
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

            return ((x, y), action, self.agent_pos, reward, done)
        else:
            # no location update + punishment for illegal action
            reward = -1 
            return (self.agent_pos, action, self.agent_pos, reward, done)

    def render(self):
        """Simple text base rendering"""
        symbols = {
            "agent": "🤖",
            "goal": "🏁",
            "death": "☠️",
            "wall": "🧱",
            "empty": "·" }

        for x in reversed(range(self.width)):
            row = ""
            for y in range(self.height):
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

class Model:
    """MM generator"""
    def __init__(self, tref=False):
        self.nodes = set()
        self.edges = set()
    
    def generate(self, data):
        # extract states
        current_nodes = set()
        current_edges = set()

        # extract nodes and edges form current batch
        for batch in data: 
            for xp in batch:
                edge = (xp[0], xp[1])
                current_nodes.update([edge[0], edge[1]])
                current_nodes.add(edge)
        


def main():
    # init grid world
    gw = GridWorld()
    replaybuffer = []

    # experiment info
    n = int(input("Number of epochs? "))
    render = (input("Render experiment (y/n)? ").lower() == "y")

    # collect data for n epoch
    for _ in range(n):
        # act until reaching some terminal state 
        while True:
            # select random action 
            action = np.random.choice(len(gw.actions))
            # get and save experience 
            xp = gw.step(action)
            replaybuffer.append(xp)

            # render grid
            if render:
                gw.render()

            # end epoch if agend reaches terminal state 
            if xp[-1] == True:
                break
        
        # reset agent for new epoch
        gw.reset()


# program execution 
main()