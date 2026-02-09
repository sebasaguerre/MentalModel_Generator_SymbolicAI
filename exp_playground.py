import numpy as np

class GridWorld:
    """Simple implementation of GridWorld for testing"""
    def __init__(self, height=10, width=10, seed=44):
        rng = np.random.default_rng(seed=seed)
        self.height = height
        self.width = width
        self.start_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)
        self.grid = np.zeros((height, width))
        self.death_pits = self.gen_death_pits(n=3)
        self.reset()

        def reset(self):
            self.agent_pos = list(self.start_pos)
            return tuple(self.agent_pos)
        
        def gen_death_pits(self, n):
            """Generate n death pits"""
            return tuple(rng.integers(low=0, high=9, size=2) for _ in range(n))
        
        def step(self, action):
            #  coordinate change 
            dx, dy = action
            x, y = self.agent_pos
            nx, ny = x + dx, y + dy

            # default return values
            reward = 0 
            done = False

            # check if action is action is valid
            if nx in range(self.width) and ny in range(self.width):
                self.agent_pos = (nx, ny)

                # check if agent fell into death pit or reached the goal 
                if self.agent_pos in self.death_pits:
                    reward = -10
                    done = True 
                elif self.agent_pos == self.goal_pos:
                    reward = 10
                    done = True 

                return ((x, y), action, self.agent_pos, reward, done)
            else:
                # no location update + punishment for illegal action
                reward = -1 
                return (self.agent_pos, action, self.agent_pos, reward, done)

        def render(self):
            pass


def main():
    # init grid world
    gw_agent = GridWorld()
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    replaybuffer = []
    n = input("Nuumbr of epochs? ")

    # collect data for n epoch
    for _ in range(n):
        # act unti end of task
        while True:
            
            # select random action & check if 
            action = actions[np.random.choice(len(actions))]
            xp = gw_agent.step(action)
            # save xp 
            replaybuffer.append(xp)

            # end epoch if agend reaches terminal state 
            if xp[-1] == True:
                break
        
        # reset 