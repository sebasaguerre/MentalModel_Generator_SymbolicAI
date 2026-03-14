import numpy as np 

class XP_Replay:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = []
        self.rng = np.random.default_rng(seed=33)

    def push(self, xp):
        """Add experience to experiecne replay"""

        # check for capacity limits 
        if len(self.memory) < self.capacity: 
            self.memory.append(xp)
        else: 
            # replace oldest experience with a new one
            self.memory.remove(self.memory[0])
            self.memory.append(xp)
    
    def sample_xp(self, sample_size):
        """Randomly sample experience for training"""

        sample = np.random.choice(self.memory, size=sample_size, replace=True)

        return sample 
    
    def __len__(self):
        return len(self.memory)

class Policy:
    def __init__(self, Q, state, rand=False):
        self.Q = Q
        self.state = state

    def sample_action(self, obs):
        pass