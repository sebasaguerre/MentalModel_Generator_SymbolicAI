
### long live dead functions

def gen_death_pits(rng, grid_size, n, min_dist=3):
    """
    Generate n deathe pits which are properly distributed
    using a simplified poisson disk method (Blue Noise)
    """
    death_pits = []

    # minimum distanxce aiming for 50%-75% theoretical maximum spacing 
    min_dist = 0.6 * (grid_size / np.sqrt(n))

    print(min_dist)

    while len(death_pits) < n:
        new_position = (rng.integers(0, grid_size), rng.integers(0, grid_size))

        # check if new position is too close to previous points
        distances = [np.linalg.norm(np.array(new_position) - np.array(dp)) for dp in death_pits]
        print(distances)
        too_close = any(dist < min_dist for dist in distances)
        
        if not too_close:
            if new_position != start_pos and new_position != goal_pos:
                death_pits.append(new_position)

    return death_pits

# NOTE: this function might not be useful anymore... premap computed at generation
def get_premap(self):
    self.premap = dict()

    # loop over the entire relation structure 
    for world, successors in self.edges.items():
        for s in successors:
            self.preimage[s].add(world)
