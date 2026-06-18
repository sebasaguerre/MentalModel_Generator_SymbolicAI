
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


      
    def generate_labels(self):
        """
        Generate higher order labels lables that change dynamically.
        Labels like: 
            - Level of entropy
            - Proximity to goal 
            - Proximity to terminal states 
        """
        # optimize to avoid lookups
        relations = self.abst.labels
        labels = self.abst.labels 
        n_actions = self.struct.n_actions
        

        for s in relations.keys():

            s_labels = labels[s]

            bounds = any(label == "bound" for label in s_labels)
            terminal = any(label in ["TS", "Goal"] for label in s_labels)
            
           # compute proportion of actions explored 
            act_exp = len(s_labels.values()) / n_actions

            # set exploration quotient 
            if act_exp < 1.0:
                entropy = f"E_{'high' if act_exp <= 0.33 else 'mid' if act_exp <= 0.66 else 'low'}"
            else:
                entropy = f"E_none"

            # add dynamic entropy label and zones-labels 
            if not terminal:
                
                # check if state has an entopy level 
                current_entropy = [label for label in labels[s] if label.startswith("E")]
                
                # add or update entropy
                if not current_entropy:
                    self.abst.labels[s].add(entropy)
                elif current_entropy[0] != entropy:
                    self.abst.labels[s].remove(current_entropy[0])
                    self.abst.labels[s].add(entropy)

                # add zone label if applicable 
                for zone, label in self.zones.items():
                    if zone not in labels[s]:
                        if self.within_radious_dfs(s, label, self.zone_radious):
                            self.abst.labels[s].add(zone) 
