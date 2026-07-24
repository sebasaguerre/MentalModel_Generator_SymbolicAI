from collections import defaultdict
import random 

values = range(1, 6)
size = random.randint(min(values), max(values) - 2)

x = defaultdict(lambda: defaultdict(list))

for level in ["top", "bellow"]:    
    for rank in range(1,3): 
        if level == "top" and rank == 1: 
            continue 
        else:
            n = 4 if level == "bellow" else 1
            for _ in range(n):
                x[level][rank].append(set(random.sample(values, random.randint(1, 5))))


for level, rank, val in ((level, rank, v) for level, dic in x.items() for rank, v in dic.items()):
    print(level, rank, val)

print()

intersect_per_rank = defaultdict(set)
union_per_rank = defaultdict(set)
    
# extract path obj props 
for level, rank_dict in x.items():
    for rank, prop_list in rank_dict.items():

        # compute intersection
        if not intersect_per_rank[rank]:
            intersect_per_rank[rank] = set.intersection(*prop_list)
        else:
            intersect_per_rank[rank] &= set.intersection(*prop_list)

        # compute union per rank
        union_per_rank[rank] |= set.union(*prop_list)

print(f"Intersections per rank: {intersect_per_rank}")
print(f"Union per rank: {union_per_rank}")



###### graveyard 


         # uni_path_props = set.intersection(parent_props, *obj_props["child"], *obj_props['nxt'])

            # # paths of union
            # union_path_props = set.union(parent_props, *obj_props["child"], *obj_props['nxt']) - uni_path_props
             # # existential paths
            # union_path_props = set.union(parent_props, *obj_props["child"], *obj_props['nxt']) - uni_path_props


   # # objective props as primary driver
            # intersect_per_rank = defaultdict(set)
            # union_per_rank = defaultdict(set)
            
            # # extract path obj props 
            # for level, rank_dict in obj_props.items():
            #     for rank, prop_list in rank_dict.items():

            #         # compute intersection
            #         if not intersect_per_rank[rank]:
            #             intersect_per_rank[rank] = set.intersection(*prop_list)
            #         else:
            #             intersect_per_rank[rank] &= set.intersection(*prop_list)

            #         # compute union per rank
            #         intersect_per_rank[rank] |= set.union(*prop_list)
    
            # path sets 
            uni_path_props = set.intersection(parent_props, *obj_props["child"], *obj_props['nxt'])
            exi_path_props = set.union(parent_props, *obj_props["child"], *obj_props['nxt']) - uni_path_props
            uni_nxt_goal = set.intesection(*obj_props["nxt"].values())
            exi_nxt_goal = set.union(*obj_props["nxt"].values()) - uni_nxt_goal

            # Until or Global formula update: path check
            if match_child:
                # Global formula update: checking match_nxt
                if match_nxt:
                    
                    # NOTE: may be restrictive to constrain on objectives
                    # global path constrained on objectives 
                    if global_path_props :=  set.intersection(*match_nxt, *path_goal_props):

                        # quantifier attribution, checking if all paths match 
                        if len(match_nxt["full"]) == len(nxt_formulas):
                            updated_formula.append(G(self.wrap_elems(match_child), action=action, quant="A"))
                        else:
                            updated_formula.append(G(self.wrap_elems(match_child), action=action, quant="E"))
                        continue
                
                # Unitl formula update 
                else: 
                    nxt_obj_props = obj_props['nxt']
                    intersect_nxt_goal = set.intersection(obj)
                # here we 

            # check for other operators based on objectives 
            for rank, props in obj_props:
                # check if props of current rank have been found 
                if props:
                    unique_props = set.union(*props)
                    shared_props = set.intersection(*props)

                    # check for quantifiers 
                    if len(props) == len(nxt_formulas):
                        updated_formula.append(F(self.wrap_elems(shared_props), action=action, quant="A"))
                    else:
                        updated_formula.append(F(self.wrap_elems(shared_props), action=action, quant="E"))
                    continue

            new_af = []