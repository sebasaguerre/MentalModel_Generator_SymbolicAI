    
def update_formulas(self, formula, goals, ignore=set()):
    """
    Refine formulas for more expresivity and taylor to objective
    """
    #NOTE: for later: consider the possibility to also track rank 1 goals for U & G
    #NOTE: this does not consider non-determinisitic systems: take care later

    # deconstruct parent fomrula: state prop + action formulas
    parent_props, action_formulas = self.decouple_state_formula(formula)
    updated_formula = deque()
    
    goal_props = tuple(rank_prop for rank_props in goals.values() for rank_prop in rank_props)

    # update action formulas
    for _ , af in enumerate(action_formulas):

        # get formula details
        temp_operat = type(af)
        action = af.action
        quant = af.quant
        child_props, nxt_formulas = self.decouple_state_formula(af.f)

        # track objective props across paths for (per action)
        obj_props = defaultdict(list)
        
        # check if next prop are in objective, if primary => nXt, else store
        for rank, objectives in goals.items():
            child_obj_props = set(prop for prop in child_props if prop in objectives)
            # if rank 1 objectives then update immediately 
            if child_obj_props and rank == 1:
                updated_formula.append(X(self.wrap_elems(child_obj_props), action, quant))
                # if immediate update (?)
                continue 
            else:
                obj_props["child"][rank].append(child_obj_props)

        # collect the operators that the child successors have for formula update
        nxt_ops = dict()

        # iterate over temporal subformulas: successors of child
        for i, nxt_elem in enumerate(nxt_formulas):

            # nxt fromula relevant details
            nxt_operat = type(nxt_elem)
            nxt_quant = nxt_elem.quant

            # NOTE: this might not be useful, possibly eliminate 
            # chekc dim of temporal fomula 
            if not (until := isinstance(nxt_elem, U)):
                props = (self.get_props(nxt_elem.f), )
            else:
                props = (self.get_props(nxt_elem.f), self.get(nxt_elem.g))

            # check if props in trajectory are part of objective (per rank)
            for rank, objectives in goals.items():
                nxt_obj_props = set(prop for prop in props[0] if prop in objectives)
                obj_props["nxt"][rank].append(nxt_obj_props)
                # if props[1] in objectives:
                #     nxt_obj_props = set(prop for prop in props[1] if prop in objectives)    

            # store operators, props and quant
            nxt_ops[(i, nxt_operat)] = (nxt_quant, props)

        #### Action formula Update ####

        # check for the possible operators we can merge into 
        merge_options = set().intersection(
            {self.merge_options(operat[1]) for operat in nxt_operat.keys()}
            )
        
        # extract the qunatifier for merge update 
        merge_quant = "E" if any("E" in tpl for tpl in nxt_ops.values()) else "A"

        # all the information 
        update_info = {"action": action, "options": merge_options, "merge_quant": merge_quant}

        # update action formula
        # NOTE: maybe there is a more efficient way of doing this 
        self.update_action_formula(updated_formula, update_info, goals, obj_props, parent_props)
        continue
            
            # intersection of paths per rank for all nxt paths 
            exi_paths = {
                        rank: [
                            set.intersection(parent_props, *obj_props["child"][rank], path_set)
                            for path_set in obj_props["nxt"][rank]
                        ]
                        for rank in obj_props["nxt"].keys()
                        }
            
            # set of objc props that are true across all paths after action was taken 
            uni_paths = {rank: set.intersection(*exi_paths[rank]) for rank in exi_paths}

   
            # check for AG
            if any(uni_paths.values()):
                updated_formula.append(G(self.wrap_elems(uni_paths), action, quant="A"))
                continue
            
            # parent to child path intersection and nxt goal props that hold in all nxt
            child_uni_paths = set.intersection(parent_props, *obj_props["child"])
            uni_nxt_props = set.intesection(*obj_props["nxt"].values())           # box phi 

            # checking for AU
            if child_uni_paths & uni_nxt_props: 
                updated_formula.append(U(self.wrap_elems(child_uni_paths),
                                         self.wrap_elems(uni_nxt_props), action, quant="A"))
                continue 
            
           # check for AX
            for rank_props in obj_props["child"].items():
                if rank_props:
                    updated_formula.append(X(self.wrap_elems(rank_props), action, quant="A"))

            # existing variables per rank in nxt successors 
            nxt_exi_props = {rank : set.union(*obj_props["nxt"][rank]) for rank in obj_props["nxt"].keys()}

            # check for EG and EU: heuristic based for ranks
            for rank, prop_list in exi_paths.items():
                if rank_exi_path := [props for props in prop_list if props]:
                    updated_formula.append(G(self.wrap_elems(rank_exi_path[0], conjunct=False),
                                            action, quant="A"))
                    continue 
                elif child_uni_paths & nxt_exi_props[rank]:
                    updated_formula.append(U(self.wrap_elems(child_uni_paths),
                                             self.wrap_elems(next(iter(nxt_exi_props[rank]))),
                                             action, quant="E"))

            # check for AF
            if uni_nxt_props:
                updated_formula.append(F(self.wrap_elems(uni_nxt_props), action, quant="A"))
                continue

            # check for EF 
            for rank, prop_list in obj_props["nxt"].items():
                    if exi_props := [props for props in prop_list if props]:
                        updated_formula.append(F(self.wrap_elems(exi_props[0]),
                                                action, quant="E"))
                        continue 


        # prepend state props to new formula then return ocnjunction
        updated_formula.appendleft(formula.conjuncts[0])

        return AND(updated_formula)
    
    
def update_action_formula(self, update_ctx, goal_props):
    "Iterative checking formula update for the most meaningul update"

    # # existential path props & universal path props for action
    # exi_paths = [obj_props["child"].intersection(nxt_props) for nxt_props in obj_props["nxt"]]
    # uni_paths = set.intersection(*exi_paths)

    # # existential and universal props at nxt states
    # exi_nxt = set.union(*obj_props["nxt"])
    # uni_nxt = set.intersection(*obj_props["nxt"])

    for g_prop in goal_props:

        # skip g_props not present 
        if not ((g_prop in update_ctx.exi_nxt) or (g_prop in update_ctx.child_props)):
            continue

        # iterate over possible oprators in order of importance
        for operator in self.op_imporance:

            if not (operator in update_ctx.op_options):
                continue 

            # check if operator fits 
            if self.check_update(operator, update_ctx, g_prop):
                return 


    # # iterate by rank to get most meaningful proposition
    # for rank, rank_props in goals.items():

    #     # iterate over sub_goals within the rank in order of importance
    #     for rank_prop in rank_props:

    #         # check if prop is prent in at least one child successor:
    #         if  any(rank_prop in nxt_prop for nxt_prop in obj_props["nxt"][rank]):

    #             # any(rank_prop in path for path in exi_paths[rank])
    #             # iterate over operatorce in order of imporance & check only those that are possible 
    #             for operator in self.op_imporance:

    #                 # check if operator fit update:
    #                 if operator in update_info["options"]:
                        
    #                     # check if update is possible, if possible check for other operators
    #                     self.check_update(operator, rank, rank_prop, exi_paths, uni_paths)

    # if no meaningful merge was possible we choose the most relevant formula from before that still holds