import gurobipy as gp
from gurobipy import GRB
from temporal_decoupling import TemporalDecoupling, obtain_proof
from temporal_network import SimpleTemporalConstraint, SimpleContingentTemporalConstraint
from dc_milp import add_dc_constraints_to_model

# NOTE: We handle numeric instability by rounding to 4 decimal points
MAX_NUMERIC_BOUND = 100000
ROUND_TO = 4

# Solver Options
FIX_CONTINGENT_BOUNDS = True
DISABLE_CUTS = True

#  Objective functions
NONE = 0
MIN_TIME_SPAN = 1
MAX_FLEXIBILITY = 2
MAX_FLEXIBILITY_NAIVE = 3
MAX_FLEXIBILITY_NEG_CKJ = 4
MIN_LB_TIME_SPAN = 5
MIN_LB_UB_TIME_SPAN = 6
MIN_BIJ = 7

def decouple_MILP(mastnu, shared_events=None, conflicts=[], objective=NONE, timeout=None, outputIIS=False, encode_agent_networks=False):

    ref_event = mastnu.ref_event
    event2agent = mastnu.event_to_agent
    ext_conts = mastnu.external_contingents
    ext_reqs = mastnu.external_requirements
    if shared_events is None:
        shared_events = mastnu.get_shared_events()
    num_agents = mastnu.num_agents
    H = max(len(ext_conts), num_agents - 2)

    try:
        # Create a new model
        m = gp.Model("Decoupling")
        m.setParam('OutputFlag', False)
        if timeout is not None:
            m.setParam('TimeLimit', timeout)
        if DISABLE_CUTS:
            m.setParam('Cuts', 0)

        # Create variables
        l, u, b, z, c, h = add_variables_to_model(m, ref_event, shared_events, event2agent, ext_conts, H, encode_agent_networks=encode_agent_networks)

        # Add external constraints
        add_constraints_to_model(m, l, u, b, z, c, h, ext_reqs, ext_conts, H)
        # print("Global decoupling constraints encoded.")

        # Add other constraints that do not affect correctness
        add_other_constraints_to_model(m, l, u, b, z, c, h, event2agent, shared_events, ref_event)

        # Encode conflict resolution
        if conflicts:
            add_conflicts_to_model(m, conflicts, l, u, c)
            # print("Conflicts added.")

        # Encode DC for agent networks if using centralized MILP
        if encode_agent_networks:
            add_dc_constraints_to_model(m, mastnu.agent2network, u, c, event2agent)
            # print("Agent DC constraints encoded.")

        # Set objectives
        add_objective(m, l, u, c, b, z, h, ref_event, objective)

        # Optimize model
        m.optimize()

        # print("Num vars: {}".format(m.numVars))
        # print("Num constraints: {}".format(m.numConstrs))

        # Write the LP model
        # m.write('model.lp')

        if m.status == GRB.Status.INFEASIBLE:
            # print('No feasible solution. See infeasible.ilp for conflict.')
            if outputIIS:
                m.computeIIS()
                m.write("infeasible.ilp")
            return None
        elif m.status == GRB.Status.TIME_LIMIT:
            print("Reached time limit.")
            raise TimeoutError('Timeout')
        else:
            assert(m.status == GRB.Status.OPTIMAL)
            # print('Solution found.')

            #  obj_val = None
            #  if not objective == NONE:
            #      obj_val = m.objVal
            #  decoupling = TemporalDecoupling.from_raw_data(compile_raw_data(u, z, c, b), ext_reqs, ext_conts, event2agent, obj_val)

            decoupling = compile_temporal_decoupling(l, u, c, event2agent)
            if not objective == NONE:
                decoupling.objective_value = m.objVal
            decoupling.add_raw_data(compile_raw_data(u, z, c, b))
            return decoupling

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

def add_variables_to_model(m, ref_event, events, event2agent, ext_conts, H, encode_agent_networks=False):
    l, u = create_lij_uij(m, events)
    b, h = create_bij_hij(m, events, ext_conts, event2agent, H)
    z = create_zijkl(m, b, ext_conts, event2agent, ref_event)
    c = create_ckj(m, events, ext_conts, event2agent, ref_event, encode_agent_networks=encode_agent_networks)
    return l, u, b, z, c, h

def create_lij_uij(m, events):
    # (vi, vj) => variable
    l = {}
    # (vi, vj) => variable
    u = {}

    for vi in events:
        for vj in events:
            # Add uij
            if vi == vj:
                uij = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="u({}, {})".format(vi, vj))
            else:
                uij = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="u({}, {})".format(vi, vj))
            u[(vi, vj)] = uij

            # Add lij
            if vi == vj:
                lij = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="l({}, {})".format(vi, vj))
            else:
                lij = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="l({}, {})".format(vi, vj))
            l[(vi, vj)] = lij

    return l, u

def create_bij_hij(m, events, ext_conts, event2agent, H):
    # (vi, vj), where owner(vi) != onwer(vj), and (vi, vj) not part of ext contingent link
    cont_map = {}
    for c in ext_conts:
        cont_map[(c.s, c.e)] = True
        cont_map[(c.e, c.s)] = True

    b = {}
    h = {}
    for vi in events:
        for vj in events:
            # owner(vi) != onwer(vj)
            if event2agent(vi) is not None and event2agent(vj) is not None and event2agent(vi) != event2agent(vj):
                # (vi, vj) not part of ext contingent link
                if (vi, vj) not in cont_map:
                    bij = m.addVar(vtype=GRB.BINARY, name="b({}, {})".format(vi, vj))
                    b[(vi, vj)] = bij
                    hij = m.addVar(lb=0, ub=H, vtype=GRB.INTEGER, name="h({}, {})".format(vi, vj))
                    h[(vi, vj)] = hij
    return b, h

def create_zijkl(m, b, ext_conts, event2agent, ref_event):
    # (vi, vj) is part of bij,
    # (vk, vl) satisfies either vk == vl == ref_event,
    # or satisfies owner(vi) == owner(vk) and (vk, vl) part of contingent
    z = {}

    for (vi, vj) in b:
        # Create zijkl when vk == vl == ref_event
        zijzz = m.addVar(vtype=GRB.BINARY, name="z({}, {}, {}, {})".format(vi, vj, ref_event, ref_event))
        z[(vi, vj, ref_event, ref_event)] = zijzz

        # Create zijkl that goes through contingent links
        for c in ext_conts:
            if event2agent(vi) == event2agent(c.s):
                zijkl = m.addVar(vtype=GRB.BINARY, name="z({}, {}, {}, {})".format(vi, vj, c.s, c.e))
                z[(vi, vj, c.s, c.e)] = zijkl
            if event2agent(vi) == event2agent(c.e):
                zijkl = m.addVar(vtype=GRB.BINARY, name="z({}, {}, {}, {})".format(vi, vj, c.e, c.s))
                z[(vi, vj, c.e, c.s)] = zijkl
    return z

def create_ckj(m, events, ext_conts, event2agent, ref_event, encode_agent_networks=False):
    # (vi, vj) is contingent
    # owner(vk) == owner(vj) and vk != vj

    # If encode-agent-networks = True with centralized MILP formulation,
    # ckj should not be instantiated where vk and vj are both target
    # events for ext_conts.
    # Proprocess-agent-network should take care of creating copy
    # variables for completeness of alg.
    if encode_agent_networks:
        # Record all uncontrollable events from ext_conts
        u_events = {}
        for l in ext_conts:
            u_events[l.e] = True

    c = {}
    for l in ext_conts:
        for vk in events:
            if vk == ref_event or event2agent(vk) == event2agent(l.e):
                if vk != l.e:
                    if not encode_agent_networks or vk not in u_events:
                        ckj = m.addVar(vtype=GRB.BINARY, name="c({}, {}, {})".format(vk, l.e, l.s))
                        c[(vk, l.e, l.s)] = ckj
    return c

def add_constraints_to_model(m, l, u, b, z, c, h, ext_reqs, ext_conts, H):
    """Encode decoupling constraints according Casanova's paper."""

    # Enforce uij = -lji
    for (vi, vj) in l:
        lji = l[(vj, vi)]
        uij = u[(vi, vj)]
        m.addConstr(uij + lji == 0, 'u=-l({},{})'.format(vi, vj))

    # Enforce uij >= lij, non-negative cycle
    for (vi, vj) in l:
        lij = l[(vi, vj)]
        uij = u[(vi, vj)]
        m.addConstr(uij - lij >= 0, 'nonnegcycle({},{})'.format(vi, vj))

    # EQ (9) Bounds on contingent links
    # For (vi, vj) in contingents
    # (lij <= Lij) and (uij >= Uij)
    for link in ext_conts:
        vi = link.s
        vj = link.e
        lij = l[(vi, vj)]
        uij = u[(vi, vj)]
        Lij = link.lb
        Uij = link.ub
        if FIX_CONTINGENT_BOUNDS:
            m.addConstr(lij == Lij, 'contlb({},{})'.format(vi, vj))
            m.addConstr(uij == Uij, 'contub({},{})'.format(vi, vj))
        else:
            m.addConstr(lij <= Lij, 'contlb({},{})'.format(vi, vj))
            m.addConstr(uij >= Uij, 'contub({},{})'.format(vi, vj))

    # Bounds on requirement links
    # For (vi, vj) in requirements
    # (lij >= Lij) and (uij <= Uij)
    for link in ext_reqs:
        vi = link.s
        vj = link.e
        lij = l[(vi, vj)]
        uij = u[(vi, vj)]
        Lij = link.lb
        Uij = link.ub
        if Lij is not None:
            m.addConstr(lij >= Lij, 'reqlb({},{})'.format(vi, vj))
        if Uij is not None:
            m.addConstr(uij <= Uij, 'requb({},{})'.format(vi, vj))

    # EQ (10),(11) Need to decouple external requirement links
    # For (vi, vj) in requirements
    # If Uij, then bij = 1
    # If Lij, then bji = 1
    for link in ext_reqs:
        vi = link.s
        vj = link.e
        Lij = link.lb
        Uij = link.ub
        if Uij is not None:
            bij = b[(vi, vj)]
            m.addConstr(bij == 1, 'decouplerequb({},{})'.format(vi, vj))
        if Lij is not None:
            bji = b[(vj, vi)]
            m.addConstr(bji == 1, 'decouplereqlb({},{})'.format(vj, vi))

    # EQ (12) Decouple if bij = 1
    # For (vi, vj) in bij
    # bij = sum(zijkl)
    for (vi, vj) in b:
        bij = b[(vi, vj)]
        zijkl_sum = []
        for (zi, zj, zk, zl) in z:
            if zi == vi and zj == vj:
                zijkl_sum.append(z[(zi, zj, zk, zl)])
        m.addConstr(bij == gp.quicksum(zijkl_sum), 'bijsumzijkl({},{})'.format(vi, vj))

    # EQ(13) Path constraint for zijkl
    # For each zijkl
    # uij >= uik + ukl + ulj + (zijkl - 1)*M
    for (vi, vj, vk, vl) in z:
        zijkl = z[(vi, vj, vk, vl)]
        uij = u[(vi, vj)]
        uik = u[(vi, vk)]
        ukl = u[(vk, vl)]
        ulj = u[(vl, vj)]
        m.addConstr(uij >= uik + ukl + ulj + (zijkl-1)*MAX_NUMERIC_BOUND, 'pathzijkl({},{},{},{})'.format(vi, vj, vk, vl))

    # EQ (14),(15) Introduction of new external requirement links
    # Height variable to avoid cycles
    # For each zijkl, if (vl, vj) in b
    # blj >= zijkl
    # hij + (1-zijkl)*(H+1) >= hlj + 1
    for (vi, vj, vk, vl) in z:
        zijkl = z[(vi, vj, vk, vl)]
        if (vl, vj) in b:
            blj = b[(vl, vj)]
            m.addConstr(blj >= zijkl, 'newblj({},{},{},{})'.format(vi, vj, vk, vl))
            hij = h[(vi, vj)]
            hlj = h[(vl, vj)]
            m.addConstr(hij + (1-zijkl)*(H+1) >= hlj + 1, 'height({},{},{},{})'.format(vi, vj, vk, vl))

    # EQ (16) Need to decouple external contingent links
    # For (vi, vj) in contingents
    # sum(ckj) = 1
    for link in ext_conts:
        vi = link.s
        vj = link.e
        ckj_sum = []
        for (ck, cj, ci) in c:
            if cj == vj:
                ckj_sum.append(c[(ck, cj, ci)])
        m.addConstr(gp.quicksum(ckj_sum) == 1, 'ckjsum({},{})'.format(vi, vj))

    # EQ (17) Path constraint for ckj
    # For each ckj
    # ukj >= uki + uij + (ckj-1)*M
    # 0 <= lkj <= lki + lij + (1-ckj)*M
    for (vk, vj, vi) in c:
        ckj = c[(vk, vj, vi)]
        ukj = u[(vk, vj)]
        uki = u[(vk, vi)]
        uij = u[(vi, vj)]
        if FIX_CONTINGENT_BOUNDS:
            m.addConstr(ukj >= uki + uij + (ckj-1)*MAX_NUMERIC_BOUND, 'pathckj0({},{},{})'.format(vk, vj, vi))
            m.addConstr(ukj <= uki + uij - (ckj-1)*MAX_NUMERIC_BOUND, 'pathckj0({},{},{})'.format(vk, vj, vi))
        else:
            m.addConstr(ukj >= uki + uij + (ckj-1)*MAX_NUMERIC_BOUND, 'pathckj0({},{},{})'.format(vk, vj, vi))
        lkj = l[(vk, vj)]
        lki = l[(vk, vi)]
        lij = l[(vi, vj)]
        m.addConstr(lkj >= 0 + (ckj-1)*MAX_NUMERIC_BOUND, 'pathckj1({},{},{})'.format(vk, vj, vi))
        if FIX_CONTINGENT_BOUNDS:
            m.addConstr(lkj <= lki + lij + (1-ckj)*MAX_NUMERIC_BOUND, 'pathckj2({},{},{})'.format(vk, vj, vi))
            m.addConstr(lkj >= lki + lij - (1-ckj)*MAX_NUMERIC_BOUND, 'pathckj2({},{},{})'.format(vk, vj, vi))
        else:
            m.addConstr(lkj <= lki + lij + (1-ckj)*MAX_NUMERIC_BOUND, 'pathckj2({},{},{})'.format(vk, vj, vi))

    # EQ (18) Introduce new external requirements due to ckj
    # For each ckj
    # If (vi, vk) in b, bik >= ckj
    # If (vk, vi) in b, bki >= ckj
    for (vk, vj, vi) in c:
        ckj = c[(vk, vj, vi)]
        if (vi, vk) in b:
            bik = b[(vi, vk)]
            m.addConstr(bik >= ckj, 'bik>=ckj({},{},{})'.format(vk, vj, vi))
        if (vk, vi) in b:
            bki = b[(vk, vi)]
            m.addConstr(bki >= ckj, 'bki>=ckj({},{},{})'.format(vk, vj, vi))

def add_other_constraints_to_model(m, l, u, b, z, c, h, event2agent, events, ref_event):
    """
    Encode internal constraints between shared events
    We assume only requirement constraints
    """

    # Enforce uij <= uik + ukj, shortest path for each agent's shared events
    for vi in events:
        for vj in events:
            for vk in events:
                if not vi == vj and not vi == vk and not vj == vk:
                    if event2agent(vi) is None or event2agent(vj) is None or event2agent(vi) == event2agent(vj):
                        if event2agent(vi) is None or event2agent(vk) is None or event2agent(vi) == event2agent(vk):
                            if event2agent(vk) is None or event2agent(vj) is None or event2agent(vk) == event2agent(vj):

                                uij = u[(vi, vj)]
                                ujk = u[(vj, vk)]
                                uik = u[(vi, vk)]
                                m.addConstr(uik <= uij + ujk, 'int_shortestpath({},{},{})'.format(vi, vj, vk))

    #  Enforce lzi >= 0, reference event always procede all other events
    for vi in events:
        if not vi == ref_event:
            lzi = l[(ref_event, vi)]
            m.addConstr(lzi >= 0, 'ref_proceed({})'.format(vi))

    #  # Get agent to events
    #  agent2events = {}
    #  for e in events:
    #      if not e == ref_event:
    #          agent = event2agent(e)
    #          if agent not in agent2events:
    #              agent2events[agent] = []
    #          agent2events[agent].append(e)

    #  # Encode for each agent
    #  for agent, events in agent2events.items():
    #      if agent in int_reqs:
    #          constraints = int_reqs[agent]
    #          for link in constraints:
    #              vi = link.s
    #              vj = link.e
    #              lij = l[(vi, vj)]
    #              uij = u[(vi, vj)]
    #              Lij = link.lb
    #              Uij = link.ub
    #              if Lij is not None:
    #                  m.addConstr(lij >= Lij, 'int_reqlb({},{})'.format(vi, vj))
    #              if Uij is not None:
    #                  m.addConstr(uij <= Uij, 'int_requb({},{})'.format(vi, vj))
    #      events.append(ref_event)
    #      for vi in events:
    #          for vj in events:
    #              for vk in events:
    #                  if not vi == vj and not vi == vk and not vj == vk:
    #                      uij = u[(vi, vj)]
    #                      ujk = u[(vj, vk)]
    #                      uik = u[(vi, vk)]
    #                      m.addConstr(uik <= uij + ujk, 'int_shortestpath({},{},{})'.format(vi, vj, vk))

def add_conflicts_to_model(m, conflicts, l, u, c):
    """
    For an example privacy-preserving conflict of the form:
    [[6, -5, [c2(vk, vj), 'UB-', 'LB+']],
     [-5]]
    Identify if any decoupling constraint, such as c2(vk, vj), is contingent.
    If so, add to conflict [ckj = 1], that is,
    [[6, -5, [c2(vk, vj), 'UB-', 'LB+']],
     [-5],
     [ckj = 1]]
    The conflict can also be resolved by setting ckj = 0.
    """
    # Preprocess the set of local contingent constraints
    contingent_map = {}
    for (vk, vj, vi) in c:
        contingent_map[(vk, vj)] = (vk, vj, vi)

    count = 0
    # Encode all conflicts
    for conflict in conflicts:
        # Encode conflict
        sum_indicator = []
        ckj_added = {}
        # Encode all the inequalities
        for inequality in conflict:
            # Inequality e.g. [6, -5, [c2(vk, vj), 'UB-', 'LB+']
            lhs_sum = 0
            sum_pos_bound_vars = []
            sum_neg_bound_vars = []
            for c_pair in inequality:
                # c_pair is either:
                # + A value, e.g. -5
                # + A constraint boundtype tuple, e.g. [c2(vk, vj), 'UB-', 'LB+']
                if isinstance(c_pair, list):
                    constraint = c_pair[0]
                    vi = constraint.s
                    vj = constraint.e
                    # If c is contingent, record it in ckj_added
                    if isinstance(constraint, SimpleContingentTemporalConstraint):
                        ckj_added[(vi, vj)] = True
                    for boundtype in c_pair[1:]:
                        if boundtype == 'UB+':
                            sum_pos_bound_vars.append(u[(vi, vj)])
                        elif boundtype == 'UB-':
                            sum_neg_bound_vars.append(u[(vi, vj)])
                        elif boundtype == 'LB+':
                            sum_pos_bound_vars.append(l[(vi, vj)])
                        elif boundtype == 'LB-':
                            sum_neg_bound_vars.append(l[(vi, vj)])
                        else:
                            # boundtype must be one of the above
                            raise ValueError
                else:
                    lhs_sum += c_pair
            # If indicate = True, inequality must be >= 0
            indicator_var = m.addVar(vtype=GRB.BINARY, name="conflict_ind_{}".format(count))
            sum_indicator.append(indicator_var)
            m.addConstr(lhs_sum + gp.quicksum(sum_pos_bound_vars) - gp.quicksum(sum_neg_bound_vars) >= 0 + (indicator_var - 1)*MAX_NUMERIC_BOUND, "conflict_{}".format(count))
            count += 1
        # Encode all the ckj
        for (vk, vj) in ckj_added:
            (vk, vj, vi) = contingent_map[(vk, vj)]
            ckj = c[(vk, vj, vi)]
            # If indicator = True, ckj = 0
            indicator_var = m.addVar(vtype=GRB.BINARY, name="conflict_ind_{}".format(count))
            sum_indicator.append(indicator_var)
            m.addConstr(ckj <= (1 - indicator_var), "conflict_{}".format(count))
            count += 1
        # At least one of the inequalities or ckj has to be resolved
        m.addConstr(gp.quicksum(sum_indicator) >= 1, "conflict_{}".format(count))
        count += 1

def add_objective(m, l, u, c, b, z, h, ref_event, objective):
    if objective == NONE:
        pass
    elif objective == MAX_FLEXIBILITY:
        # Hunsberger's flexibility
        # max sum(uij)
        m.setObjective(gp.quicksum(l.values()), GRB.MINIMIZE)
    elif objective == MIN_TIME_SPAN:
        # Casanova's min time span
        # min max(uzj)
        span = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="time_span")
        for (vi, vj) in u:
            if vi == ref_event:
                uij = u[(vi, vj)]
                m.addConstr(span >= uij, 'span_u({},{})'.format(vi, vj))
        m.setObjective(span, GRB.MINIMIZE)
    elif objective == MAX_FLEXIBILITY_NAIVE:
        # Naive flexibility measure (cite Wilson)
        # min sum(uzj + ujz)
        sum_vars = []
        for (vi, vj) in u:
            if vi == ref_event:
                sum_vars.append(u[(vi, vj)])
            if vj == ref_event:
                sum_vars.append(u[(vi, vj)])
        #  m.setObjective(gp.quicksum(sum_vars), GRB.MAXIMIZE)
        m.setObjective(-gp.quicksum(sum_vars), GRB.MINIMIZE)
    elif objective == MAX_FLEXIBILITY_NEG_CKJ:
        # Hunsberger's flexibility except if contingent ckj = 1, then neg ukj + neg ujk
        # TODO: could also encode in linear expr
        expr = gp.QuadExpr()
        ckj = {}
        for (vk, vj, vi) in c:
            ckj[(vk, vj)] = c[(vk, vj, vi)]
        for (vi, vj) in u:
            uij = u[(vi, vj)]
            added = 0
            if (vi, vj) in ckj and (vj, vi) in ckj:
                cij = ckj[(vi, vj)]
                cji = ckj[(vj, vi)]
                new_c = m.addVar(vtype=GRB.BINARY)
                m.addConstr(new_c <= cij + cji)
                m.addConstr(new_c >= cij)
                m.addConstr(new_c >= cji)
                expr.add((1-2*new_c)*uij)
                added += 1
            elif (vi, vj) in ckj:
                cij = ckj[(vi, vj)]
                expr.add((1-2*cij)*uij)
                added += 1
            elif (vj, vi) in ckj:
                cij = ckj[(vj, vi)]
                expr.add((1-2*cij)*uij)
                added += 1
            assert(added <= 1)
            if added == 0:
                expr.add(uij)
        #  m.setObjective(expr, GRB.MAXIMIZE)
        m.setObjective(-expr, GRB.MINIMIZE)
    elif objective == MIN_LB_TIME_SPAN:
        # My own objective to minimize the lb of time span, which is,
        # assuming all obs is ideal, and execution of cont is fast
        # It should provide posibility of a more flexible execution
        # min max(lzj)
        span = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="lb_time_span")
        for (vi, vj) in l:
            if vi == ref_event:
                lij = l[(vi, vj)]
                m.addConstr(span >= lij, 'span_l({},{})'.format(vi, vj))
        m.setObjective(span, GRB.MINIMIZE)
    elif objective == MIN_LB_UB_TIME_SPAN:
        # Weighting MIN_LB_TIME_SPAN and MIN_TIME_SPAN
        lb_span = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="lb_time_span")
        for (vi, vj) in l:
            if vi == ref_event:
                lij = l[(vi, vj)]
                m.addConstr(lb_span >= lij, 'span_l({},{})'.format(vi, vj))
        ub_span = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="ub_time_span")
        for (vi, vj) in u:
            if vi == ref_event:
                uij = u[(vi, vj)]
                m.addConstr(ub_span >= uij, 'span_u({},{})'.format(vi, vj))
        m.setObjective(lb_span + ub_span, GRB.MINIMIZE)
    elif objective == MIN_BIJ:
        # Objective to minimize the number of ext requirement links that
        # need decoupling
        m.setObjective(gp.quicksum(b.values()), GRB.MINIMIZE)
    elif isinstance(objective, list):
        # TODO: Hack
        shared_events = objective.copy()
        shared_events.pop()
        print(shared_events)
        obj_var = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name='obj_var')
        for ev in shared_events:
            m.addConstr(obj_var <= u[(ref_event, ev)] + u[(ev, ref_event)])
        m.setObjective(obj_var, GRB.MAXIMIZE)
    else:
        raise ValueError

def compile_temporal_decoupling(l, u, c, event2agent):
    """
    Compile the optimization solution into temporal decoupling
    """
    # Preprocess the set of local contingent constraints
    contingent_map = {}
    for (vk, vj, vi) in c:
        ckj = c[(vk, vj, vi)]
        if round(ckj.x, 0) == 1:
            contingent_map[(vk, vj)] = True

    # Compile agent to decoupling constraints
    agent2constraints = {}
    added_map = {}
    for (vi, vj) in u:
        # vi and vj are different, otherwise uii = 0
        # (vi, vj) or (vj, vi) has not been added
        if vi != vj and (vi, vj) not in added_map:
            # Record only local constraints, discard external constraints
            if event2agent(vi) == event2agent(vj) or event2agent(vi) is None or event2agent(vj) is None:
                added_map[(vi, vj)] = True
                added_map[(vj, vi)] = True
                # Get lb, ub value
                uij = u[(vi, vj)].x
                lij = l[(vi, vj)].x
                # NOTE: Handle numeric instability by rounding to 6 decimal points
                uij = round(uij, ROUND_TO)
                lij = round(lij, ROUND_TO)
                # Instantiate decoupling constraint
                # NOTE: we allow contingent links with lb == ub
                if (vi, vj) in contingent_map:
                    #  print("vi, vj: {}, {}, {}, {}".format(vi, vj, lij, uij))
                    constraint = SimpleContingentTemporalConstraint(vi, vj, lij, uij, 'decoupling:cont({},{})'.format(vi, vj))
                elif (vj, vi) in contingent_map:
                    constraint = SimpleContingentTemporalConstraint(vj, vi, -uij, -lij, 'decoupling:cont({},{})'.format(vj, vi))
                else:
                    constraint = SimpleTemporalConstraint(vi, vj, lij, uij, 'decoupling:req({},{})'.format(vi, vj))

                # Add constraint to agent
                if event2agent(vi) is not None:
                    agent_name = event2agent(vi)
                else:
                    agent_name = event2agent(vj)
                assert(agent_name is not None)
                if agent_name not in agent2constraints:
                    agent2constraints[agent_name] = []
                agent2constraints[agent_name].append(constraint)
    return TemporalDecoupling(agent2constraints)

def compile_raw_data(u, z, c, b):
    # Raw data is a dictionary with
    # {'u': {(vi, vj): real},
    #  'z': {(vi, vj, vk, vl): boolean},
    #  'c': {(vk, vj, vi): boolean}，
    #  'b': {(vi, vj): boolean}
    # }
    # NOTE: Handle numeric instability by rounding to decimal points
    u = {v_pair: round(var.x, ROUND_TO) for v_pair, var in u.items()}
    z = {v_pair: round(var.x, 0) for v_pair, var in z.items()}
    c = {v_pair: round(var.x, 0) for v_pair, var in c.items()}
    b = {v_pair: round(var.x, 0) for v_pair, var in b.items()}
    raw_data = {'u': u, 'z': z, 'c': c, 'b': b}
    return raw_data
