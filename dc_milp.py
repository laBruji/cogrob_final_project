import gurobipy as gp
from gurobipy import GRB
from temporal_network import TemporalNetwork, SimpleTemporalConstraint, SimpleContingentTemporalConstraint

MAX_NUMERIC_BOUND = 100000

def preprocess_agent_network(tn, ext_conts, shared_events):
    '''
    Given an agent network and external constraints, preprocess it
    so that no contingent constraint starts from an uncontrollable event.
    Returns a shallow copy of the network, with new constraints for
    the modified part of the network.
    E.g. Shared events denoted with (V)
    A ==> B ==> (C) --> (D) ---> (E)
                            ===> G
    where (F, D) is part of ext_conts.
    Result: A ==> B-copy -0,0-> B ==> C-copy -0,0-> (C) --> (D) ---> (E)
                                                             -0,0-> (D-copy)
    + B-copy is necessary because contingent constraints cannot start
    from uncontrollable node.
    + C-copy is necessary if ext_conts contain a link with target in
    the agent network, because a shared event may become start of a
    contingent link.
    + D-copy is necessary if there is another local cont constraint (D, G)
    in the agent network, or if there is another ext cont that ends at
    the agent network (the latter case also makes D-copy shared, and ckj
    cannot use D itself as vk).
    '''
    all_events = tn.get_events()
    constraints = tn.get_constraints()
    shared_events = list(set.intersection(set(all_events), set(shared_events)))

    additional_shared_events = []

    network = TemporalNetwork()
    network.add_events(shared_events)

    uncontrollable_events = {}
    num_ext_conts_at_agent = 0
    for c in constraints:
        if isinstance(c, SimpleContingentTemporalConstraint):
            uncontrollable_events[c.e] = 'local'
    for c in ext_conts:
        if c.e in all_events:
            uncontrollable_events[c.e] = 'external'
            num_ext_conts_at_agent += 1

    need_copy_events = {}
    # Handle case for B-copy and D-copy due to local cont (D, G)
    for c in constraints:
        if isinstance(c, SimpleContingentTemporalConstraint):
            # c.s is an uncontrollable event
            if c.s in uncontrollable_events:
                # Contingent link starts from an uncontrollable event
                # Need copy of the uncontrollable event
                need_copy_events[c.s] = True
    # Handle case for C-copy
    if num_ext_conts_at_agent > 0:
        for e in shared_events:
            if e in uncontrollable_events and uncontrollable_events[e] == 'local':
                need_copy_events[e] = True
    # Handle case for (D-copy)
    if num_ext_conts_at_agent > 1:
        for e in shared_events:
            if e in uncontrollable_events and uncontrollable_events[e] == 'external':
                need_copy_events[e] = 'shared'

    # Add constraints to network
    # Handle all copy of events except D
    for c in constraints:
        if isinstance(c, SimpleContingentTemporalConstraint):
            if c.e in need_copy_events and uncontrollable_events[c.e] == 'local':
                # Create copy of the uncontrollable event
                u_event_copy = c.e + "-copy"
                equality = SimpleTemporalConstraint(u_event_copy, c.e, 0, 0, 'equality({},{})'.format(u_event_copy, c.e))
                new_c = SimpleContingentTemporalConstraint(c.s, u_event_copy, c.lb, c.ub, c.name)
                network.add_constraints([equality, new_c])
            else:
                network.add_constraint(c)
        else:
            network.add_constraint(c)

    # Handle events like D
    for e in shared_events:
        if e in need_copy_events and uncontrollable_events[e] == 'external':
            u_event_copy = e + "-copy"
            equality = SimpleTemporalConstraint(u_event_copy, e, 0, 0, 'equality({},{})'.format(u_event_copy, e))
            network.add_constraint(equality)
            # If another external contingent may use e as vk in ckj
            if need_copy_events[e] == 'shared':
                additional_shared_events.append(u_event_copy)
            # If there are contingent links that starts at e, modify its start
            network_constraints = network.get_constraints()
            for c in network_constraints:
                if isinstance(c, SimpleContingentTemporalConstraint):
                    if c.s == e:
                        network.remove_constraint(c, remove_events=False)
                        new_c = SimpleContingentTemporalConstraint(u_event_copy, c.e, c.lb, c.ub, c.name)
                        network.add_constraint(new_c)

    return network, additional_shared_events

def add_dc_constraints_to_model(m, agent2network, u, c, event2agent):
    for agent, network in agent2network.items():
        agent_u = filter_by_agent(agent, u, event2agent)
        agent_c = filter_by_agent(agent, c, event2agent)
        encode_dc_milp(m, network, agent_u, agent_c)

def filter_by_agent(agent_name, v, event2agent):

    def is_local_edge(pair):
        vi = pair[0]
        vj = pair[1]
        ai = event2agent(vi)
        aj = event2agent(vj)
        return (ai is None and aj is None) or\
               (ai is None and aj == agent_name) or\
               (ai == agent_name and aj is None) or\
               (ai == agent_name and aj == agent_name)

    return {pair: var for pair, var in v.items() if is_local_edge(pair)}

def encode_dc_milp(m, network, u, c):
    """
    This is a variant of dc_milp where additional constraints
    are introduced because of ckj = 1
    """
    # Create additional local variables
    u, w, x, b = add_variables_to_model(m, network, u)

    # Add constraints
    add_constraints_to_model(m, network, u, w, x, b, c)

def add_variables_to_model(m, network, u):

    # Get all local events for agent
    events = network.get_events()

    # (vi, vj) => variable, vi != vj
    for vi in events:
        for vj in events:
            # Add if not already in u
            if not vi == vj and (vi, vj) not in u:
                uij = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="u({},{})".format(vi, vj))
                u[(vi, vj)] = uij


    # (vi, vj, vk) => variable
    w = {}

    # (vi, vj, vk) => boolean variable, for each wait var
    x = {}

    # (vi, vj, vk) => boolean variable, for precede or not
    b = {}

    # If only checking dc, for each (vi, vk) contingent, vj in V
    # In general, with decoupling MILP, ckj may cause local contingent
    # constraints to change. Therefore, we instantiate for all (vi, vj, vk)
    for vi in events:
        for vj in events:
            for vk in events:
                if not vi == vj and not vj == vk and not vi == vk:
                    wijk = m.addVar(lb=-MAX_NUMERIC_BOUND, ub=MAX_NUMERIC_BOUND, vtype=GRB.CONTINUOUS, name="w({},{},{})".format(vi, vj, vk))
                    w[(vi, vj, vk)] = wijk
                    xijk = m.addVar(vtype=GRB.BINARY, name="x({}, {}, {})".format(vi, vj, vk))
                    x[(vi, vj, vk)] = xijk
                    bijk = m.addVar(vtype=GRB.BINARY, name="b({}, {}, {})".format(vi, vj, vk))
                    b[(vi, vj, vk)] = bijk

    return u, w, x, b

def add_constraints_to_model(m, network, u, w, x, b, cc):

    # Get all local events for agent
    events = network.get_events()

    # Get constraints for agent
    constraints = network.get_constraints()
    contingent_constraints = [c for c in constraints if isinstance(c, SimpleContingentTemporalConstraint)]

    #====================================================
    # The following constraints are the same as dc_milp,
    # With some slight modification on loop conditions.
    #====================================================

    # Non-negative cycle constraint
    # uij + uji >= 0
    visited = {}
    for (vi, vj) in u:
        if not (vi, vj) in visited and not (vj, vi) in visited:
            visited[(vi, vj)] = True
            uij = u[(vi, vj)]
            uji = u[(vj, vi)]
            m.addConstr(uij + uji >= 0, 'nonneg({},{})'.format(vi, vj))

    # (1), (2) Bounds for requirement and contingent constraints
    for c in constraints:
        # uij <= Uij, lij >= Lij (uji <= -Lij)
        uij = u[(c.s, c.e)]
        uji = u[(c.e, c.s)]
        if not c.ub is None:
            m.addConstr(uij <= c.ub, 'upperbound({},{})'.format(c.s, c.e))
        if not c.lb is None:
            m.addConstr(uji <= - c.lb, 'lowerbound({},{})'.format(c.e, c.s))

        # if contingent, uij = Uij, lij = Lij
        if isinstance(c, SimpleContingentTemporalConstraint):
            m.addConstr(uij >= c.ub, 'u({}, {}) >= U'.format(c.s, c.e))
            m.addConstr(uji >= - c.lb, 'u({}, {}) >= -L'.format(c.e, c.s))

    # (3) Shortest path constraint
    # uik <= uij + ujk
    for vi in events:
        for vj in events:
            for vk in events:
                if not vi == vj and not vi == vk and not vj == vk:
                    uij = u[(vi, vj)]
                    ujk = u[(vj, vk)]
                    uik = u[(vi, vk)]
                    m.addConstr(uik <= uij + ujk, 'shortestpath({},{},{})'.format(vi, vj, vk))

    # (4) Precede constraint
    # ljk > 0 => uij <= lik - ljk (uij <= -uki + ukj) and lij >= uik - ujk (-uji >= uik - ujk)
    for c in contingent_constraints:
        vi = c.s
        vk = c.e
        for vj in events:
            if not vj == vi and not vj == vk:
                # When (vi, vk) is contingent and ljk > 0 means vj precedes vk for sure
                bijk = b[(vi, vj, vk)]
                uij = u[(vi, vj)]
                uji = u[(vj, vi)]
                uik = u[(vi, vk)]
                uki = u[(vk, vi)]
                ujk = u[(vj, vk)]
                ukj = u[(vk, vj)]
                # If b = 0, ljk <= 0 (ukj >= 0)
                m.addConstr(ukj + bijk * MAX_NUMERIC_BOUND >= 0, 'precede-b0({},{},{})'.format(vi, vj, vk))
                # If b = 1, the other two constraints
                m.addConstr(uij - (1-bijk) * MAX_NUMERIC_BOUND <= -uki + ukj, 'precede-b1-a({},{},{})'.format(vi, vj, vk))
                m.addConstr(-uji + (1-bijk) * MAX_NUMERIC_BOUND >= uik - ujk, 'precede-b1-b({},{},{})'.format(vi, vj, vk))

    # (5) Wait constraint
    # uik - ujk <= wijk
    for c in contingent_constraints:
        vi = c.s
        vk = c.e
        for vj in events:
            if not vj == vi and not vj == vk:
                # When (vi, vk) is contingent and vj, vk unordered
                uik = u[(vi, vk)]
                ujk = u[(vj, vk)]
                wijk = w[(vi, vj, vk)]
                # uik - ujk <= wijk
                m.addConstr(uik - ujk <= wijk, 'wait({},{},{})'.format(vi, vj, vk))

    # (6) Wait constraint
    # min(lik, wijk) <= lij
    # wijk <= uij should hold according to Cui's, but missed by Casanova
    # if (vi, vj) is contingent, lij >= wijk, missed by both Cui and Casanova
    # NOTE: We instantiate for all (vi, vj, vk) since EQ (5) and wait regression
    # provides a lower bound on wijk. If wijk for some (vi, vj, vk) is unbounded,
    # wijk can take arbitrarily small values and satisfy the above constraints.
    for vi in events:
        for vj in events:
            for vk in events:
                if not vi == vj and not vi == vk and not vj == vk:
                    uij = u[(vi, vj)]
                    uji = u[(vj, vi)]
                    uik = u[(vi, vk)]
                    uki = u[(vk, vi)]
                    ujk = u[(vj, vk)]
                    wijk = w[(vi, vj, vk)]

                    # wijk <= uij
                    # See Wah and Xin's NLP encoding for why
                    # If (vi, vj) is requirement link, then it's possible that needs to wait, if so, wait should be smaller than uij.
                    m.addConstr(wijk <= uij, 'wait<ub({},{},{})'.format(vi, vj, vk))

                    # min(lik, wijk) <= lij
                    # (lij >= lik and wijk >= lik) or (lij >= wijk and wijk <= lik)
                    # Reason: If wijk >= lik, we need that ukj + uji (uji = -lij <= -wijk) + lik(should be uik, but ik is contingent) >= 0 based on shortest path,
                    # then ukj >= -lik - uji >= -lik + wijk >= 0, meaning vj can happen after vk,
                    # Should be fine without comparing wijk >=/<= lik, but (7) uses it so we add it
                    xijk = x[(vi, vj, vk)]
                    # If x = 0, lij >= lik (uji <= uki) and -uki <= wijk
                    m.addConstr(uji - xijk * MAX_NUMERIC_BOUND <= uki, 'waitcond0({},{},{})'.format(vi, vj, vk))
                    m.addConstr(wijk + xijk * MAX_NUMERIC_BOUND >= -uki, 'waitcond0+({},{},{})'.format(vi, vj, vk))
                    # If x = 1, lij >= wijk (-uji >= wijk) and -uki >= wijk
                    m.addConstr(-uji + (1-xijk) * MAX_NUMERIC_BOUND >= wijk, 'waitcond1({},{},{})'.format(vi, vj, vk))
                    m.addConstr(wijk - (1-xijk) * MAX_NUMERIC_BOUND <= -uki, 'waitcond1+({},{},{})'.format(vi, vj, vk))

                    # if (vi, vj) is contingent, lij >= wijk
                    # In Wah and Xin's paper, there is a constraint wijk = lij if (vi, vj) is contingent
                    # We can preprocess the network to avoid contingent links sharing same source event,
                    # by adding a copy of the source event and add equality constraint.
                    # Cui's paper may have assumed so, which is why this constraint is not added.
                    # We add this constraint to avoid extra preprocessing.
                    for c in contingent_constraints:
                        if vi == c.s and vj == c.e:
                            # (vi, vj) is contingent! lij >= wijk (-uji >= wijk)
                            m.addConstr(-uji >= wijk, 'waitcondcontingent({},{},{})'.format(vi, vj, vk))

    # (8) wait regression
    # wijk − umj <= wimk
    for c in contingent_constraints:
        vi = c.s
        vk = c.e
        for vj in events:
            for vm in events:
                if not vj == vi and not vj == vk and not vm == vi and not vm == vk and not vm == vj:
                    wijk = w[(vi, vj, vk)]
                    wimk = w[(vi, vm, vk)]
                    umj = u[(vm, vj)]
                    m.addConstr(wijk - umj <= wimk, 'regression({},{},{},{})'.format(vi, vj, vk, vm))

    # (7) wait regression for contingent constraint
    # (wijk <= 0) or (wijk − lmj <= wimk)
    # According to Cui's paper, can be strengthed to
    # (wijk >= lik) => (wijk − lmj <= wimk)
    # That is xijk = 0 => (wijk + ujm <= wimk)
    for c1 in contingent_constraints:
        for c2 in contingent_constraints:
            vi = c1.s
            vk = c1.e
            vm = c2.s
            vj = c2.e
            if not c1 == c2 and not vm == vi:
                # If vm == vi, contingent links starts at same source event, no need to propagate further
                wijk = w[(vi, vj, vk)]
                xijk = x[(vi, vj, vk)]
                wimk = w[(vi, vm, vk)]
                ujm = u[(vj, vm)]
                m.addConstr(wijk + ujm - xijk * MAX_NUMERIC_BOUND <= wimk, 'regression-contingent({},{},{},{})'.format(vi, vj, vk, vm))

    #====================================================
    # The following constraints are additional due to ckj
    #====================================================

    # (4) additional Precede constraint
    # ljk > 0 => uij <= lik - ljk (uij <= -uki + ukj) and lij >= uik - ujk (-uji >= uik - ujk)
    for (vi, vk, vm) in cc:
        for vj in events:
            if not vj == vi and not vj == vk:
                # When (vi, vk) is contingent and ljk > 0 means vj precedes vk for sure
                cik = cc[(vi, vk, vm)]
                bijk = b[(vi, vj, vk)]
                uij = u[(vi, vj)]
                uji = u[(vj, vi)]
                uik = u[(vi, vk)]
                uki = u[(vk, vi)]
                ujk = u[(vj, vk)]
                ukj = u[(vk, vj)]
                # If cik = 1, b = 0, ljk <= 0 (ukj >= 0)
                m.addConstr(ukj + bijk * MAX_NUMERIC_BOUND + (1-cik)*MAX_NUMERIC_BOUND >= 0, 'ckj-precede-b0({},{},{})'.format(vi, vj, vk))
                # If cik = 1, b = 1, the other two constraints
                m.addConstr(uij - (1-bijk) * MAX_NUMERIC_BOUND <= -uki + ukj + (1-cik)*MAX_NUMERIC_BOUND, 'ckj-precede-b1-a({},{},{})'.format(vi, vj, vk))
                m.addConstr(-uji + (1-bijk) * MAX_NUMERIC_BOUND + (1-cik)*MAX_NUMERIC_BOUND >= uik - ujk, 'ckj-precede-b1-b({},{},{})'.format(vi, vj, vk))

    # (5) additional Wait constraint
    # uik - ujk <= wijk
    for (vi, vk, vm) in cc:
        for vj in events:
            if not vj == vi and not vj == vk:
                # When (vi, vk) is contingent and vj, vk unordered
                cik = cc[(vi, vk, vm)]
                uik = u[(vi, vk)]
                ujk = u[(vj, vk)]
                wijk = w[(vi, vj, vk)]
                # uik - ujk <= wijk
                m.addConstr(uik - ujk <= wijk + (1-cik)*MAX_NUMERIC_BOUND, 'ckj-wait({},{},{})'.format(vi, vj, vk))

    # (6) additional Wait constraint
    # if (vi, vj) is contingent, lij >= wijk, missed by both Cui and Casanova
    for (vi, vj, vm) in cc:
        for vk in events:
            if not vk == vi and not vk == vj:
                cij = cc[(vi, vj, vm)]
                uji = u[(vj, vi)]
                wijk = w[(vi, vj, vk)]
                # lij >= wijk (-uji >= wijk)
                m.addConstr(-uji + (1-cij)*MAX_NUMERIC_BOUND >= wijk, 'ckj-waitcondcontingent({},{},{})'.format(vi, vj, vk))

    # (8) additional Wait regression
    # wijk − umj <= wimk
    for (vi, vk, cp) in cc:
        for vj in events:
            for vm in events:
                if not vj == vi and not vj == vk and not vm == vi and not vm == vk and not vm == vj:
                    cik = cc[(vi, vk, cp)]
                    wijk = w[(vi, vj, vk)]
                    wimk = w[(vi, vm, vk)]
                    umj = u[(vm, vj)]
                    m.addConstr(wijk - umj <= wimk + (1-cik)*MAX_NUMERIC_BOUND, 'ckj-regression({},{},{},{})'.format(vi, vj, vk, vm))

    # (7) additional wait regression for contingent constraint (1)
    # (wijk >= lik) => (wijk − lmj <= wimk)
    # That is xijk = 0 => (wijk + ujm <= wimk)
    for c1 in contingent_constraints:
        for (vm, vj, vq) in cc:
            vi = c1.s
            vk = c1.e
            if not vm == vi:
                # If vm == vi, contingent links starts at same source event, no need to propagate further
                cmj = cc[(vm, vj, vq)]
                wijk = w[(vi, vj, vk)]
                xijk = x[(vi, vj, vk)]
                wimk = w[(vi, vm, vk)]
                ujm = u[(vj, vm)]
                m.addConstr(wijk + ujm - xijk * MAX_NUMERIC_BOUND <= wimk + (1-cmj)*MAX_NUMERIC_BOUND, 'ckj(1)-regression-contingent({},{},{},{})'.format(vi, vj, vk, vm))

    # (7) additional wait regression for contingent constraint (2)
    # (wijk >= lik) => (wijk − lmj <= wimk)
    # That is xijk = 0 => (wijk + ujm <= wimk)
    for (vi, vk, vp) in cc:
        for c2 in contingent_constraints:
            vm = c2.s
            vj = c2.e
            if not vm == vi:
                # If vm == vi, contingent links starts at same source event, no need to propagate further
                cik = cc[(vi, vk, vp)]
                wijk = w[(vi, vj, vk)]
                xijk = x[(vi, vj, vk)]
                wimk = w[(vi, vm, vk)]
                ujm = u[(vj, vm)]
                m.addConstr(wijk + ujm - xijk * MAX_NUMERIC_BOUND <= wimk + (1-cik)*MAX_NUMERIC_BOUND, 'ckj(2)-regression-contingent({},{},{},{})'.format(vi, vj, vk, vm))


    # (7) additional wait regression for contingent constraint (3)
    # (wijk >= lik) => (wijk − lmj <= wimk)
    # That is xijk = 0 => (wijk + ujm <= wimk)
    for (vi, vk, vp) in cc:
        for (vm, vj, vq) in cc:
            # cik and cmk only one can be true
            if not vm == vi and not vk == vj:
                # If vm == vi, contingent links starts at same source event, no need to propagate further
                cik = cc[(vi, vk, vp)]
                cmj = cc[(vm, vj, vq)]
                wijk = w[(vi, vj, vk)]
                xijk = x[(vi, vj, vk)]
                wimk = w[(vi, vm, vk)]
                ujm = u[(vj, vm)]
                m.addConstr(wijk + ujm - xijk * MAX_NUMERIC_BOUND <= wimk + (1-cik)*MAX_NUMERIC_BOUND + (1-cmj)*MAX_NUMERIC_BOUND, 'ckj(3)-regression-contingent({},{},{},{})'.format(vi, vj, vk, vm))
