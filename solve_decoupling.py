from temporal_decoupling import TemporalDecoupling
from dc_be import DCCheckerBE
from temporal_network import TemporalNetwork, SimpleTemporalConstraint, SimpleContingentTemporalConstraint
from decouple_milp import decouple_MILP, NONE, MIN_TIME_SPAN
from dc_milp import preprocess_agent_network
from networks import MaSTNU

from timeit import default_timer as timer

def solve_decoupling(mastnu, shared_events=None, output_stats=False, objective=NONE, timeout=None):
    """Solve the temporal decoupling problem for MaSTNU.

    Args:
        mastnu: MaSTNU instance
        shared_events: Optional; A set of shared events, default to necessary ones
        output_stats: Optional; Bool, whether stats is outputted or not
        objective: Optional; Option for objective function for the decoupling

    Returns:
        decoupling: None or of type TemporalDecoupling
        conflicts: The set of conflicts extracted
        Stats: Optional; A dict of solver stats
    """

    if output_stats:
        num_iterations = 0
        total_task_time = 0
        total_agent_time = {a: 0 for a in mastnu.agent2network.keys()}

    # Initialize conflicts
    conflicts = []

    # Solve for decoupling iteratively as conflicts are discovered.
    while True:
        if output_stats:
            start = timer()

        decoupling = decouple_MILP(mastnu, shared_events, conflicts=conflicts, objective=objective, timeout=timeout)
        #  if decoupling:
        #      print(decoupling.pprint())
        #  else:
        #      print("No decoupling found.")

        if output_stats:
            end = timer()
            total_task_time += end - start
            last_iteration_time = end - start
            num_iterations += 1
            # print("iter: {}".format(num_iterations))
            # print("time: {}".format(end - start))

        if decoupling is None:
            if output_stats:
                return None, conflicts, {'num_iterations': num_iterations,
                                         'total_task_time': total_task_time,
                                         'total_agent_time': total_agent_time,
                                         'last_iteration_time': last_iteration_time}
            else:
                return None, conflicts
        else:
            # Check if agent networks are controllable under the decoupling,
            # and extract privacy-preserving conflict if not.
            suc = True
            for agent, network in mastnu.agent2network.items():
                decoupling_constraints = decoupling.agent_to_decoupling(agent)
                if output_stats:
                    start = timer()
                controllable, conflict = check_dc_under_decoupling(network, decoupling_constraints)
                #  print("Agent: {}".format(agent))
                #  print("Controllable: {}, conflict: {}".format(controllable, conflict))
                if output_stats:
                    end = timer()
                    total_agent_time[agent] += end - start
                if not controllable:
                    suc = False
                    conflicts.append(conflict)

            if suc:
                if output_stats:
                    return decoupling, conflicts, {'num_iterations': num_iterations,
                                                 'total_task_time': total_task_time,
                                                 'total_agent_time': total_agent_time,
                                                 'last_iteration_time': last_iteration_time}
                else:
                    return decoupling, conflicts

def check_dc_under_decoupling(network, decoupling_constraints):
    network.add_constraints(decoupling_constraints)
    checker = DCCheckerBE(network)
    controllable, conflict = checker.is_controllable()
    network.remove_constraints(decoupling_constraints, remove_events=False)
    # Ensure the conflict is private by projecting out local constraints
    if not controllable:
        assert conflict is not None
        conflict = ensure_conflict_privacy(conflict, decoupling_constraints)
    return controllable, conflict

def ensure_conflict_privacy(conflict, decoupling_constraints):
    """
    A conflict is of the form:
    [[[c1, 'UB+'], [c1, 'LB-'], [c2, 'UB-', 'LB+']],
     [[c1, 'LB-']]]
    Each row is an inequality that can be resolved by setting inequality >= 0.
    The first row is a negative cycle, the second row and onwards are supporting
    conditions. Resolving any row resolves the entire conflict.
    To ensure privacy, if any constraint, e.g. c1 is private, we replace the
    element [c1, 'LB-'] to the actual value -c1.lb.
    """
    new_conflict = []
    for inequality in conflict:
        new_inequality = []
        for c_pair in inequality:
            c = c_pair[0]
            if c in decoupling_constraints:
                new_inequality.append(c_pair)
            else:
                # Private constraint, project it out!
                value = 0
                for boundtype in c_pair[1:]:
                    if boundtype == 'UB+':
                        value += c.ub
                    elif boundtype == 'UB-':
                        value += -c.ub
                    elif boundtype == 'LB+':
                        value += c.lb
                    elif boundtype == 'LB-':
                        value += -c.lb
                    else:
                        # boundtype must be one of the above
                        raise ValueError
                new_inequality.append(value)
        new_conflict.append(new_inequality)
    return new_conflict

def solve_decoupling_milp(mastnu, shared_events=None, output_stats=False, objective=NONE, timeout=None):
    """Solve the temporal decoupling problem for MaSTNU using centralized MILP formulation."""

    new_mastnu, shared_events = preprocess_networks(mastnu, shared_events)
    start = timer()
    decoupling = decouple_MILP(new_mastnu, shared_events, encode_agent_networks=True, objective=objective, timeout=timeout)
    end = timer()
    # print("time: {}".format(end - start))
    if output_stats:
        return decoupling, {'total_time': end - start}
    else:
        return decoupling

def preprocess_networks(mastnu, shared_events=None):
    # Preprocess all agent networks
    if shared_events is None:
        shared_events = mastnu.get_shared_events()
    event2agent = mastnu.event2agent
    ref_event = mastnu.ref_event
    ext_conts = mastnu.external_contingents
    processed_agent2network = {}
    for agent, network in mastnu.agent2network.items():
        processed_network, additional_shared_events = preprocess_agent_network(network, ext_conts, shared_events)
        processed_agent2network[agent] = processed_network
        shared_events += additional_shared_events
    return MaSTNU(processed_agent2network, mastnu.external_requirements, mastnu.external_contingents, mastnu.ref_event), shared_events
