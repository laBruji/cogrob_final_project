from collections import namedtuple
import networkx as nx
import numpy as np
from dispatcher import Dispatcher
from stnu import Stnu
from fast_dc import EdgeType, FastDc

from temporal_network import (SimpleContingentTemporalConstraint,
                                          SimpleTemporalConstraint,
                                          TemporalNetwork)
from networks import MaSTNU
from solve_decoupling import solve_decoupling
import threading

StnuEdge = namedtuple('StnuEdge', ['fro', 'to', 'lower_bound', 'upper_bound'])

def example_mastnu():
    """Test example in Casanova's paper."""

    ext_conts = [SimpleContingentTemporalConstraint('a1', 'b1', 1, 4, 'c1'),
            SimpleContingentTemporalConstraint('b3', 'a3', 1, 5, 'c3')]
    ext_reqs = [SimpleTemporalConstraint('b2', 'a2', 6, 8, 'c2')]

    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_network.add_constraint(SimpleTemporalConstraint('z', 'a1', 0, 5, 'c4'))
    agent_a_network.add_constraint(SimpleTemporalConstraint('a2', 'a3', 1, 10, 'c5'))
    agent_a_network.add_event('z')
    agent2network['agent-a'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_network.add_constraint(SimpleTemporalConstraint('b1', 'b2', 4, 6, 'c6'))
    agent_b_network.add_constraint(SimpleTemporalConstraint('b2', 'b3', 6, 12, 'c7'))
    agent_b_network.add_event('z')
    agent2network['agent-b'] = agent_b_network

    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'z')

    agents = list(mastnu.agent2network.keys())

    # print(mastnu.event2agent)
    # print(mastnu.external_contingents)
    # print(mastnu.get_shared_events())
    # print(compile_event_to_agent(agent2network, 'z'))

    return mastnu, agents

def get_independent_STNU(decoupling, agent):
    decoupled = decoupling.agent_to_decoupling(agent)
    controllable_edges = []
    uncontrollable_edges = []
    nodes = set()
    for a in decoupled:
        nodes.add(a.s)
        nodes.add(a.e)
        new_edge = StnuEdge(a.s, a.e, a.lb, a.ub)
        if type(a) == SimpleContingentTemporalConstraint:
            uncontrollable_edges.append(new_edge)
        else:
            controllable_edges.append(new_edge)
    return controllable_edges, uncontrollable_edges

def online_dispatch(dispatchable_form, events, dispatcher):
    """Algorithm to dispatch online. It should use the Python time module and make
    sure that events are executed as close as possible to the proper times. Method
    should not return until enough real-world time has passed to execute the entire plan.
    """
    exec_windows = {k:(-np.inf, np.inf) for k in events}
    enabled_events = []
    executed_events = set()
    
    def propagate(event, t_i):
        in_edges = list(dispatchable_form.in_edges(event, data=True))
        out_edges = list(dispatchable_form.out_edges(event, data=True))
        for i in out_edges:
            # get associated event
            e = i[1]
            if e not in executed_events:
                new_up = min(exec_windows[e][1], t_i + i[2]["weight"])
                exec_windows[e] = (exec_windows[e][0], new_up)

        for i in in_edges:
            # get associated event
            e = i[0]
            if e not in executed_events:
                new_low = max(exec_windows[e][0], t_i - i[2]["weight"])
                exec_windows[e] = (new_low, exec_windows[e][1])
            
    def get_predecessors(event):
        out_edges = list(dispatchable_form.out_edges(event, data=True))
        pred = set()
        for i in out_edges:
            if i[2]["weight"] < 0:
                pred.add(i[1])
        return pred
    
    predecessors = {}
    for e in events:
        predecessors[e] = get_predecessors(e)
        if len(predecessors[e]) == 0:
            enabled_events.append(e)
    
    # Start dispatching!
    dispatcher.start()
    
    while len(executed_events) != len(events):
        # sort current events
        enabled_events = sorted(enabled_events, key=lambda x: x[0])
        # get event that should happen first
        current_event = enabled_events[0]
        # wait until it's time to dispatch it
        to_wait = exec_windows[current_event][0] - dispatcher.time()
        if to_wait > 0:
            dispatcher.sleep(to_wait)
        
        if current_event not in executed_events:
            t_i = dispatcher.dispatch(current_event)
            exec_windows[current_event] = (t_i, t_i)
            propagate(current_event, t_i)
            executed_events.add(current_event)
            enabled_events.remove(current_event)

            # update enabled events 
            for e in events:
                if e not in executed_events and e not in enabled_events:
                    if predecessors[e].issubset(executed_events):
                        enabled_events.append(e)

        
    # All done!
    dispatcher.done()


def main():
    mastnu, agents = example_mastnu()
    decoupling, conflicts, stats = solve_decoupling(mastnu, output_stats=True)

    dispatchable_forms = {}
    agent_nodes = {}
    for agent in agents:
        # get independent STNUs
        controllable_edges, uncontrollable_edges = get_independent_STNU(decoupling, agent)
        stnu = Stnu()
        stnu.set_values(controllable_edges, uncontrollable_edges)

        # run DC Checking
        algo = FastDc()
        all_edges = algo.solve(stnu)

        # delete unnecessary edges for dispatching
        to_delete = set()
        nodes = set()
        for e in all_edges:
            nodes.add(e.renaming[e.fro])
            nodes.add(e.renaming[e.to])
            if e.type == EdgeType.UPPER_CASE:
                to_delete.add(frozenset([e.renaming[e.fro], e.renaming[e.to]]))
        
        dispatchable = nx.DiGraph()
        for e in all_edges:
            current_edge = frozenset([e.renaming[e.fro], e.renaming[e.to]])
            if current_edge in to_delete and e.type != EdgeType.UPPER_CASE:
                continue
            elif e.type == EdgeType.UPPER_CASE:
                dispatchable.add_edges_from([(e.renaming[e.fro], e.renaming[e.to], {"weight": e.value, "letter": e.maybe_letter})])
            else:
                dispatchable.add_edges_from([(e.renaming[e.fro], e.renaming[e.to], {"weight" : e.value})])
        
        dispatchable_forms[agent] = dispatchable
        agent_nodes[agent] = nodes
    
    
    # dispatch events
    for agent in agents:
        threading.Thread(target = online_dispatch, args = (dispatchable_forms[agent], agent_nodes[agent], Dispatcher())).start()

if __name__ == "__main__":
    main()
