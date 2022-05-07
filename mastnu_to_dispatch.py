import networkx as nx
import numpy as np
from dispatcher import Dispatcher

from temporal_network import (SimpleContingentTemporalConstraint,
                                          SimpleTemporalConstraint,
                                          TemporalNetwork)
from networks import MaSTNU
from solve_decoupling import solve_decoupling


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

def get_independent_STNUs(decoupling, agent):
    decoupled = decoupling.agent_to_decoupling(agent)
    d_graph = nx.DiGraph()
    for a in decoupled:
        d_graph.add_edge(a.s, a.e, weight=a.ub)
        d_graph.add_edge(a.e, a.s, weight=-a.lb)
        # add special edges
        if type(a) == SimpleContingentTemporalConstraint:
            attrs = {(a.s, a.e): {"weight": a.lb, "case": "lower", "letter": a.e}, (a.e, a.s): {"weight": -a.ub, "case": "upper", "letter": a.e}}
            nx.set_edge_attributes(d_graph, attrs)
    return d_graph

def dc_checking(d_graph):
    # probably use what I found
    pass

def online_dispatch(dispatchable_form, dispatcher):
    """Algorithm to dispatch online. It should use the Python time module and make
    sure that events are executed as close as possible to the proper times. Method
    should not return until enough real-world time has passed to execute the entire plan.
    """
    N = dispatchable_form.number_of_nodes()
    events = list(dispatchable_form.nodes())
    unscheduled_events = set(dispatchable_form.nodes())
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
    
    while len(executed_events) != N:
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
    d_graphs = {}
    for agent in agents:
        # get independent STNUs
        d_graph = get_independent_STNUs(decoupling, agent)
        d_graphs[agent] = d_graph # unnecessary but here just in case I might need it later

        # dc-checking algorithm
        dispatchable = dc_checking(d_graph)
        
        dispatcher = Dispatcher()
        online_dispatch(dispatchable, dispatcher)

if __name__ == "__main__":
    main()
