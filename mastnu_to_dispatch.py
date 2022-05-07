import networkx as nx
import numpy as np

from temporal_network import (SimpleContingentTemporalConstraint,
                                          SimpleTemporalConstraint,
                                          TemporalNetwork)
from networks import MaSTNU
from solve_decoupling import solve_decoupling


def example_stnu():
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
            attrs = {(a.s, a.e): {"weight": a.lb, "case": "lower"}, (a.e, a.s): {"weight": -a.ub, "case": "upper"}}
            nx.set_edge_attributes(d_graph, attrs)
    return d_graph

def dc_checking(d_graph):
    pass


def main():
    mastnu, agents = example_stnu()
    decoupling, conflicts, stats = solve_decoupling(mastnu, output_stats=True)
    d_graphs = {}
    for agent in agents:
        # get independent STNUs
        d_graph = get_independent_STNUs(decoupling, agent)
        d_graphs[agent] = d_graph # unnecessary but here just in case I might need it later

        # dc-checking algorithm
        dispatchable = dc_checking(d_graph)

if __name__ == "__main__":
    main()
