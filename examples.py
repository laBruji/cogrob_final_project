
from networks import MaSTNU, compile_event_to_agent
from temporal_network import SimpleContingentTemporalConstraint, SimpleTemporalConstraint, TemporalNetwork


def example_mastnu():
    """Test example in Casanova's paper."""
    constraints = []
    ext_conts = [SimpleContingentTemporalConstraint('a1', 'b1', 1, 4, 'c1'),
            SimpleContingentTemporalConstraint('b3', 'a3', 1, 5, 'c3')]
    ext_reqs = [SimpleTemporalConstraint('b2', 'a2', 6, 8, 'c2')]
    constraints += ext_conts + ext_reqs
    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_constraints = [SimpleTemporalConstraint('z', 'a1', 0, 5, 'c4'),SimpleTemporalConstraint('a2', 'a3', 1, 10, 'c5')]
    for constraint in agent_a_constraints:
        agent_a_network.add_constraint(constraint)
    agent_a_network.add_event('z')
    agent2network['agent-a'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_constraints = [SimpleTemporalConstraint('b1', 'b2', 4, 6, 'c6'), SimpleTemporalConstraint('b2', 'b3', 6, 12, 'c7')]
    for constraint in agent_b_constraints:
        agent_b_network.add_constraint(constraint)
    agent_b_network.add_event('z')
    agent2network['agent-b'] = agent_b_network

    constraints += agent_a_constraints + agent_b_constraints
    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'z')

    agents = list(mastnu.agent2network.keys())

    # print(mastnu.event2agent)
    # print(mastnu.external_contingents)
    # print(mastnu.get_shared_events())
    # print(compile_event_to_agent(agent2network, 'z'))

    return mastnu, agents, constraints

def test_centralized_milp_preprocess_2():

    # Testcase4:
    """
                  ====> F
                (Ce) --> (Ds) ---> (E)
                 ||       |         ||
    A ==> B ==> (Cs) --> (De) ---> (E)
    For agent-a, C should have copy
    For agent-b, B, C should have copies
    """
    constraints = []
    ext_reqs = [SimpleTemporalConstraint('aD', 'bD', 1, 5, 'c3')]
    ext_conts = [SimpleContingentTemporalConstraint('bC', 'aC', 1, 4, 'c1'),
            SimpleContingentTemporalConstraint('aE', 'bE', 6, 8, 'c2')]
    constraints += ext_conts + ext_reqs
    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_constraints = [SimpleTemporalConstraint('aC', 'aD', 0, 5, 'c4'),
                    SimpleTemporalConstraint('aD', 'aE', 1, 10, 'c5'), 
                    SimpleContingentTemporalConstraint('aC', 'aF', 0, 5, 'c10'),
                    SimpleTemporalConstraint('z', 'aC', lb=0, name='ref_preceding_a')]
    for constraint in agent_a_constraints:
        agent_a_network.add_constraint(constraint)

    agent_a_network.add_event('z')
    agent2network['agent-a'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_constraints = [SimpleContingentTemporalConstraint('bA', 'bB', 6, 12, 'c6'),
                    SimpleContingentTemporalConstraint('bB', 'bC', 6, 12, 'c7'),
                    SimpleTemporalConstraint('bC', 'bD', 4, 6, 'c8'),
                    SimpleTemporalConstraint('bD', 'bE', 6, 12, 'c9'),
                    SimpleTemporalConstraint('z', 'bA', lb=0, name='ref_preceding_b')]

    for constraint in agent_b_constraints:
        agent_a_network.add_constraint(constraint)
    agent_b_network.add_event('z')
    agent2network['agent-b'] = agent_b_network

    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'z')
    agents = list(mastnu.agent2network.keys())
    constraints += agent_a_constraints + agent_b_constraints
    return mastnu, agents, constraints

def test_nikhil_example_delay_5():
    # NOTE: delay should be [5, 5+epsilon]
    # A ==> B (observable)
    #          C ---> D
    # A --> D and B ---> D

    constraints = []
    ext_conts = [SimpleContingentTemporalConstraint('B', 'B-obs', 5, 5.1, 'c2')]
    ext_reqs = [SimpleTemporalConstraint('B', 'D', 30, 45, 'c4')]
    constraints += ext_conts + ext_reqs
    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_constraints = [SimpleContingentTemporalConstraint('A', 'B', 20, 40, 'c1')]
    for constraint in agent_a_constraints:
        agent_a_network.add_constraint(constraint)
    agent2network['agent-a'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_constraints = [SimpleTemporalConstraint('C', 'D', 15, 15, 'c8'),
                    SimpleTemporalConstraint('A', 'D', 60, 75, 'c3')]
    for constraint in agent_b_constraints:
        agent_b_network.add_constraint(constraint)
    agent_b_network.add_event('B-obs')
    agent2network['agent-b'] = agent_b_network

    constraints += agent_a_constraints + agent_b_constraints
    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'A')
    agents = list(mastnu.agent2network.keys())
    return mastnu, agents, constraints

def test_nikhil_example_obs_or_not():
    # Test under max flex metric, whether obs is taken or not, when w or w/o obs are both feasible. Seems that it would prefer using the comm link is uncertainty is low in this case.
    # AB is now [20, 30] instead of [20, 40]
    # A ==> B (observable)
    #          C ---> D
    # A --> D and B ---> D
    
    constraints = []
    ext_conts = [SimpleContingentTemporalConstraint('B', 'B-obs', 0, 5, 'c2')]
    ext_reqs = [SimpleTemporalConstraint('B', 'D', 30, 45, 'c4')]
    constraints += ext_conts + ext_reqs
    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_constraints = [SimpleContingentTemporalConstraint('A', 'B', 20, 30, 'c1')]
    for constraint in agent_a_constraints:
        agent_a_network.add_constraint(constraint)
    agent2network['agent-a'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_constraints = [SimpleTemporalConstraint('C', 'D', 15, 15, 'c8'), 
                    SimpleTemporalConstraint('A', 'D', 60, 75, 'c3')]
    for constraint in agent_b_constraints:
        agent_b_network.add_constraint(constraint)
    agent_b_network.add_event('B-obs')
    agent2network['agent-b'] = agent_b_network

    constraints += agent_a_constraints + agent_b_constraints
    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'A')
    agents = list(mastnu.agent2network.keys())
    return mastnu, agents, constraints
    
def test_nikhil_example_delay_5_icaps():
    # delay should be [0, 5]
    # A ==> B (observable)
    #          C ---> D
    # A --> D and B ---> D

    ext_conts = [SimpleContingentTemporalConstraint('B', 'B-obs', 0, 5, 'c2')]
    ext_reqs = [SimpleTemporalConstraint('B', 'D', 30, 45, 'c4')]

    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_network.add_constraint(SimpleContingentTemporalConstraint('A', 'B', 20, 40, 'c1'))
    agent2network['agent-a'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_network.add_constraint(SimpleTemporalConstraint('C', 'D', 15, 15, 'c8'))
    agent_b_network.add_constraint(SimpleTemporalConstraint('A', 'D', 60, 75, 'c3'))
    agent_b_network.add_event('B-obs')
    agent2network['agent-b'] = agent_b_network

    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'A')
    agents = list(mastnu.agent2network.keys())
    return mastnu, agents

def lecture_example():
    constraints = []
    ext_conts = [SimpleContingentTemporalConstraint('a', 'a_', 0, 5, 'c2')]
    ext_reqs = [SimpleTemporalConstraint('a', 'c', 30, 45, 'c3')]
    constraints += ext_conts + ext_reqs
    agent2network = {}

    agent_a_network = TemporalNetwork()
    agent_a_constraints = [SimpleContingentTemporalConstraint('z', 'a', 20, 40, 'c1')]
    for constraint in agent_a_constraints:
        agent_a_network.add_constraint(constraint)
    agent_a_network.add_event('z')
    agent2network['alice'] = agent_a_network

    agent_b_network = TemporalNetwork()
    agent_b_constraints = [SimpleTemporalConstraint('b', 'c', 15, 15, 'c4'),
                    SimpleTemporalConstraint('z', 'c', 60, 75, 'c5')]
    for constraint in agent_b_constraints:
        agent_b_network.add_constraint(constraint)
    agent_b_network.add_event('z')
    agent_b_network.add_event('a_')
    agent2network['bob'] = agent_b_network

    constraints += agent_a_constraints + agent_b_constraints
    mastnu = MaSTNU(agent2network, ext_reqs, ext_conts, 'z')

    agents = list(mastnu.agent2network.keys())

    return mastnu, agents, constraints