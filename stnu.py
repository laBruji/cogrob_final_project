from collections import namedtuple, defaultdict


StnuEdge = namedtuple('StnuEdge', ['fro', 'to', 'lower_bound', 'upper_bound'])

class Stnu(object):
    def __init__(self):
        self.reset()
        self._renaming = {}
        self._inverse_renaming = {}


    def reset(self):
        self.num_nodes = 0
        self.controllable_edges = []
        self.uncontrollable_edges = []
    
    def set_values(self, c_edges, uc_edges):
        self.reset()
        next_number = [1]
        def read_edges(edge_list, where_to):
            for edge in edge_list:
                start_name, end_name, lower_bound, upper_bound = edge.fro, edge.to, edge.lower_bound, edge.upper_bound
                for name in [start_name, end_name]:
                    if name not in self._renaming:
                        self._renaming[name] = next_number[0]
                        self._inverse_renaming[next_number[0]] = name
                        next_number[0] += 1
                where_to.append(StnuEdge(self._renaming[start_name],
                                          self._renaming[end_name],
                                          lower_bound,
                                          upper_bound))

        read_edges(c_edges, self.controllable_edges)
        read_edges(uc_edges, self.uncontrollable_edges)
        self.num_nodes = len(self._renaming)
        self.verify_contraints()

    def verify_contraints(self):
        pairs = []
        for edge in self.controllable_edges:
            # check bounds on edges
            assert edge.lower_bound <= edge.upper_bound
            pairs.append((edge.fro, edge.to))

        # Controllable edges are not repeated (not two edges connect 
        # the same nodes). This could in principle be allowed and then we
        # could normalize by replacing all the edges between a given pair
        # of nodes by a single edge with bound interval equal to an
        # intersection of bound intervals for all the edges connecting
        # that pair of nodes.
        assert len(pairs) == len(set(pairs))

        pairs = []
        input_degree = defaultdict(lambda: 0)
        uncontrollable_starts = set()
        uncontrollable_ends = set()
        for edge in self.uncontrollable_edges:
            # Check bounds on edges. Notice that below 0 is disallowed in
            # contrast to controllable edges
            uncontrollable_starts.add(edge.fro)
            uncontrollable_ends.add(edge.to)
            input_degree[edge.to] += 1
            assert 0 <= edge.lower_bound and edge.lower_bound <= edge.upper_bound            
            pairs.append((edge.fro, edge.to))

        # no two uncontrollable edges are one after another in the network.
        assert len(uncontrollable_starts.intersection(uncontrollable_ends)) == 0
        # Controllable edges are not repeated (not two edges connect 
        # the same nodes). Here we really mean it. If two edges between 
        # a pair of nodes had different bounds network is immediately
        # not DC.
        assert len(pairs) == len(set(pairs))

        # Two uncontrollable edges cannot end in the same node.
        for node in input_degree:
            assert input_degree[node] <= 1

    @property
    def num_edges(self):
        return len(self.uncontrollable_edges) + len(self.controllable_edges)

