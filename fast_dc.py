from collections import namedtuple, defaultdict
from queue import Queue, PriorityQueue

from stnu import StnuEdge


class EdgeType(object):
    SIMPLE = 1
    LOWER_CASE = 2
    UPPER_CASE = 3

class Edge(object):
    def __init__(self, fro, to, value, typ, maybe_letter=None,renaming=None):
        self.fro = fro
        self.to = to
        self.value = value
        self.type = typ
        self.maybe_letter = maybe_letter
        self.renaming = renaming

    def hash(self):
        return hash((self.fro, self.to, self.value, self.type, self.maybe_letter))
    
    def __hash__(self):
        return self.hash()

    def __eq__(self, other):
        return self.hash() == other.hash()
    

    def _printit(self, fro, to, maybe_letter):
        type_str = ''
        if self.type == EdgeType.UPPER_CASE:
            type_str = 'UC(%s):' % maybe_letter
        elif self.type == EdgeType.LOWER_CASE:
            type_str = 'LC(%s):' % maybe_letter
        return '%s....%s%.1f....>%s' % (fro,
                                    type_str,
                                    self.value,
                                    to)

    def __unicode__(self):
        if self.renaming is None:
            return self._printit(self.fro, self.to, self.maybe_letter)
        else:
            return self._printit(self.renaming[self.fro],
                          self.renaming[self.to],
                          self.renaming[self.maybe_letter] 
                          if self.maybe_letter is not None else None)


    def __str__(self):
        return self.__unicode__()

class DcTester(object):
    def __init__(self, stnu):
        self.stnu = stnu
        self._is_dc = None
        self._update_dc = True
        self._first_time = True

    def is_dynamically_controllable(self):
        # if nothing changed return cached result
        if not self._update_dc:
            return self._is_dc

        if self._first_time:
            # if we calculate DC from the first time use nonincremental
            alg = FastDc()
            self._is_dc = alg.solve(self.stnu)
            self._first_time = False
        else:
            # stub: use incremental algorithm here
            self._is_dc = None 

        self._update_dc = False
        return self._is_dc

class FastDc(object):
    """Implementation based on paper "A Structural Characterization of Temporal
    Dynamic Controllability" by Paul Morris"""

    def generate_graph(self, network):
        """Generates graph of edges as in section 2.2"""
        num_nodes = network.num_nodes
        edge_list = []

        renaming = { k:v for k,v in network._inverse_renaming.items() }

        def add_controllable(e):        
            edge_list.append(Edge(e.fro, e.to, e.upper_bound, EdgeType.SIMPLE, renaming=renaming))
            edge_list.append(Edge(e.to, e.fro, -e.lower_bound, EdgeType.SIMPLE, renaming=renaming))

        def add_uncontrollable(e):
            edge_list.append(Edge(e.fro, e.to, e.upper_bound, EdgeType.SIMPLE, renaming=renaming))
            edge_list.append(Edge(e.to, e.fro, -e.lower_bound, EdgeType.SIMPLE, renaming=renaming))
            edge_list.append(Edge(e.to, e.fro, -e.upper_bound, EdgeType.UPPER_CASE, e.to, renaming=renaming))
            edge_list.append(Edge(e.fro, e.to, e.lower_bound, EdgeType.LOWER_CASE, e.to, renaming=renaming))

        for e in network.controllable_edges:
            add_controllable(e)

        new_index = 1
        for e in network.uncontrollable_edges:
            # contingent edges with bounds [l, u] can be normalized to edge
            # can be replaced by requirement edge [l,l] followed by upper case edge
            if e.lower_bound == 0:
                add_uncontrollable(e)
            else:
                new_node = num_nodes + 1
                num_nodes += 1
                renaming[new_node] = '%s_%d' % (renaming[e.fro], new_index)
                new_index += 1 
                add_controllable(StnuEdge(e.fro, new_node, e.lower_bound, e.lower_bound))
                add_uncontrollable(StnuEdge(new_node, e.to, 0, e.upper_bound - e.lower_bound))

        return num_nodes, edge_list

    def allmax(self, num_nodes, edge_list):
        """Calculates allmax projection of STNU (see section 2.2). 

        Returns two parameters:
            success - true if consistent
            potentials - potentials for use in Dijkstra algorithm
        """
        weights = {}
        neighbor_list = defaultdict(lambda: set())

        for e in edge_list:
            if e.type != EdgeType.LOWER_CASE:
                pair = (e.fro, e.to)
                neighbor_list[e.fro].add(e.to)
                if not pair in weights or weights[pair] > e.value:
                    weights[pair] = e.value
        
        # Like in Johnson's algorithm we add node 0 artificially 
        for node in range(1, num_nodes + 1):
            weights[(0,node)] = 0
            neighbor_list[0].add(node)

        source = 0
        terminated, distances = self.spfa(source = 0,
                                          num_nodes=num_nodes + 1,
                                          weights=weights,
                                          neighbor_list=neighbor_list)

        return (terminated, distances)
        
    def spfa(self, source, num_nodes, weights, neighbor_list):
        """Shortest Paths Fastests Algorithm - think optimized Bellman-Ford,
        
        Assumes nodes have numbers from 0 to num_nodes (inclusive).

        Returns two variables:
            success - true if no negative cycle
            distances - shortest distance from the source to each node
                        or None if negative cycle
        """
        # None is Infinity
        distance = [None] * num_nodes
        currently_in_queue = [False] * num_nodes
        times_in_queue = [0] * num_nodes
        q = Queue()

        distance[source] = 0
        currently_in_queue[source] = True
        times_in_queue[source] = 1
        q.put(0)

        negative_cycle = False

        while not q.empty() and not negative_cycle:
            node = q.get()
            currently_in_queue[node] = False
            for neighbor in neighbor_list[node]:
                if (distance[neighbor] is None or
                        distance[neighbor] > distance[node] + weights[(node, neighbor)]):
                    distance[neighbor] = distance[node] + weights[(node, neighbor)]
                    if not currently_in_queue[neighbor]:
                        currently_in_queue[neighbor] = True
                        times_in_queue[neighbor] += 1
                        if times_in_queue[neighbor] > num_nodes:
                            negative_cycle = True
                            break
                        q.put(neighbor)

        if negative_cycle:
            return (False, None)
        else:
            return (True, distance)

    def reduce_edge(self, edge1, edge2):
        # X ---edge1----> Y ----edge2----> Z
        assert edge2.fro == edge1.to
        new_fro = edge1.fro
        new_to = edge2.to
        new_value = edge1.value + edge2.value
        new_type = None
        new_maybe_letter = None
        # UPPER CASE REDUCTION
        if edge1.type == EdgeType.SIMPLE and edge2.type == EdgeType.UPPER_CASE:
            new_maybe_letter = edge2.maybe_letter
            new_type = EdgeType.UPPER_CASE
        # LOWER CASE REDUCTION
        elif (edge1.type == EdgeType.LOWER_CASE and edge2.type == EdgeType.SIMPLE and
                edge2.value < 0):
            new_type = EdgeType.SIMPLE
        # CROSS CASE REDUCTION
        elif (edge1.type == EdgeType.LOWER_CASE and edge2.type == EdgeType.UPPER_CASE and
                edge2.value < 0 and edge1.maybe_letter != edge2.maybe_letter):
            new_maybe_letter = edge2.maybe_letter
            new_type = EdgeType.UPPER_CASE
        # NO-CASE REDUCITON
        elif edge1.type == EdgeType.SIMPLE and edge2.type == EdgeType.SIMPLE:
            new_type = EdgeType.SIMPLE

        if new_type is None:
            # not reduction matched
            return None
        
        if new_type == EdgeType.UPPER_CASE:
            # try applying LABEL REMOVAL
            # If you thing about this for our purposes SIMPLE edge is strictly
            # better than upper case.
            if new_value >= 0:
                new_type = EdgeType.SIMPLE
                new_maybe_letter = None

        new_edge = Edge(new_fro, new_to, new_value, new_type, new_maybe_letter, renaming=edge1.renaming)
        #print 'combine \t%s \twith \t%s \tto get \t%s' %(edge1, edge2, new_edge)
        return new_edge

    def reduce_lower_case(self, num_nodes, edge_list, potentials, lc_edge):
        new_edges = set()

        # Notice that here we are going to be using Johnson's algorithm in
        # a nonintuitive way, we will remove some edges from the original
        # graph which we use to calculate potentials. If you look through
        # proof of Johnson's algorithms you will notice that removing edges
        # never invalidate the key properties of potentials
        outgoing_edges = defaultdict(lambda: [])
        for edge in edge_list:
            # Ignore lower case edges and upper-case edges with the same letter as lc_edge
            # (paper terminology: breach)
            if (edge.type == EdgeType.LOWER_CASE or
                    (edge.type == EdgeType.UPPER_CASE and
                    edge.maybe_letter == lc_edge.maybe_letter)):
                continue
            outgoing_edges[edge.fro].append(edge)
        # distance in shortest path's graph
        reduced_edge = [None] * (num_nodes + 1)
        distance = [None] * (num_nodes + 1)
        visited = [False] * (num_nodes + 1)

        source = lc_edge.to
        distance[source] = 0

        q = PriorityQueue()
        q.put((0, source))

        #print 'processing LCE %s' % (lc_edge,)

        while not q.empty():
            _, node = q.get()
            if visited[node]:
                continue
            #print 'visiting %d' % node
            visited[node] = True
            for edge in outgoing_edges[node]:
                neighbor = edge.to
                edge_value_potential = edge.value + potentials[edge.fro] - potentials[edge.to]
                if (distance[neighbor] is None or
                        distance[neighbor] > distance[node] + edge_value_potential):
                    # add calculate reduced edge that lead us here
                    if reduced_edge[node] is None:
                        new_reduced_edge = edge
                    else:
                        new_reduced_edge = self.reduce_edge(reduced_edge[node], edge)
                    if new_reduced_edge is None:
                        # cannot make a reduction
                        continue
                    reduced_edge[neighbor] = new_reduced_edge
                    distance[neighbor] = distance[node] + edge_value_potential
                    q.put((distance[neighbor], neighbor))
                    # This the reduced distance as described in the book, excluding the effect of
                    # potentials
                    real_reduced_distance = distance[neighbor] + potentials[neighbor] - potentials[source]
                    # check if we have a moat
                    if real_reduced_distance < 0:
                        relevant_edge = self.reduce_edge(lc_edge, reduced_edge[neighbor])
                        
                        if relevant_edge is not None:
                            #print '^^ moat ^^'
                            new_edges.add(relevant_edge)

        #for edge in list(new_edges):
        #    print '   %s' % (edge,)

        return list(new_edges)

    def solve(self, network):
        """Implementation of pseudocode from end of section 3"""
        K = len(network.uncontrollable_edges)
        num_nodes, new_edges = self.generate_graph(network)
        #print 'start graph (%d nodes):' % num_nodes
        #for edge in new_edges:
        #    print '    %s' % (edge,)
        completed_iterations = 0
        all_edges = []
        while len(new_edges) > 0 and completed_iterations <= (K+1):
            #print 'iteration %d' % (completed_iterations,)
            all_edges.extend(new_edges)
            new_edges = []
            consistent, potentials = self.allmax(num_nodes, all_edges)
            #print "iteration %d - %s" %(completed_iterations, consistent)
            #print '   allmax check %s' % ('succeeded' if consistent else 'failed')
            if not consistent:
                return False
            for e in all_edges:
                if e.type == EdgeType.LOWER_CASE:
                    reduced_edges = self.reduce_lower_case(num_nodes,
                                                           all_edges,
                                                           potentials,
                                                           e)
                    ##print "considering lc_edge %s" % (e,)
                    ##for e in reduced_edges:
                    ##    print '    adding edge: %s (hash: %d)' % (e, hash(e))
                    new_edges.extend(reduced_edges)
            new_edges = [edge for edge in new_edges if edge not in all_edges]
           
            completed_iterations += 1
        # Assuming the theory from the paper checks out. We need one extra
        # iteration to verify that no edge was actually added.
        assert completed_iterations <= (K+1)
        return all_edges