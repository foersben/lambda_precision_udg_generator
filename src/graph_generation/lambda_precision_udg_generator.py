from typing import Any

from joblib import Parallel, delayed
import networkx as nx
import numpy as np
from random import choice, choices
from scipy.spatial.distance import euclidean as dist
from scipy.spatial import cKDTree, KDTree
import itertools as it
from math import inf

from .random_points_generator import RandomPointsGenerator


class LambdaPrecisionUDG:
    def __init__(self, lambda_precision_udg, lambda_precision_points, radius):
        if lambda_precision_udg is None:
            points = lambda_precision_points.get_lambda_precision_points()
            pos = {i: (points[i][0], points[i][1]) for i in range(len(points))}
            lambda_precision_udg = nx.random_geometric_graph(len(points), radius, pos=pos)

        self.graph = lambda_precision_udg
        self.lambda_precision_points = lambda_precision_points
        self.radius = radius

    def clone(self):
        return self.__class__(self.graph.copy(), self.lambda_precision_points, self.radius)

    def get_geometric_graph(self):
        """
        Returns the NetworkX geometric graph based on the lambda precision points and radius.

        """
        points = self.lambda_precision_points.get_lambda_precision_points()
        return nx.random_geometric_graph(len(points), self.radius,
                                         pos={i: (points[i][0], points[i][1]) for i in range(len(points))}, )

    def average_degree(self):
        """Returns the average node degree of a graph

        :return: average node degree
        """
        return np.mean([degree for _, degree in self.graph.degree()])

    def graph_connectivity(self, n_node_connectivity=None, n_edge_connectivity=None):
        """
        Determines node and edge connectivity of a graph

        :param n_node_connectivity:
        :param n_edge_connectivity:
        :return:
        """
        if not (n_node_connectivity or n_edge_connectivity):
            return list(nx.bridges(self.graph))
        elif isinstance(n_node_connectivity, int) and n_node_connectivity > 0 and not n_edge_connectivity:
            # determines node connectivity
            return nx.node_connectivity(self.graph)
        elif isinstance(n_edge_connectivity, int) and n_edge_connectivity > 0 and not n_node_connectivity:
            # determines edge-connectivity
            return nx.edge_connectivity(self.graph)

    def _removable_edges(self, bridges=True):
        """
        Determines list of edges removable so graph stays bridge-free/connected

        :param bridges:     Bool to determine whether bridges should be removed
        :return:            List of edges removable so graph stays bridge-free/connected
        """
        if bridges:
            copy = self.graph.copy()
            copy.remove_edges_from(nx.bridges(copy))
            return list(copy.edges())
        else:
            """
            Determines list of edges removable so graph stays bridge-free
            or doesn't add bridges
            """
            graph = self.graph
            edges = list(graph.subgraph([node for node, degree in graph.degree() if degree > 2]).edges())
            removable = []
            while edges:
                edge = list(edges.pop())
                sg = graph.copy()
                sg.remove_edge(*edge)
                sg_bridges = list(nx.bridges(sg))
                if sg_bridges and sg_bridges != list(nx.bridges(graph)):
                    # print(f"Bridges present in graph: {list(nx.bridges(graph))}")
                    # print(f"Edge: {edge}, bridges found: {sg_bridges}")
                    for e in sg_bridges:
                        if e in edges:
                            edges.remove(e)
                else:
                    removable.append(edge)
            # print(f"Removable edges (_edge_removal): {len(removable)}")
            return removable

    def _edge_weights(self, edges, exponent=1):
        node_ids, node_coords = list(zip(*self.graph.subgraph(list(it.chain(*edges))).nodes(data="pos")))
        edge_dist = {tuple(sorted(edge)): dist(node_coords[node_ids.index(edge[0])],
                                               node_coords[node_ids.index(edge[1])], ) ** exponent for edge in edges}
        return list(edge_dist.keys()), list(edge_dist.values())

    def _random_choice(self, edges=[], exponent=1, weights=False):
        return (choices(*self._edge_weights(edges, exponent))[0] if weights else choice(edges))

    def reduce_avg_degree(self, target_avg_deg, weights=False, exponent=1, attempts=3, bridges=True, ):
        """
        The algorithm can enter an endless loop at this point because the requirements of the target_avg_deg and the
        achievable degree based on given constraints, seem to be mutually exclusive at times
        => therefore, another escape path noticing such a loop state is mandatory

        :param target_avg_deg:  target average degree
        :param weights:         bool to determine whether weights should be used
        :param exponent:        exponent to penalise edge length
        :param attempts:        number of attempts to reach target_avg_deg
        :param bridges:         bool to determine whether only edges not causing bridges should be removed
        """
        graph_memory = self.clone()
        for _ in range(attempts):
            clone = self.clone()
            removable_edges = clone._removable_edges(bridges=bridges)
            if bridges:
                while clone.average_degree() > target_avg_deg and removable_edges:
                    selected_edge = clone._random_choice(removable_edges, exponent=exponent, weights=weights)
                    clone.graph.remove_edge(*selected_edge)
                    removable_edges.remove(selected_edge)
                    removable_edges = clone._removable_edges(bridges=bridges)
            else:
                bridge_hits = 0
                # n = 10
                n = 1
                while clone.average_degree() > target_avg_deg and removable_edges:
                    selected_edges = [clone._random_choice(removable_edges, exponent=exponent, weights=weights) for _ in
                                      range(int(n))]
                    copy = clone.graph.copy()
                    copy.remove_edges_from(selected_edges)
                    if (  # compare_list_of_tuples(
                            #     list(nx.bridges(copy)), list(nx.bridges(self.graph))
                            # )
                            set([frozenset(elem) for elem in list(nx.bridges(copy))]) == set(
                        [frozenset(elem) for elem in
                         list(nx.bridges(self.graph))]) and clone.average_degree() > target_avg_deg):
                        clone.graph = copy
                        list(map(lambda edge: removable_edges.remove(edge), selected_edges))
                        # n += 5
                        bridge_hits = 0
                    else:
                        n = 1  # max(n / 2, 1)
                        bridge_hits += 1
                    if bridge_hits > 3 or not removable_edges:
                        removable_edges = clone._removable_edges(bridges=bridges)
                        n = 1
                        bridge_hits = 0
                    if clone.average_degree() < graph_memory.average_degree():
                        graph_memory.graph = clone.graph.copy()
            if clone.average_degree() > target_avg_deg:
                clone = graph_memory
                print(f"Target average degree couldn't be achieved for given parameters. \
                            Minimum achieved average degree {clone.average_degree()}")
            return clone

    @staticmethod
    def _k_nearest_neighbours(sg1, sg2, knn, knn_queue=None):
        """Computes k-nearest-neighbours using K-d Tree algorithm on two subgraphs of a
            larger graph and returns a list of edges as tuples of nodeIds

        @param sg1  NetworkX graph as subgraph representing a connected component of a
                    larger graph
        @param sg2  NetworkX graph as subgraph representing a connected component of a
                    larger graph
        @param knn  Number of nearest neighbours to determine
        @return     Edges as tuples of nodeIds
        """

        def _queue_in_order(lst, elem):
            """Manages list as descending ordered queue of floating numbers while
                maintaining its length.

            @param lst List that will be managed as descending ordered queue of
                floating numbers
            @param elem Element that shall be added to the list
            """
            for i in range(len(lst)):
                if elem[1] < lst[i][1]:
                    lst.insert(i, elem)
                    lst.pop()
                    return
            return

        # if sg1 has less nodes than sg2 swap them
        if sg1.number_of_nodes() < sg2.number_of_nodes():
            sg1, sg2 = sg2, sg1

        sg1_nodes_pos = sg1.nodes(data="pos")
        sg1_nodes, sg1_coords = list(zip(*sg1_nodes_pos))
        sg2_nodes_pos = sg2.nodes(data="pos")
        sg2_nodes, sg2_coords = list(zip(*sg2_nodes_pos))

        kdtree = cKDTree(sg1_coords)
        if knn_queue is None:
            knn_queue = [((-1, -1), inf)] * knn
        nn = kdtree.query(sg2_coords, k=knn)
        if knn == 1:
            for i in range(len(nn[0])):
                _queue_in_order(knn_queue, ((sg2_nodes[i], sg1_nodes[nn[1][i]]), nn[0][i]))
        else:
            for i in range(len(nn[0])):
                for j in range(len(nn[0][0])):
                    _queue_in_order(knn_queue, ((sg2_nodes[i], sg1_nodes[nn[1][i][j]]), nn[0][i][j]))
        return knn_queue

    @staticmethod
    def _knn(graph: nx.Graph, node: int, k: int = 1):
        node_pos = [graph.nodes[node]['pos'] for node in graph.nodes]
        kdtree = KDTree(node_pos)
        _, neighbor_indices = kdtree.query(node_pos[node], k=k + 1)

        nearest_neighbors = [list(graph.nodes)[i] for i in neighbor_indices[1:]]
        return nearest_neighbors

    @staticmethod
    def _subgraph_boundaries(graph: nx.Graph, boundaries, outer=False):
        """Returns a subgraph with nodes that fulfil given boundary conditions

        @param graph        Networkx Graph
        @param boundaries   List of Lists of coordinates
                                outer list - size coordinate dimensions
                                inner lists - min and max value for each dimension
        @param outer        Bool to determines whether we want the subgraph within or
                            outside of the given boundaries
        @return             Networkx graph, a subgraph of the given graph with nodes
                            within given boundaries
        """

        def check_boundaries(node):
            """Returns whether a node is within given boundaries

            @param node List of tuples, contains nodeId and tuple of node coordinates
            @return     True if node is within given boundaries else False
            """
            for i in range(len(boundaries)):
                if not (boundaries[i][0] <= node[1][i] <= boundaries[i][1]):
                    return False
            return True

        if not outer:
            return graph.subgraph([node[0] for node in graph.nodes(data="pos") if check_boundaries(node)])
        else:
            return graph.subgraph([node[0] for node in graph.nodes(data="pos") if not (check_boundaries(node))])

    def _connect_components(self, lcc_sg, cc_sg, knn):
        """Connects two disjoint connected components of the same graph using K-d Tree
            and k-nearest-neighbour search
        https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        Furthermore, we space partition the graph components by considering the smaller
        components min/max x/y value -/+ radius of rgg and check first k-NN of the nodes
        of the remaining nodes within this area. Afterwards when at least k nodes have
        been found within we extend the area by the largest distance of the k determined
        node pairs and determine whether there is a possible candidate replacing
        previously found k-NNs node pairs

        @param lcc_sg   Networkx graph as subgraph of the largest connected component of
                        a given graph
        @param cc_sg    Networkx graph as subgraph of a connected component of a given
                        graph
        @param knn      k-nearest-neighbours, number of edges that have to be found
                        between given connected components
        @return         k-nearest-neighbours determined edges
        """
        if len(cc_sg.nodes) < 4:
            return np.array(LambdaPrecisionUDG._k_nearest_neighbours(cc_sg, lcc_sg, knn), dtype=object, )[:, 0]

        radius = self.radius
        csg_nodes_pos = cc_sg.nodes(data="pos")
        _, csg_coords = list(zip(*csg_nodes_pos))
        x, y = list(zip(*csg_coords))
        boundaries = [(max(min(x) - radius, 0), min(max(x) + radius, 1)),
                      (max(min(y) - radius, 0), min(max(y) + radius, 1)), ]
        lcsg_bound = self._subgraph_boundaries(lcc_sg, boundaries)

        if lcsg_bound.number_of_nodes() == 0:
            knn_edges = LambdaPrecisionUDG._k_nearest_neighbours(cc_sg, lcc_sg, knn)
        else:
            knn_edges = LambdaPrecisionUDG._k_nearest_neighbours(cc_sg, lcsg_bound, knn)
            if (np.array(knn_edges, dtype=object)[:, 0][-1])[1] == inf:
                lcsg_bound = LambdaPrecisionUDG._subgraph_boundaries(lcc_sg, boundaries, outer=True)
                knn_edges = LambdaPrecisionUDG._k_nearest_neighbours(cc_sg, lcsg_bound, knn, knn_edges)
            else:
                nn = (np.array(knn_edges, dtype=object)[:, 0][-1])[1]
                sg_bound = LambdaPrecisionUDG._subgraph_boundaries(lcc_sg, boundaries, outer=True)
                boundaries = [(max(min(x) - radius - nn, 0), min(max(x) + radius + nn, 1)),
                              (max(min(y) - radius - nn, 0), min(max(y) + radius + nn, 1)), ]
                lcsg_bound = LambdaPrecisionUDG._subgraph_boundaries(sg_bound, boundaries)
                if lcsg_bound.number_of_nodes() != 0:
                    knn_edges = LambdaPrecisionUDG._k_nearest_neighbours(cc_sg, lcsg_bound, knn, knn_edges)
        return np.array(knn_edges, dtype=object)[:, 0]

    def connect_graph_components(self, knn=1, connect_to_largest_connected_subgraph=False,
                                 connect_to_center_of_gravity=False):
        """
        Connects all connected components of a given graph

        @param graph                                    NetworkX random geometric graph
        @param knn                                      Number of edges with whom the
                                                        largest connected component of
                                                        the given graph will be
                                                        connected to each of the
                                                        possible other connected
                                                        components
        @param connect_to_largest_connected_subgraph    Connect all connected components
                                                        directly with largest connected
                                                        subgraph
        @param connect_to_center_of_gravity             Connect all connected components
                                                        with the closest connected
                                                        component recursively
        @return                                         Returns a connected random
                                                        geometric graph
        """

        def center_of_gravity(graph):
            """Determines the center of gravity of a given graph

            @param graph    NetworkX Graph
            @return         Coordinate of the center of gravity
            """
            _, coords = list(zip(*graph.nodes(data="pos")))
            return tuple([sum(x[i] for x in coords) / len(coords) for i in range(len(coords[0]))])

        graph = self.graph
        connected_sgs = [graph.subgraph(cc).copy() for cc in nx.connected_components(graph)]
        if len(connected_sgs) == 1:
            return
        if connect_to_largest_connected_subgraph:
            largest_connected_sg = connected_sgs.pop(connected_sgs.index(max(connected_sgs, key=len)))
            edges = []
            for sg in connected_sgs:
                edges.append(self._connect_components(largest_connected_sg, sg, knn))
        elif connect_to_center_of_gravity:
            # BUG includes self ref edges - reason unknown - further resulting errors
            # can't be excluded
            cog = [center_of_gravity(csg) for csg in connected_sgs]
            pairs = list(set(
                tuple(sorted([center, (cKDTree(cog[:center] + cog[center + 1:])).query(cog[center])[1], ])) for center
                in range(len(cog))))
            edges = []
            for pair in pairs:
                edges.append(self._connect_components(connected_sgs[pair[0]], connected_sgs[pair[1]], knn))
        else:
            smallest_connected_sg = connected_sgs.pop(connected_sgs.index(min(connected_sgs, key=len)))
            graph_copy = graph.copy()
            graph_copy.remove_nodes_from(smallest_connected_sg.nodes())
            edges = [self._connect_components(graph_copy, smallest_connected_sg, knn)]

        graph.add_edges_from([tuple(edge[0]) for edge in edges])
        self.connect_graph_components(knn, connect_to_largest_connected_subgraph, connect_to_center_of_gravity, )

    def augment_bridges_knn(self):
        """
        Augments bridges by adding additional edges.

        There are two cases:
        1. The bridge connects two connected components with more than one node each.
           - an edge is added between the two closest nodes (excluding the nodes of the bridge) of the components (KNN)
        2. The bridge connects two connected components with only one node each.
              - explained in the paper - something termed chain of bridges / bridge path or similar
              - specifically handled joining every other node of the path with an additional edge
        """
        from networkx.algorithms.connectivity.edge_kcomponents import bridge_components
        from networkx.exception import NetworkXError

        graph = self.graph
        bridges = list(nx.bridges(graph))
        if bridges:
            edges = [tuple(set(i).symmetric_difference(set(j))) for i in bridges for j in bridges if
                     len(set(i).intersection(set(j))) == 1]
            # for i in bridges for j in bridges if set(i).intersection(set(j))]
            self.graph.add_edges_from(edges)
            for bridge in list(nx.bridges(graph)):
                cc_nodes = [cc for cc in list(bridge_components(graph)) if set(bridge).intersection(set(cc))]
                for i in range(len(cc_nodes)):
                    if len(cc_nodes[i]) > 1:
                        cc_nodes[i] = set(cc_nodes[i]).difference(set(bridge))
                edges.append(tuple(*self._connect_components(*[graph.subgraph(cc) for cc in cc_nodes], knn=1)))
                graph.add_edge(*edges[-1])
            node_ids, node_coords = list(zip(*graph.subgraph(list(it.chain(*edges))).nodes(data="pos")))
            edge_dist = {edge: dist(node_coords[node_ids.index(edge[0])], node_coords[node_ids.index(edge[1])], ) for
                         edge in edges}
            for edge in sorted(edge_dist, key=edge_dist.get, reverse=True):
                graph_copy = graph.copy()
                try:
                    graph_copy.remove_edge(*edge)
                    if len(list(nx.bridges(graph_copy))) == 0:
                        graph.remove_edge(*edge)
                except NetworkXError:
                    pass

    def draw_random_geometric_graph(self, filepath=None, custom="", partitioning=None):
        """
        Plots networkx graphs

        :param filepath: path to save the plot
        """
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib import pyplot as plt
        import os

        print("Drawing graph...")
        pos = nx.get_node_attributes(self.graph, "pos")

        # find node near center (0.5,0.5)
        d_min = 1
        for node in pos:
            x, y = pos[node]
            d = (x - 0.5) ** 2 + (y - 0.5) ** 2
            if d < d_min:
                node_center = node
                d_min = d

        # color by path length from node near center
        # colors = dict(nx.single_source_shortest_path_length(self.graph, node_center))
        node_color = None
        if partitioning:
            from distinctipy import distinctipy
            partitioning = dict(partitioning)
            print(f"Partitioning: {partitioning}")
            colors = distinctipy.get_colors(len(set(partitioning.values())))
            print(f"Colors: {colors}")
            # node_colors = [colors[partitioning.get(node) - 1] for node in self.graph.nodes]
            node_color = []
            for node in self.graph.nodes:
                print(f"partition_set_id: {partitioning.get(node) - 1}")
                print(f"partition node color: {node}: {colors[partitioning.get(node) - 1]}")
                node_color.append(colors[partitioning.get(node) - 1])

            # cmap = plt.get_cmap("tab10")
            # node_color = [cmap(color) for color in node_color]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(self.graph, pos)  # , alpha=0.4)
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            # nodelist=list(colors.keys()),
            node_size=80,
            # node_color=list(colors.values()),
            node_color=node_color,
            cmap=plt.cm.Reds_r
        )

        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.axis("off")

        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            # filename = f"{filepath}/udg_{str(self.graph.number_of_nodes())}_nodes_{str(self.radius)}_radius{'_' if custom else ''}{custom}.svg"
            filename = f"{filepath}/udg_{str(self.graph.number_of_nodes())}_nodes_{str(self.lambda_precision_points.get_min_dist())}_lambda{'_' if custom else ''}{custom}.svg"

            print(f"Filename: {filename}")
            plt.savefig(filename, format="svg")
        else:
            plt.show()
            plt.close()

    def serialize(self, path):
        """
        Serializes the object

        :param path:    path to save the object
        """
        import pickle

        with open(f"{path}/{id(self)}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(cls, filepath):
        """
        Deserializes the object

        :param filepath:    path to load the object from
        :return:            deserialized object
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)


class LambdaPrecisionUDGGenerator:
    """
    Generates random geometric graphs with lambda-precision (minimal distance) in between nodes
    """

    def __init__(self, random_points_generator: RandomPointsGenerator, radius: float):
        self.random_points_generator = random_points_generator
        self.radius = radius

    def generate_graph(self, connected: bool = False):
        """
        Generates a random geometric graph with a minimum distance between nodes

        :param connected: whether the generation is repeated until a connected graph is generated
        :return: LambdaPrecisionUDG
        """
        while True:
            lpp = self.random_points_generator.generate_points()
            while not lpp:
                lpp = self.random_points_generator.generate_points()
            points = lpp.get_lambda_precision_points()
            pos = {i: (points[i][0], points[i][1]) for i in range(len(points))}
            graph = nx.random_geometric_graph(len(points), self.radius, pos=pos)
            if not connected or nx.is_connected(graph):
                print(f"connected = True: {nx.is_connected(graph)}")
                return LambdaPrecisionUDG(graph, lpp, self.radius)

    def generate_graphs_parallel(self, number: int, prefer: Any = None, connected: bool = False):
        """
        Generates for a given number as many graphs in parallel
        using the Joblib library

        :param number:      number of graphs to generate
        :param prefer:      argument for joblib about the preferred way to parallelise
        :param connected:   whether to generate specifically connected graphs

        :return:            list of generated graphs
        """

        return Parallel(n_jobs=-1, prefer=prefer)(delayed(self.generate_graph)(connected) for _ in range(number))

    def generate_graphs(self, number: int, connected: bool = False):
        """
        Generates for a given number as many graphs

        :param number:      number of graphs to generate
        :param connected:   whether to generate specifically connected graphs
        :return:            list of generated graphs
        """
        return [self.generate_graph(connected) for _ in range(number)]


if __name__ == "__main__":
    # from .random_points_generator import RandomPointsGenerator

    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.048588),
                                                radius=0.065625)
        graph = generator.generate_graph()

        print(f"Is connected: {nx.is_connected(graph.graph)}")
        print(f"Average degree: {graph.average_degree()}")
        print(f"Coverage: {graph.lambda_precision_points.get_density()}")

    # if not nx.is_connected(graph.graph):
    #     graph.connect_graph_components()

    # print(f"Is connected: {nx.is_connected(graph.graph)}")

    # print(f"Bridges: {list(nx.bridges(graph.graph))}")

    # if list(nx.bridges(graph.graph)):
    #     graph.augment_bridges_knn()

    # print(f"Bridges: {list(nx.bridges(graph.graph))}")
