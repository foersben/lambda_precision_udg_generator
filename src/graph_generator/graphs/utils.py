import networkx as nx
from networkx.algorithms.connectivity.edge_kcomponents import bridge_components
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean as dist
from random import choice, choices
from itertools import combinations

from graph_generator.graphs.lambda_precision_udg import LambdaPrecisionUDG

""" This module contains utility functions for graph analysis and manipulation, specifically for LambdaPrecisionUDG graphs. It includes functions for calculating graph connectivity, determining removable edges, computing edge weights, and connecting graph components using k-nearest neighbors.

TODO: still connected to the original class (self) - adjust to be independent functions instead of methods

NOTES: 
    * list_bridges() replaced by nx.bridges(graph)
    * compute_node_connectivity() replaced by nx.node_connectivity(graph)
    * compute_edge_connectivity() replaced by nx.edge_connectivity(graph)
"""


def _removable_edges(graph: nx.Graph, preserve_bridges: bool = False) -> list[tuple[int, int]]:
    """ Determines the list of edges that can be removed from the graph either to ensure it remains bridge-free or, if specified, to prevent the addition of any bridges.

    If the parameter `bridges` is set to True, the method returns a list of edges that do not form bridges and are removable while keeping the graph bridge-free. If `bridges` is False, the method computes edges that can be safely removed based on subgraphs such that any removal avoids introducing any additional bridges.

    Args:
        preserve_bridges: A boolean flag indicating whether to preserve bridge-free characteristics in the graph after edges are removed. If set to True, edges form no bridges. If set to False, computes removable edges through subgraph processing.

    Returns:
        List of removable edges from the graph based on the `bridges` flag.
    """

    bridges = {tuple(sorted(edge)) for edge in nx.bridges(graph)}

    removable = []
    if preserve_bridges:
        # Determines list of edges removable so graph stays bridge-free or doesn't add bridges
        # edges = list(graph.subgraph([node for node, degree in graph.degree() if degree > 2]).edges())
        edges = {tuple(sorted(edge)) for edge in
                 graph.edges(nbunch=[node for node, degree in graph.degree() if degree > 2])
                 if tuple(sorted(edge)) not in bridges}

        while edges:
            edge = tuple(edges.pop())
            working_graph = graph.copy()
            working_graph.remove_edge(*edge)
            new_bridges = {tuple(sorted(edge)) for edge in nx.bridges(working_graph)}
            if new_bridges > bridges:
                for bridge in bridges:
                    if bridge in edges:
                        edges.remove(bridge)
            else:
                removable.append(edge)
    else:
        working_graph = graph.copy()
        working_graph.remove_edges_from(list(bridges))
        removable = list(working_graph.edges())

    return removable


def _edge_weights(
        graph: nx.Graph,
        edges: list[tuple[int, int]],
        exponent: int = 1
) -> tuple[list[tuple[int, ...]], list[float]]:
    """ Calculates weights for the given set of edges in the graph based on the distance between the nodes of each # # edge. The distances are raised to the power of the provided exponent value.

    Args:
        graph:      The input graph from which edges are selected.
        edges:      A list of tuples, where each tuple represents an edge by specifying a pair of connected nodes.
        exponent:   The power to which the calculated distances are raised for generating the weights. Defaults to 1.
    Returns:
        A tuple containing two elements:
            1. A list of edges as tuples of node IDs
            2. A list of numeric weights corresponding to each edge
    """

    pos = nx.get_node_attributes(graph, 'pos')
    edge_dist = {tuple(sorted((u, v))): dist(pos[u], pos[v]) ** exponent for (u, v) in edges}
    return list(edge_dist.keys()), list(edge_dist.values())


def _random_choice(
        graph: nx.Graph,
        edges: list[tuple[int, int]],
        exponent: int = 1,
        use_weights: bool = False
) -> tuple[int, int]:
    """ Selects a random edge from a given list of edges, optionally based on weighted probabilities, and raises it#  to a specified exponent power if weights are used.

    Args:
        graph:          The input graph from which edges are selected.
        edges:          List of edges from which to select a random edge.
        exponent:       The exponent to which weights are raised when calculating probabilities, applicable only if#  weights are enabled.
        use_weights:    A flag indicating whether the selection process should use weighted probabilities.

    Returns:
        The selected edge chosen randomly from the provided edges, either weighted or unweighted based on the # weights flag.

    Raises:
        ValueError: If the list of edges is empty, indicating that no edges are available for selection.
    """

    if not edges:
        raise ValueError("Cannot choose from empty edge list")

    return choices(*_edge_weights(graph, edges, exponent))[0] if use_weights else choice(edges)


def reduce_avg_degree(
        graph: LambdaPrecisionUDG,
        target_avg_deg: float,
        use_edge_weights: bool = False,
        edge_len_exponent: int = 1,
        attempts: int = 3,
        preserve_bridges: bool = False
) -> LambdaPrecisionUDG | None:
    """ Reduce graph's average degree toward target over multiple attempts.

    Args:
        graph: Input graph to prune
        target_avg_deg: average degree to reach by removing edges
        use_edge_weights: Weight removals by distance if True.
        edge_len_exponent: Exponent for weighting function.
        attempts: Number of independent trials to avoid bad local minima.
        preserve_bridges: If True, never remove or create additional bridges, if False, maintain connectedness of # the graph.

    Returns:
        The best-pruned graph, or None, if target average degree couldn't be achieved.
    """

    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    best_graph = graph.clone()
    for _ in range(attempts):
        working_graph = graph.clone()
        removable_edges = _removable_edges(working_graph, preserve_bridges=preserve_bridges)
        if preserve_bridges:
            bridge_hits = 0
            n = 1
            while working_graph.average_degree() > target_avg_deg and removable_edges:
                selected_edges = [
                    _random_choice(working_graph, removable_edges, exponent=edge_len_exponent,
                                   use_weights=use_edge_weights)
                    for _ in range(int(n))]
                trial_graph = working_graph.copy()
                trial_graph.remove_edges_from(selected_edges)
                if {frozenset(edge) for edge in nx.bridges(trial_graph)} == bridges:
                    working_graph = trial_graph
                    if working_graph.average_degree() < best_graph.average_degree():
                        best_graph = working_graph.copy()
                    list(map(lambda edge: removable_edges.remove(edge), selected_edges))
                    # n += 5
                    bridge_hits = 0
                else:
                    n = 1  # max(n / 2, 1)
                    bridge_hits += 1
                if bridge_hits > 3 or not removable_edges:
                    removable_edges = _removable_edges(working_graph, preserve_bridges=preserve_bridges)
                    n = 1
                    bridge_hits = 0
        else:
            while working_graph.average_degree() > target_avg_deg and removable_edges:
                selected_edge = _random_choice(working_graph, removable_edges, exponent=edge_len_exponent,
                                               use_weights=use_edge_weights)
                working_graph.remove_edge(*selected_edge)
                removable_edges.remove(selected_edge)
                removable_edges = _removable_edges(working_graph, preserve_bridges=preserve_bridges)
            best_graph = working_graph

        best_avg_deg = best_graph.average_degree()
        if best_avg_deg > target_avg_deg:
            print(f"Target average degree couldn't be achieved for given parameters. \
                        Minimum achieved average degree {best_avg_deg}")
        return best_graph
    # TODO: always return best_graph, even if the target_avg_deg couldn't be achieved - before changing the usages have to be corrected in the code everywhere else
    return None


def _k_nearest_neighbors(graph: nx.Graph, nodes1: list[int], nodes2: list[int], k: int) -> list[tuple[int, int]]:
    """ Identifies and connects k-nearest neighbours between two sets of nodes in a graph based on their spatial positions. The function utilises a k-d tree for fast nearest neighbour search and returns edges sorted by distance.

    Args:
        graph: A graph where each node has a 'pos' attribute representing its position in space.
        nodes1: A list of node IDs in the graph from which the neighbours are queried.
        nodes2: A list of node IDs for which the nearest neighbours are identified in nodes1.
        k: The number of nearest neighbours to find for each node in nodes2.

    Returns:
        A list of tuples representing edges between nodes2 and their k-nearest neighbours in nodes1.
    """

    pos1 = np.array([graph.nodes[n]['pos'] for n in nodes1])
    pos2 = np.array([graph.nodes[n]['pos'] for n in nodes2])

    tree = cKDTree(pos1)
    distances, indices = tree.query(pos2, k=k)

    # Handle k=1 case by adding dimension
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    edges = []
    for i, node2 in enumerate(nodes2):
        for d, idx in zip(distances[i], indices[i]):
            edges.append((node2, nodes1[idx]))

    # Sort by distance and return top k
    return sorted(edges, key=lambda x: x[1])[:k]


def connect_components(graph: LambdaPrecisionUDG, k: int = 1, strategy: str = "largest") -> None:
    """ Connects disconnected components of a given graph using a specified strategy.

    This function operates on a graph with disconnected components and connects
    them based on one of the predefined strategies: 'largest', 'cog', or
    'progressive'. The function modifies the graph in place by adding edges.

    Args:
        graph: The graph where disconnected components need to be connected. The graph is assumed to be an undirected graph with positional node data.
        k: The number of neighbors to consider when determining connections between components. Default is 1.
        strategy: The strategy to use for connecting components. Options include:
            - 'largest': Connect all components to the largest one.
            - 'cog': Connect components based on the center of geometry.
            - 'progressive': Iteratively connect components to their closest neighbor.
            Default is 'largest'.
    """

    components = list(nx.connected_components(graph))

    if len(components) == 1:
        return

    if strategy == "largest":
        largest = max(components, key=len)
        for component in components:
            if component != largest:
                edges = _k_nearest_neighbors(graph, list(largest), list(component), k)
                graph.add_edges_from(edges)

    elif strategy == "cog":
        cogs = []
        for component in components:
            positions = [graph.nodes[n]['pos'] for n in component]
            cog = np.mean(positions, axis=0)
            cogs.append(cog)

        tree = cKDTree(cogs)
        _, pairs = tree.query(cogs, k=2)

        for i, j in pairs:
            if i != j and i < j:
                edges = _k_nearest_neighbors(graph, list(components[i]), list(components[j]), k)
                graph.add_edges_from(edges)

    else:  # Progressive connection
        while len(components) > 1:
            closest = []
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components[i + 1:]):
                    edges = _k_nearest_neighbors(graph, list(comp1), list(comp2), 1)
                    if edges:
                        closest.append((edges[0], comp1 | comp2))

            if not closest:
                break

            # Add closest connection
            closest_edge = min(closest, key=lambda x: x[0][1])
            graph.add_edge(*closest_edge[0])

            # Update components
            components = list(nx.connected_components(graph))


def _augment_bridge_chains(graph: nx.Graph, bridges: list[tuple[int, int]]) -> None:
    """ Augments bridge chains in a given graph by creating additional edges derived from initial bridges. This process involves identifying potential chains from the bridge connections and adding them to the graph. Finally, the graph is visually depicted with segmented nodes for further analysis.

    Args:
        graph: The graph on which bridge chains are to be augmented.
        bridges: A list of 2-tuples representing the bridge connections in the graph.
    """

    chain_pairs = [(a, b) for node in {node for edge in bridges for node in edge} for a, b in
                   combinations([v if u == node else u for u, v in bridges if u == node or v == node], 2)]
    graph.add_edges_from(chain_pairs)

    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test1_5.png")


def _select_largest_strategy(
        graph: nx.Graph,
        pos: dict[int, tuple[float, float]],
) -> tuple[int, int]:
    """ Select the best edge to connect a non-largest graph component to the largest component in a graph, with the aim of minimizing the number of bridges in the graph.

    Args:
        graph: A graph representing the network.
        pos: A dictionary where keys are node identifiers and values are tuples of floats representing the positions (coordinates) of nodes.

    Returns:
        A tuple containing two integers, representing the nodes of the edge that connects the largest component to a non-largest component and reduces the bridge count.

    Raises:
        RuntimeError: If no suitable edge can be found that reduces the number of bridges.
    """

    bridges = list(nx.bridges(graph))
    components = list(bridge_components(graph))
    largest_component = max(components, key=len)
    other_components = [component for component in components if component != largest_component]

    candidates = []
    for component in other_components:
        for u in component:
            for v in largest_component:
                if not graph.has_edge(u, v):
                    distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                    candidates.append((distance, u, v))

    candidates.sort()
    for _, u, v in candidates:
        graph.add_edge(u, v)
        if len(list(nx.bridges(graph))) < len(bridges):
            return u, v
        graph.remove_edge(u, v)

    raise RuntimeError("No edge found in 'largest' strategy that reduces bridge count.")


def _select_cog_strategy(
        graph: nx.Graph,
        pos: dict[int, tuple[float, float]],
) -> tuple[int, int]:
    """ Selects two nodes from a given graph's components to connect based on the "cog" strategy.

    This function calculates the geometric centre of each connected component within a graph and employs a k-d tree for efficiently determining the closest pair of components. It then selects two nodes, one from each of the closest components, to potentially form an edge while avoiding direct connections to existing edges. The selection minimises the Euclidean distance between these nodes' positions.

    Args:
        graph: The input graph, represented as a NetworkX Graph object.
        pos: A dictionary mapping node IDs to their coordinates as tuples of floats.

    Returns:
        A tuple containing two integers, each representing a node ID, corresponding to the pair of nodes selected for connection.

    Raises:
        RuntimeError: If no suitable edge can be found that reduces the number of bridges in the graph.
    """

    bridges = list(nx.bridges(graph))
    components = list(bridge_components(graph))

    if len(components) < 2:
        raise RuntimeError("Need ≥2 components for 'cog'")

    centroids = np.vstack([np.mean([pos[n] for n in comp], axis=0) for comp in components])

    i, j = min(combinations(range(len(centroids)), 2),
               key=lambda pair: np.linalg.norm(centroids[pair[0]] - centroids[pair[1]]))
    c1, c2 = components[i], components[j]

    candidates = [(np.linalg.norm(np.array(pos[u]) - pos[v]), u, v) for u in c1 for v in c2 if not graph.has_edge(u, v)]
    if not candidates:
        raise RuntimeError("No available edge for 'cog' strategy")

    candidates.sort()
    for _, u, v in candidates:
        graph.add_edge(u, v)
        if len(list(nx.bridges(graph))) < len(bridges):
            return u, v
        graph.remove_edge(u, v)
    raise RuntimeError("No edge found in 'largest' strategy that reduces bridge count.")


def _select_smallest_strategy(
        graph: nx.Graph,
        pos: dict[int, tuple[float, float]],
) -> tuple[int, int]:
    """ Selects and adds the edge which minimises the number of graph bridges and connects the smallest component in `components` to one of the others by utilising graphs and positional distances between nodes. Sorting and evaluating potential candidates ensures the optimal selection meeting the criteria. Raises a RuntimeError if no edge can provide the desired reduction in bridges.

    Args:
        graph: The undirected graph to be analysed and modified.
        pos: Dictionary mapping node IDs to their (x, y) coordinates, aiding in the calculation of distances between nodes.

    Returns:
        A tuple containing the node IDs of the newly added edge that minimises the number of bridges in the graph.

    Raises:
        RuntimeError: If no edge can be found that reduces the number of bridges in the graph.
    """

    bridges = list(nx.bridges(graph))
    components = list(bridge_components(graph))

    smallest_component = min(components, key=len)
    other_components = [component for component in components if component != smallest_component]
    candidates = [(np.linalg.norm(np.array(pos[u]) - np.array(pos[v])), u, v) for component in other_components
                  for u in component for v in smallest_component if not graph.has_edge(u, v)]
    candidates.sort()
    for _, u, v in candidates:
        graph.add_edge(u, v)
        if len(list(nx.bridges(graph))) < len(bridges):
            return u, v
        graph.remove_edge(u, v)
    raise RuntimeError("No edge found in 'largest' strategy that reduces bridge count.")


def _prune_redundant_edges(
        graph: nx.Graph,
        initial_edges: set[frozenset[int]],
        pos: dict[int, tuple[float, float]]
) -> None:
    """ Prunes redundant edges from the provided graph while preserving its connected components. Redundant edges are removed based on their Euclidean distance, starting with the longest. The function ensures no bridges are introduced as a result of edge removal.

    Args:
        graph: The graph from which to prune edges.
        initial_edges: The set of edges present initially in the graph, used to identify newly added edges that can potentially be pruned.
        pos: A dictionary mapping each node in the graph to its position in a Cartesian coordinate system.
    """

    def euclid(a: int, b: int) -> float:
        return float(np.linalg.norm(np.array(pos[a]) - np.array(pos[b])))

    added_edges = [edge for edge in graph.edges() if frozenset(edge) not in initial_edges]
    for u, v in sorted(added_edges, key=lambda edge: euclid(*edge), reverse=True):
        graph.remove_edge(u, v)
        if list(nx.bridges(graph)):
            graph.add_edge(u, v)


def augment_bridges_knn(graph: nx.Graph, strategy: str = "largest") -> None:
    """ Augments bridges in a graph by iteratively linking components based on the selected strategy (`"cog"`, `"largest"`, or `"smallest"`). This function modifies the graph in place to eliminate bridges and enhance connectivity between its components.

    Args:
        graph: The graph for which bridges need to be augmented. The input graph should be a NetworkX graph object with coordinates for its nodes stored under the attribute `'pos'`.
        strategy: The method for linking components. Options include:
            - 'cog': Connect based on centres of gravity of components.
            - 'largest': Prioritise linking to the largest component.
            - 'smallest': Prioritise linking from the smallest component.
            Defaults to 'largest'.

    Raises:
        ValueError: If an unknown strategy is provided.
        RuntimeError: If no available edge can be found that reduces the number of bridges in the graph.
    """

    initial_edges = {frozenset(edge) for edge in graph.edges()}
    pos = nx.get_node_attributes(graph, 'pos')

    if bridges := list(nx.bridges(graph)):
        _augment_bridge_chains(graph, bridges)

        while len(list(nx.bridges(graph))):
            try:
                u, v = {
                    'largest': _select_largest_strategy,
                    'cog': _select_cog_strategy,
                    'smallest': _select_smallest_strategy,
                }[strategy](graph, pos)
            except KeyError:
                raise ValueError(f"Unknown strategy: {strategy}")

            graph.add_edge(u, v)

    _prune_redundant_edges(graph, initial_edges, pos)

    print("Bridges eliminated; total edges added and kept:",
          len([edge for edge in graph.edges() if frozenset(edge) not in initial_edges]))
