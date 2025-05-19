import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean as dist
from random import choice, choices

from src.graph_generator.graphs.lambda_precision_udg2 import LambdaPrecisionUDG

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


def augment_bridges_knn(graph: nx.Graph, strategy: str = "cog") -> None:
    """ Augment a connected graph by adding the minimum number of edges needed to eliminate all bridges.

    This routine operates in three phases:

      1. **Bridge‐chain linking**
         Finds all “chain” bridges (two bridges sharing a single node) and connects their far endpoints.
      2. **Iterative component merging**
         While any bridges remain, it:
           - Identifies the current bridge‐connected components.
           - Selects two components based on `strategy`:
             - `'largest'`: the two largest components by node count.
             - `'cog'`: the two whose centroids (mean of 2D node positions) are closest.
             - `'progressive'`: the smallest component paired with its nearest neighbour.
           - Computes up to the top‐K nearest neighbour node‐pairs across these components,
             sorted by Euclidean distance, and adds the first edge not already in the graph.
      3. **Pruning**
         Any added edges whose removal does not reintroduce bridges are removed in
         descending order of their length.

    Args:
        graph: networkx.Graph
            A connected, undirected graph. Each node must have a 2‑tuple or list‐like `'pos'`
            attribute giving its (x, y) coordinates.
        strategy: {'cog', 'largest', 'progressive'}, optional
            Determines how to select which pair of bridge‐connected components to connect:
            - `'largest'`: merge the two largest components.
            - `'cog'`: merge the two components whose centroids are closest.
            - `'progressive'`: merge the smallest component with its nearest neighbour.
            Defaults to `'cog'`.

    Returns:
        None
            The input `graph` is modified in place.  No value is returned.

    Raises:
        RuntimeError: If at any iteration no new edge can be found between the selected components (e.g., all candidate edges already exist), this error is raised.

    Notes:
        - Runs in roughly O(B + Σᵥ dᵥ²) per iteration, where B is the number of bridges
          and dᵥ the number of bridges incident on node v.
        - The final pruning step ensures minimality: only those edges strictly necessary
          to maintain a bridge‐free graph are kept.
    """

    from networkx.algorithms.connectivity.edge_kcomponents import bridge_components
    from itertools import combinations

    initial_edges = {frozenset(edge) for edge in graph.edges()}
    bridges = list(nx.bridges(graph))
    pos = nx.get_node_attributes(graph, 'pos')
    if bridges:
        # bridges is your list of 2‑tuples
        chain_pairs = [(a, b) for node in {n for edge in bridges for n in edge}
                       # collect all “other” endpoints of bridges touching `node`
                       for a, b in
                       combinations([v for u, v in bridges if u == node] + [u for u, v in bridges if v == node], 2)]
        graph.add_edges_from(chain_pairs)

        while list(nx.bridges(graph)):
            components = list(bridge_components(graph))
            # Select component pair
            if strategy == 'largest':
                # pick two largest by size
                c1, c2 = sorted(components, key=len, reverse=True)[0:2]
            elif strategy == 'cog':
                # center of gravity
                centers = []
                valid = []
                for component in components:
                    pts = np.array([pos[node] for node in component])
                    centers.append(pts.mean(axis=0))
                    valid.append(component)
                # find two closest centers
                tree = cKDTree(centers)
                dists, idxs = tree.query(centers, k=2)
                # find global minimum pair (skip self at idx 0)
                min_i = np.argmin(dists[:, 1])
                c1 = valid[min_i]
                c2 = valid[idxs[min_i, 1]]
            elif strategy == 'progressive':
                # smallest to nearest other
                smallest = min(components, key=len)
                others = [component for component in components if component is not smallest]
                pts_s = np.array([pos[node] for node in smallest]).mean(axis=0)
                centers = [np.array([pos[node] for node in component]).mean(axis=0) for component in others]
                tree = cKDTree(centers)
                _, idx = tree.query(pts_s)
                c1, c2 = smallest, others[idx]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Pick closest node pair between c1 and c2
            pts1 = np.array([pos[node] for node in c1])
            pts2 = np.array([pos[node] for node in c2])
            nodes1 = list(c1)
            nodes2 = list(c2)

            # 1) ask for the two nearest neighbours
            k = min(8, len(pts2))
            dists, idxs = cKDTree(pts2).query(pts1, k=k)

            # 2) force into 2‑D arrays
            dists = np.atleast_2d(dists)
            idxs = np.atleast_2d(idxs)

            # 3) build a global sorted list of (distance, i, neighbor_j)
            candidates = [(dists[i, m], i, idxs[i, m]) for i in range(dists.shape[0]) for m in range(dists.shape[1])]
            candidates.sort(key=lambda x: x[0])

            # 4) pick the first one that isn’t already an edge
            for _, i, j in candidates:
                u, v = nodes1[i], nodes2[j]
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    break
            else:
                # If *none* of the top‑2 worked, you can either raise or increase the size of `k`.
                raise RuntimeError(f"No new edge available between {c1} and {c2}")

    # Final prune: remove added edges if unnecessary
    added = [edge for edge in graph.edges() if frozenset(edge) not in initial_edges]
    print("Number of edges added to augment all bridges: ", len(added))

    def euclid(a: int, b: int) -> float:
        """ Augments a graph by connecting unconnected components through nearest neighbors based on a selected strategy. This function modifies the input graph in place.

        Args:
            a: The first node in the edge.
            b: The second node in the edge.

        Returns:
            The Euclidean distance between two nodes in the graph.
        """

        return float(np.linalg.norm(np.array(pos[a]) - np.array(pos[b])))

    print("Number of edges before pruning:", len(graph.edges()))
    for u, v in sorted(added, key=lambda edge: euclid(*edge), reverse=True):
        graph.remove_edge(u, v)
        if list(nx.bridges(graph)):
            graph.add_edge(u, v)
    print("Number of edges after pruning:", len(graph.edges()))
