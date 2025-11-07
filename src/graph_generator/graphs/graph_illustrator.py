import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Wedge
from matplotlib.colors import to_hex
import random

from graph_generator.graphs.lambda_precision_udg import LambdaPrecisionUDG

BASE_COLOR = 'red'
FALLBACK_COLOR = 'gray'
BASE_RADIUS = 0.004
NODE_RADIUS = 0.008
RING_WIDTH = 3
BUFFER_RADIUS = 0.002


def _assign_bandwidth_usage(ax: plt.Axes, graph: LambdaPrecisionUDG, max_bandwidth: int = 100) -> None:
    """ Randomly assigns bandwidth usage to each edge in the given graph. The `bandwidth_usage` is added as an edge attribute where its value is a random integer between 0 and the specified maximum bandwidth.

    Args:
        ax: The matplotlib axes object where the graph will be drawn.
        graph: A graph object representing a LambdaPrecisionUDG that contains the edges to which bandwidth usage will be assigned.
        max_bandwidth: The maximum bandwidth usage value, defaulting to 100. This value determines the upper limit for randomly assigned bandwidth usage.
    """

    pos = nx.get_node_attributes(graph, "pos")

    for u, v in graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        bandwidth_usage = graph.edges[u, v].get('bandwidth_usage')
        if bandwidth_usage is not None:
            edge_color = _get_edge_color(bandwidth_usage, max_bandwidth)
        else:
            edge_color = FALLBACK_COLOR
        ax.plot([x1, x2], [y1, y2], color=edge_color, lw=0.8, zorder=1)


def _generate_distinguishable_colormap(mean_types: int) -> dict[int, str]:
    """ Generates a mapping from integer identifiers to hexadecimal colour codes.

    This method creates a dictionary that assigns unique hexadecimal colour codes derived from a colourmap to each integer identifier. The number of unique integer identifiers is determined by the `means` attribute from the `network_config` object. The colour codes are generated using the `tab20` colourmap from Matplotlib.

    Args:
        mean_types: Integer representing the number of distinguishable colours to generate. Must be greater than 0.

    Returns:
        dict[int, str]: A dictionary where the keys are integer identifiers (ranging from 0 to `mean_types` - 1) and the values are hexadecimal colour codes.
    """

    return {i: to_hex(plt.cm.tab20(i / mean_types)) for i in range(mean_types)}


def _assign_colors_to_nodes(graph: LambdaPrecisionUDG, colours: list[str]) -> dict[int, str]:
    """ Randomly assigns one of the provided colours to each node in the graph.

    This function iterates through all nodes in the input graph and assigns a randomly chosen colours from the given list of colours to each node. The result is a dictionary where each key is a node and its corresponding value is the randomly chosen colours from the colours list. The same colours may be assigned to multiple nodes as the selection is randomised.

    Args:
        graph: The input graph whose nodes will have colours assigned.
        colours: A list of colours from which each node is randomly assigned one.

    Returns:
        A dictionary where keys are graph nodes, and values are the assigned colours.
    """

    return {node: random.choices(colours, k=1)[0] for node in graph.nodes()}


def _get_edge_color(bandwidth_usage: int, max_bandwidth: int = 100) -> tuple[float, float, float]:
    """ Compute the colour representation of the edge based on its bandwidth usage, maximum bandwidth, and overload threshold. The function uses a provided colourmap to determine the colours for edges within safe usage and sets red colour for overloaded edges.

    Args:
        bandwidth_usage: The current bandwidth usage of the edge.
        max_bandwidth: The maximum permissible bandwidth for the edge, default is 100.
        colormap: A colourmap identifier string to map normalised usage to a colour, default is 'viridis'.

    Returns:
        The RGB colour representation as a tuple for the edge.
    """

    return plt.cm.plasma(bandwidth_usage / max_bandwidth) if bandwidth_usage <= max_bandwidth else (1, 0, 0, 1)


def _assign_means_randomly(graph: nx.Graph, mean_types: int) -> None:
    """ Randomly assigns a list of mean types to each node in the graph.

    This function iterates through all nodes in the input graph and assigns a randomly chosen list of mean types from the range 0 to `means` - 1. The result is a dictionary where each key is a node and its corresponding value is the randomly chosen list of mean types. The same mean types may be assigned to multiple nodes as the selection is randomised.

    Args:
        graph: The input graph whose nodes will have mean types assigned.
        mean_types: The number of distinct mean types available for assignment.

    Returns:
        A dictionary where keys are graph nodes, and values are lists of assigned mean types.
    """
    for _, data in graph.nodes(data=True):
        data['means'] = random.sample(range(mean_types), k=random.randint(1, mean_types))


def _draw_segmented_nodes(ax: plt.Axes, graph: nx.Graph, colormap: dict[int, str]) -> None:
    """ Draws a segmented circular node on the given axes at a specified position. The node is divided into a random number of segments (between 1 and k), and each segment is coloured using a randomly selected subset of n colours.

    This function is used to visualise a node with multiple attributes or properties, represented by the segments and their respective colours.

    Args:
        ax: The matplotlib axes object where the node should be drawn.
        graph: The graph object containing the node to be drawn.
        colormap: A dictionary mapping integer identifiers to hexadecimal colour codes, used for segment colouring.
    """

    if not any(data.get('means', []) for _, data in graph.nodes(data=True)):
        _assign_means_randomly(graph, mean_types=len(colormap))

    if not any('means' in data for _, data in graph.nodes(data=True)):
        print("No mean types found in nodes. Please assign mean types to nodes before drawing.")

    pos = nx.get_node_attributes(graph, "pos")

    for node, data in graph.nodes(data=True):
        segments = len(data.get('means', []))

        theta = np.linspace(0, 2 * np.pi, segments + 1)

        for i in range(segments):
            # Create a wedge (segment) for each colour
            wedge = Wedge(
                center=pos[node],
                r=NODE_RADIUS,
                theta1=np.degrees(theta[i]),
                theta2=np.degrees(theta[i + 1]),
                color=colormap.get(data['means'][i], FALLBACK_COLOR),
                zorder=2
            )
            ax.add_patch(wedge)


def draw_graph_with_segmented_nodes(
        graph: nx.Graph,
        mean_types: int,
        max_bandwidth: int = 100,
        save_path: str = None
) -> None:
    """ Draws a graph representation with segmented nodes and edge colouring based on bandwidth usage.

    This function visualises a graph in which the edges are colourised depending on their
    bandwidth usage relative to specified thresholds and maximum bandwidth. Nodes are
    represented as segmented circles for differentiation, and colours correspond to groups
    or categories defined by the number of segments.

    Args:
        graph: The graph represented as a networkx.Graph object where node positions and edge attributes are expected to be pre-defined.
        mean_types: The total number of distinct colours to use for the segmented nodes.
        max_bandwidth: The maximum potential bandwidth of an edge. Default is 100.
        save_path: Optional path to save the graph as an image. If None, the graph is displayed.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes

    _assign_bandwidth_usage(ax, graph, max_bandwidth)

    _draw_segmented_nodes(ax, graph, colormap=_generate_distinguishable_colormap(mean_types))

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    else:
        plt.show(block=True)
