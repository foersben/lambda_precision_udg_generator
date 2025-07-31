import logging

import networkx as nx
from copy import deepcopy

from src.graph_generator.points.lambda_precision_points import LambdaPrecisionPoints


class LambdaPrecisionUDG(nx.Graph):
    """Unit disk graph (UDG) over lambda-precision points, inheriting from networkx.Graph.

    Attributes:
        logger (Logger): Logger instance for logging graph generation events.
        graph (nx.Graph): The underlying undirected graph.
        radius (float): Communication radius for connecting nodes.
        points_metadata (dict[str, any]): Metadata about the lambda-precision points.
    """

    def __init__(self, points: LambdaPrecisionPoints, radius: float, logger: logging.Logger = None) -> None:
        """Initialises the unit disk graph with lambda-precision points.

        Args:
            points: A NumPy array of shape (n, 2) with (x, y) coordinates.
            radius: Communication radius for connecting nodes.
            logger: Logger instance for logging graph generation events.
        """

        self.logger = logger or logging.getLogger(__name__)

        # points = points.get_lambda_precision_points()
        # pos = {i: (points[i][0], points[i][1]) for i in range(len(points))}
        graph = nx.random_geometric_graph(len(points), radius,
                                          pos=dict(enumerate(points.get_lambda_precision_points())))
        self.logger.info(
            f"Number of nodes: {len(graph.nodes())}, "
            f"Number of edges: {len(graph.edges())}"
        )

        super().__init__()

        # Fast transfer of structure and attributes
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))

        self.radius = radius
        self.points_metadata = points.get_metadata()

    def subgraph(self, nodes) -> "LambdaPrecisionUDG":
        """Returns a subgraph as a new LambdaPrecisionUDG instance (deep copy, not a view).

        Args:
            nodes: Iterable of node labels to induce the subgraph on.

        Returns:
            LambdaPrecisionUDG instance representing the subgraph.
        """
        induced_subgraph = super().subgraph(nodes).copy()  # A true copy, not a view

        # Create new LambdaPrecisionUDG with dummy points, will override structure next
        new_udg = self.clone()
        new_udg.clear()  # Clear all nodes and edges before adding new ones

        new_udg.add_nodes_from(induced_subgraph.nodes(data=True))
        new_udg.add_edges_from(induced_subgraph.edges(data=True))

        return new_udg

    def get_networkx_graph(self) -> nx.Graph:
        """ Returns the underlying networkx graph.

        Returns:
            The underlying networkx graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes(data=True))
        graph.add_edges_from(self.edges(data=True))

        return deepcopy(graph)

    def get_lambda_precision_points(self) -> LambdaPrecisionPoints:
        """ Returns the lambda-precision points.

        Returns:
            LambdaPrecisionPoints object containing the points.
        """

        return LambdaPrecisionPoints.from_metadata(self)

    def clone(self) -> "LambdaPrecisionUDG":
        """ Creates a deep copy of the LambdaPrecisionUDG instance.

        Returns:
            A deep copy of the current LambdaPrecisionUDG instance.
        """

        return deepcopy(self)

    def copy(self, as_view=False):
        """Returns a copy of the graph.

        Overrides the nx.Graph.copy() method to handle the required constructor arguments.

        Parameters
        ----------
        as_view : bool, optional (default=False)
            If True, the returned graph-view provides a read-only view
            of the original graph without actually copying any data.

        Returns
        -------
        G : LambdaPrecisionUDG
            A copy of the graph.
        """
        if as_view is True:
            # will still fail e.g. if a subgraph is created
            return super().copy(as_view=True)
            # return nx.graphviews.generic_graph_view(self)

        # Use our existing clone method which performs a proper deep copy
        return self.clone()

    def average_degree(self) -> float:
        """ Returns the average node degree of an undirected graph

        Returns:
            Average degree of the graph as a float.
        """

        return 2.0 * self.number_of_edges() / float(self.number_of_nodes())

    def serialize(self, path: str) -> None:
        """ Serializes the object

        Args:
            path: path to save the object
        """

        import pickle

        with open(f"{path}/{id(self)}.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def deserialize(cls, filepath: str) -> "LambdaPrecisionUDG":
        """ Deserializes the object

        Args:
            cls: class to deserialize
            filepath: path to load the object from
        Returns:
            deserialized object
        """

        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __setitem__(self, key: any, value: any):
        """ Sets the value for the specified key in the internal dictionary. If the internal dictionary (`_properties`) does not exist, it initialises it.

        Args:
            key: Key used to store the value in the dictionary.
            value: Value to associate with the key in the dictionary.
        """

        if not hasattr(self, '_properties'):
            self._properties = {}
