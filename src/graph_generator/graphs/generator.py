import logging
from typing import Any

import networkx as nx
from joblib import Parallel, delayed

from src.graph_generator.graphs.lambda_precision_udg import LambdaPrecisionUDG
from src.graph_generator.points.generator import RandomPointsGenerator
from src.utils.logging_config import setup_logging


class LambdaPrecisionUDGGenerator:
    """ Handles the generation of random geometric graphs using a lambda-precision based uniform distribution and facilitates both serial and parallel generation of these graphs. Intended for use in simulations or experiments where random network structures of this type are required.

    This class allows for the creation of graphs with specified connectivity, distance constraints, and provides parallelised generation for performance optimisations in cases requiring multiple graphs.

    Attributes:
        logger (Logger): Logger instance for logging graph generation events.
        random_points_generator (RandomPointsGenerator): Instance of RandomPointsGenerator used for generating random points in space.
        radius (float): Distance up to which nodes are connected in the graph.
        _lpp (LambdaPrecisionPoints): Instance of LambdaPrecisionPoints, will be set during graph generation.
    """

    def __init__(self, random_points_generator: RandomPointsGenerator, radius: float, logger: logging.Logger = None):
        """ Initialises the LambdaPrecisionUDGGenerator with a RandomPointsGenerator instance and a connection radius.

        Args:
            random_points_generator: Instance of RandomPointsGenerator used for generating random points in space.
            radius: Distance up to which nodes are connected in the graph.
            logger: Logger instance for logging graph generation events.
        """

        self.logger = logger or logging.getLogger(__name__)

        self.random_points_generator = random_points_generator
        self.radius = radius
        self._lpp = None  # LambdaPrecisionPoints instance, will be set during graph generation

    def generate_graph(self, connected: bool = False) -> LambdaPrecisionUDG:
        """ Generates a random geometric graph with certain properties, based on lambda-precision points and a specified radius. The generated graph can optionally be ensured to be connected.

        Args:
            connected: If True, ensures the generated graph is connected. Defaults to False.

        Returns:
            New LambdaPrecisionUDG instance containing the generated graph, lambda precision points, and the used radius.
        """

        while True:
            self._lpp = self.random_points_generator.generate_points()
            while not self._lpp:
                self._lpp = self.random_points_generator.generate_points()
            # points = lpp.get_lambda_precision_points()
            # pos = {i: (points[i][0], points[i][1]) for i in range(len(points))}
            # graph = nx.random_geometric_graph(len(points), self.radius, pos=pos)
            graph = LambdaPrecisionUDG(self._lpp, self.radius)
            if not connected or nx.is_connected(graph):
                self.logger.info(f"connected = {nx.is_connected(graph)}")
                return graph

    def generate_graphs_parallel(self, number: int, prefer: Any = None, connected: bool = False) -> list[
        LambdaPrecisionUDG]:
        """ Generates for a given number as many graphs in parallel using the Joblib library for performance optimisation. The generated graphs can optionally be ensured to be connected.

        Args:
            number: number of graphs to generate
            prefer: argument for joblib about the preferred way to parallelise
            connected: whether to generate specifically connected graphs

        Returns:
            List of generated graphs
        """

        return Parallel(n_jobs=-1, prefer=prefer, initializer=setup_logging)(
            delayed(self.generate_graph)(connected) for _ in range(number))

    def generate_graphs(self, number: int, connected: bool = False) -> list[LambdaPrecisionUDG]:
        """ Generates for a given number as many graphs in serial. The generated graphs can optionally be ensured to be connected.

        Args:
            number: number of graphs to generate
            connected: whether to generate specifically connected graphs

        Returns:
            List of generated graphs
        """

        return [self.generate_graph(connected) for _ in range(number)]
