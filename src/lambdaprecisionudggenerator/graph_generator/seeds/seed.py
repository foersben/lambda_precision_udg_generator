from dataclasses import dataclass, field
from copy import deepcopy
import networkx as nx
import numpy as np
import logging

from lambdaprecisionudggenerator.graph_generator.graphs.generator import LambdaPrecisionUDGGenerator
from lambdaprecisionudggenerator.graph_generator.graphs.lambda_precision_udg import LambdaPrecisionUDG
from lambdaprecisionudggenerator.graph_generator.points.generator import RandomPointsGenerator


@dataclass
class GeneratorSeed:
    """ A class for generating and managing unit disk graph (UDG) samples with specified properties.

    This class provides methods to configure, generate, and analyse graphs based on node layout, connection rules, and other graph properties. It supports serialisation and deserialisation for persistence and repeated usage.

    Attributes:
        node_number: Number of nodes in the generated graph.
        min_distance: Minimum distance between nodes in the graph.
        radius: Connection radius for determining edges in the graph.
        coverage_bound: Range of allowed coverage densities for generated graphs.
        avg_deg_bound: Range of allowed average degrees for generated graphs.
        prob_connected: Probability that a generated graph is connected.
        sample_size: Number of graphs to generate.
        graphs: List of generated graphs.
        logger: Logger instance for logging information about graph generation and properties.
    """

    node_number: int
    min_distance: float
    radius: float
    coverage_bound: tuple[float, float]
    avg_deg_bound: tuple[float, float]
    probability_connected: float
    sample_size: int = 20
    graphs: list[LambdaPrecisionUDG] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def generate_graphs(self, sample_size: int, new: bool = True, connected: bool = False,
                        bounds: bool = False) -> None:
        """ Generates multiple graphs using the parameters provided and applies optional filtering criteria such as connectivity and bounds. This process can initialise a fresh set of graphs or append to the existing set.

        Args:
            sample_size: The number of graphs to generate.
            new: Whether to initialise a new graph list or append to the existing one.
            connected: Ensures the generated graphs are connected if set to True.
            bounds: Filters the resulting graphs based on average degree and coverage bounds if set to True.
        """

        if new:
            self.graphs = []

        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(self.node_number, self.min_distance), self.radius)

        while len(self.graphs) < sample_size:
            self.logger.info("Generate graphs...")
            graphs = generator.generate_graphs_parallel(max(sample_size - len(self.graphs), 10), connected=connected)
            self.graphs.extend(graphs)
            self.logger.debug(
                f"Graph: Node Number: {self.node_number}, "
                f"Lambda: {self.min_distance}, Radius: {self.radius}, "
                f"Avg Degree: {str(self.avg_deg_bound)}"
                f"Connected: {sum([nx.is_connected(graph) for graph in self.graphs])}"
            )
            for graph in self.graphs:
                self.logger.debug(
                    f"Average Degree: {self.avg_deg_bound[0]} <= {graph.average_degree()} <= {self.avg_deg_bound[1]} "
                    f"Coverage: {self.coverage_bound[0]} <= {graph.get_lambda_precision_points().get_density()} <= {self.coverage_bound[1]}"
                )
            if bounds:
                self.graphs = list(filter(
                    lambda graph: self.avg_deg_bound[0] <= graph.average_degree() <= self.avg_deg_bound[1] and
                                  self.coverage_bound[0] <= graph.get_lambda_precision_points().get_density() <=
                                  self.coverage_bound[1], self.graphs))
            self.logger.info(f"Graphs successfully generated: {len(self.graphs)} / {sample_size}")
        self.graphs = self.graphs[:sample_size]

    def copy(self) -> "GeneratorSeed":
        """ Creates a copy of the current instance of UDGGeneratorSeed, maintaining the same parameter values.

        Returns:
            A new instance of UDGGeneratorSeed initialized with the same attributes as the original instance.
        """

        return GeneratorSeed(self.node_number, self.min_distance, self.radius, deepcopy(self.coverage_bound),
                             deepcopy(self.avg_deg_bound), self.probability_connected, self.sample_size)

    def probability_connected(self) -> float:
        """ Calculates the probability of connectivity for a set of graphs.

        This function computes the average of connectivity statuses for a collection of graphs stored in the `self.graphs` attribute. Connectivity is determined using the NetworkX `is_connected` function, which checks if a graph is fully connected (every node is reachable from every other node). The result is stored in the `self.prob_connected` attribute and returned.

        Returns:
            The calculated probability of connectivity.
        """

        self.probability_connected = float(np.mean([nx.is_connected(graph) for graph in self.graphs]))
        return self.probability_connected

    def get_avg_coverage(self) -> float:
        """ Calculate the average coverage across all graphs.

        This method computes the mean density of lambda precision points for all graphs stored in the `self.graphs` attribute. Each graph is expected to have a `lambda_precision_points` property with a method `get_density()` that provides the density value.

        Returns:
            The average density of lambda precision points across all graphs.
        """

        return float(np.mean([graph.get_lambda_precision_points().get_density() for graph in self.graphs]))

    def get_avg_degree(self) -> float:
        """ Calculates the average degree of a series of graphs.

        This method iterates over a collection of graphs, computes the average degree for each graph, and returns the mean of those values as a float.

        Returns:
            A float representing the average degree of the graphs.
        """

        return float(np.mean([graph.average_degree() for graph in self.graphs]))

    def degree_distribution(self, sample_size: int = 0, new: bool = False) -> list[list[int]]:
        """ Computes the degree distribution for the generated graphs. Degree distribution is represented as a list of lists, where each inner list contains the degrees of nodes for a specific graph in the set of generated graphs.

        This method optionally supports specifying the number of graphs to consider and whether to regenerate the graphs before computation.

        Args:
            sample_size: The number of graphs for which the degree distribution is to be computed. Defaults to 0, which uses the total sample size defined in the class.
            new: Boolean indicating whether to generate new graphs instead of using the current set. Defaults to False.

        Returns:
            A list of lists where each inner list contains the degree values of all nodes in the corresponding graph.
        """

        if not sample_size:
            sample_size = self.sample_size
        self.generate_graphs(sample_size, new)
        return [list(np.array(graph.degree())[:, 1]) for graph in self.graphs[:sample_size]]

    def median_node_degree_distribution(self, sample_size: int = 0) -> list[float]:
        """ Calculates the median of the node degree distribution for a graph.

        This function computes and returns a list of median values from the degree distribution of nodes in a graph. The computation can be optionally carried out for a random subset of nodes specified by the sample size.

        Args:
            sample_size: Optional. The number of nodes to sample for computing the degree distribution. If set to zero, the entire set of nodes will be considered.

        Returns:
            A list of median values corresponding to the degree distribution of nodes in the graph.
        """

        return list(map(float, map(np.median, self.degree_distribution(sample_size))))

    def variance_node_degree_distribution(self, sample_size: int = 0) -> list[float]:
        """ Calculate the variance of the node degree distribution.

        This method computes the variance of the degree distribution for a given sample size of the graph's nodes. It utilises the degree distribution values to derive the variance and returns them as a list of floats. This can be useful to analyse the spread or dispersion of node connectivity in the graph structure.

        Args:
            sample_size: Number of nodes to sample for the degree distribution. If set to 0, the computation will consider all nodes.

        Returns:
            A list containing the variances of the sampled degree distribution.
        """

        # return np.var(self.degree_distribution(sample_size))
        return list(map(float, map(np.var, self.degree_distribution(sample_size))))

    def local_clustering(self, sample_size: int = 0, new: bool = False) -> list[dict[int, float]]:
        """ Calculates the local clustering coefficient for a collection of graph samples.

        The function computes the clustering coefficient for all nodes in each of the sampled graphs. It allows specifying the number of graphs to sample or utilising the pre-configured `sample_size`. Optionally, it can generate new graph samples if the `new` parameter is set.

        Args:
            sample_size: The number of graph samples to evaluate. If not specified or set to zero, the default sample size defined in the object is used.
            new: Indicates whether new graphs should be generated for the clustering computation. Set to ``True`` to generate fresh samples.

        Returns:
            A list of dictionaries, where each dictionary maps node indices to their respective clustering coefficients for each sampled graph.
        """

        if not sample_size:
            sample_size = self.sample_size
        self.generate_graphs(sample_size, new)
        return list(nx.clustering(graph) for graph in self.graphs[:sample_size])

    def variance_local_clustering(self, sample_size: int = 0) -> list[float]:
        """ Computes the variance of local clustering coefficients for a given sample of nodes.

        The method calculates the variance of local clustering values from a sample that is either randomly selected or encompasses the entire node set if `sample_size` is set to 0. The method utilises pre-computed local clustering coefficients and determines variance values for each sample set.

        Args:
            sample_size: Number of nodes to sample for computing local clustering. Defaults to 0, meaning all nodes are included.

        Returns:
            A list of float values representing the variances of local clustering coefficients computed for the sample(s).
        """

        local_clustering = self.local_clustering(sample_size)
        return [float(np.var(list(clustering.values()))) for clustering in local_clustering]

    def median_local_clustering(self, sample_size: int = 0) -> float:
        """ Calculate the median local clustering coefficient of a network.

        This method computes the local clustering coefficient for all nodes in the network or for a specified sample of nodes and then determines the median value of these coefficients.

        Args:
            sample_size: The number of nodes to sample for the local clustering calculation. If set to 0, all nodes are included in the calculation. Defaults to 0.

        Returns:
            The median local clustering coefficient of the network based on the computed or sampled node values.
        """

        return float(np.median(self.local_clustering(sample_size)))

    def global_clustering(self, sample_size: int = 0, new: bool = False) -> float:
        """ Computes the global clustering coefficient for a set of generated graphs.

        The global clustering coefficient is a measure of the overall tendency of nodes in a graph to form tightly connected clusters. This method first generates multiple graph instances based on the given sample size and then computes the transitivity (global clustering coefficient) for each graph. The final result is the mean value of the computed transitivity across all sampled graphs.

        Args:
            sample_size: Number of graph samples to generate. If set to 0, the method defaults to using the object's stored sample size.
            new: Flag indicating whether to regenerate new graphs or use cached graphs. If True, new graphs will be generated.

        Returns:
            The mean global clustering coefficient computed across the sampled graphs.
        """

        if not sample_size:
            sample_size = self.sample_size
        self.generate_graphs(sample_size, new)
        return float(np.mean([nx.transitivity(graph) for graph in self.graphs[:sample_size]]))

    def __str__(self) -> str:
        """ Provides a string representation of the instance by formatting and displaying the values of its attributes. This method ensures that the object is represented as a readable and easily understandable string.

        Returns:
            A string that includes formatted attributes and their values to represent the object's current state.
        """
        return (f"node_number: {self.node_number}\n"
                f"min_dist: {self.min_distance}\n"
                f"radius: {self.radius}\n"
                f"coverage_bound: {self.coverage_bound}\n"
                f"avg_deg_bound: {self.avg_deg_bound}\n"
                f"prob_connected: {self.probability_connected}\n"
                f"sample_size: {self.sample_size}\n"
                f"graphs: {str((len(self.graphs), self.graphs))}")

    def serialize(self, path: str) -> None:
        """ Serialises the current instance of the class to a file using the pickle module. The file is stored at the given path with a unique file name based on the object's ID to avoid collisions.

        Args:
            path: The directory path where the serialized file will be saved.
        """

        import pickle

        self.logger.info(f"UDGGeneratorSeed.serialize {path}/{id(self)}.pkl")
        with open(f"{path}/{id(self)}.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def deserialize(cls, filepath: str) -> "GeneratorSeed":
        """ Deserialises a `UDGGeneratorSeed` object from a file using the pickle module. The method expects the file to be in binary format. This operation will read the contents of the specified file and return a `UDGGeneratorSeed` object that was previously serialised.

        Args:
            filepath: The file path where the serialised `UDGGeneratorSeed` object is stored.

        Returns:
            A `UDGGeneratorSeed` object deserialised from the specified file.
        """

        import pickle

        logging.getLogger(__name__).info(f"UDGGeneratorSeed.deserialize {filepath}")
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def get_metadata(self) -> dict:
        """ Returns a dictionary containing all metadata except graphs and logger.

        This provides a space-efficient representation of the seed configuration without the generated graphs or logger instance.
        """

        return {
            "node_number": self.node_number,
            "min_distance": self.min_distance,
            "radius": self.radius,
            "coverage_bound": self.coverage_bound,
            "avg_deg_bound": self.avg_deg_bound,
            "probability_connected": self.__dict__['probability_connected'],
            "sample_size": self.sample_size
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "GeneratorSeed":
        """ Recreates a GeneratorSeed instance from metadata dictionary.

        The new instance will have empty graphs list and default logger.
        """

        return cls(
            node_number=metadata["node_number"],
            min_distance=metadata["min_distance"],
            radius=metadata["radius"],
            coverage_bound=metadata["coverage_bound"],
            avg_deg_bound=metadata["avg_deg_bound"],
            probability_connected=metadata["probability_connected"],
            sample_size=metadata["sample_size"]
        )
