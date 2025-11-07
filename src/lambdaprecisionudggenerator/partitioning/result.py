import pyomo.environ as pyo
from pyomo.opt import SolverResults
import numpy as np
import logging
import cloudpickle
import lzma
import os
from networkx import Graph
from collections import Counter

from lambdaprecisionudggenerator.graph_generator.seeds.seed import GeneratorSeed

logger = logging.getLogger(__name__)


class BaseResult:
    """ Base class for all partitioning results.

    Attributes:
        graph (LambdaPrecisionUDG): The graph being partitioned/having means assigned to.
        model (pyo.ConcreteModel): Pyomo concrete model used for partitioning
        partition_size (int): Number of partitions/domains
        opt_type (str): Type of optimisation ('opt', 'max', 'var', 'spread') - should be deprecated in future versions
        _seed (dict[str, Any]): Seed used to generate the graph
        objective (float): Objective value from the Pyomo model's objective function.
        wallclock_time (float): Solver wallclock time in seconds
        aborted (bool): Flag indicating if solver aborted before finding an optimal solution
        graph_id (int): Unique hash of the graph
        lbound (float): Lower bound of solution
        ubound (float): Upper bound of solution
        mipgap (float): MIP gap percentage
    """

    def __init__(self, graph: Graph, result: SolverResults, model: pyo.ConcreteModel, partition_size: int,
                 opt_type: str, seed: GeneratorSeed) -> None:
        """ Initialises an object that encapsulates the optimisation result and relevant metadata.

        This constructor stores the provided optimisation input parameters, calculates metrics such as the objective value, lower and upper bounds, and captures additional solver-related outcomes. Following initialisation, information about the created result type is logged.

        Args:
            graph: Graph structure instance representing the problem to be solved.
            model: The Pyomo ConcreteModel instance that represents the optimisation model.
            partition_size: Size of each partition to be considered within the graph.
            opt_type: The type of optimisation performed (should be deprecated in future versions).
            seed: Seed for random number generation used within the optimisation process.
        """

        self.graph = graph.copy()
        self.model = model
        self.partition_size = partition_size
        self.opt_type = opt_type
        self._seed = seed.get_metadata() if seed else None

        # Post-initialisation
        self.objective = pyo.value(model.objective)
        self.wallclock_time = result.solver.wallclock_time
        self.aborted = result.solver.status == pyo.SolverStatus.aborted
        self.graph_id = hash(self.graph)

        # Store solution bounds
        self.lbound = result.Problem.lower_bound
        self.ubound = result.Problem.upper_bound
        self.mipgap = abs(self.ubound - self.lbound) / abs(self.ubound) \
            if self.ubound != 0 else float('inf')

        self._extract_assignment()

        logger.info(f"Created {self.opt_type} result for graph {self.graph_id}")

    def _extract_assignment(self) -> None:
        """ Extracts the partitioning/assignment of nodes and means from the model's variables and assigns it to the `assignment` attribute of the class. This method evaluates a binary threshold on the variables and filters accordingly.

        Raises:
            AttributeError: If model variables are not properly defined or accessible.
        """

        for node in self.graph.nodes():
            try:
                self.graph.nodes[node]['means'] = [
                    mean for (node_id, mean), var in self.model.x.items()
                    if node == node_id and pyo.value(var) > 0.5]
            except AttributeError as error:
                logger.error(f"Error extracting partitioning: {error}")
                self.graph.nodes[node]['means'] = []

    @property
    def seed(self) -> GeneratorSeed:
        """ Returns the seed used to generate the graph.

        Returns:
            The seed metadata as a dictionary.
        """

        return GeneratorSeed.from_metadata(self._seed)

    def serialize(self, path: str, compress: bool = True) -> None:
        """ Serialises the current instance to a file at the specified path. The object can be compressed using the `lzma` compression algorithm if the `compress` flag is set to True.

        Args:
            path: The directory path where the serialised object will be stored.
            compress: A flag indicating whether to compress the serialised object using `lzma`. Defaults to True.

        Raises:
            Exception: If the serialisation process fails due to any reason.
        """

        try:
            os.makedirs(path, exist_ok=True)
            filename = f"{id(self)}.pkl{'.xz' if compress else ''}"
            filepath = os.path.join(path, filename)

            # logger.info(f"Serialising result to {filepath}")

            with (lzma.open if compress else open)(filepath, "wb") as f:
                cloudpickle.dump(self, f)

            # logger.info(f"Successfully serialised result")
        except Exception as e:
            logger.error(f"Serialisation failed: {e}")
            raise

    @classmethod
    def deserialize(cls, filepath: str, compressed: bool = True) -> 'BaseResult':
        """ Deserialise result from file

        Args:
            filepath: Path to serialised file
            compressed: Whether file is compressed

        Returns:
            Deserialized result object
        """

        try:
            logger.info(f"Deserialising result from {filepath}")

            with (lzma.open if compressed else open)(filepath, "rb") as f:
                result = cloudpickle.load(f)

            logger.info(f"Successfully deserialized {type(result).__name__}")
            return result
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise

    def _neighbour_means(self, node: int) -> list[int]:
        """ Returns the means for a given node's neighbourhood in the graph.

        Args:
            node: The node for which to find neighbours.

        Returns:
            A list of neighbour nodes.
        """

        means = list(self.graph.nodes[node]['means'])  # Include the node itself
        for neighbour in self.graph.neighbors(node):
            for mean in self.graph.nodes[neighbour]['means']:
                means.append(mean)

        return means

    def calculate_errors(self) -> int:
        """ Calculates the number of errors in the given graph by evaluating each node's neighbours and summing up the conditions based on the partition size.

        This method computes the number of errors in the nodes of the graph based on whether the number of neighbours meets a specific condition relative to the partition size. It logs errors if any attributes required for calculations are missing.

        Returns:
            The total count of errors identified in the graph
        """

        return np.sum([self.partition_size - len(set(self._neighbour_means(node))) for node in self.graph.nodes])

    def calculate_incomplete_nodes(self) -> int:
        """ Calculates the number of incomplete nodes in the model.

        An incomplete node is defined as a node that has no means assigned to it. This method iterates through all nodes in the graph and counts those that have an empty 'means' list.

        Returns:
            The count of incomplete nodes.
        """

        return sum(1 for node in self.graph.nodes if len(set(self._neighbour_means(node))) < self.partition_size)

    def calculate_variance(self) -> float:
        """ Calculates the variance for each node in the model and updates the graph with this information.

        The variance for a node is computed by evaluating the means assigned to it and its neighbourhood expected and actual coverage over parts of the model. It is utilised for measuring distribution deviations in the network.

        Returns:
            The average variance across all nodes in the graph.
        """

        frequency = lambda node: list(Counter(self._neighbour_means(node)).values())

        return np.mean([
            np.var(frequency(node) + [0] * (self.partition_size - len(frequency(node))))
            for node in self.graph.nodes
        ], dtype=np.float64)

    def _spread(self, means: list[int]) -> int:
        """ Calculates the spread of means in a given set.

        Args:
            means: A list of means assigned to nodes for which to calculate the spread.

        Returns:
            The spread value, defined as the difference between the maximum and minimum means.
        """

        frequency = Counter(means).values()
        max_freq = max(frequency)
        min_freq = 0 if len(frequency) < self.partition_size else min(frequency)

        return max_freq - min_freq

    def calculate_spread(self) -> float:
        """ Calculates the spread for each node in the model and assigns it to the corresponding nodes in the graph.

        The spread is calculated as the difference between the high and low values (`xh` and `xl`) for each node. It then updates the graph with the calculated spread values. If any error occurs during the calculation (e.g., missing attributes), an error is logged, and the spread per node dictionary is reset to an empty state.
        """

        return np.mean([self._spread(self._neighbour_means(node)) for node in self.graph.nodes], dtype=np.float64)

    def _calculate_residues(self) -> tuple[float, ...]:
        """ Calculate and manage the residues of resources for each node within the model.

        This method computes the remaining resources (residues) on each node in the model after deducting the resources utilised for some assigned tasks. The residues data is stored and updated in `self.residue_per_node` and also assigned as attributes to the nodes in the graph. Logging is used to provide insight into the calculation process or potential errors.
        """

        graph = self.graph

        try:
            for node in graph.nodes:
                graph.nodes[node]['residue'] = (
                        graph.graph["node_resources"]
                        - tuple(sum(self.model.mean_cost.values()[mean])
                                for mean in graph.nodes[node]['means'])
                )
            logger.debug(f"The mean residue of the graph amounts to: "
                         f"{np.mean(graph.nodes[node]['residue'] for node in graph.nodes)}")
            return tuple(sum(values) for values in zip(*(graph.nodes[node]['residue'] for node in graph.nodes)))
        except (AttributeError, IndexError) as e:
            logger.debug(f"Error calculating residue: {e}")
            for node in graph.nodes:
                graph.nodes[node]['residue'] = tuple(0.0, )
            return tuple(0.0, )


class OptSoftDomaticPartitionResult(BaseResult):
    """ Represents the results from a model analysis, focusing on minimising the sum of errors.

    This class extends the functionality of BaseResult by including operations specifically aimed at evaluating and quantifying errors of a mean assignment on lambda-precision UDG graph model.
    """

    def __init__(self, graph: Graph, result: SolverResults, model: pyo.ConcreteModel, partition_size: int,
                 opt_type: str, seed: GeneratorSeed) -> None:
        """ Initialises an object of `OptSoftDomaticPartitionResult`, actual initialisation is passed to `BaseResult`.

        Args:
            graph: The graph structure being analysed.
            result: The results from the solver after running the optimisation.
            model: The Pyomo model used for the optimisation.
            partition_size: The size of each partition in the graph.
            opt_type: The type of optimisation performed (should be deprecated in future versions).
            seed: The generator seed used for reproducibility.

        Raises:
            ValueError: If the partition size is not defined in the model.
        """

        super().__init__(graph, result, model, partition_size, opt_type, seed)

        logger.info(f"MinErrorsResult: {self.calculate_errors()} errors, "
                    f"{self.calculate_incomplete_nodes()} incomplete nodes")


class MaxSoftDomaticPartitionResult(BaseResult):
    """ Represents the results from a model analysis, focusing on minimising the sum of incompletely covered nodes.

    This class extends the functionality of BaseResult by including operations specifically aimed at evaluating and quantifying the incompletely covered nodes of a mean assignment on lambda-precision UDG graph model.
    """

    def __init__(self, graph: Graph, result: SolverResults, model: pyo.ConcreteModel, partition_size: int,
                 opt_type: str, seed: GeneratorSeed) -> None:
        """ Initialises an object of `MaxSoftDomaticPartitionResult`, actual initialisation is passed to `BaseResult`.

        Args:
            graph: The graph structure being analysed.
            result: The results from the solver after running the optimisation.
            model: The Pyomo model used for the optimisation.
            partition_size: The size of each partition in the graph.
            opt_type: The type of optimisation performed (should be deprecated in future versions).
            seed: The generator seed used for reproducibility.

        Raises:
            ValueError: If the partition size is not defined in the model.
        """

        super().__init__(graph, result, model, partition_size, opt_type, seed)
        logger.info(f"MinErrorsResult: {self.calculate_errors()} errors, "
                    f"{self.calculate_incomplete_nodes()} incomplete nodes")


class MinVarianceResult(BaseResult):
    """ Represents the result of a computation aimed at minimising variance within a graph partitioning problem. This class calculates and stores results related to the variance of graph node coverage and the overall objective value derived from a Pyomo model.

    The `MinVarianceResult` class extends functionality from its parent class to specifically handle variance computation and logging while ensuring compatibility with graph representations and Pyomo modelling constructs.
    """

    def __init__(self, graph: Graph, result: SolverResults, model: pyo.ConcreteModel, partition_size: int,
                 opt_type: str, seed: GeneratorSeed) -> None:
        """ Initialises an object of `MinVarianceResult` while most initialisation is passed to `BaseResult`.

        Args:
            graph: The graph structure being analysed.
            result: The results from the solver after running the optimisation.
            model: The Pyomo model used for the optimisation.
            partition_size: The size of each partition in the graph.
            opt_type: The type of optimisation performed (should be deprecated in future versions).
            seed: The generator seed used for reproducibility.

        Raises:
            ValueError: If the partition size is not defined in the model.
        """

        super().__init__(graph, result, model, partition_size, opt_type, seed)
        logger.info(f"MinVarianceResult: objective={self.objective:.4f}")


class MinSpreadResult(BaseResult):
    """ Represents the result of a computation aimed at minimising spread within a graph partitioning problem. This class calculates and stores results related to the spread of means across nodes in a graph, derived from a Pyomo model.

    The `MinSpreadResult` class extends functionality from its parent class to specifically handle spread computation and logging while ensuring compatibility with graph representations and Pyomo modelling constructs.
    """

    def __init__(self, graph: Graph, result: SolverResults, model: pyo.ConcreteModel, partition_size: int,
                 opt_type: str, seed: GeneratorSeed) -> None:
        """ Initialises an object of `MinSpreadResult`, actual initialisation is passed to `BaseResult`.

        Args:
            graph: The graph structure being analysed.
            result: The results from the solver after running the optimisation.
            model: The Pyomo model used for the optimisation.
            partition_size: The size of each partition in the graph.
            opt_type: The type of optimisation performed (should be deprecated in future versions).
            seed: The generator seed used for reproducibility.

        Raises:
            ValueError: If the partition size is not defined in the model.
        """

        super().__init__(graph, result, model, partition_size, opt_type, seed)
        logger.info(f"MinSpreadResult: objective={self.objective:.4f}")


class MinSpreadResourceResult(BaseResult):
    """ Result class for resource-based spread minimisation """

    mean_cost: tuple[tuple[float, ...], ...]

    def __init__(self, graph: Graph, result: SolverResults, model: pyo.ConcreteModel, partition_size: int,
                 opt_type: str, seed: GeneratorSeed) -> None:
        """ Initialises an object of `MinSpreadResourceResult`, actual initialisation is passed to `BaseResult`.

        Args:
            graph: The graph structure being analysed.
            result: The results from the solver after running the optimisation.
            model: The Pyomo model used for the optimisation.
            partition_size: The size of each partition in the graph.
            opt_type: The type of optimisation performed (should be deprecated in future versions).
            seed: The generator seed used for reproducibility.

        Raises:
            ValueError: If the partition size is not defined in the model.
        """

        super().__init__(graph, result, model, partition_size, opt_type, seed)
        self.mean_cost = model.mean_cost.values()
        logger.info(f"MinSpreadResourceResult: resource-based spread calculated")
