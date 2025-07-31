from dataclasses import dataclass, field
from typing import Callable
import logging
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverResults
from networkx import adjacency_matrix, Graph

from src.graph_generator.seeds.seed import GeneratorSeed
from src.partitioning.result2 import MinSpreadResult, MinVarianceResult, OptSoftDomaticPartitionResult, \
    MaxSoftDomaticPartitionResult, MinSpreadResourceResult

logging.getLogger('pyomo').setLevel(logging.CRITICAL)
logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

ResourceVec = tuple[float, ...]  # resource cost for nodes
ResCostMat = tuple[tuple[float, ...], ...]  # resource cost of means
Sense = Callable[..., pyo.Expression]  # pyo.minimize / pyo.maximize


@dataclass(slots=True, frozen=True)
class SolverConfig:
    """ Represents the configuration for a solver.

    This class provides configuration options for defining the behaviour of a solver. It uses a dataclass with slots and is immutable due to the frozen parameter. The primary purpose is to offer structured and type-safe storage for solver configuration properties.

    Attributes:
        name (str): The solver to use for solving the optimisation problem. Defaults to 'gurobi'.
        io (str): The input/output interface for the solver. Defaults to 'python'.
        stream_output (bool): A boolean flag indicating whether the solver's output should be displayed on the screen during execution. Defaults to False.
        keepfiles (bool): A boolean flag indicating whether intermediate solver files (e.g., .nl, .sol) should be retained after execution. Defaults to False.
        mip_focus (int): The focus of the MIP solver, which can influence the solver's behaviour in terms of speed and solution quality. Defaults to 3, which is a common setting for Gurobi.
        mip_gap (float): The acceptable gap between the best known solution and the optimal solution. Defaults to 0, meaning no gap is allowed.
        time_limit (float): The maximum time in seconds that the solver is allowed to run before it is terminated. Defaults to 600 seconds (10 minutes).
    """

    name: str = 'gurobi'
    io: str = 'python'
    stream_output: bool = False
    keepfiles: bool = False
    mip_focus: int = 0  # Default MIP focus for Gurobi, can be adjusted based on solver capabilities
    mip_gap: float = 1e-4  # Default MIP gap for Gurobi, can be adjusted based on solver capabilities
    time_limit: int = 600.0


@dataclass(slots=True, frozen=True)
class PyomoConfig:
    """ Represents the configuration for a Pyomo model.

    This class provides configuration options for defining the behaviour of a Pyomo model. It uses a dataclass with slots and is immutable due to the frozen parameter. The primary purpose is to offer structured and type-safe storage for Pyomo model configuration properties.
    """

    sense: Sense
    solver_config: SolverConfig


@dataclass(slots=True, frozen=True)
class DomaticPartitionConfig(PyomoConfig):
    """ Consolidated configuration for spread- and resource-based mean distribution. """

    partition_size: int = field(default=1)
    mean_count_per_node: int = field(default=1),
    solver_config: SolverConfig = field(default_factory=SolverConfig, init=True)
    sense: Sense = field(default=pyo.maximize, init=True)
    weight: tuple[float, ...] = field(default_factory=tuple)
    lower_bound: tuple[float, ...] = field(default_factory=tuple)
    upper_bound: tuple[float, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class VarianceDistributionConfig(PyomoConfig):
    """ Configuration for variance-based mean distribution. """

    partition_size: int = field(default=1)
    mean_count_per_node: int = field(default=1),
    solver_config: SolverConfig = field(default_factory=SolverConfig, init=True)
    sense: Sense = field(default=pyo.minimize, init=True)
    weight: tuple[float, ...] = field(default_factory=tuple)
    lower_bound: tuple[float, ...] = field(default_factory=tuple)
    upper_bound: tuple[float, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class SpreadResourceDistributionConfig(PyomoConfig):
    """ Consolidated configuration for spread- and resource-based mean distribution. """

    means: ResCostMat = field(default_factory=tuple)
    solver_config: SolverConfig = field(default_factory=SolverConfig, init=True)
    sense: Sense = field(default=pyo.minimize, init=True)


def _create_neighbourhood_dict(model: pyo.ConcreteModel) -> None:
    """ Creates and initialises the neighbourhood dictionary for a Pyomo ConcreteModel.

    Args:
        model: The Pyomo ConcreteModel whose neighbourhood dictionary is to be created.
    """

    neighbours_dict = {v: [w for w in model.Nodes if model.links[v, w] > 0] for v in model.Nodes}

    def get_neighbours(model: pyo.ConcreteModel, v: int) -> list[int]:
        return neighbours_dict[v]

    model.neighbours = pyo.Param(model.Nodes, initialize=get_neighbours, within=pyo.Any)


def _create_base_model(graph: Graph) -> pyo.ConcreteModel:
    """ Creates a Pyomo ConcreteModel for a graph optimisation problem. The model is built based on the provided graph structure, including sets, parameters, and variables needed for optimisation. The graph neighbourhood data is incorporated into the model, ensuring a direct mapping to the problem's constraints and objectives.

    Args:
        graph: The input graph as a networkx graph with nodes and edges used to define the problem structure and neighbourhood data.

    Returns:
        A Pyomo ConcreteModel instance configured for the given graph.
    """

    model = pyo.ConcreteModel()
    nodes = list(graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)

    adj = adjacency_matrix(graph)
    adj.setdiag(1)
    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        initialize={(v, w): adj[i, j] for i, v in enumerate(nodes) for j, w in enumerate(nodes)}
    )
    _create_neighbourhood_dict(model)

    return model


def _create_base_partition_model(graph: Graph,
                                 config: VarianceDistributionConfig | DomaticPartitionConfig) -> pyo.ConcreteModel:
    """ Creates a Pyomo ConcreteModel for a partitioning problem based on the provided graph and partition size. The model includes sets, parameters, and variables needed for optimisation, ensuring a direct mapping to the problem's constraints and objectives.

    Args:
        graph: The input graph as a networkx graph with nodes and edges used to define the problem structure.
        partition_size: The size of the partitions to be created in the optimisation model.

    Returns:
        A Pyomo ConcreteModel instance configured for the given graph and partition size.
    """

    model = _create_base_model(graph)
    model.PartSize = pyo.Set(initialize=range(1, config.partition_size + 1))
    model.part_size = pyo.Param(within=pyo.PositiveIntegers, initialize=config.partition_size)
    model.node_degrees = pyo.Param(
        model.Nodes,
        within=pyo.PositiveIntegers,
        initialize={v: sum(model.links[v, w] for w in model.Nodes) for v in model.Nodes}
    )
    model.mean_count_per_node = pyo.Param(within=pyo.PositiveIntegers, initialize=config.mean_count_per_node)
    return model


def _create_base_resource_model(graph: Graph, means: ResCostMat) -> pyo.ConcreteModel:
    """ Creates a Pyomo ConcreteModel for a graph optimisation problem. The model is built based on the provided graph and a resource-cost matrix (means). It includes sets, parameters, and variables needed for optimisation. The graph neighbourhood data and resource configuration are incorporated into the model, ensuring a direct mapping to the problem's constraints and objectives.

    Args:
        graph: The input graph as a networkx graph with nodes and edges used to define the problem structure and neighbourhood data.
        means: A 2D cost matrix where the rows represent different means (e.g., strategies) and the columns correspond to various resources.

    Returns:
        A Pyomo ConcreteModel instance configured for the given graph and cost matrix.
    """

    model = _create_base_model(graph)
    node_resources: ResourceVec = graph.graph.get("node_resources", (1.0,) * len(means[0]))
    model.Resources = pyo.Set(initialize=range(1, len(node_resources) + 1))
    model.Means = pyo.Set(initialize=range(1, len(means) + 1))
    model.node_resources = pyo.Param(
        model.Resources,
        within=pyo.NonNegativeReals,
        initialize={r: node_resources[r - 1] for r in model.Resources}
    )

    model.mean_cost = pyo.Param(
        model.Means, model.Resources,
        initialize={(i, r): means[i - 1][r - 1] for i in model.Means for r in model.Resources}
    )

    model.x = pyo.Var(model.Nodes, model.Means, within=pyo.Binary)
    model.xl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.xh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    return model


def _solve_assignment_model(
        model: pyo.ConcreteModel,
        config: PyomoConfig,
        dist_type: str
) -> SolverResults:
    """ Solve an assignment model using the specified solver configuration and optional distribution type.

    This function initialises a solver using the configuration, adjusts solver options such as time limits and output controls, and executes the solving process of the provided Pyomo model.

    Args:
        model: The Pyomo ConcreteModel representing the assignment problem to solve.
        config: The configuration settings for the resource distribution, including solver specifics.
        dist_type: An optional string indicating the type of distribution being processed. If provided, this will also set the solver's log file name to include the distribution type.
    Returns:
        The results of the solver execution wrapped in a SolverResults object. If the solver fails, `None` is returned instead.
    """

    solver = SolverFactory(config.solver_config.name, solver_io=config.solver_config.io)
    options = {
        'TimeLimit': config.solver_config.time_limit,
        'MIPFocus': config.solver_config.mip_focus,
        'MIPGap': config.solver_config.mip_gap,
        'LogFile': f"{dist_type}_solver.log",
        'OutputFlag': 1 if config.solver_config.stream_output else 0
    }

    try:
        result = solver.solve(model, options=options, tee=config.solver_config.stream_output,
                              keepfiles=config.solver_config.keepfiles)
    except Exception as e:
        logger.error(f"Solver failed: {str(e)}")
        result = None

    return result


def _spread_resource_constraints(model: pyo.ConcreteModel) -> None:
    """ Spread resource constraints across nodes based on their neighbours within the given model.

    The function defines and adds two constraint rules (`lower_bound` and `upper_bound`) to the passed optimisation model. These constraints relate the variables of each node to the sum of variables from their neighbouring nodes.

    Args:
        model: Instance of a pyomo `ConcreteModel` optimisation model.
    """

    def lower_bound(model, v, i):
        return model.xl[v] <= sum(model.x[w, i] for w in model.neighbours[v])

    model.lower_bound = pyo.Constraint(model.Nodes, model.Means, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.xh[v] >= sum(model.x[w, i] for w in model.neighbours[v])

    model.upper_bound = pyo.Constraint(model.Nodes, model.Means, rule=upper_bound)


def spread_based_max_resource_utilisation_distribution(
        graph: Graph,
        config: SpreadResourceDistributionConfig,
        seed: GeneratorSeed = None
) -> MinSpreadResourceResult:
    """ Implements the resource-based distribution scheme from MILP (3) (ITNAC Paper).

    This formulation uses auxiliary variables to enforce maximal resource utilisation per node through logical disjunctions. It ensures that for each node, no additional security mean type can be applied without exceeding at least one resource capacity. The optimisation minimises the sum of spreads across all nodes' inclusive neighbourhoods while maintaining exact resource exhaustion constraints.

    Args:
        graph: NetworkX graph representing the wireless sensor network topology.
        config: Configuration containing security means parameters and solver settings.
        seed: Optional generator seed for reproducibility of the optimisation process.

    Returns:
        MinSpreadResourceResult: Object containing optimisation results and metadata.

    Raises:
        ValueError: If security means cost matrix is not provided in the configuration.
    """

    if not config.means:
        raise ValueError("Security means cost matrix must be provided")

    model = _create_base_resource_model(graph, config.means)
    model.epsilon = pyo.Param(within=pyo.NonNegativeReals, initialize=0.01)
    model.y = pyo.Var(model.Nodes, model.Means, model.Resources, within=pyo.Binary)

    _spread_resource_constraints(model)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=config.sense)

    def indicator_var(model, v, i):
        return 1 - model.x[v, i] <= sum(model.y[v, i, r] for r in model.Resources)

    model.indicator_var = pyo.Constraint(model.Nodes, model.Means, rule=indicator_var)

    def resource_usage(model, v, i, r):
        return model.y[v, i, r] * (model.node_resources[r] - model.mean_cost[i, r]) <= sum(
            model.mean_cost[j, r] * model.x[v, j] for j in model.Means) - model.epsilon

    model.resource_usage = pyo.Constraint(model.Nodes, model.Means, model.Resources, rule=resource_usage)

    def resource_constraint(model, v, r):
        return sum(model.mean_cost[i, r] * model.x[v, i] for i in model.Means) <= model.node_resources[r]

    model.resource_constraint = pyo.Constraint(model.Nodes, model.Resources, rule=resource_constraint)

    return MinSpreadResourceResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='resource'),
        model=model,
        partition_size=len(config.means),
        seed=seed,
        opt_type='spread'
    )


def spread_resource_based_distribution(
        graph: Graph,
        config: SpreadResourceDistributionConfig,
        reward_factor: float = 1.0,
        seed: GeneratorSeed = None
) -> MinSpreadResourceResult:
    """Implements the resource utilisation maximisation scheme from MILP (2) (ITNAC paper, reference coming).

    This formulation minimises the sum of spreads while incorporating a reward factor that balances resource utilisation. The objective function trades off neighbourhood spread minimisation against resource capacity utilisation. A higher reward factor prioritises resource utilisation over spread minimisation, while a lower factor does the opposite.

    Args:
        graph: NetworkX graph representing the wireless sensor network topology.
        config: Configuration containing security means parameters and solver settings.
        reward_factor: Weighting parameter balancing spread minimisation and resource utilisation. Higher values emphasise resource utilisation. Defaults to 1.0.
        seed: Optional generator seed for reproducibility of the optimisation process.

    Returns:
        MinSpreadResourceResult: Object containing optimisation results and metadata.

    Raises:
        ValueError: If security means cost matrix is not provided in the configuration.
    """

    if not config.means:
        raise ValueError("Security means cost matrix must be provided")

    model = _create_base_resource_model(graph, config.means)
    model.reward_factor = pyo.Param(within=pyo.NonNegativeReals, initialize=reward_factor)
    _spread_resource_constraints(model)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes) + model.reward_factor * sum(
            model.node_resources[r] - sum(model.mean_cost[i, r] * model.x[v, i] for i in model.Means) for r in
            model.Resources for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=config.sense)

    def resource_constraint(model, v, r):
        return sum(model.mean_cost[i, r] * model.x[v, i] for i in model.Means) <= model.node_resources[r]

    model.resource_constraint = pyo.Constraint(model.Nodes, model.Resources, rule=resource_constraint)

    def mean_assignment(model, v):
        return sum(model.x[v, i] for i in model.Means) >= 1

    model.mean_assignment = pyo.Constraint(model.Nodes, rule=mean_assignment)

    return MinSpreadResourceResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='resource'),
        model=model,
        partition_size=len(config.means),
        seed=seed,
        opt_type='spread'
    )


def _max_packings(
        means: tuple[tuple[int, ...], ...],
        resources: tuple[int, ...],
        n: int,
        config: list[tuple[int, ...]] = None,
) -> set[tuple[tuple[int, ...], ...]]:
    """ Finds all maximal packings using integer arithmetic.

    Args:
        means: Tuple of mean resource requirements as integers
        resources: Available resources as integers
        n: Number of means to consider
        config: Current packing configuration

    Returns:
        Set of maximal packings (each packing is a tuple of mean tuples)
    """

    if config is None:
        config = []

    # Base case: processed all means
    if n == 0:
        # Check if configuration is maximal
        for mean in means:
            if mean in config:
                continue
            # Check only resource dimensions (ignore index dimension)
            if all(a <= r for a, r in zip(mean[:-1], resources)):
                return set()  # Not maximal
        return {tuple(config)}  # Valid maximal packing

    # Current mean being considered
    mean = means[n - 1]
    # Recursive call without current mean
    packings_without = _max_packings(means, resources, n - 1, config)

    # Check if mean fits in resource dimensions
    if all(a <= r for a, r in zip(mean[:-1], resources)):
        # Update resources by subtracting mean's requirements
        new_res = tuple(r - a for r, a in zip(resources, mean[:-1]))
        # Recursive call with current mean
        packings_with = _max_packings(means, new_res, n - 1, config + [mean])
    else:
        packings_with = set()

    return packings_without | packings_with


def _max_packings_matrix(means: ResCostMat, resources: ResourceVec) -> tuple[
    tuple[ResCostMat, ...], tuple[tuple[int, ...], ...]]:
    """ Computes the maximum packings matrix and corresponding packing configurations based on provided means and resources. This function applies a packing algorithm to find the optimal packing configurations and represents the result in a binary matrix format.

    Args:
        means: A tuple of tuples, where each inner tuple represents a mean value configuration. Each mean tuple includes numerical values used for packing.
        resources: A tuple representing the numerical constraints or capacities for packing. Each value denotes the available capacity of a resource.

    Returns:
        A tuple containing two elements:
            - The first element is a tuple of packing configurations, where each configuration is represented as a tuple of tuples, excluding index information.
            - The second element is a tuple of tuples in binary matrix form, where each sub-tuple represents a row and indicates whether a mean is included in a particular packing configuration (1) or not (0).
    """

    # Scaling factor to convert floats to integers
    SCALE = 10 ** 6
    scaled_resources = tuple(int(r * SCALE) for r in resources)

    # Create indexed means: (scaled_resource1, ..., scaled_resourceN, index)
    indexed_means = []
    for i, mean in enumerate(means):
        scaled_mean = tuple(int(m * SCALE) for m in mean)
        indexed_means.append(scaled_mean + (i,))
    indexed_means = tuple(indexed_means)

    # Compute maximal packings with integer arithmetic
    packing_result = _max_packings(indexed_means, scaled_resources, len(indexed_means), [])

    # Create binary matrix representation
    matrix = [[0] * len(means) for _ in range(len(packing_result))]
    # Map indices to original means
    index_to_mean = {i: mean for i, mean in enumerate(means)}

    # Prepare packing configurations in original float format
    float_packings = []
    for i, config in enumerate(packing_result):
        # Get original mean using the last element (index)
        orig_config = tuple(index_to_mean[mean[-1]] for mean in config)
        float_packings.append(orig_config)

        # Update matrix representation
        for mean in config:
            mean_idx = mean[-1]
            matrix[i][mean_idx] = 1

    return tuple(float_packings), tuple(tuple(row) for row in matrix)


def spread_based_configurations_distribution(
        graph: Graph,
        config: SpreadResourceDistributionConfig,
        seed: GeneratorSeed = None
) -> MinSpreadResourceResult:
    """ Implements the precomputed configurations scheme from MILP (4) (ITNAC paper, reference coming).

    This formulation precomputes all maximal security mean configurations that exhaust node resources, then selects configurations to minimise neighbourhood spread. It reduces the solution space by considering only resource-exhausting combinations, but configuration count may grow exponentially with security mean options. The approach maintains the minimal constraint count while accurately modelling the distribution concept.

    Args:
        graph: NetworkX graph representing the wireless sensor network topology.
        config: Configuration containing security means parameters and solver settings.
        seed: Optional generator seed for reproducibility of the optimisation process.

    Returns:
        MinSpreadResourceResult: Object containing optimisation results and metadata.

    Raises:
        ValueError: If security means cost matrix is not provided in the configuration.
    """

    if not config.means:
        raise ValueError("Security means cost matrix must be provided")

    model = _create_base_resource_model(graph, config.means)
    node_resources = graph.graph.get("node_resources", (1.0,) * len(config.means[0]))
    packings, mapping_matrix = _max_packings_matrix(config.means, node_resources)
    model.Configurations = pyo.Set(initialize=range(1, len(packings) + 1))
    model.mean_mapping = pyo.Param(
        model.Configurations,
        model.Means,
        within=pyo.Binary,
        initialize={(i, j): mapping_matrix[i - 1][j - 1] for i in model.Configurations for j in model.Means}
    )
    model.y = pyo.Var(model.Nodes, model.Configurations, within=pyo.Binary)

    _spread_resource_constraints(model)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=config.sense)

    def mapping(model, v, j):
        return model.x[v, j] == sum(model.mean_mapping[i, j] * model.y[v, i] for i in model.Configurations)

    model.mapping = pyo.Constraint(model.Nodes, model.Means, rule=mapping)

    def configuration_assignment(model, v):
        return sum(model.y[v, i] for i in model.Configurations) == 1

    model.part = pyo.Constraint(model.Nodes, rule=configuration_assignment)

    return MinSpreadResourceResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='resource'),
        model=model,
        partition_size=len(config.means),
        seed=seed,
        opt_type='spread'
    )


def min_spread_partition(
        graph: Graph,
        config: VarianceDistributionConfig,
        seed: GeneratorSeed = None,
) -> MinSpreadResult:
    """ The MILP computes for a graph the minimal sum of the spread of each node's inclusive neighbourhood's mean assignment. This ensures for a given graph in each node's neighbourhood there is a balanced number of means available. One mean is assigned per each node.

    Args:
        graph: The input graph structure representing nodes and links.
        config: Configuration containing partition size, mean count per node, and solver settings.
        seed: The seed used to generate the graph.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    model = _create_base_partition_model(graph, config)
    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.yl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.yh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    def objective(model):
        return sum(model.yh[v] - model.yl[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=config.sense)

    def lower_bound(model, v, i):
        return model.yl[v] <= sum(model.x[w, i] for w in model.neighbours[v])

    model.lower_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.yh[v] >= sum(model.x[w, i] for w in model.neighbours[v])

    model.upper_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=upper_bound)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.mean_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    return MinSpreadResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='spread'),
        model=model,
        partition_size=config.partition_size,
        seed=seed,
        opt_type='spread'
    )


def min_variance_partition(
        graph: Graph,
        config: VarianceDistributionConfig,
        seed: GeneratorSeed = None
) -> MinVarianceResult:
    """ Computes an MIQP for a given graph that minimises the sum of the variances of the means assigned to each node's inclusive neighbourhood in the graph.

    Args:
        graph: NetworkX Graph
        config: Configuration containing partition size, mean count per node, and solver settings.
        seed: Seed used to generate the graph.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    model = _create_base_partition_model(graph, config)
    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)

    def objective(model):
        return sum(1 / model.part_size * sum(((model.node_degrees[v]) / model.part_size - sum(
            model.x[w, i] for w in model.neighbours[v]
        )) ** 2 for i in model.PartSize) for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=config.sense)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.mean_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    return MinVarianceResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='variance'),
        model=model,
        partition_size=config.partition_size,
        seed=seed,
        opt_type='var'
    )


def min_spread_squared_partition(
        graph: Graph,
        config: VarianceDistributionConfig,
        seed: GeneratorSeed = None
) -> MinSpreadResult:
    """ Computes an MIQP for a given graph that minimises the sum of the squared spreads of the means assigned to each node's inclusive neighbourhood in the graph.

    Args:
        graph: NetworkX Graph
        config: Configuration containing partition size, mean count per node, and solver settings.
        seed: Seed used to generate the graph.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    model = _create_base_partition_model(graph, config)
    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.yl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.yh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    def objective(model):
        return sum((model.yh[v] - model.yl[v]) ** 2 for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=config.sense)

    def lower_bound(model, v, i):
        return model.yl[v] <= sum(model.x[w, i] for w in model.neighbours[v])

    model.lower_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.yh[v] >= sum(model.x[w, i] for w in model.neighbours[v])

    model.upper_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=upper_bound)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.mean_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    return MinSpreadResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='spread_squared'),
        model=model,
        partition_size=config.partition_size,
        seed=seed,
        opt_type='spread'
    )


def _create_domatic_partition_model(graph: Graph, config: DomaticPartitionConfig) -> pyo.ConcreteModel:
    """ Creates a Pyomo ConcreteModel for a domatic partitioning problem based on the provided graph and configuration.

    Args:
        graph: The input graph as a networkx graph with nodes and edges used to define the problem structure.
        config: Configuration containing partition size, mean count per node, and solver settings.

    Returns:
        A Pyomo ConcreteModel instance configured for the given graph and partition size.
    """

    weight = config.weight or (1.0,) * config.partition_size
    lower_bound = config.lower_bound or (0,) * config.partition_size
    upper_bound = config.upper_bound or (1,) * config.partition_size

    model = _create_base_partition_model(graph, config)
    model.c = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: weight[i - 1] for i in model.PartSize}
    )
    model.l = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: lower_bound[i - 1] for i in model.PartSize}
    )
    model.u = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: upper_bound[i - 1] for i in model.PartSize}
    )
    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.y = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.mean_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    def neighbourhood(model, v, i):
        return model.y[v, i] <= sum([model.x[w, i] for w in model.neighbours[v]])

    model.neighbourship = pyo.Constraint(model.Nodes, model.PartSize, rule=neighbourhood)

    def bounds(model, i):
        return pyo.inequality(model.l[i], sum(model.x[v, i] for v in model.Nodes) / len(model.Nodes), model.u[i])

    model.bounds = pyo.Constraint(model.PartSize, rule=bounds)

    return model


def opt_soft_domatic_partition(
        graph: Graph,  # links
        config: DomaticPartitionConfig,
        seed: GeneratorSeed = None
) -> OptSoftDomaticPartitionResult:
    """ Computes an optimal n-soft domatic partition for a given graph using the given MILP formulation.

    Args:
        graph: NetworkX graph representing the wireless sensor network topology.
        config: Configuration for the optimisation, including partition size, weights, bounds, and solver settings.
        seed: Optional generator seed for reproducibility of the optimisation process.

    Returns:
        Result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    model = _create_domatic_partition_model(graph, config)

    def coverage(model):
        return sum(model.c[i] * sum(model.y[v, i] for v in model.Nodes) for i in model.PartSize)

    model.objective = pyo.Objective(rule=coverage, sense=config.sense)

    return OptSoftDomaticPartitionResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='opt'),
        model=model,
        partition_size=config.partition_size,
        seed=seed,
        opt_type='opt'
    )


def max_soft_domatic_partition(
        graph: Graph,  # links
        config: DomaticPartitionConfig,
        seed: GeneratorSeed = None
) -> MaxSoftDomaticPartitionResult:
    """ Computes a maximal n-soft domatic partition for a given graph in form of an MILP.

    Args:
        graph: NetworkX graph to be partitioned.
        config: Configuration for the optimisation, including partition size, weights, bounds, and solver settings.
        seed: Seed used to generate the graph.

    Returns:
        Result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    model = _create_domatic_partition_model(graph, config)
    model.z = pyo.Var(model.Nodes, within=pyo.Binary)

    def coverage(model):
        return sum(sum(model.c[i] * model.x[v, i] for i in model.PartSize) * model.z[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=coverage, sense=config.sense)

    def fulfilling_nodes(model, v, i):
        return model.z[v] <= model.y[v, i]

    model.fulfilling_nodes = pyo.Constraint(model.Nodes, model.PartSize, rule=fulfilling_nodes)

    return MaxSoftDomaticPartitionResult(
        graph=graph,
        result=_solve_assignment_model(model, config, dist_type='max'),
        model=model,
        partition_size=config.partition_size,
        seed=seed,
        opt_type='max'
    )
