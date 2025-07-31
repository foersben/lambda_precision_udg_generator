from dataclasses import dataclass
from typing import Callable
import logging

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from networkx import adjacency_matrix, Graph

from src.graph_generator.graphs.lambda_precision_udg2 import LambdaPrecisionUDG
from src.graph_generator.seeds.seed import GeneratorSeed
from src.partitioning.result import MinSpreadResult, MinVarianceResult, MinIncompleteNodesResult, MinErrorsResult, \
    MinSpreadResourceResult

logger = logging.getLogger(__name__)

ResourceVec = tuple[float, ...]
ResCostMat = tuple[tuple[float, ...], ...]
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
    """

    name: str = 'gurobi'
    io: str = 'python'
    stream_output: bool = False
    keepfiles: bool = False


def min_spread_resource_multi_distribution_3(
        graph: Graph,  # links
        seed: GeneratorSeed,
        sm_perf_cost: ResCostMat = None,  # m
        sense: Sense = pyo.minimize,
        solver_config: SolverConfig = SolverConfig(),
) -> MinSpreadResourceResult:
    """ The MILP computes for a graph the minimal sum of the spread of each node's inclusive neighbourhood's mean assignments. This ensures for a given graph in each node's neighbourhood there is a balanced number of means available. It further takes into account the resources available to each node and the resources required by a mean. The resource demand and availability is static at every point in time. The resources available to a node and the resources required by a mean do not change during the optimisation process.

    The algorithm computes a trade-off between the resource exhaustion of nodes and the spread of performance costs in the optimisation objective. It assigns each node a mean combination that exhausts its resources. Every mean can only be assigned once per node.

    Args:
        graph: The input graph structure representing nodes and links.
        seed: The seed used to generate the graph.
        sm_perf_cost: A matrix represented as a nested tuple, where the element at position (i, r) specifies the performance cost of allocating resource r to security mechanism i. This parameter must be explicitly provided.
        sense: The direction of optimisation, either minimising or maximising the objective. Provided using Pyomo's ``ObjectiveSense`` enumeration. Defaults to minimisation.
        solver_config: Configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    if sm_perf_cost is None:
        raise ValueError("Performance cost matrix must be provided.")

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)

    model = pyo.ConcreteModel()

    nodes = list(graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.SecMeans = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    node_resources = graph.graph.get("node_resources", (1.0,) * len(sm_perf_cost))
    model.Resources = pyo.Set(initialize=range(1, len(node_resources) + 1))
    # unused
    model.PartSize = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.node_degrees = None

    am = adjacency_matrix(graph)
    am.setdiag(1)

    # unused
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=len(sm_perf_cost)
    )
    model.epsilon = pyo.Param(within=pyo.NonNegativeReals, initialize=0.01)
    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.sm_perf_cost = pyo.Param(
        model.SecMeans,
        model.Resources,
        within=pyo.NonNegativeReals,
        initialize={(i, r): sm_perf_cost[i - 1][r - 1] for i in model.SecMeans for r in model.Resources}
    )
    model.sm_node_resources = pyo.Param(
        model.Resources,
        within=pyo.NonNegativeReals,
        initialize={r: node_resources[r - 1] for r in model.Resources}
    )
    neighbours_dict = {v: [w for w in model.Nodes if model.links[v, w] > 0] for v in model.Nodes}

    def get_neighbours(model, v):
        return neighbours_dict[v]

    model.neighbours = pyo.Param(model.Nodes, initialize=get_neighbours, within=pyo.Any)

    model.x = pyo.Var(model.Nodes, model.SecMeans, within=pyo.Binary)
    model.xl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.xh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)
    model.y = pyo.Var(model.Nodes, model.SecMeans, model.Resources, within=pyo.Binary)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=sense)

    def indicator_var(model, v, i):
        return 1 - model.x[v, i] <= sum(model.y[v, i, r] for r in model.Resources)

    model.indicator_var = pyo.Constraint(model.Nodes, model.SecMeans, rule=indicator_var)

    def resource_usage(model, v, i, r):
        return model.y[v, i, r] * (model.sm_node_resources[r] - model.sm_perf_cost[i, r]) <= sum(
            model.sm_perf_cost[j, r] * model.x[v, j] for j in model.Means) - model.epsilon

    model.resource_usage = pyo.Constraint(model.Nodes, model.SecMeans, model.Resources, rule=resource_usage)

    def lower_bound(model, v, i):
        return model.xl[v] <= sum(model.x[w, i] for w in model.neighbours[v])

    model.lower_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.xh[v] >= sum(model.x[w, i] for w in model.neighbours[v])

    model.upper_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=upper_bound)

    def resource_constraint(model, v, r):
        return sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.Means) <= model.sm_node_resources[r]

    model.resource_constraint = pyo.Constraint(model.Nodes, model.Resources, rule=resource_constraint)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 600.0,
        'MIPFocus': 3,
        # 'MIPGap': 0.1
    })

    print(f"Result: {result}")

    return MinSpreadResourceResult(
        graph=graph,
        result=result,
        model=model,
        partition_size=len(sm_perf_cost),
        seed=seed,
        sm_perf_cost=sm_perf_cost,
        packings=None,
        packings_matrix=None,
        opt_type='spread'
    )


def min_spread_resource_multi_distribution_2(
        graph: Graph,  # links
        seed: GeneratorSeed,
        sm_perf_cost: ResCostMat = None,  # m
        sense: Sense = pyo.minimize,
        reward_factor: float = 1.0,
        solver_config: SolverConfig = SolverConfig()
) -> MinSpreadResourceResult:
    """ The MILP computes for a graph the minimal sum of the spread of each node's inclusive neighbourhood's mean assignments. This ensures for a given graph in each node's neighbourhood there is a balanced number of means available. It further takes into account the resources available to each node and the resources required by a mean. The resource demand and availability is static at every point in time. The resources available to a node and the resources required by a mean do not change during the optimisation process.

    The algorithm assigns each node a mean combination that exhausts its resources. Every mean can only be assigned once per node.

    Args:
        graph: The input graph structure representing nodes and links.
        seed: The seed used to generate the graph.
        sm_perf_cost: A matrix represented as a nested tuple, where the element at position (i, r) specifies the performance cost of allocating resource r to security mechanism i. This parameter must be explicitly provided.
        sense: The direction of optimisation, either minimising or maximising the objective. Provided using Pyomo's ``Sense`` enumeration. Defaults to minimisation.
        reward_factor: A multiplier that influences the trade-off between resource exhaustion of nodes and the spread of performance costs in the optimisation objective. Defaults to 1.0.
        solver_config: Configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    if sm_perf_cost is None:
        raise ValueError("Performance cost matrix must be provided.")

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)

    model = pyo.ConcreteModel()

    nodes = list(graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.SecMeans = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.PartSize = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    # unused
    node_resources = graph.graph.get("node_resources", (1.0,) * len(sm_perf_cost))
    model.Resources = pyo.Set(initialize=range(1, len(node_resources) + 1))
    model.node_degrees = None

    am = adjacency_matrix(graph)
    am.setdiag(1)

    # unused
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=len(sm_perf_cost)
    )
    model.reward_factor = pyo.Param(within=pyo.NonNegativeReals, initialize=reward_factor)
    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.sm_perf_cost = pyo.Param(
        model.SecMeans,
        model.Resources,
        within=pyo.NonNegativeReals,
        initialize={(i, r): sm_perf_cost[i - 1][r - 1] for i in model.SecMeans for r in model.Resources}
    )
    model.sm_node_resources = pyo.Param(
        model.Resources,
        within=pyo.NonNegativeReals,
        initialize={r: node_resources[r - 1] for r in model.Resources}
    )
    neighbours_dict = {v: [w for w in model.Nodes if model.links[v, w] > 0] for v in model.Nodes}

    def get_neighbours(model, v):
        return neighbours_dict[v]

    model.neighbours = pyo.Param(model.Nodes, initialize=get_neighbours, within=pyo.Any)

    model.x = pyo.Var(model.Nodes, model.SecMeans, within=pyo.Binary)
    model.xl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.xh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    # model.y = pyo.Var(within=pyo.PositiveIntegers)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes) + model.reward_factor * sum(
            model.sm_node_resources[r] - sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.Means) for r
            in model.Resources for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=sense)

    # def resource_usage(model):
    #     # return model.y == len(model.Nodes) * len(model.Resources) - sum(
    #     #     model.sm_perf_cost[i, r] * model.x[v, i] for i in model.SecMeans for r in model.Resources for v in
    #     #     model.Nodes)
    #     return model.y == sum(
    #         model.sm_node_resources[r] - sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.SecMeans) for r
    #         in model.Resources for v in model.Nodes)

    # model.resource_usage = pyo.Constraint(rule=resource_usage)

    def lower_bound(model, v, i):
        return model.xl[v] <= sum(model.x[w, i] for w in model.neighbours[v])

    model.lower_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.xh[v] >= sum(model.x[w, i] for w in model.neighbours[v])

    model.upper_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=upper_bound)

    def resource_constraint(model, v, r):
        return sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.Means) <= model.sm_node_resources[r]

    model.resource_constraint = pyo.Constraint(model.Nodes, model.Resources, rule=resource_constraint)

    def sec_per_node(model, v):
        return sum(model.x[v, i] for i in model.Means) >= 1

    model.sec_per_node = pyo.Constraint(model.Nodes, rule=sec_per_node)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 600.0,
        'MIPFocus': 3,
        # 'MIPGap': 0.2
    })

    print(f"Result: {result}")

    return MinSpreadResourceResult(
        graph=graph,
        result=result,
        model=model,
        partition_size=len(sm_perf_cost),
        seed=seed,
        sm_perf_cost=sm_perf_cost,
        packings=None,
        packings_matrix=None,
        opt_type='spread'
    )


def min_spread_resource_multi_distribution_1(
        graph: Graph,  # links
        seed: GeneratorSeed,
        # sm_node_resources: ResourceVec = (1,),  # k
        sm_perf_cost: ResCostMat = None,  # m
        sense: Sense = pyo.minimize,  # pyo.minimize
        solver_config: SolverConfig = SolverConfig()
) -> MinSpreadResourceResult:
    """ The MILP computes for a graph the minimal sum of the spread of each node's inclusive neighbourhood's mean assignments. This ensures for a given graph in each node's neighbourhood there is a balanced number of means available. It further takes into account the resources available to each node and the resources required by a mean. The resource demand and availability is static at every point in time. The resources available to a node and the resources required by a mean do not change during the optimisation process.

    This particular algorithm pre-computes the combinations of means assignable to a node that exhaust its given resources for this purpose. Hence, only mean combinations that exhaust the resources are considered in the optimisation process. Every mean can only be assigned once per node.

    Args:
        graph: NetworkX graph object representing the network.
        seed: Random seed for reproducibility.
        # sm_node_resources: A tuple of resource capacities corresponding to each type of security mean resource available in the system. Defaults to a single resource of capacity 1.
        sm_perf_cost: A matrix represented as a nested tuple, where the element at position (i, r) specifies the performance cost of allocating resource r to security mean i. This parameter must be explicitly provided.
        sense: The direction of optimisation, either minimising or maximising the objective. Provided using Pyomo's ``Sense`` enumeration. Defaults to minimisation.
        solver_config: Configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    if sm_perf_cost is None:
        raise ValueError("Performance cost matrix must be provided.")

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)

    model = pyo.ConcreteModel()

    node_resources = graph.graph.get("node_resources", (1.0,) * len(sm_perf_cost))
    packings, mapping_matrix = _max_packings_matrix(sm_perf_cost, node_resources)

    # TODO test in context
    # mapping_matrix = (
    #     (0, 0, 1, 1, 0, 0),
    #     (0, 0, 0, 0, 1, 0),
    #     (1, 1, 0, 0, 0, 0),
    #     (1, 0, 1, 0, 0, 1),
    #     # (1, 0, 0, 0, 0, 1),
    #     (0, 0, 0, 1, 0, 1),
    #     # (1, 0, 1, 0, 0, 0),
    #     (0, 1, 0, 0, 0, 1),
    #     (0, 1, 0, 1, 0, 0),
    #     (0, 1, 1, 0, 0, 0)
    # )
    # packings = (
    #     ((0.7, 0.2, 0.3), (0.3, 0.2, 0.1)),
    #     ((1.0, 0.5, 0.8),),
    #     ((0.2, 0.6, 0.3), (0.6, 0.1, 0.5)),
    #     ((0.1, 0.3, 0.4), (0.3, 0.2, 0.1), (0.6, 0.1, 0.5)),
    #     # ((0.1, 0.3, 0.4), (0.6, 0.1, 0.5)),
    #     ((0.1, 0.3, 0.4), (0.7, 0.2, 0.3)),
    #     # ((0.3, 0.2, 0.1), (0.6, 0.1, 0.5)),
    #     ((0.1, 0.3, 0.4), (0.2, 0.6, 0.3)),
    #     ((0.7, 0.2, 0.3), (0.2, 0.6, 0.3)),
    #     ((0.3, 0.2, 0.1), (0.2, 0.6, 0.3))
    # )

    model.Nodes = pyo.Set(initialize=list(graph.nodes()))
    model.SecMeans = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.SMPackings = pyo.Set(initialize=range(1, len(packings) + 1))

    am = adjacency_matrix(graph)
    am.setdiag(1)

    model.sm_perf_cost = pyo.Param(
        model.SecMeans,
        model.Resources,
        within=pyo.NonNegativeReals,
        initialize={(i, r): sm_perf_cost[i - 1][r - 1] for i in model.SecMeans for r in model.Resources}
    )
    model.sm_to_packings = pyo.Param(
        model.SMPackings,
        model.SecMeans,
        within=pyo.Binary,
        initialize={(i, j): mapping_matrix[i - 1][j - 1] for i in model.SMPackings for j in model.SecMeans}
    )

    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    neighbours_dict = {v: [w for w in model.Nodes if model.links[v, w] > 0] for v in model.Nodes}

    def get_neighbours(model, v):
        return neighbours_dict[v]

    model.neighbours = pyo.Param(model.Nodes, initialize=get_neighbours, within=pyo.Any)

    model.y = pyo.Var(model.Nodes, model.SMPackings, within=pyo.Binary)
    model.x = pyo.Var(model.Nodes, model.SecMeans, within=pyo.Binary)
    model.xl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.xh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=sense)

    def mapping(model, v, j):
        return model.x[v, j] == sum(model.sm_to_packings[i, j] * model.y[v, i] for i in model.SMPackings)

    model.mapping = pyo.Constraint(model.Nodes, model.SecMeans, rule=mapping)

    def lower_bound(model, v, i):
        return model.xl[v] <= sum(model.x[w, i] for w in model.neighbours[v])

    model.lower_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.xh[v] >= sum(model.x[w, i] for w in model.neighbours[v])

    model.upper_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=upper_bound)

    def packing_distribution(model, v):
        return sum(model.y[v, i] for i in model.SMPackings) == 1

    model.part = pyo.Constraint(model.Nodes, rule=packing_distribution)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 600.0,
        'MIPFocus': 3,
        # 'MIPGap': 0.2
    })

    print(f"Result: {result}")

    return MinSpreadResourceResult(
        graph=graph,
        result=result,
        model=model,
        partition_size=len(sm_perf_cost),
        seed=seed,
        sm_perf_cost=sm_perf_cost,
        packings=packings,
        packings_matrix=mapping_matrix,
        opt_type='spread'
    )


def min_spread_n_partition(
        graph: Graph,  # links
        partition_size: int,  # dom_part_size
        seed: GeneratorSeed,
        sm_count_per_node: int = 1,  # k
        sm_perf_cost: ResourceVec = None,  # m
        sense: Sense = pyo.minimize,  # pyo.minimize
        solver_config: SolverConfig = SolverConfig()
) -> MinSpreadResult:
    """ The MILP computes for a graph the minimal sum of the spread of each node's inclusive neighbourhood's mean assignment. This ensures for a given graph in each node's neighbourhood there is a balanced number of means available. One mean is assigned per each node.

    Args:
        graph: The input graph structure representing nodes and links.
        partition_size: The number of partitions to create in the optimisation process.
        seed: The seed used to generate the graph.
        sm_count_per_node: The number of security mechanisms to assign to each node. Defaults to 1.
        sm_perf_cost: A list of performance costs for each partition size. If not provided, defaults to a list of ones with length equal to the partition size.
        sense: The direction of optimisation, either minimising or maximising the objective. Provided using Pyomo's ``Sense`` enumeration. Defaults to minimisation.
        solver_config: Configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)

    model = pyo.ConcreteModel()

    nodes = list(graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph)
    am.setdiag(1)

    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.node_degrees = pyo.Param(
        model.Nodes,
        within=pyo.NonNegativeIntegers,
        initialize={v: sum(model.links[v, w] for w in model.Nodes) for v in model.Nodes}
    )
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=partition_size
    )
    model.sm_count_per_node = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=sm_count_per_node
    )
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]
    model.sm_perf_cost = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: sm_perf_cost[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )

    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.yl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.yh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    def objective(model):
        return sum(model.yh[v] - model.yl[v] for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=sense)

    # def nodes(model, v):
    #     return sum(model.sm_perf_cost[i] * model.x[v, i] for i in model.PartSize) <= model.sm_count_per_node

    # model.nodes = pyo.Constraint(model.Nodes, rule=nodes)

    def lower_bound(model, v, i):
        return model.yl[v] <= sum(
            model.x[w, i] for w in [neighbours for neighbours in model.Nodes if model.links[v, neighbours] > 0])

    model.lower_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.yh[v] >= sum(
            model.x[w, i] for w in [neighbour for neighbour in model.Nodes if model.links[v, neighbour] > 0])

    model.upper_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=upper_bound)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.sm_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 1200.0,
        'MIPFocus': 2,
        # 'MIPGap': max(0.01, 2 / len(model.Nodes))
    })

    return MinSpreadResult(
        graph=graph,
        result=result,
        model=model,
        partition_size=partition_size,
        seed=seed,
        opt_type='spread'
    )


def min_variance_n_partition(
        graph: Graph,  # links
        partition_size: int,  # dom_part_size
        seed: GeneratorSeed,
        sm_count_per_node: int = 1,  # k
        sm_perf_cost: list[float] = None,  # m
        sense: Sense = pyo.minimize,  # pyo.minimize
        solver_config: SolverConfig = SolverConfig()
) -> MinVarianceResult:
    """ Computes an MIQP for a given graph that minimises the sum of the variances of the means assigned to each node's inclusive neighbourhood in the graph. Exactly one mean is assigned to each node.

    Args:
        graph: NetworkX Graph
        partition_size: Number of partitions to create in the optimisation process.
        seed: Seed used to generate the graph.
        sm_count_per_node: Number of security means to assign to each node. Defaults to 1.
        sm_perf_cost: A list of performance costs for each partition size. If not provided, defaults to a list of ones with length equal to the partition size.
        sense: The direction of optimisation, either minimising or maximising the objective. Provided using Pyomo's ``Sense`` enumeration. Defaults to minimisation.
        solver_config: Configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        An object containing the result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)
    # opt.options['OutputFlag'] = 1  # tell gurobi to be verbose with output
    # opt.options['SensitivityAnalysis'] = 1
    # opt.options['SolCount'] = 1  # number of solutions to be found
    # opt.options['BestObjStop'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph)
    am.setdiag(1)

    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.node_degrees = pyo.Param(
        model.Nodes,
        within=pyo.PositiveIntegers,
        initialize={v: sum(model.links[v, w] for w in model.Nodes) for v in model.Nodes}
    )
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=partition_size
    )
    model.sm_count_per_node = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=sm_count_per_node
    )
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]
    model.sm_perf_cost = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: sm_perf_cost[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )

    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)

    # def get_neighbors(model, v, i):
    #     return [neighbour for neighbour in model.Nodes if
    #             model.links[v, neighbour] == 1 and model.x[v, i] == 1]

    # def objective(model):
    #     return sum((1 / model.part_size) * sum(((model.node_degrees[v] + 1) / model.part_size -
    #                                             1 + len(get_neighbors(model, v, i)))
    #                                            ** 2 for i in model.PartSize) for v in model.Nodes)

    # model.coverage = pyo.Objective(rule=objective, sense=sense)

    def objective(model):
        return sum(1 / model.part_size * sum(((model.node_degrees[v]) / model.part_size - sum(
            model.x[w, i] for w in [neighbours for neighbours in model.Nodes if model.links[v, neighbours] > 0]
        )) ** 2 for i in model.PartSize) for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=sense)

    # def nodes(model, v):
    #     return sum(model.sm_perf_cost[i] * model.x[v, i] for i in model.PartSize) <= model.sm_count_per_node

    # model.nodes = pyo.Constraint(model.Nodes, rule=nodes)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.sm_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 1200.0,
        'MIPFocus': 2,
        # 'MIPGap': max(0.01, 2 / len(model.Nodes))
    })

    return MinVarianceResult(
        graph=graph,
        result=result,
        model=model,
        partition_size=partition_size,
        seed=seed,
        opt_type='var'
    )


def opt_n_soft_domatic_partition(
        graph: Graph,  # links
        partition_size: int,  # dom_part_size
        seed: GeneratorSeed,
        weight: list[float] = None,  # c
        lowerBound: list[float] = None,  # l
        upperBound: list[float] = None,  # u
        sm_count_per_node: int = 1,  # k
        sm_perf_cost: list[float] = None,  # m
        sense: Sense = pyo.maximize,  # pyo.minimize
        solver_config: SolverConfig = SolverConfig()
) -> MinErrorsResult:
    """ Computes an optimal n-soft domatic partition for a given graph using the given MILP formulation.

    Args:
        graph:               graph to be partitioned
        partition_size:      partition size
        seed:                seed used to generate the graph
        weight:              weight of the sets in the partition
        lowerBound:          lower bound for each set in the partition
        upperBound:          upper bound for each set in the partition
        sm_count_per_node:   number of security means per node
        sm_perf_cost:        relative performance cost of each security mean
        sense:               pyo.minimize or pyo.maximize
        solver_config:       configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        Result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """
    weight = weight if weight else [1 for _ in range(partition_size)]
    lowerBound = lowerBound if lowerBound else [0 for _ in range(partition_size)]
    upperBound = upperBound if upperBound else [1 for _ in range(partition_size)]
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    model.Nodes = pyo.Set(initialize=list(graph.nodes()))
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph)
    am.setdiag(1)

    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.node_degrees = pyo.Param(
        model.Nodes,
        within=pyo.NonNegativeIntegers,
        initialize={v: sum(model.links[v, w] for w in model.Nodes) for v in model.Nodes}
    )
    model.dom_num = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=partition_size
    )
    model.sm_count_per_node = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=sm_count_per_node
    )
    model.sm_perf_cost = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: sm_perf_cost[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )
    model.c = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: weight[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )
    model.l = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: lowerBound[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )
    model.u = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: upperBound[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )

    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.y = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)

    def coverage(model):
        return sum(model.c[i] * sum(model.y[v, i] for v in model.Nodes)
                   for i in model.PartSize)

    model.coverage = pyo.Objective(rule=coverage, sense=sense)

    # def nodes(model, v):
    #     return sum(model.sm_perf_cost[i] * model.x[v, i] for i in model.PartSize) \
    #         <= model.sm_count_per_node

    # model.nodes = pyo.Constraint(model.Nodes, rule=nodes)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.sm_count_per_node

    # def nodes(model, v):
    #     return sum(model.sm_perf_cost[i] * model.x[v, i] for i in model.PartSize) \
    #         <= model.sm_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    def neighbourship(model, v, i):
        return model.y[v, i] <= sum([model.x[w, i]
                                     for w in [neighbours
                                               for neighbours in model.Nodes if model.links[v, neighbours] > 0]])

    model.neighbourship = pyo.Constraint(
        model.Nodes,
        model.PartSize,
        rule=neighbourship
    )

    def bounds(model, i):
        return pyo.inequality(
            model.l[i],
            sum(model.x[v, i] for v in model.Nodes) / len(model.Nodes),
            model.u[i]
        )

    model.bounds = pyo.Constraint(model.PartSize, rule=bounds)

    # TimeLimit in seconds: 40min = 2400s
    return MinErrorsResult(
        graph=graph,
        result=opt.solve(
            model,
            report_timing=True,
            options={
                # 'TimeLimit': 1200.0,
                'MIPFocus': 2,
                # 'MIPGap': max(0.01, 2 / len(model.Nodes))
            }
        ),
        model=model,
        partition_size=partition_size,
        seed=seed,
        opt_type='opt'
    )


def max_n_soft_domatic_partition(
        graph: Graph,  # links
        partition_size: int,  # dom_part_size
        seed: GeneratorSeed,
        weight: list[float] = None,  # c
        lowerBound: list[float] = None,  # l
        upperBound: list[float] = None,  # u
        sm_count_per_node: int = 1,  # k
        sm_perf_cost: list[float] = None,  # m
        sense: Sense = pyo.maximize,  # pyo.minimize
        solver_config: SolverConfig = SolverConfig()
) -> MinIncompleteNodesResult:
    """ Computes a maximal n-soft domatic partition for a given graph in form of an MILP.

    Args:
        graph:               graph to be partitioned
        partition_size:      partition size
        seed:                seed used to generate the graph
        weight:              weight of the sets in the partition
        lowerBound:          lower bound for each set in the partition
        upperBound:          upper bound for each set in the partition
        sm_count_per_node:   number of security means per node
        sm_perf_cost:        relative performance cost of each security mean
        sense:               pyo.minimize or pyo.maximize
        solver_config:       configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:
        Result of the optimisation process, including the graph, the optimisation results, the Pyomo model, partition size, and other relevant metadata.
    """

    weight = weight if weight else [1 for _ in range(partition_size)]
    lowerBound = lowerBound if lowerBound else [0 for _ in range(partition_size)]
    upperBound = upperBound if upperBound else [1 for _ in range(partition_size)]
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    model.Nodes = pyo.Set(initialize=list(graph.nodes()))
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph)
    am.setdiag(1)

    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.dom_num = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=partition_size
    )
    model.sm_count_per_node = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=sm_count_per_node
    )
    model.sm_perf_cost = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={
            i: sm_perf_cost[i - 1] for i in range(1, len(model.PartSize) + 1)
        }
    )
    model.c = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={
            i: weight[i - 1] for i in range(1, len(model.PartSize) + 1)
        }
    )
    model.l = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={
            i: lowerBound[i - 1] for i in range(1, len(model.PartSize) + 1)
        }
    )
    model.u = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={
            i: upperBound[i - 1] for i in range(1, len(model.PartSize) + 1)
        }
    )

    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.y = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.z = pyo.Var(model.Nodes, within=pyo.Binary)

    def coverage(model):
        return sum(sum(model.c[i] * model.x[v, i] for i in model.PartSize) * model.z[v]
                   for v in model.Nodes)

    model.coverage = pyo.Objective(rule=coverage, sense=sense)

    # def nodes(model, v):
    #     return sum(model.sm_perf_cost[i] * model.x[v, i]
    #                for i in model.PartSize) <= model.sm_count_per_node

    # model.nodes = pyo.Constraint(model.Nodes, rule=nodes)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.sm_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    def neighbourship(model, v, i):
        return model.y[v, i] <= sum(
            [model.x[w, i] for w in [neighbours
                                     for neighbours in model.Nodes if model.links[v, neighbours] > 0]]
        )

    model.neighbourship = pyo.Constraint(
        model.Nodes,
        model.PartSize,
        rule=neighbourship
    )

    def fulfilling_nodes(model, v, i):
        return model.z[v] <= model.y[v, i]

    model.fulfilling_nodes = pyo.Constraint(
        model.Nodes,
        model.PartSize,
        rule=fulfilling_nodes
    )

    def bounds(model, i):
        return pyo.inequality(
            model.l[i],
            sum(model.x[v, i] for v in model.Nodes) / len(model.Nodes),
            model.u[i])

    model.bounds = pyo.Constraint(model.PartSize, rule=bounds)

    # TimeLimit in seconds: 40min = 2400s
    return MinIncompleteNodesResult(
        graph=graph,
        result=opt.solve(
            model,
            report_timing=True,
            options={
                'TimeLimit': 1200.0,
                'MIPFocus': 2,
                # 'MIPGap': max(0.01, 2 / len(model.Nodes))}
            }
        ),
        model=model,
        partition_size=partition_size,
        seed=seed,
        opt_type='max'
    )


def min_spread_squared_n_partition(
        graph: Graph,  # links
        partition_size: int,  # dom_part_size
        seed: GeneratorSeed,
        sm_count_per_node: int = 1,  # k
        sm_perf_cost: list[float] = None,  # m
        sense: Sense = pyo.minimize,  # pyo.minimize
        solver_config: SolverConfig = SolverConfig()
) -> MinSpreadResult:
    """

    Args:
        graph:
        partition_size:
        seed:
        sm_count_per_node:
        sm_perf_cost:
        sense:
        solver_config: configuration for the solver, including the name, input/output interface, and flags for output streaming and file retention. Defaults to a standard Gurobi configuration.

    Returns:

    """

    opt = SolverFactory(solver_config.name, solver_io=solver_config.io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph)
    am.setdiag(1)

    model.links = pyo.Param(
        model.Nodes, model.Nodes,
        within=pyo.Binary,
        initialize={
            (v, w): am[i, j]
            for j, w in enumerate(model.Nodes)
            for i, v in enumerate(model.Nodes)
        }
    )
    model.node_degrees = pyo.Param(
        model.Nodes,
        within=pyo.NonNegativeIntegers,
        initialize={v: sum(model.links[v, w] for w in model.Nodes) for v in model.Nodes}
    )
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=partition_size
    )
    model.sm_count_per_node = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=sm_count_per_node
    )
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]
    model.sm_perf_cost = pyo.Param(
        model.PartSize,
        within=pyo.NonNegativeReals,
        initialize={i: sm_perf_cost[i - 1] for i in range(1, len(model.PartSize) + 1)}
    )

    model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
    model.yl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.yh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    def objective(model):
        return sum((model.yh[v] - model.yl[v]) ** 2 for v in model.Nodes)

    model.objective = pyo.Objective(rule=objective, sense=sense)

    # def nodes(model, v):
    #     return sum(model.sm_perf_cost[i] * model.x[v, i] for i in model.PartSize) <= model.sm_count_per_node

    # model.nodes = pyo.Constraint(model.Nodes, rule=nodes)

    def lower_bound(model, v, i):
        return model.yl[v] <= sum(
            model.x[w, i] for w in [neighbours for neighbours in model.Nodes if model.links[v, neighbours] > 0])

    model.lower_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=lower_bound)

    def upper_bound(model, v, i):
        return model.yh[v] >= sum(
            model.x[w, i] for w in [neighbour for neighbour in model.Nodes if model.links[v, neighbour] > 0])

    model.upper_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=upper_bound)

    def part(model, v):
        return sum(model.x[v, i] for i in model.PartSize) == model.sm_count_per_node

    model.part = pyo.Constraint(model.Nodes, rule=part)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 1200.0,
        'MIPFocus': 2,
        # 'MIPGap': max(0.01, 2 / len(model.Nodes))
    })

    return MinSpreadResult(
        graph=graph,
        result=result,
        model=model,
        partition_size=partition_size,
        seed=seed,
        opt_type='spread'
    )


# def min_spread_n_resource_partition(
#         graph,  # links
#         partition_size: int,  # dom_part_size
#         seed,
#         sm_count_per_node=1,  # k
#         sm_perf_cost=None,  # m
#         sense=pyo.minimize,  # pyo.minimize
#         solver='gurobi',
#         solver_io='python',
#         stream_solver=False,  # True prints solver output to screen
#         keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
#     """
#
#     :param graph:
#     :param partition_size:
#     :param seed:
#     :param sm_count_per_node:
#     :param sm_perf_cost:
#     :param sense:
#     :param solver:
#     :param solver_io:
#     :param stream_solver:
#     :param keepfiles:
#     :return:
#     """
#     opt = SolverFactory(solver, solver_io=solver_io)
#     opt.options['outlev'] = 1  # tell gurobi to be verbose with output
#     opt.options['solnsens'] = 1
#     opt.options['SolCount'] = 1  # number of solutions to be found
#     opt.options['bestbound'] = 1
#
#     model = pyo.ConcreteModel()
#
#     nodes = list(graph.nodes())
#     model.Nodes = pyo.Set(initialize=nodes)
#     model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))
#
#     am = adjacency_matrix(graph)
#     am.setdiag(1)
#
#     model.links = pyo.Param(
#         model.Nodes, model.Nodes,
#         within=pyo.Binary,
#         initialize={
#             (v, w): am[i, j]
#             for j, w in enumerate(model.Nodes)
#             for i, v in enumerate(model.Nodes)
#         }
#     )
#     model.node_degrees = pyo.Param(
#         model.Nodes,
#         within=pyo.NonNegativeIntegers,
#         initialize={v: sum(model.links[v, w] for w in model.Nodes) for v in model.Nodes}
#     )
#     model.part_size = pyo.Param(
#         within=pyo.PositiveIntegers,
#         initialize=partition_size
#     )
#     model.sm_count_per_node = pyo.Param(
#         within=pyo.PositiveIntegers,
#         initialize=sm_count_per_node
#     )
#     sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]
#     model.sm_perf_cost = pyo.Param(
#         model.PartSize,
#         within=pyo.NonNegativeReals,
#         initialize={i: sm_perf_cost[i - 1] for i in range(1, len(model.PartSize) + 1)}
#     )
#
#     model.x = pyo.Var(model.Nodes, model.PartSize, within=pyo.Binary)
#     model.yl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
#     model.yh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)
#
#     def objective(model):
#         return sum(model.yh[v] - model.yl[v] for v in model.Nodes)
#
#     model.objective = pyo.Objective(rule=objective, sense=sense)
#
#     # def nodes(model, v):
#     #     return sum(model.sm_perf_cost[i] * model.x[v, i] for i in model.PartSize) <= model.sm_count_per_node
#
#     # model.nodes = pyo.Constraint(model.Nodes, rule=nodes)
#
#     def lower_bound(model, v, i):
#         return model.yl[v] <= sum(
#             model.x[w, i] for w in [neighbours for neighbours in model.Nodes if model.links[v, neighbours] > 0])
#
#     model.lower_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=lower_bound)
#
#     def upper_bound(model, v, i):
#         return model.yh[v] >= sum(
#             model.x[w, i] for w in [neighbour for neighbour in model.Nodes if model.links[v, neighbour] > 0])
#
#     model.upper_bound = pyo.Constraint(model.Nodes, model.PartSize, rule=upper_bound)
#
#     def part(model, v):
#         return sum(model.x[v, i] for i in model.PartSize) == model.sm_count_per_node
#
#     model.part = pyo.Constraint(model.Nodes, rule=part)
#
#     result = opt.solve(model, report_timing=True, options={
#         'TimeLimit': 1200.0,
#         'MIPFocus': 2,
#         # 'MIPGap': max(0.01, 2 / len(model.Nodes))
#     })
#
#     return MinSpreadResult(
#         graph=graph,
#         result=result,
#         model=model,
#         partition_size=partition_size,
#         seed=seed,
#         opt_type='spread'
#     )

def _max_packings(
        means: tuple[tuple[int, ...], ...],
        resources: tuple[int, ...],
        n: int,
        config: list[tuple[int, ...]] = None,
) -> set[tuple[tuple[int, ...], ...]]:
    """Finds all maximal packings using integer arithmetic.

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


def _max_packings_matrix(
        means: tuple[tuple[float, ...], ...],
        resources: tuple[float, ...]
) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], tuple[tuple[int, ...], ...]]:
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

    # Scale resources to integers
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


if __name__ == '__main__':
    means = (
        (0.6, 0.1, 0.5),
        (0.2, 0.6, 0.3),
        (0.3, 0.2, 0.1),
        (0.7, 0.2, 0.3),
        (1.0, 0.5, 0.8),
        (0.1, 0.3, 0.4),
    )
    resources = (1.0, 1.0, 1.0)
    # apps = tuple(tuple(app + (i,) for app, i in zip(apps, range(len(apps)))))
    # resources += (np.inf,)
    # packing_result = {{app[-1]: app[:-1] for app in config}
    #                   for config in _max_packings(apps, resources, len(apps))
    #                   }
    # print(packing_result)
    # print(len(packing_result))

    packings, mapping_matrix = _max_packings_matrix(means, resources)
    print(f"Packings: {packings}")
    print(f"Mapping Matrix: {mapping_matrix}")

    # print(len(_max_packings(applications, resources, len(applications))))
