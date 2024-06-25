from typing import Tuple, Set, Any

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from networkx import adjacency_matrix
import numpy as np

# from .result import MinSpreadResult, MinVarianceResult, MinIncompleteNodesResult, MinErrorsResult, \
#     MinSpreadResourceResult

# PartitioningResult

__all__ = [
    'opt_n_soft_domatic_partition',
    'max_n_soft_domatic_partition',
    'min_variance_n_partition',
    'min_spread_n_partition',
    'min_spread_resource_multi_distribution_1',
    'min_spread_resource_multi_distribution_2',
    'min_spread_resource_multi_distribution_3',
]


def min_spread_resource_multi_distribution_3(
        graph,  # links
        seed,
        sm_node_resources=(1,),  #: tuple[float, ... | tuple[float, ...]] = (1,),  # k
        sm_perf_cost: tuple[tuple[float, ...], ...] = None,  # m
        sense=pyo.minimize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    # _max_packings_matrix(apps: tuple[tuple, ...], resources: tuple) -> tuple[tuple, ...]:

    if sm_perf_cost is None:
        raise ValueError("Performance cost matrix must be provided.")

    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.SecMeans = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.Resources = pyo.Set(initialize=range(1, len(sm_node_resources) + 1))
    # unused
    model.PartSize = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.node_degrees = None

    am = adjacency_matrix(graph.graph)
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
        initialize={r: sm_node_resources[r - 1] for r in model.Resources}
    )

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
            model.sm_perf_cost[j, r] * model.x[v, j] for j in model.SecMeans) - model.epsilon

    model.resource_usage = pyo.Constraint(model.Nodes, model.SecMeans, model.Resources, rule=resource_usage)

    def lower_bound(model, v, i):
        # neighbours =
        return model.xl[v] <= sum(model.x[w, i] for w in [w for w in model.Nodes if model.links[v, w] > 0])

    model.lower_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=lower_bound)

    def upper_bound(model, v, i):
        # neighbours =
        return model.xh[v] >= sum(model.x[w, i] for w in [w for w in model.Nodes if model.links[v, w] > 0])

    model.upper_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=upper_bound)

    def resource_constraint(model, v, r):
        return sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.SecMeans) <= model.sm_node_resources[r]

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
        graph,  # links
        seed,
        sm_node_resources=(1,),  #: tuple[float, ... | tuple[float, ...]] = (1,),  # k
        sm_perf_cost: tuple[tuple[float, ...], ...] = None,  # m
        sense=pyo.minimize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    # _max_packings_matrix(apps: tuple[tuple, ...], resources: tuple) -> tuple[tuple, ...]:

    if sm_perf_cost is None:
        raise ValueError("Performance cost matrix must be provided.")

    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.SecMeans = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.Resources = pyo.Set(initialize=range(1, len(sm_node_resources) + 1))
    # unused
    model.PartSize = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.node_degrees = None

    am = adjacency_matrix(graph.graph)
    am.setdiag(1)

    # unused
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=len(sm_perf_cost)
    )
    model.reward_factor = pyo.Param(within=pyo.NonNegativeReals, initialize=0.1)
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
        initialize={r: sm_node_resources[r - 1] for r in model.Resources}
    )

    model.x = pyo.Var(model.Nodes, model.SecMeans, within=pyo.Binary)
    model.xl = pyo.Var(model.Nodes, within=pyo.NonNegativeIntegers)
    model.xh = pyo.Var(model.Nodes, within=pyo.PositiveIntegers)

    # model.y = pyo.Var(within=pyo.PositiveIntegers)

    def objective(model):
        return sum(model.xh[v] - model.xl[v] for v in model.Nodes) + model.reward_factor * sum(
            model.sm_node_resources[r] - sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.SecMeans) for r
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
        # neighbours =
        return model.xl[v] <= sum(model.x[w, i] for w in [w for w in model.Nodes if model.links[v, w] > 0])

    model.lower_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=lower_bound)

    def upper_bound(model, v, i):
        # neighbours =
        return model.xh[v] >= sum(model.x[w, i] for w in [w for w in model.Nodes if model.links[v, w] > 0])

    model.upper_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=upper_bound)

    def resource_constraint(model, v, r):
        return sum(model.sm_perf_cost[i, r] * model.x[v, i] for i in model.SecMeans) <= model.sm_node_resources[r]

    model.resource_constraint = pyo.Constraint(model.Nodes, model.Resources, rule=resource_constraint)

    # def sec_per_node(model, v):
    #     return sum(model.x[v, i] for i in model.SecMeans) >= 1

    # model.sec_per_node = pyo.Constraint(model.Nodes, rule=sec_per_node)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 60.0,
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
        graph,  # links
        seed,
        sm_node_resources=(1,),  #: tuple[float, ... | tuple[float, ...]] = (1,),  # k
        sm_perf_cost: tuple[tuple[float, ...], ...] = None,  # m
        sense=pyo.minimize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    # _max_packings_matrix(apps: tuple[tuple, ...], resources: tuple) -> tuple[tuple, ...]:
    """
    Computes a minimal spread resource n-partition for a given graph.
    The result class/container is wrong,  it should contain a list of  security means applied to each node, instead it
    shows only one security mean per node.

    :param graph:
    :param seed:
    :param sm_node_resources:
    :param sm_perf_cost:
    :param sense:
    :param solver:
    :param solver_io:
    :param stream_solver:
    :param keepfiles:
    :return:
    """

    if sm_perf_cost is None:
        raise ValueError("Performance cost matrix must be provided.")

    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    packings, mapping_matrix = _max_packings_matrix(sm_perf_cost, sm_node_resources)

    nodes = list(graph.graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.SecMeans = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.SMPackings = pyo.Set(initialize=range(1, len(packings) + 1))
    # unused
    model.PartSize = pyo.Set(initialize=range(1, len(sm_perf_cost) + 1))
    model.node_degrees = None

    am = adjacency_matrix(graph.graph)
    am.setdiag(1)

    # unused
    model.part_size = pyo.Param(
        within=pyo.PositiveIntegers,
        initialize=len(sm_perf_cost)
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
        neighbours = [w for w in model.Nodes if model.links[v, w] > 0]
        return model.xl[v] <= sum(model.x[w, i] for w in neighbours)

    model.lower_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=lower_bound)

    def upper_bound(model, v, i):
        neighbours = [w for w in model.Nodes if model.links[v, w] > 0]
        return model.xh[v] >= sum(model.x[w, i] for w in neighbours)

    model.upper_bound = pyo.Constraint(model.Nodes, model.SecMeans, rule=upper_bound)

    def packing_distribution(model, v):
        return sum(model.y[v, i] for i in model.SMPackings) == 1

    model.part = pyo.Constraint(model.Nodes, rule=packing_distribution)

    result = opt.solve(model, report_timing=True, options={
        'TimeLimit': 600.0,
        'MIPFocus': 3,
        'MIPGap': 0.2
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
        graph,  # links
        partition_size: int,  # dom_part_size
        seed,
        sm_count_per_node=1,  # k
        sm_perf_cost=None,  # m
        sense=pyo.minimize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    """

    :param graph:
    :param partition_size:
    :param seed:
    :param sm_count_per_node:
    :param sm_perf_cost:
    :param sense:
    :param solver:
    :param solver_io:
    :param stream_solver:
    :param keepfiles:
    :return:
    """
    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph.graph)
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
        graph,  # links
        partition_size: int,  # dom_part_size
        seed,
        sm_count_per_node=1,  # k
        sm_perf_cost=None,  # m
        sense=pyo.minimize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    """

    :param graph:
    :param partition_size:
    :param seed:
    :param sm_count_per_node:
    :param sm_perf_cost:
    :param sense:
    :param solver:
    :param solver_io:
    :param stream_solver:
    :param keepfiles:
    :return:
    """

    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph.graph)
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
        graph,  # links
        partition_size: int,  # dom_part_size
        seed,
        weight=None,  # c
        lowerBound=None,  # l
        upperBound=None,  # u
        sm_count_per_node=1,  # k
        sm_perf_cost=None,  # m
        sense=pyo.maximize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    """
    Computes an optimal n-soft domatic partition for a given graph.

    :param graph:               graph to be partitioned
    :param partition_size:      partition size
    :param seed:                seed used to generate the graph
    :param weight:              weight of the sets in the partition
    :param lowerBound:          lower bound for each set in the partition
    :param upperBound:          upper bound for each set in the partition
    :param sm_count_per_node:   number of security means per node
    :param sm_perf_cost:        relative performance cost of each security mean
    :param sense:               pyo.minimize or pyo.maximize
    :param solver:              gurobi
    :param solver_io:           python or mps
    :param stream_solver:       True prints solver output to screen
    :param keepfiles:           True prints intermediate file names (.nl,.sol,...)
    :return:                    PartitioningResult
    """
    weight = weight if weight else [1 for _ in range(partition_size)]
    lowerBound = lowerBound if lowerBound else [0 for _ in range(partition_size)]
    upperBound = upperBound if upperBound else [1 for _ in range(partition_size)]
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for _ in range(partition_size)]

    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    model.Nodes = pyo.Set(initialize=list(graph.graph.nodes()))
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph.graph)
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
        graph,  # links
        partition_size: int,  # dom_part_size
        seed,
        weight=None,  # c
        lowerBound=None,  # l
        upperBound=None,  # u
        sm_count_per_node=1,  # k
        sm_perf_cost=None,  # m
        sense=pyo.maximize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    """
    Computes a maximal n-soft domatic partition for a given graph.

    :param graph:               graph to be partitioned
    :param partition_size:      partition size
    :param seed:                seed used to generate the graph
    :param weight:              weight of the sets in the partition
    :param lowerBound:          lower bound for each set in the partition
    :param upperBound:          upper bound for each set in the partition
    :param sm_count_per_node:   number of security means per node
    :param sm_perf_cost:        relative performance cost of each security mean
    :param sense:               pyo.minimize or pyo.maximize
    :param solver:              gurobi
    :param solver_io:           python or mps
    :param stream_solver:       True prints solver output to screen
    :param keepfiles:           True prints intermediate file names (.nl,.sol,...)
    :return:                    PartitioningResult
    """

    weight = weight if weight else [1 for i in range(partition_size)]
    lowerBound = lowerBound if lowerBound else [0 for i in range(partition_size)]
    upperBound = upperBound if upperBound else [1 for i in range(partition_size)]
    sm_perf_cost = sm_perf_cost if sm_perf_cost else [1 for i in range(partition_size)]

    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    model.Nodes = pyo.Set(initialize=list(graph.graph.nodes()))
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph.graph)
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
        graph,  # links
        partition_size: int,  # dom_part_size
        seed,
        sm_count_per_node=1,  # k
        sm_perf_cost=None,  # m
        sense=pyo.minimize,  # pyo.minimize
        solver='gurobi',
        solver_io='python',
        stream_solver=False,  # True prints solver output to screen
        keepfiles=False):  # True prints intermediate file names (.nl,.sol,...)
    """

    :param graph:
    :param partition_size:
    :param seed:
    :param sm_count_per_node:
    :param sm_perf_cost:
    :param sense:
    :param solver:
    :param solver_io:
    :param stream_solver:
    :param keepfiles:
    :return:
    """
    opt = SolverFactory(solver, solver_io=solver_io)
    opt.options['outlev'] = 1  # tell gurobi to be verbose with output
    opt.options['solnsens'] = 1
    opt.options['SolCount'] = 1  # number of solutions to be found
    opt.options['bestbound'] = 1

    model = pyo.ConcreteModel()

    nodes = list(graph.graph.nodes())
    model.Nodes = pyo.Set(initialize=nodes)
    model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

    am = adjacency_matrix(graph.graph)
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
#     nodes = list(graph.graph.nodes())
#     model.Nodes = pyo.Set(initialize=nodes)
#     model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))
#
#     am = adjacency_matrix(graph.graph)
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


def _max_packings(apps: tuple[tuple[float, ...], ...], res: tuple[float, ...], n: int,
                  config: list[tuple[float, ...], ...] = []) -> set[tuple[tuple[float, ...], ...], ...] | set[Any]:
    """
    Computes the maximal packings of a set of applications given a set of resources.
    :param apps:
    :param res:
    :param n:
    :param config:
    :return:
    """
    if any(j < 0 for j in res):
        if all(any(app_comp > res_comp for app_comp, res_comp in zip(app, np.array(res) + config[-1])) for app in
               set(apps) - set(config[:-1])):
            return {tuple(config[:-1])}
        else:
            return set()
    elif not n:
        if all(any(app_comp > res_comp for app_comp, res_comp in zip(app, res))
               for app in set(apps) - set(config)):
            return {tuple(config)}
        else:
            return set()
    return _max_packings(apps, res, n - 1, config).union(_max_packings(
        apps,
        tuple(j - i for i, j in zip(apps[n - 1], res)),
        n - 1,
        config + [apps[n - 1]]
    ))


def _max_packings_matrix(apps: tuple[tuple[float, ...], ...], resources: tuple[float, ...]) -> tuple[
    tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...]]:
    """
    Computes the maximal packings of a set of applications given a set of resources.

    :param apps:
    :param resources:
    :return:
    """
    apps = tuple(tuple(app + (i,) for app, i in zip(apps, range(len(apps)))))
    resources += (np.inf,)
    packing_result = _max_packings(apps, resources, len(apps))

    matrix = np.zeros((len(packing_result), len(apps)), dtype=int)
    for i, config in enumerate(packing_result):
        for app in config:
            matrix[i, app[-1]] = 1

    return (
        tuple(tuple(app[:-1] for app in config) for config in packing_result),
        tuple(tuple(config) for config in matrix)
    )


if __name__ == '__main__':
    apps = (
        (0.6, 0.1, 0.5),
        (0.2, 0.6, 0.3),
        (0.3, 0.2, 0.1),
        (0.7, 0.2, 0.3),
        (0.1, 0.3, 0.4),
    )
    # apps = tuple(tuple(app + (i,) for app, i in zip(apps, range(len(apps)))))
    resources = (1.0, 1.0, 1.0)
    # resources += (np.inf,)
    # packing_result = {{app[-1]: app[:-1] for app in config}
    #                   for config in _max_packings(apps, resources, len(apps))
    #                   }
    # print(packing_result)
    # print(len(packing_result))

    packings, mapping_matrix = _max_packings_matrix(apps, resources)
    print(f"Packings: {packings}")
    print(f"Mapping Matrix: {mapping_matrix}")

    # print(len(_max_packings(applications, resources, len(applications))))
