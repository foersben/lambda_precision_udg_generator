import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from networkx import adjacency_matrix

from .result import MinSpreadResult, MinVarianceResult, MinIncompleteNodesResult, MinErrorsResult

# PartitioningResult

__all__ = [
    'opt_n_soft_domatic_partition',
    'max_n_soft_domatic_partition',
    'min_variance_n_partition',
    'min_spread_n_partition',
]


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


if __name__ == '__main__':
    pass
