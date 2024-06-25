from src.partitioning.partitioning import opt_n_soft_domatic_partition, max_n_soft_domatic_partition, \
    min_variance_n_partition, min_spread_n_partition, min_spread_resource_multi_distribution_1, \
    min_spread_resource_multi_distribution_2, min_spread_resource_multi_distribution_3
from src.partitioning.result import PartitioningResultDB

from src.graph_generation.generator_seeds import (
    UDGGeneratorSeedDB,
    UDGSeedGenerator
)


def test_partitioning():
    generator = UDGSeedGenerator(sample_size=10)
    seeds = generator.generate_seeds(
        avg_degs=[3, 4],  # , 5, 6],
        node_numbers=[
            20,
            40,
            # 60,
            # 80,
            # 100,
            # 120,
            # 140,
            # 160,
            # 180,
            # 200,
            # 220,
            # 240,
            # 260,
            # 280,
            # 300,
        ],
    )
    for seed in seeds:
        seed.generate_graphs(5)
    seed_db = UDGGeneratorSeedDB(*seeds)
    # seed_db.serialize(f"test_output/{id(seed_db)}")
    # seed_db = UDGGeneratorSeedDB.deserialize(f"test_output/{id(seed_db)}")
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)

    partitioning_result_db = PartitioningResultDB()

    print(f'Number of Seeds: {len(seed_db.seeds)}')
    for seed in seed_db.seeds:
        print(f'Number of nodes in seed: {seed.node_number}')
        print(f'Number of graphs in seed: {len(seed.graphs)}')
        for graph in seed.graphs:
            res = opt_n_soft_domatic_partition(graph=graph, partition_size=3, seed=seed)
            partitioning_result_db.append(res)
            print(res)
            res = opt_n_soft_domatic_partition(graph=graph, partition_size=4, seed=seed)
            partitioning_result_db.append(res)
            print(res)
            # res = opt_n_soft_domatic_partition(graph=graph, partition_size=5, seed=seed)
            # partitioning_result_db.append(res)
            # print(res)
            # res = opt_n_soft_domatic_partition(graph=graph, partition_size=6, seed=seed)
            # partitioning_result_db.append(res)
            # print(res)
    # differentiate optimisation types
    try:
        partitioning_result_db.plot(
            data_key=1,
            partition_size=4,
            filepath="test_output/test"
        )
    except Exception as e:
        print(e)
    try:
        partitioning_result_db.plot(
            data_key=2,
            partition_size=4,
            filepath="test_output/test"
        )
    except Exception as e:
        print(e)
    try:
        partitioning_result_db.plot(
            data_key=0,
            partition_size=3,
            filepath="test_output/test"
        )
    except Exception as e:
        print(e)

    partitioning_result_db.latex_table_opt_max()


def test_partitioning_latex_table():
    generator = UDGSeedGenerator(sample_size=10)
    seeds = generator.generate_seeds(
        avg_degs=[3, 4],
        node_numbers=[20, 40],
    )
    for seed in seeds:
        seed.generate_graphs(5)
    seed_db = UDGGeneratorSeedDB(*seeds)
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)

    partitioning_result_db = PartitioningResultDB()

    print(f'Number of Seeds: {len(seed_db.seeds)}')
    for seed in seed_db.seeds:
        print(f'Number of nodes in seed: {seed.node_number}')
        print(f'Number of graphs in seed: {len(seed.graphs)}')
        for graph in seed.graphs:
            res = opt_n_soft_domatic_partition(graph=graph, partition_size=3, seed=seed)
            partitioning_result_db.append(res)
            print(res)
            res = opt_n_soft_domatic_partition(graph=graph, partition_size=4, seed=seed)
            partitioning_result_db.append(res)
            print(res)
    # differentiate optimisation types
    print(str(partitioning_result_db.latex_table_opt_max()))
    print(partitioning_result_db.latex_table_opt_max())


def test_partitioning_variance():
    generator = UDGSeedGenerator(sample_size=10)
    seeds = generator.generate_seeds(
        avg_degs=[3, 4],
        node_numbers=[100, 200],
    )
    for seed in seeds:
        seed.generate_graphs(3, connected=True)
    seed_db = UDGGeneratorSeedDB(*seeds)
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)

    partitioning_result_db = PartitioningResultDB()

    print(f'Number of Seeds: {len(seed_db.seeds)}')
    for seed in seed_db.seeds:
        print(f'Number of nodes in seed: {seed.node_number}')
        print(f'Number of graphs in seed: {len(seed.graphs)}')
        for graph in seed.graphs:
            res = min_variance_n_partition(graph=graph, partition_size=4, seed=seed)
            partitioning_result_db.append(res)
            print(res)


def test_partitioning_opt():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_n_soft_domatic_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_opt")
            seeds[i] = None


def test_partitioning_opt():
    for partition_size in [3, 4]:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            if seed.avg_deg_bound[0] < 4:
                print("================")
                print(f"Skipped AVG DEG BOUND: {seed.avg_deg_bound[0]}")
                print("================")
                seeds[i] = None
                continue
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_n_soft_domatic_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning3_opt_var")
            seeds[i] = None


def test_partitioning_opt_wo_bridges():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_bridges").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_n_soft_domatic_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_opt_wo_bridges")
            seeds[i] = None


def test_partitioning_max():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = max_n_soft_domatic_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_max")
            seeds[i] = None


def test_partitioning_max_wo_bridges():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_bridges").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_n_soft_domatic_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_max_wo_bridges")
            seeds[i] = None


def test_partitioning_var():
    partition_sizes = [3, 4, 5]
    for partition_size in partition_sizes:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            # if seed.node_number != 300:
            #     seeds[i] = None
            #     continue
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for i, graph in enumerate(seed.graphs):
                # if i == 0 or seed.avg_deg_bound[0] == 3:
                #     continue
                res = min_variance_n_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_var")
            seeds[i] = None


def test_partitioning_var_deg3():
    partition_sizes = [3, 4]
    for partition_size in partition_sizes:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            if seed:
                if seed.avg_deg_bound[0] < 4 or seed.node_number not in [200, 300]:
                    seeds[i] = None
                    continue
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for i, graph in enumerate(seed.graphs):
                res = min_variance_n_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_var_deg3_1")
            seeds[i] = None


def test_partitioning_opt_deg3():
    partition_sizes = [3, 4]
    for partition_size in partition_sizes:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3").seeds,
            key=lambda seed: seed.node_number)
        seeds = list(filter(lambda seed: seed.node_number not in [300], seeds))
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            print(f"Seed: {i}, Avg Deg Bound: {seed.avg_deg_bound[0]}")
            # if seed.avg_deg_bound[0] > 3 or seed.node_number not in [100, 200, 300]:
            #     seeds[i] = None
            #     continue
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for j, graph in enumerate(seed.graphs):
                print(f"Graph: {j}")
                res = opt_n_soft_domatic_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_opt_deg3_1")
            seeds[i] = None


def test_partitioning_spread():
    partition_sizes = [3, 4, 5]
    for partition_size in partition_sizes:
        seeds = sorted(
            UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3").seeds,
            key=lambda seed: seed.node_number)
        print(f'Number of Seeds: {len(seeds)}')
        for i, seed in enumerate(seeds):
            # if seed.node_number != 300:
            #     seeds[i] = None
            #     continue
            print(f'Number of nodes in seed: {seed.node_number}')
            print(f'Number of graphs in seed: {len(seed.graphs)}')
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = min_spread_n_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_spread_squared")
            seeds[i] = None


def test_partitioning_spread_resource_based():
    sm_perf_costs = (
        # (1.0, 1.0, 1.0),
        # (1.0, 1.0, 1.0),
        # (1.0, 1.0, 1.0),
        (0.6, 0.1, 0.5),
        (0.2, 0.6, 0.3),
        (0.3, 0.2, 0.1),
        (0.7, 0.2, 0.3),
        (1.0, 0.5, 0.8),
        (0.1, 0.3, 0.4),
    )
    sm_node_resources = (1.0, 1.0, 1.0)
    seeds = sorted(
        UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3").seeds,
        key=lambda seed: seed.node_number)
    for i, seed in enumerate(seeds):
        if seed:
            if seed.node_number not in [100, 200, 300]:
                seeds[i] = None
                continue
        print(f'Number of nodes in seed: {seed.node_number}')
        print(f'Number of graphs in seed: {len(seed.graphs)}')
        partitioning_result_db1 = PartitioningResultDB()
        partitioning_result_db2 = PartitioningResultDB()
        partitioning_result_db3 = PartitioningResultDB()
        for j, graph in enumerate(seed.graphs):
            print(f"Graph: {j}")
            print(f"Seed: {i}, Avg Deg Bound: {seed.avg_deg_bound[0]}, Security Mean Count: {len(sm_perf_costs)}")
            partitioning_result_db1.append(
                min_spread_resource_multi_distribution_1(
                    graph=graph,
                    seed=seed,
                    sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                )
            )
            partitioning_result_db2.append(
                min_spread_resource_multi_distribution_2(
                    graph=graph,
                    seed=seed,
                    sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                )
            )
            partitioning_result_db3.append(
                min_spread_resource_multi_distribution_3(
                    graph=graph,
                    seed=seed,
                    sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                )
            )
            print(f"Result1: {partitioning_result_db1.results[-1]}")
            print(f"Result2: {partitioning_result_db2.results[-1]}")
            print(f"Result3: {partitioning_result_db3.results[-1]}")
        partitioning_result_db1.serialize(path="../test_output/test_partitioning_opt_spread_resource/1")
        partitioning_result_db2.serialize(path="../test_output/test_partitioning_opt_spread_resource/2")
        partitioning_result_db3.serialize(path="../test_output/test_partitioning_opt_spread_resource/3")
        seeds[i] = None


def test_serialize_results():
    results = PartitioningResultDB.deserialize("../test_output/test_partitioning_opt")
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning_max")
    results.latex_table_opt_max()
    # for result in results.results:


def test_serialize_results2():
    results = PartitioningResultDB.deserialize("../test_output/test_partitioning_var")
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning_spread")
    results.latex_table_var_spread()


def test_eval_results_opt():
    results = PartitioningResultDB.deserialize("../test_output/test_partitioning3_opt/")
    # for data_key in [0, 1, 2]:
    #     for partition_size in [3, 4, 5]:
    #         results.plot(opt="opt", data_key=data_key, partition_size=partition_size,
    #                      filepath=f"../test_output/test_partitioning3_opt/data_key_{str(data_key)}_partition_size_{str(partition_size)}.tex")
    print(results.latex_table_opt_max())
    results_max = PartitioningResultDB.deserialize("../test_output/test_partitioning3_max/")
    results.compare_mean_miss_cov(results_max)
    results.compare_mean_incomplete_nodes(results_max)


def test_eval_results_max():
    results = PartitioningResultDB.deserialize("../test_output/test_partitioning3_max/")
    # for data_key in [0, 1, 2]:
    #     for partition_size in [3, 4, 5]:
    #         results.plot(opt="max", data_key=data_key, partition_size=partition_size,
    #                      filepath=f"../test_output/test_partitioning3_max/data_key_{str(data_key)}_partition_size_{str(partition_size)}.tex")
    print(results.latex_table_opt_max())


def test_eval_results_opt_max_inc():
    results = PartitioningResultDB.deserialize("../test_output/test_results_opt_max_14_10_23/test_partitioning_max/")
    for data_key in [0, 1, 2]:
        for partition_size in [3, 4, 5]:
            results.plot(opt="max", data_key=data_key, partition_size=partition_size,
                         filepath=f"../test_output/test_results_opt_max_14_10_23/data_key_{str(data_key)}_partition_size_{str(partition_size)}.tex")


def test_eval_results_var_spread_opt():
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning3_spread2/")
    results = PartitioningResultDB.deserialize("../test_output/test_partitioning3_var2/")
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning_var_deg3_1")
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning_opt_deg3_1/")
    results.results = list(filter(lambda result: result.seed.avg_deg_bound[0] > 3, results.results))
    results.results = list(filter(lambda result: result.partition_size < 5, results.results))
    # for res in results.results:
    #     print(f"Partitioning: {res.partitioning}")
    #     res.graph().draw_random_geometric_graph(partitioning=res.partitioning)
    #     break
    print(results.latex_table_var_spread())

    # results2 = PartitioningResultDB.deserialize("../test_output/test_partitioning3_opt_var")
    # results2 = PartitioningResultDB.deserialize("../test_output/test_partitioning3_opt")
    # results2.results = list(filter(lambda result: result.seed.avg_deg_bound[0] > 3, results2.results))
    # results2.results = list(filter(lambda result: result.partition_size < 5, results2.results))

    # for res in results2.results:
    #     print(f"Partitioning: {res.partitioning}")
    #     res.graph().draw_random_geometric_graph(partitioning=res.partitioning)
    #     break
    # print("Opt:")
    # print(results2.latex_table_opt_max(miss_cov=False, inc_nodes=True))
    # print(results2.latex_table_var_spread())
    # print("Variance:")
    # print(results.latex_table_var_spread())

    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning3_spread/")
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning3_opt/")
    # print(results.latex_table_opt_max())
    # for data_key in [0, 1, 2]:
    #     for partition_size in [3, 4, 5]:
    #         results.plot(opt="opt", data_key=data_key, partition_size=partition_size,
    #                      filepath=f"../test_output/test_partitioning3_var/data_key_{str(data_key)}_partition_size_{str(partition_size)}.tex")

# import networkx as nx
# import pyomo.environ as pyo
# from networkx import adjacency_matrix

# graph = nx.random_geometric_graph(10, 0.1)
# partition_size = 4
# model = pyo.ConcreteModel()

# nodes = list(graph.nodes())
# model.Nodes = pyo.Set(initialize=nodes)
# model.PartSize = pyo.Set(initialize=range(1, partition_size + 1))

# am = adjacency_matrix(graph)
# am.setdiag(1)

# model.links = pyo.Param(
#     model.Nodes, model.Nodes,
#     within=pyo.Binary,
#     initialize={
#         (model.Nodes[i], model.Nodes[j]):
#             am[i - 2, j - 1]
#         for j in range(1, len(model.Nodes) + 1)
#         for i in range(1, len(model.Nodes) + 1)
#     }
# )
# for v in model.Nodes:
#     print("=============")
#     for i, link in enumerate(model.links[v, :]):
#         print(f"v={v}: i={i}:{link}")
