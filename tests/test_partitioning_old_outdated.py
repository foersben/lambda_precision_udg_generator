import pytest

from lambdaprecisionudggenerator.graph_generator.seeds.database import (
    GeneratorSeed,
    GeneratorSeedDB,
)
from lambdaprecisionudggenerator.graph_generator.seeds.generator import SeedGenerator
from lambdaprecisionudggenerator.partitioning import PartitioningResultDB
from lambdaprecisionudggenerator.partitioning.partitioning import (
    _max_packings_matrix,
    max_soft_domatic_partition,
    min_spread_partition,
    min_variance_partition,
    opt_soft_domatic_partition,
    spread_based_configurations_distribution,
    spread_based_max_resource_utilisation_distribution,
    spread_resource_based_distribution,
)


def test_partitioning():
    generator = SeedGenerator(sample_size=10)
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
    seed_db = GeneratorSeedDB(*seeds)
    # seed_db.serialize(f"test_output/{id(seed_db)}")
    # seed_db = UDGGeneratorSeedDB.deserialize(f"test_output/{id(seed_db)}")
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)

    partitioning_result_db = PartitioningResultDB()

    print(f"Number of Seeds: {len(seed_db.seeds)}")
    for seed in seed_db.seeds:
        print(f"Number of nodes in seed: {seed.node_number}")
        print(f"Number of graphs in seed: {len(seed.graphs)}")
        for graph in seed.graphs:
            res = opt_soft_domatic_partition(graph=graph, partition_size=3, seed=seed)
            partitioning_result_db.append(res)
            print(res)
            res = opt_soft_domatic_partition(graph=graph, partition_size=4, seed=seed)
            partitioning_result_db.append(res)
            print(res)
            # res = opt_soft_domatic_partition(graph=graph, partition_size=5, seed=seed)
            # partitioning_result_db.append(res)
            # print(res)
            # res = opt_soft_domatic_partition(graph=graph, partition_size=6, seed=seed)
            # partitioning_result_db.append(res)
            # print(res)
    # differentiate optimisation types
    try:
        partitioning_result_db.plot(data_key=1, partition_size=4, filepath="test_output/test")
    except Exception as e:
        print(e)
    try:
        partitioning_result_db.plot(data_key=2, partition_size=4, filepath="test_output/test")
    except Exception as e:
        print(e)
    try:
        partitioning_result_db.plot(data_key=0, partition_size=3, filepath="test_output/test")
    except Exception as e:
        print(e)

    partitioning_result_db.latex_table_opt_max()


def test_partitioning_latex_table():
    generator = SeedGenerator(sample_size=10)
    seeds = generator.generate_seeds(
        avg_degs=[3, 4],
        node_numbers=[20, 40],
    )
    for seed in seeds:
        seed.generate_graphs(5)
    seed_db = GeneratorSeedDB(*seeds)
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)

    partitioning_result_db = PartitioningResultDB()

    print(f"Number of Seeds: {len(seed_db.seeds)}")
    for seed in seed_db.seeds:
        print(f"Number of nodes in seed: {seed.node_number}")
        print(f"Number of graphs in seed: {len(seed.graphs)}")
        for graph in seed.graphs:
            res = opt_soft_domatic_partition(graph=graph, partition_size=3, seed=seed)
            partitioning_result_db.append(res)
            print(res)
            res = opt_soft_domatic_partition(graph=graph, partition_size=4, seed=seed)
            partitioning_result_db.append(res)
            print(res)
    # differentiate optimisation types
    print(str(partitioning_result_db.latex_table_opt_max()))
    print(partitioning_result_db.latex_table_opt_max())


def test_partitioning_variance():
    generator = SeedGenerator(sample_size=10)
    seeds = generator.generate_seeds(
        avg_degs=[3, 4],
        node_numbers=[100, 200],
    )
    for seed in seeds:
        seed.generate_graphs(3, connected=True)
    seed_db = GeneratorSeedDB(*seeds)
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)

    partitioning_result_db = PartitioningResultDB()

    print(f"Number of Seeds: {len(seed_db.seeds)}")
    for seed in seed_db.seeds:
        print(f"Number of nodes in seed: {seed.node_number}")
        print(f"Number of graphs in seed: {len(seed.graphs)}")
        for graph in seed.graphs:
            res = min_variance_partition(graph=graph, partition_size=4, seed=seed)
            partitioning_result_db.append(res)
            print(res)


def test_partitioning_opt():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_soft_domatic_partition(
                    graph=graph, partition_size=partition_size, seed=seed
                )
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_opt")
            seeds[i] = None


def test_partitioning_opt():
    for partition_size in [3, 4]:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            if seed.avg_deg_bound[0] < 4:
                print("================")
                print(f"Skipped AVG DEG BOUND: {seed.avg_deg_bound[0]}")
                print("================")
                seeds[i] = None
                continue
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_soft_domatic_partition(
                    graph=graph, partition_size=partition_size, seed=seed
                )
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning3_opt_var")
            seeds[i] = None


def test_partitioning_opt_wo_bridges():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_bridges"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_soft_domatic_partition(
                    graph=graph, partition_size=partition_size, seed=seed
                )
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_opt_wo_bridges")
            seeds[i] = None


def test_partitioning_max():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = max_soft_domatic_partition(
                    graph=graph, partition_size=partition_size, seed=seed
                )
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_max")
            seeds[i] = None


def test_partitioning_max_wo_bridges():
    for partition_size in [3, 4, 5]:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_bridges"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = opt_soft_domatic_partition(
                    graph=graph, partition_size=partition_size, seed=seed
                )
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_max_wo_bridges")
            seeds[i] = None


def test_partitioning_var():
    partition_sizes = [3, 4, 5]
    for partition_size in partition_sizes:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            # if seed.node_number != 300:
            #     seeds[i] = None
            #     continue
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for i, graph in enumerate(seed.graphs):
                # if i == 0 or seed.avg_deg_bound[0] == 3:
                #     continue
                res = min_variance_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_var")
            seeds[i] = None


def test_partitioning_var_deg3():
    partition_sizes = [3, 4]
    for partition_size in partition_sizes:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            if seed:
                if seed.avg_deg_bound[0] < 4 or seed.node_number not in [200, 300]:
                    seeds[i] = None
                    continue
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for i, graph in enumerate(seed.graphs):
                res = min_variance_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_var_deg3_1")
            seeds[i] = None


def test_partitioning_opt_deg3():
    partition_sizes = [3, 4]
    for partition_size in partition_sizes:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        seeds = list(filter(lambda seed: seed.node_number not in [300], seeds))
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            print(f"Seed: {i}, Avg Deg Bound: {seed.avg_deg_bound[0]}")
            # if seed.avg_deg_bound[0] > 3 or seed.node_number not in [100, 200, 300]:
            #     seeds[i] = None
            #     continue
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for j, graph in enumerate(seed.graphs):
                print(f"Graph: {j}")
                res = opt_soft_domatic_partition(
                    graph=graph, partition_size=partition_size, seed=seed
                )
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_opt_deg3_1")
            seeds[i] = None


def test_partitioning_spread():
    partition_sizes = [3, 4, 5]
    for partition_size in partition_sizes:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            # if seed.node_number != 300:
            #     seeds[i] = None
            #     continue
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                res = min_spread_partition(graph=graph, partition_size=partition_size, seed=seed)
                partitioning_result_db.append(res)
                print(res)
            partitioning_result_db.serialize(path="../test_output/test_partitioning_spread_squared")
            seeds[i] = None


def test_partitioning_spread_advanced():
    partition_sizes = [3, 4, 5]
    for partition_size in partition_sizes:
        seeds = sorted(
            GeneratorSeedDB.deserialize(
                "../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3"
            ).seeds,
            key=lambda seed: seed.node_number,
        )
        print(f"Number of Seeds: {len(seeds)}")
        for i, seed in enumerate(seeds):
            # if seed.node_number != 300:
            #     seeds[i] = None
            #     continue
            print(f"Number of nodes in seed: {seed.node_number}")
            print(f"Number of graphs in seed: {len(seed.graphs)}")
            partitioning_result_db = PartitioningResultDB()
            for graph in seed.graphs:
                partitioning_result_db.append(
                    min_spread_partition(graph=graph, partition_size=partition_size, seed=seed)
                )
                print(partitioning_result_db.results[-1])
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
    seeds: list[GeneratorSeed] = sorted(
        GeneratorSeedDB.deserialize(
            "../test_output/test_seed_generator3/test_UDGGeneratorSeedDB3"
        ).seeds,
        key=lambda seed: seed.node_number,
    )
    for i, seed in enumerate(seeds):
        if seed:
            if seed.node_number not in [20, 40]:
                seeds[i] = None
                continue
        print(f"Number of nodes in seed: {seed.node_number}")
        print(f"Number of graphs in seed: {len(seed.graphs)}")
        partitioning_result_db1 = PartitioningResultDB()
        partitioning_result_db2 = PartitioningResultDB()
        partitioning_result_db3 = PartitioningResultDB()
        partitioning_result_db4 = PartitioningResultDB()
        for j, graph in enumerate(seed.graphs):
            print(f"Graph: {j}")
            graph.graph["node_resources"] = sm_node_resources
            print(
                f"Seed: {i}, Avg Deg Bound: {seed.avg_deg_bound[0]}, Security Mean Count: {len(sm_perf_costs)}"
            )
            partitioning_result_db1.append(
                spread_based_max_resource_utilisation_distribution(
                    graph=graph,
                    seed=seed,
                    # sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                )
            )
            partitioning_result_db2.append(
                spread_resource_based_distribution(
                    graph=graph,
                    seed=seed,
                    # sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                    reward_factor=1.0,
                )
            )
            partitioning_result_db3.append(
                spread_based_configurations_distribution(
                    graph=graph,
                    seed=seed,
                    # sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                )
            )
            partitioning_result_db4.append(
                spread_resource_based_distribution(
                    graph=graph,
                    seed=seed,
                    # sm_node_resources=sm_node_resources,
                    sm_perf_cost=sm_perf_costs,
                    reward_factor=10.0,
                )
            )
            print(f"Result1: {partitioning_result_db1.results[-1]}")
            print(f"Result2: {partitioning_result_db2.results[-1]}")
            print(f"Result3: {partitioning_result_db3.results[-1]}")
            print(f"Result4: {partitioning_result_db4.results[-1]}")
        partitioning_result_db1.serialize(
            path="../test_output/test_partitioning_opt_spread_resource1/1"
        )
        partitioning_result_db2.serialize(
            path="../test_output/test_partitioning_opt_spread_resource1/2"
        )
        partitioning_result_db3.serialize(
            path="../test_output/test_partitioning_opt_spread_resource1/3"
        )
        partitioning_result_db4.serialize(
            path="../test_output/test_partitioning_opt_spread_resource1/4"
        )
        seeds[i] = None


def test_serialize_partitioning_resources():
    results1 = PartitioningResultDB.deserialize(
        "../test_output/test_partitioning_opt_spread_resource/1.1"
    )
    results2 = PartitioningResultDB.deserialize(
        "../test_output/test_partitioning_opt_spread_resource/2.1"
    )
    results3 = PartitioningResultDB.deserialize(
        "../test_output/test_partitioning_opt_spread_resource/3.1"
    )
    results4 = PartitioningResultDB.deserialize(
        "../test_output/test_partitioning_opt_spread_resource/4.1"
    )
    for results in [results1, results2, results3, results4]:
        for result in results.results:
            # result.mean_res = (
            #     (0.6, 0.1, 0.5),
            #     (0.2, 0.6, 0.3),
            #     (0.3, 0.2, 0.1),
            #     (0.7, 0.2, 0.3),
            #     (1.0, 0.5, 0.8),
            #     (0.1, 0.3, 0.4),
            # )
            result.node_res = (1.0, 1.0, 1.0)
    # results = PartitioningResultDB.deserialize("../test_output/test_partitioning_spread")
    print(results1.latex_table_var_spread())
    print(results2.latex_table_var_spread())
    print(results3.latex_table_var_spread())
    print(results4.latex_table_var_spread())


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
    results = PartitioningResultDB.deserialize(
        "../test_output/test_results_opt_max_14_10_23/test_partitioning_max/"
    )
    for data_key in [0, 1, 2]:
        for partition_size in [3, 4, 5]:
            results.plot(
                opt="max",
                data_key=data_key,
                partition_size=partition_size,
                filepath=f"../test_output/test_results_opt_max_14_10_23/data_key_{data_key!s}_partition_size_{partition_size!s}.tex",
            )


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


def test_packings_maximality():
    # Test cases with different dimensions
    test_cases = [
        # 2D means
        {
            "means": (
                (0.3, 0.4),
                (0.5, 0.3),
                (0.2, 0.6),
                (0.4, 0.2),
            ),
            "resources": (1.0, 1.0),
        },
        # 3D means
        {
            "means": (
                (0.3, 0.4, 0.2),
                (0.5, 0.3, 0.4),
                (0.2, 0.6, 0.3),
                (0.4, 0.2, 0.5),
                (0.1, 0.3, 0.4),
            ),
            "resources": (1.0, 1.0, 1.0),
        },
        # 4D means
        {
            "means": (
                (0.3, 0.4, 0.2, 0.1),
                (0.5, 0.3, 0.4, 0.2),
                (0.2, 0.6, 0.3, 0.3),
                (0.4, 0.2, 0.5, 0.4),
                (0.1, 0.3, 0.4, 0.2),
                (0.2, 0.2, 0.3, 0.5),
            ),
            "resources": (1.0, 1.0, 1.0, 1.0),
        },
    ]

    for test_case in test_cases:
        means = test_case["means"]
        resources = test_case["resources"]

        print(f"\nTesting {len(resources)}D case:")
        print(f"Means: {means}")

        packings, _ = _max_packings_matrix(means, resources)
        print(f"Found packings: {packings}")

        for packing in packings:
            remaining_means = set(means) - set(packing)
            current_usage = [sum(p[i] for p in packing) for i in range(len(resources))]
            print(f"\nChecking packing: {packing}")
            print(f"Current resource usage: {current_usage}")

            for mean in remaining_means:
                would_fit = True
                new_usage = current_usage.copy()

                for i, usage in enumerate(mean):
                    new_usage[i] += usage
                    if new_usage[i] > resources[i]:
                        would_fit = False
                        break

                if would_fit:
                    print(f"Could add mean: {mean}")
                    print(f"Resulting usage would be: {new_usage}")
                    pytest.fail(
                        f"Found non-maximal packing in {len(resources)}D case:\n"
                        f"Current packing: {packing}\n"
                        f"Could add mean: {mean}\n"
                        f"Resources: {resources}\n"
                        f"Current usage: {current_usage}\n"
                        f"New usage would be: {new_usage}"
                    )


def test_packings_maximality_edge_cases():
    # Edge cases with means very close to resource limits
    edge_cases = [
        # Case with means almost filling the resources
        {
            "means": (
                (0.99, 0.99),
                (0.02, 0.02),
                (0.98, 0.01),
                (0.01, 0.98),
            ),
            "resources": (1.0, 1.0),
        },
        # Case with very small means
        {
            "means": (
                (0.01, 0.01, 0.01),
                (0.02, 0.02, 0.02),
                (0.03, 0.03, 0.03),
                (0.04, 0.04, 0.04),
                (0.05, 0.05, 0.05),
            ),
            "resources": (1.0, 1.0, 1.0),
        },
    ]

    for test_case in edge_cases:
        means = test_case["means"]
        resources = test_case["resources"]

        packings, _ = _max_packings_matrix(means, resources)

        for packing in packings:
            remaining_means = set(means) - set(packing)

            for mean in remaining_means:
                current_usage = [sum(p[i] for p in packing) for i in range(len(resources))]
                would_fit = all(
                    current_usage[i] + mean[i] <= resources[i] for i in range(len(resources))
                )

                if would_fit:
                    dimension = len(resources)
                    pytest.fail(
                        f"Found non-maximal packing in edge case {dimension}D:\n"
                        f"Current packing: {packing}\n"
                        f"Could add mean: {mean}\n"
                        f"Resources: {resources}\n"
                        f"Current usage: {current_usage}"
                    )


def test_packings_maximality_random():
    import random

    random.seed(42)  # For reproducibility

    # Generate random test cases
    for dimension in [2, 3, 4]:
        for _ in range(5):  # 5 random tests per dimension
            # Generate random means
            means = tuple(
                tuple(random.uniform(0.1, 0.6) for _ in range(dimension))
                for _ in range(dimension + 3)  # number of means increases with dimension
            )
            resources = tuple(1.0 for _ in range(dimension))

            packings, _ = _max_packings_matrix(means, resources)

            for packing in packings:
                remaining_means = set(means) - set(packing)

                for mean in remaining_means:
                    current_usage = [sum(p[i] for p in packing) for i in range(dimension)]
                    would_fit = all(
                        current_usage[i] + mean[i] <= resources[i] for i in range(dimension)
                    )

                    if would_fit:
                        pytest.fail(
                            f"Found non-maximal packing in random {dimension}D case:\n"
                            f"Current packing: {packing}\n"
                            f"Could add mean: {mean}\n"
                            f"Resources: {resources}\n"
                            f"Current usage: {current_usage}"
                        )


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
