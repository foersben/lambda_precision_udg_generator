import numpy as np

from src.graph_generation.generator_seeds import UDGSeedGenerator, UDGGeneratorSeedDB


def test_seed_generator():
    seeds = UDGSeedGenerator(sample_size=10).generate_seeds(node_numbers=[i for i in range(280, 320, 20)],
                                                            coverage_bound=[0.85, 0.9], avg_degs=[3, 4, 5, 6])
    db = UDGGeneratorSeedDB(*seeds)
    db.serialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_graphs")
    for seed in db.seeds:
        seed.generate_graphs(20, bounds=True, connected=True, new=True)
        for i, graph in enumerate(seed.graphs):
            custom = f"deg_{seed.avg_deg_bound[0]}_{seed.avg_deg_bound[1]}_num_{i}"
            print("Initialise drawing:")
            graph.draw_random_geometric_graph(filepath="../test_output/test_seed_generator2", custom=custom)
    print(db.latex_table())
    db.serialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
    db.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
    print(db.latex_table())


def test_seed_wo_bridges():
    db = UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
    for seed in db.seeds:
        for graph in seed.graphs:
            graph.augment_bridges_knn()
            graph.reduce_avg_degree(target_avg_deg=seed.avg_deg_bound[0], bridges=True)
    db.serialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_bridges")


def test_eval_seeds():
    db = UDGGeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
    print(db.latex_table())
    print("Seed lengths:")
    for seed in db.seeds:
        print(len(seed.graphs))


def test_seed_generation():
    seeds = UDGSeedGenerator(sample_size=10).generate_seeds(avg_degs=[3],
                                                            node_numbers=[100, 200])
    db = UDGGeneratorSeedDB(*seeds)
    list(map(lambda seed: seed.generate_graphs(3), seeds))
    db.serialize("../test_output/test_UDGGeneratorSeedDB")
    db.deserialize("../test_output/test_UDGGeneratorSeedDB")
    print(seeds.latex_table())


def test_seed_correctness():
    seeds = UDGSeedGenerator(sample_size=10).generate_seeds(avg_degs=[3, 4],
                                                            node_numbers=[100, 200])
    db = UDGGeneratorSeedDB(*seeds)
    for seed in db.seeds:
        seed.generate_graphs(10, connected=False, bounds=True)
        print(f"Seed: {str(seed.avg_deg_bound)} Avg Sample Degree: {seed.get_avg_degree()}")
        print(f"Seed: {str(seed.coverage_bound)} Avg Sample Coverage: {seed.get_avg_coverage()}")
        for graph in seed.graphs:
            graph.draw_random_geometric_graph(filepath="../test_output")
        assert seed.avg_deg_bound[0] - 0.125 <= seed.get_avg_degree() <= seed.avg_deg_bound[1]
        assert seed.coverage_bound[0] <= seed.get_avg_coverage() <= seed.coverage_bound[1]


def test_coverage_bound_results():
    coverage_range = np.arange(0.85, 0.975, 0.025).round(3)
    coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))

    for coverage_bound in coverage_bounds:
        seeds = UDGSeedGenerator(sample_size=40).generate_seeds(avg_degs=[4, 5], node_numbers=range(20, 320, 20),
                                                                coverage_bound=coverage_bound, padding=False)
        db = UDGGeneratorSeedDB(*seeds)
        db.serialize(f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bound[0]}_{coverage_bound[1]}")
        print(db.latex_table_coverage())
        db.mean_var_local_clustering(filepath="../test_output/test_coverage_bound_results/plots/var_local_clust")
        db.mean_var_deg_distribution(filepath="../test_output/test_coverage_bound_results/plots/var_deg_dist")


def test_eval_table():
    coverage_range = np.arange(0.85, 0.99, 0.025).round(3)
    coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))
    coverage_bounds[-1] = (coverage_bounds[-1][0], coverage_bounds[-1][1] - 0.001)

    db = UDGGeneratorSeedDB.deserialize(
        f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bounds[0][0]}_{coverage_bounds[0][1]}")
    for coverage_bound in coverage_bounds[1:]:
        db_seeds = UDGGeneratorSeedDB.deserialize(
            f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bound[0]}_{coverage_bound[1]}")
        db.append(*db_seeds.seeds)
    print(db.latex_table_coverage())
    db.mean_var_local_clustering(filepath="../test_output/test_coverage_bound_results/plots/var_local_clust")
    db.mean_var_deg_distribution(filepath="../test_output/test_coverage_bound_results/plots/var_deg_dist")


def test_eval_table_seeds():
    db = UDGGeneratorSeedDB.deserialize(f"../test_output/table_seed_values")
    print(db.latex_table())


def test_gen_images():
    coverage_range = np.arange(0.85, 0.99, 0.025).round(3)
    coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))
    coverage_bounds[-1] = (coverage_bounds[-1][0], coverage_bounds[-1][1] - 0.001)

    filepaths = [
        "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[0][0]}_{coverage_bounds[0][1]}",
        "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[2][0]}_{coverage_bounds[2][1]}",
        "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[3][0]}_{coverage_bounds[3][1]}",
        "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[5][0]}_{coverage_bounds[5][1]}"
    ]
    for filepath in filepaths:
        db = UDGGeneratorSeedDB.deserialize(filepath)
        for seed in db.seeds:
            seed.generate_graphs(5, bounds=True, connected=True, new=True)
            seed.graphs[3].draw_random_geometric_graph(filepath=filepath, custom=f"{seed.avg_deg_bound[0]}_avg_deg")
    # db = UDGGeneratorSeedDB.deserialize(
    #     f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bounds[2][0]}_{coverage_bounds[2][1]}")
    # db = UDGGeneratorSeedDB.deserialize(
    #     f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bounds[3][0]}_{coverage_bounds[3][1]}")
    # db = UDGGeneratorSeedDB.deserialize(
    #     f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bounds[5][0]}_{coverage_bounds[5][1]}")


def test_cmp_miss_cov_inc_nodes():
    pass
