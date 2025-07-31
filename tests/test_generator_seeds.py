from src.utils.logging_config import setup_logging

# Initialize logging BEFORE any other imports
setup_logging()

import pytest
import numpy as np
import logging
import sys

from src.graph_generator.seeds.database import GeneratorSeedDB
from src.graph_generator.seeds.generator import SeedGenerator
from src.graph_generator.points.generator import RandomPointsGenerator


# root = logging.getLogger()
# root.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
# root.handlers.clear()  # remove the handler pytest/PyCharm adds
# root.addHandler(handler)
#
# # If your modules create their own loggers, make sure they propagate
# logging.getLogger("RandomPointsGenerator").propagate = True
# logging.getLogger("LambdaPrecisionPoints").propagate = True
# logging.getLogger("LambdaPrecisionUDGGenerator").propagate = True
# logging.getLogger("LambdaPrecisionUDG").propagate = True
# logging.getLogger("SeedGenerator").propagate = True
# logging.getLogger("GeneratorSeed").propagate = True
# logging.getLogger("GeneratorSeedDB").propagate = True

@pytest.fixture
def test_logger():
    """Fixture providing a configured logger for tests"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.WARNING)

    logger.handlers.clear()  # remove the handler pytest/PyCharm adds

    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

    return logger


def test_seed_generator(test_logger: logging.Logger):
    """ Generates seeds with the following properties:
        - Sample size: 10
        - Node numbers: [20, 40, ..., 300]
        - Coverage bounds: [0.85, 0.875] (taking into account the padding)
        - Average degrees: [3, 4, 5, 6]

    The generated seeds are then serialised to a specified directory, and graphs are generated for each seed. The results are saved to:
        - "../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_graphs"

    The function performs the following steps:
    1. Creates a SeedGenerator instance with a specified sample size and generates seeds using specified node numbers, coverage bounds, and average degrees.
    2. Constructs a GeneratorSeedDB using the generated seeds and serialises it to a specified directory.
    3. Iterates over the seeds, generates graphs for each, and saves the visual representation of the generated graphs to specified file paths.
    4. Prints the LaTeX table representation of the seeds in the seed database.
    5. Serialises the modified database to a new directory and subsequently deserialises it.
    6. Prints the LaTeX table representation again to verify the integrity of deserialisation.
    """

    seeds = SeedGenerator(sample_size=20).generate_seeds(
        node_numbers=[i for i in range(20, 120, 20)],
        coverage_bound=(0.85, 0.9),
        avg_degs=[3, 4, 5, 6]
    )
    db = GeneratorSeedDB(*seeds)
    db.serialize("../test_output/test_seed_generator4/test_UDGGeneratorSeedDB_wo_graphs")
    for seed in db.seeds:
        seed.generate_graphs(20, bounds=True, connected=True, new=True)
        # for i, graph in enumerate(seed.graphs):
        #     root.info("Initialise drawing:")
        #     graph.draw_random_geometric_graph(
        #         filepath="../test_output/test_seed_generator4/test_UDGGeneratorSeedDB",
        #         custom=f"deg_{seed.avg_deg_bound[0]}_{seed.avg_deg_bound[1]}_num_{i}"
        #     )
    # db.latex_table()
    db.serialize("../test_output/test_seed_generator4/test_UDGGeneratorSeedDB")
    db.deserialize("../test_output/test_seed_generator4/test_UDGGeneratorSeedDB")
    test_logger.info(db.latex_table())


def test_seed_wo_bridges():
    """ Executes a test for generating seeds without bridges by modifying graph structures in a serialised database and preserving their average degree.

    The function deserialises a database of generator seeds from a specified path. For each seed in the database, it iterates through its associated graphs, augments their structure with bridges using the k-nearest neighbours algorithm, and then reduces the average degree of the graphs while ensuring the bridges are preserved. Finally, the modified database is serialised to a new location.
    """

    db = GeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
    for seed in db.seeds:
        for graph in seed.graphs:
            graph.augment_bridges_knn()
            graph.reduce_avg_degree(target_avg_deg=seed.avg_deg_bound[0], preserve_bridges=True)
    db.serialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB_wo_bridges")


def test_eval_seeds():
    db = GeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
    print(db.latex_table())
    print("Seed lengths:")
    for seed in db.seeds:
        print(len(seed.graphs))


def test_seed_generation():
    logger = logging.getLogger("UDGGeneratorSeedDB")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    seeds = SeedGenerator(sample_size=20).generate_seeds(avg_degs=[3],
                                                         node_numbers=[100, 200])
    db = GeneratorSeedDB(*seeds)
    list(map(lambda seed: seed.generate_graphs(3), seeds))
    # db.serialize("../test_output/test_UDGGeneratorSeedDB")
    # db.deserialize("../test_output/test_UDGGeneratorSeedDB")
    print(db.latex_table())


def test_seed_correctness():
    seeds = SeedGenerator(sample_size=10).generate_seeds(avg_degs=[3, 4],
                                                         node_numbers=[100, 200])
    db = GeneratorSeedDB(*seeds)
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
        seeds = SeedGenerator(sample_size=40).generate_seeds(avg_degs=[4, 5], node_numbers=list(range(220, 320, 20)),
                                                             coverage_bound=coverage_bound, padding=False)
        db = GeneratorSeedDB(*seeds)
        db.serialize(f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bound[0]}_{coverage_bound[1]}")
        print(db.latex_table_coverage())
        db.mean_var_local_clustering(filepath="../test_output/test_coverage_bound_results/plots/var_local_clust")
        db.mean_var_deg_distribution(filepath="../test_output/test_coverage_bound_results/plots/var_deg_dist")


def test_eval_table():
    coverage_range = np.arange(0.85, 0.99, 0.025).round(3)
    coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))
    coverage_bounds[-1] = (coverage_bounds[-1][0], coverage_bounds[-1][1] - 0.001)

    db = GeneratorSeedDB.deserialize(
        f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bounds[0][0]}_{coverage_bounds[0][1]}")
    for coverage_bound in coverage_bounds[1:]:
        db_seeds = GeneratorSeedDB.deserialize(
            f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bound[0]}_{coverage_bound[1]}")
        db.append(*db_seeds.seeds)
    print(db.latex_table_coverage())
    db.mean_var_local_clustering(filepath="../test_output/test_coverage_bound_results/plots/var_local_clust")
    db.mean_var_deg_distribution(filepath="../test_output/test_coverage_bound_results/plots/var_deg_dist")


def test_eval_table_seeds():
    db = GeneratorSeedDB.deserialize(f"../test_output/table_seed_values")
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
        db = GeneratorSeedDB.deserialize(filepath)
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
