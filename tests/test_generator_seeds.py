from pathlib import Path

from src.utils.logging_config import setup_logging

setup_logging()

import pytest
import numpy as np
import logging

from src.graph_generator.seeds.database import GeneratorSeedDB
from src.graph_generator.seeds.generator import SeedGenerator

BASE_DIR = Path("../test_output/test_generator_seeds")
SEED_DB_DIR = BASE_DIR / "seeds"
GRAPH_OUTPUT_DIR = BASE_DIR / "graphs"
LATEX_OUTPUT_DIR = BASE_DIR / "latex_tables"
PATH_CONSTANTS = (SEED_DB_DIR, GRAPH_OUTPUT_DIR, LATEX_OUTPUT_DIR)  # Group paths for convenience


def create_output_dirs():
    """ Ensures that all necessary test output directories exist. """

    for path in PATH_CONSTANTS:
        path.mkdir(parents=True, exist_ok=True)


# Ensure paths are created before tests run
create_output_dirs()


@pytest.fixture(scope='function', autouse=True)
def logger(caplog: pytest.LogCaptureFixture) -> logging.Logger:
    """ Fixture providing a configured logger for tests.

    Args:
        caplog: The pytest log capture fixture to capture log messages.

    Returns:
        A configured logger instance that logs at the DEBUG level.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(handler)
    return root


@pytest.fixture(scope="module")
def sample_seeds() -> GeneratorSeedDB:
    """ Fixture to generate a small sample of seeds for testing.

    Returns:
        GeneratorSeedDB: A database containing generated seeds.
    """

    seeds = SeedGenerator(sample_size=10).generate_seeds(
        node_numbers=[20, 40],
        coverage_bound=(0.85, 0.9),
        avg_degs=[3, 4, 5]
    )
    return GeneratorSeedDB(*seeds)


@pytest.mark.usefixtures("logger")
@pytest.mark.parametrize("node_numbers, avg_degs", [
    ([20, 40, 60], [3, 4, 5]),
    ([80, 100], [6, 7]),
])
def test_seed_generation(logger: logging.Logger, node_numbers: list[int], avg_degs: list[int]) -> None:
    """ Test seed generation for different configurations of node numbers and average degrees.

    Args:
        node_numbers: The range of node numbers to test against.
        avg_degs: The average degrees range to test with.
    """

    seeds = SeedGenerator(sample_size=10).generate_seeds(
        node_numbers=node_numbers,
        coverage_bound=(0.85, 0.9),
        avg_degs=avg_degs
    )
    db = GeneratorSeedDB(*seeds)
    logger.info(db.latex_table())
    assert len(db.seeds) == len(node_numbers) * len(avg_degs)


@pytest.mark.usefixtures("logger")
def test_serialization_and_deserialization(logger: logging.Logger, sample_seeds: GeneratorSeedDB) -> None:
    """ Test the serialisation and deserialization of a seed database.

    Args:
        sample_seeds: Fixture providing a sample seed database.
    """

    db_filepath = SEED_DB_DIR / "test_seeds.db"

    # Serialize and Deserialize
    sample_seeds.serialize(str(db_filepath))
    deserialized_db = GeneratorSeedDB.deserialize(str(db_filepath))

    logger.info(deserialized_db.latex_table())
    # Compare seed count
    assert len(sample_seeds.seeds) == len(deserialized_db.seeds)


@pytest.mark.usefixtures("logger")
@pytest.mark.parametrize("coverage_bound", [
    (0.85, 0.875),
    (0.9, 0.925),
])
def test_seed_coverage_bounds(logger: logging.Logger, coverage_bound: tuple[float, float]) -> None:
    """ Test seed generation with varying coverage bounds.

    Args:
        coverage_bound: Tuple representing the lower and upper bounds for coverage.
    """

    seeds = SeedGenerator(sample_size=5).generate_seeds(
        node_numbers=[50],
        coverage_bound=coverage_bound,
        avg_degs=[3]
    )
    db = GeneratorSeedDB(*seeds)
    logger.debug(db.latex_table())  # Log coverage details
    assert all(seed.coverage_bound == coverage_bound for seed in db.seeds)


def test_database_operations(sample_seeds: GeneratorSeedDB) -> None:
    """ Test serialisation and graph generation for seeds in the sample database.

    Args:
        sample_seeds: Fixture providing a sample seed database.
    """

    db_filepath = SEED_DB_DIR / "test_seeds_operations.db"

    # Verify serialization is successful
    sample_seeds.serialize(str(db_filepath))
    assert db_filepath.is_file()

    # Perform additional operations
    for seed in sample_seeds.seeds:
        seed.generate_graphs(connected=True)
        for graph in seed.graphs:
            assert len(graph.nodes) == seed.node_number


@pytest.mark.parametrize("target_avg_deg", [3, 4, 5])
def test_graph_degree_reduction(sample_seeds: GeneratorSeedDB, target_avg_deg) -> None:
    """ Test graph degree reduction while preserving bridges.

    Args:
        sample_seeds: Fixture providing a sample seed database.
        target_avg_deg: The target average degree for reduction.
    """

    for seed in sample_seeds.seeds:
        for graph in seed.graphs:
            graph.reduce_avg_degree(target_avg_deg, preserve_bridges=True)
            avg_degree = np.mean([deg for _, deg in graph.degree])
            assert avg_degree <= target_avg_deg


@pytest.mark.parametrize("coverage_bound", [
    (0.85, 0.875),
    (0.90, 0.925),
])
@pytest.mark.parametrize("node_numbers", [
    [20, 40, 60],
    [80, 100],
])
def test_combined_seed_generation_and_latex_table(coverage_bound: tuple[float, float], node_numbers: list[int],
                                                  logger: logging.Logger) -> None:
    """ Test combined functionality of seed generation and LaTeX table creation.

    Args:
        coverage_bound: The coverage boundaries for the seeds.
        node_numbers: The list of node numbers for which seeds are generated.
    """

    seeds = SeedGenerator(sample_size=5).generate_seeds(
        node_numbers=node_numbers,
        coverage_bound=coverage_bound,
        avg_degs=[3, 4]
    )
    db = GeneratorSeedDB(*seeds)

    # Verify LaTeX table creation
    latex_table_path = LATEX_OUTPUT_DIR / f"latex_table_{coverage_bound[0]}_{coverage_bound[1]}.tex"
    latex_table = db.latex_table(filepath=str(latex_table_path))
    logger.info(f"LaTeX table saved to {latex_table_path}:\n{latex_table}")
    assert latex_table_path.is_file()


def test_seed_generator(logger: logging.Logger):
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

    initial_db_path = SEED_DB_DIR / "test_UDGGeneratorSeedDB_wo_graphs.db"
    db.serialize(str(initial_db_path))
    assert initial_db_path.exists(), "Initial seed database was not serialised."

    for seed in db.seeds:
        seed.generate_graphs(20, bounds=True, connected=True, new=True)  # Generate graphs

    modified_db_path = SEED_DB_DIR / "test_UDGGeneratorSeedDB_with_graphs.db"
    db.serialize(str(modified_db_path))
    assert modified_db_path.exists(), "Modified seed database was not serialized."

    deserialized_db = GeneratorSeedDB.deserialize(str(modified_db_path))

    logger.info("Verifying LaTeX table representation of the database...")
    latex_table = deserialized_db.latex_table()
    assert latex_table is not None, "LaTeX table generation failed."
    logger.info("\n" + latex_table)

# def test_seed_wo_bridges():
#     """ Executes a test for generating seeds without bridges by modifying graph structures in a serialised database and # preserving their average degree.
#
#     The function deserialises a database of generator seeds from a specified path. For each seed in the database, it iterates # through its associated graphs, augments their structure with bridges using the k-nearest neighbours algorithm, and then # reduces the average degree of the graphs while ensuring the bridges are preserved. Finally, the modified database is serialised#  to a new location.
#     """
#
#     db = GeneratorSeedDB.deserialize(str(SEED_DB_DIR / "seeds_without_bridges.db"))
#     for seed in db.seeds:
#         for graph in seed.graphs:
#             graph.augment_bridges_knn()
#             graph.reduce_avg_degree(target_avg_deg=seed.avg_deg_bound[0], preserve_bridges=True)
#     db = GeneratorSeedDB.deserialize(str(SEED_DB_DIR / "seeds_without_bridges_reduced_avg_deg.db"))


# def test_eval_seeds():
#     db = GeneratorSeedDB.deserialize("../test_output/test_seed_generator2/test_UDGGeneratorSeedDB")
#     print(db.latex_table())
#     print("Seed lengths:")
#     for seed in db.seeds:
#         print(len(seed.graphs))


# def test_seed_correctness():
#     seeds = SeedGenerator(sample_size=10).generate_seeds(avg_degs=[3, 4],
#                                                          node_numbers=[100, 200])
#     db = GeneratorSeedDB(*seeds)
#     for seed in db.seeds:
#         seed.generate_graphs(10, connected=False, bounds=True)
#         print(f"Seed: {str(seed.avg_deg_bound)} Avg Sample Degree: {seed.get_avg_degree()}")
#         print(f"Seed: {str(seed.coverage_bound)} Avg Sample Coverage: {seed.get_avg_coverage()}")
#         for graph in seed.graphs:
#             graph.draw_random_geometric_graph(filepath="../test_output")
#         assert seed.avg_deg_bound[0] - 0.125 <= seed.get_avg_degree() <= seed.avg_deg_bound[1]
#         assert seed.coverage_bound[0] <= seed.get_avg_coverage() <= seed.coverage_bound[1]
#
#
# def test_coverage_bound_results():
#     coverage_range = np.arange(0.85, 0.975, 0.025).round(3)
#     coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))
#
#     for coverage_bound in coverage_bounds:
#         seeds = SeedGenerator(sample_size=40).generate_seeds(avg_degs=[4, 5], node_numbers=list(range(220, 320, 20)),
#                                                              coverage_bound=coverage_bound, padding=False)
#         db = GeneratorSeedDB(*seeds)
#         db.serialize(f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bound[0]}_{coverage_bound[1]}")
#         print(db.latex_table_coverage())
#         db.mean_var_local_clustering(filepath="../test_output/test_coverage_bound_results/plots/var_local_clust")
#         db.mean_var_deg_distribution(filepath="../test_output/test_coverage_bound_results/plots/var_deg_dist")
#
#
# def test_eval_table():
#     coverage_range = np.arange(0.85, 0.99, 0.025).round(3)
#     coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))
#     coverage_bounds[-1] = (coverage_bounds[-1][0], coverage_bounds[-1][1] - 0.001)
#
#     db = GeneratorSeedDB.deserialize(
#         f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bounds[0][0]}_{coverage_bounds[0][1]}")
#     for coverage_bound in coverage_bounds[1:]:
#         db_seeds = GeneratorSeedDB.deserialize(
#             f"../test_output/test_coverage_bound_results/UDGGenSeedDB_{coverage_bound[0]}_{coverage_bound[1]}")
#         db.append(*db_seeds.seeds)
#     print(db.latex_table_coverage())
#     db.mean_var_local_clustering(filepath="../test_output/test_coverage_bound_results/plots/var_local_clust")
#     db.mean_var_deg_distribution(filepath="../test_output/test_coverage_bound_results/plots/var_deg_dist")
#
#
# def test_eval_table_seeds():
#     db = GeneratorSeedDB.deserialize(f"../test_output/table_seed_values")
#     print(db.latex_table())
#
#
# def test_gen_images():
#     coverage_range = np.arange(0.85, 0.99, 0.025).round(3)
#     coverage_bounds = list(zip(coverage_range, (coverage_range + 0.025).round(3)))
#     coverage_bounds[-1] = (coverage_bounds[-1][0], coverage_bounds[-1][1] - 0.001)
#
#     filepaths = [
#         "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[0][0]}_{coverage_bounds[0][1]}",
#         "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[2][0]}_{coverage_bounds[2][1]}",
#         "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[3][0]}_{coverage_bounds[3][1]}",
#         "../test_output/test_coverage_bound_results/"f"UDGGenSeedDB_{coverage_bounds[5][0]}_{coverage_bounds[5][1]}"
#     ]
#     for filepath in filepaths:
#         db = GeneratorSeedDB.deserialize(filepath)
#         for seed in db.seeds:
#             seed.generate_graphs(5, bounds=True, connected=True, new=True)
#             seed.graphs[3].draw_random_geometric_graph(filepath=filepath, custom=f"{seed.avg_deg_bound[0]}_avg_deg")
#
