import random

from src.utils.logging_config import setup_logging
import logging
import os
import sys
import pytest
import networkx as nx
from typing import Any
from pathlib import Path

from src.partitioning.partitioning2 import (
    opt_soft_domatic_partition,
    max_soft_domatic_partition,
    min_variance_partition,
    min_spread_partition,
    min_spread_squared_partition,
    spread_based_max_resource_utilisation_distribution,
    spread_resource_based_distribution,
    spread_based_configurations_distribution,
    SpreadResourceDistributionConfig,
    DomaticPartitionConfig, VarianceDistributionConfig, _max_packings_matrix,
)
from src.graph_generator.seeds.database import GeneratorSeedDB
from src.graph_generator.seeds.seed import GeneratorSeed
from src.partitioning.result2 import BaseResult
from src.partitioning.result_db import ResultDB, DataKey
from src.graph_generator.graphs.graph_illustrator import draw_graph_with_segmented_nodes


def pytest_sessionstart(session: pytest.Session) -> None:
    """ Fixture to set up logging before any tests are run.

    Args:
        session: The pytest session object.
    """

    setup_logging()


@pytest.fixture(autouse=True)
def logger(caplog: pytest.LogCaptureFixture) -> logging.Logger:
    """ Fixture providing a configured logger for tests.

    Args:
        caplog: The pytest log capture fixture to capture log messages.

    Returns:
        A configured logger instance that logs to stdout.
    """

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    return root


# Constants for spread resource tests
MEAN_COST = [
    (0.6, 0.1, 0.5),
    (0.2, 0.6, 0.3),
    (0.3, 0.2, 0.1),
    (0.7, 0.2, 0.3),
    (1.0, 0.5, 0.8),
    (0.1, 0.3, 0.4),
]
SM_NODE_RESOURCES = (1.0, 1.0, 1.0)

SOLVER_NAMES_OLD = ("test_partitioning_spread_resource1", "test_partitioning_spread_resource2",
                    "test_partitioning_spread_resource3", "test_partitioning_spread_resource4")
SOLVER_NAMES = ("config", "resource1", "max_util", "resource10")

BASE_DIR = Path("../test_output/test_seed_generator4")
SEED_DB_DIR = BASE_DIR / "test_UDGGeneratorSeedDB"
PARTITION_DIR = BASE_DIR  # / "partitioning"

SPREAD_VARIANTS = [
    (SOLVER_NAMES[0], spread_based_configurations_distribution, None, {"means": MEAN_COST}),
    (SOLVER_NAMES[1], spread_resource_based_distribution, 1.0, {"means": MEAN_COST}),
    (SOLVER_NAMES[2], spread_based_max_resource_utilisation_distribution, None, {"means": MEAN_COST}),
    (SOLVER_NAMES[3], spread_resource_based_distribution, 10.0, {"means": MEAN_COST}),
]
SPREAD_NAMES = [name for name, *_ in SPREAD_VARIANTS]


@pytest.fixture(scope="module")
def filtered_seeds():
    seeds: list[GeneratorSeed] = sorted(
        GeneratorSeedDB.deserialize(str(SEED_DB_DIR)).seeds,
        key=lambda seed: seed.node_number)
    return seeds


@pytest.mark.parametrize("variant,solver_fn,reward_factor,config_kwargs", SPREAD_VARIANTS)
def test_partitioning_spread_resource_based(
        filtered_seeds: list[GeneratorSeed],
        variant: str,
        solver_fn: callable,
        reward_factor: float,
        config_kwargs: dict[str, Any],
) -> None:
    """ Test for partitioning with spread resource based distribution.
    
    Args:
        filtered_seeds: List of filtered generator seeds to test on.
        solver_fn: Function implementing the solver algorithm.
        reward_factor: Factor applied to rewards, if applicable.
        config_kwargs: Additional configuration parameters for the solver.
    """

    result_db = ResultDB()
    for seed in filtered_seeds:
        if seed.node_number not in [20, 40]:
            continue
        for graph in seed.graphs:
            graph.graph['node_resources'] = SM_NODE_RESOURCES
            result = solver_fn(
                graph=graph,
                seed=seed,
                reward_factor=reward_factor,
                config=SpreadResourceDistributionConfig(**config_kwargs)
            )
            result_db.append(result)
    result_path = PARTITION_DIR / variant
    result_path.mkdir(parents=True, exist_ok=True)
    result_db.serialize(path=str(result_path))

    # assert result_path.exists()
    # assert (result_dir / "results.db").exists()  # adjust filename as needed


@pytest.fixture(scope="module", params=SOLVER_NAMES_OLD)
def result_db_spread(request: pytest.FixtureRequest) -> ResultDB:
    path = PARTITION_DIR / request.param
    return ResultDB.deserialize(str(path))


@pytest.mark.usefixtures("logger")
def test_serialize_partitioning_resources(result_db_spread: ResultDB) -> None:
    """ Test serialisation of partitioning resources and generation of LaTeX table.

    Args:
        result_db_spread: The ResultDB instance containing the results.
    """

    table = result_db_spread.get_latex_table(
        eval_data=[DataKey.ERRORS, DataKey.INCOMPLETE_NODES, DataKey.SPREAD, DataKey.VARIANCE,
                   DataKey.COMPUTATION_TIME],
        eval_method="mean",
        distinguish_optimality=True,
    )
    logging.getLogger(__name__).debug(table)
    assert table is not None


@pytest.mark.parametrize("data_key", list(DataKey))
def test_plot_all_keys(result_db_spread: ResultDB, tmp_path: Path, data_key: DataKey) -> None:
    """ Test plotting and LaTeX generation for all DataKey metrics.

    Args:
        result_db_spread: The ResultDB instance containing the results.
        tmp_path: Temporary path for saving plots and tables.
        data_key: The DataKey metric to test.
    """

    plot_fp = tmp_path / f"{data_key.name.lower()}_plot.png"
    table_fp = tmp_path / f"{data_key.name.lower()}_table.tex"
    result_db_spread.plot(data_key, partition_size=6, filepath=str(plot_fp))
    assert plot_fp.exists()
    table = result_db_spread.get_latex_table(
        eval_data=[data_key],
        eval_method="mean",
        distinguish_optimality=True,
        filepath=str(table_fp),
    )
    assert table is not None


@pytest.fixture
def mock_graph():
    """Fixture to create a mock graph with nodes and means."""

    return nx.path_graph(5)


@pytest.fixture
def mock_result(mock_graph: nx.Graph) -> callable:
    """ Fixture to create a `BaseResult` instance with mock data. """

    def _make(solver_fn):
        """ Helper function to create a `BaseResult` using a solver function. """
        config = VarianceDistributionConfig(partition_size=3, mean_count_per_node=1)
        result = solver_fn(graph=mock_graph, seed=None, config=config)

        assert isinstance(result, BaseResult)
        assert result.calculate_variance() >= 0

        return result

    return _make


@pytest.mark.parametrize("method_name, expected", [
    ("calculate_incomplete_nodes", 2),
    ("calculate_errors", 2),
    ("calculate_variance", 0.08888888888888889),
    ("calculate_spread", 0.4),
])
def test_base_result_metrics(mock_result: callable, method_name: str, expected: Any) -> None:
    result = mock_result(min_spread_partition)
    actual = getattr(result, method_name)()

    assert actual == expected, f"{method_name} returned {actual}, expected {expected}"


def test_save_graphs_per_degree() -> None:
    """ Create subdirectories for each degree and save graphs with specific mean assignments.

    Reads results from the ResultDB, iterates over all results, creates a subdirectory for each degree bound, and saves the corresponding graphs with mean assignments.
    """

    # Deserialise the result database
    result_db = ResultDB.deserialize("../test_output/test_seed_generator4/test_partitioning_spread_resource2")

    # Base directory for saving graphs
    base_dir = "../test_output/test_seed_generator4/graphs_per_degree"
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base directory exists

    # Iterate over all results in the database
    for result in result_db.results:
        degree_bound = result.seed.avg_deg_bound[0]  # Get the degree bound for the current result
        graph = result.graph  # Get the graph object from the result

        # Create subdirectory for the current degree bound
        degree_dir = os.path.join(base_dir, f"deg_{degree_bound}")
        os.makedirs(degree_dir, exist_ok=True)

        # Generate a filename for the graph
        graph_filename = os.path.join(degree_dir, f"graph_{id(result.graph)}.png")

        # Use the graph illustrator to save the graph with mean assignments
        draw_graph_with_segmented_nodes(
            graph=graph,
            mean_types=result.partition_size,
            save_path=graph_filename
        )
        print(f"Saved graph for degree {degree_bound} to {graph_filename}")


@pytest.fixture(scope="module")
def simple_graph() -> nx.Graph:
    """ Tiny graph solved in a few milliseconds irrespective of the solver.

    Using a fixture keeps set-up cost minimal when the test matrix is extended. The nodes are placed on the diagonal so that the drawing code can read the (x, y) coordinates from the `pos` attribute.

    Returns:
        A simple path graph with 3 nodes, positioned diagonally.

    """

    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])  # path with 3 vertices
    pos = {
        0: (0.0, 0.0),
        1: (0.5, 0.5),
        2: (1.0, 1.0)
    }
    nx.set_node_attributes(graph, pos, "pos")

    return graph


@pytest.mark.parametrize("solver_fn", [opt_soft_domatic_partition, max_soft_domatic_partition])
def test_partitioning_domatic_based(simple_graph: nx.Graph, solver_fn: callable):
    """ Unified test covering both optimal and maximal n-soft domatic partition MILPs.

    The assertions are deliberately lightweight so the test suits CI runs:
    – the solver returns a result object derived from `BaseResult`
    – the variance information is present and well-formed

    Args:
        simple_graph: A simple graph fixture for testing.
        solver_fn: The solver function to test, which should return a `BaseResult`.

    Raises:
        AssertionError: If the result is not a `BaseResult` or if the variance is negative.
    """

    config = DomaticPartitionConfig(partition_size=3, mean_count_per_node=2)
    result = solver_fn(graph=simple_graph, seed=None, config=config)

    assert isinstance(result, BaseResult), f"{solver_fn.__name__} did not return a BaseResult"
    assert result.calculate_variance() >= 0, "negative variance detected"

    draw_graph_with_segmented_nodes(result.graph, mean_types=result.partition_size, save_path=None)


@pytest.mark.parametrize(
    "solver_fn",
    [min_spread_squared_partition, min_spread_partition, min_variance_partition]
)
def test_variance_distribution(simple_graph: nx.Graph, solver_fn: callable):
    """ Unified test covering both ‘spread’ and ‘squared spread’ MILPs.

    The assertions are deliberately lightweight so the test suits CI runs:
    – the solver returns a result object derived from `BaseResult`
    – the variance information is present and well-formed

    Args:
        simple_graph: A simple graph fixture for testing.
        solver_fn: The solver function to test, which should return a `BaseResult`.

    Raises:
        AssertionError: If the result is not a `BaseResult` or if the variance is negative.
    """

    config = VarianceDistributionConfig(partition_size=3, mean_count_per_node=2)
    result = solver_fn(graph=simple_graph, seed=None, config=config)

    # 1. Correct polymorphic type
    assert isinstance(result, BaseResult), f"{solver_fn.__name__} did not return a BaseResult"

    assert result.calculate_variance() >= 0, "negative variance detected"


@pytest.mark.parametrize(
    "solver_fn",
    [
        spread_based_max_resource_utilisation_distribution,
        spread_resource_based_distribution,
        spread_based_configurations_distribution,
    ],
)
def test_distribution_variance_and_type(simple_graph: nx.Graph, solver_fn: callable):
    """ Parametrised test for distribution-based solvers.

    Verifies that:
    – the solver returns an instance of `BaseResult`
    – the variance is non-negative

    Args:
        simple_graph: A simple graph fixture for testing.
        solver_fn: The solver function to test, which should return a `BaseResult`.

    Raises:
        AssertionError: If the result is not a `BaseResult` or if the variance is negative.
    """

    config = SpreadResourceDistributionConfig(means=((1.0,), (0.8,)))
    result = solver_fn(graph=simple_graph, seed=None, config=config)

    assert isinstance(result, BaseResult), f"{solver_fn.__name__} did not return a BaseResult"
    assert result.calculate_variance() >= 0, "negative variance detected"


STATIC_CASES = [
    ({(0.3, 0.4), (0.5, 0.3), (0.2, 0.6), (0.4, 0.2)}, (1.0, 1.0)),
    ({(0.3, 0.4, 0.2), (0.5, 0.3, 0.4), (0.2, 0.6, 0.3), (0.4, 0.2, 0.5), (0.1, 0.3, 0.4)}, (1.0, 1.0, 1.0)),
    ({(0.3, 0.4, 0.2, 0.1), (0.5, 0.3, 0.4, 0.2), (0.2, 0.6, 0.3, 0.3), (0.4, 0.2, 0.5, 0.4), (0.1, 0.3, 0.4, 0.2),
      (0.2, 0.2, 0.3, 0.5)}, (1.0, 1.0, 1.0, 1.0))
]
EDGE_CASES = [
    ({(0.99, 0.99), (0.02, 0.02), (0.98, 0.01), (0.01, 0.98)}, (1.0, 1.0)),
    ({(0.01, 0.01, 0.01), (0.02, 0.02, 0.02), (0.03, 0.03, 0.03), (0.04, 0.04, 0.04), (0.05, 0.05, 0.05)},
     (1.0, 1.0, 1.0)),
]


@pytest.mark.parametrize("means,resources", STATIC_CASES + EDGE_CASES)
def test_packings_maximality_all(means, resources):
    """ Test to ensure that the packings found by _max_packings_matrix are maximal."""

    packings, _ = _max_packings_matrix(tuple(means), resources)
    for packing in packings:
        used = [sum(mean[resource] for mean in packing) for resource in range(len(resources))]
        for mean in set(means) - set(packing):
            assert not all(used[res] + mean[res] <= resources[res] for res in range(len(resources))), \
                f"Non-maximal packing {packing} could fit {mean}" \
                f"Non-maximal packing {packing} could fit {mean}"


def test_packings_maximality_random():
    """ Test to ensure that the packings found by _max_packings_matrix are maximal, using random test cases. """

    random.seed(42)
    for dim in [2, 3, 4]:
        for _ in range(5):
            means = tuple(
                tuple(random.uniform(0.1, 0.6) for _ in range(dim))
                for _ in range(dim + 3)
            )
            resources = (1.0,) * dim
            packings, _ = _max_packings_matrix(means, resources)
            for packing in packings:
                used = [sum(mean[resource] for mean in packing) for resource in range(dim)]
                for mean in set(means) - set(packing):
                    assert not all(used[res] + mean[res] <= resources[res] for res in range(dim)), \
                        f"Non-maximal random packing {packing} could fit {mean}"
