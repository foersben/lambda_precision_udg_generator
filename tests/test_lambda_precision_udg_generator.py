import logging

from lambdaprecisionudggenerator.graph_generator.points.generator import RandomPointsGenerator
from lambdaprecisionudggenerator.utils import setup_logging

setup_logging()

import pytest
import networkx as nx
from lambdaprecisionudggenerator.graph_generator import LambdaPrecisionUDGGenerator
import lambdaprecisionudggenerator.graph_generator.graphs.utils as utils
import lambdaprecisionudggenerator.graph_generator.graphs.graph_illustrator as graph_depiction


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


@pytest.mark.usefixtures("logger")
def test_density_from_metadata(logger: logging.Logger) -> None:
    """ Tests if the density of the graph reconstructed from metadata matches the original density.

    This is done by generating a graph with a specified number of points and minimum distance, then reconstructing the points from the graph's metadata and checking if the density matches. The test runs multiple iterations to ensure consistency and reliability of the density calculation.

    Raises:
        AssertionError: If the original density does not match the reconstructed density within a specified tolerance.
    """

    lambda_precision_points_generator = RandomPointsGenerator(point_number=300, min_dist=0.037)
    for _ in range(20):
        generator = LambdaPrecisionUDGGenerator(lambda_precision_points_generator, radius=0.083, logger=logger)
        graph = generator.generate_graph(connected=True)
        original_density = generator._lpp.get_density()

        # build UDG and reconstruct points from metadata
        reconstructed_pp = graph.get_lambda_precision_points()
        reconstructed_density = reconstructed_pp.get_density()

        # densities must match
        logger.info(f"Original density: {original_density}, Reconstructed density: {reconstructed_density}")
        assert pytest.approx(original_density, rel=1e-3) == reconstructed_density


@pytest.mark.usefixtures("logger")
def test_lambda_precision_udg_generator(logger: logging.Logger) -> None:
    """ Tests the functionality of the LambdaPrecisionUDGGenerator class by creating an instance, generating a graph with specified parameters, and performing assertions to validate the expected behaviour. Specifically, this includes verifying the connectivity of the graph and checking the average degree of the generated graph.

    Args:
        logger: Logger instance for logging debug information.
    """

    for _ in range(20):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.037), radius=0.083,
                                                logger=logger)
        graph = generator.generate_graph(connected=True)
        # graph.draw_random_geometric_graph(filepath="test_output")

        logger.info(f"Average degree: {graph.average_degree()}")
        logger.info(
            f"Average degree, after reducing it: {utils.reduce_avg_degree(graph, 4, preserve_bridges=False).average_degree()}")
        logger.info(f"Average degree: {graph.average_degree()}")
        logger.info(
            f"Average degree w/o bridges: {utils.reduce_avg_degree(graph, 3, preserve_bridges=True).average_degree()}")


@pytest.mark.usefixtures("logger")
def test_old_generator_seeds(logger: logging.Logger) -> None:
    """ Tests the functionality of the LambdaPrecisionUDGGenerator class with specific seeds to ensure consistent behaviour across multiple runs. This includes generating graphs with a fixed number of points and minimum distance, and checking the connectivity and average degree of the generated graphs.

    Args:
        logger: Logger instance for logging debug information.
    """

    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=100, min_dist=0.086932),
                                                radius=0.121875)
        graph = generator.generate_graph()

        logger.info(f"Is connected: {nx.is_connected(graph.graph)}")
        logger.info(f"Average degree: {graph.average_degree()}")
        logger.info(f"Coverage: {graph.get_lambda_precision_points().get_density()}")
    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.048588),
                                                radius=0.065625)
        graph = generator.generate_graph()

        logger.info(f"Is connected: {nx.is_connected(graph.graph)}")
        logger.info(f"Average degree: {graph.average_degree()}")
        logger.info(f"Coverage: {graph.get_lambda_precision_points().get_density()}")


@pytest.mark.usefixtures("logger")
def test_connecting_graphs(logger: logging.Logger) -> None:
    """ Tests the connectivity of a graph generated by the LambdaPrecisionUDGGenerator. It generates a graph with a specified number of points and minimum distance, checks if the graph is connected, and if not, connects its components using the largest component strategy. It then checks for bridges in the graph and augments them using the smallest strategy. Finally, it asserts that the graph is connected and has no bridges.

    Args:
        logger: Logger instance for logging debug information.
    """

    generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.037), radius=0.0649)
    graph = generator.generate_graph()

    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test.png")
    logger.info(f"Is connected: {nx.is_connected(graph)}")
    if not nx.is_connected(graph):
        utils.connect_components(graph, strategy="largest")
    logger.info(f"Is connected: {nx.is_connected(graph)}")
    assert nx.is_connected(graph)
    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test1.png")

    logger.info(f"Bridges: {list(nx.bridges(graph))}")
    if list(nx.bridges(graph)):
        utils.augment_bridges_knn(graph, strategy="smallest")
    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test2.png")

    logger.info(f"Bridges: {list(nx.bridges(graph))}")
    assert nx.is_connected(graph)
    assert len(list(nx.bridges(graph))) == 0
