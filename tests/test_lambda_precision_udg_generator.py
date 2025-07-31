import pytest
import sys
import networkx as nx
import logging

from src.graph_generator.graphs.generator import LambdaPrecisionUDGGenerator
# from src.graph_generator.graphs.lambda_precision_udg2 import LambdaPrecisionUDG
import src.graph_generator.graphs.utils as utils
import src.graph_generator.graphs.graph_illustrator as graph_depiction

# initialise logger
root = logging.getLogger()  # the root logger
root.setLevel(logging.DEBUG)  # print everything ≥ DEBUG
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
root.handlers.clear()  # remove the handler pytest/PyCharm adds
root.addHandler(handler)

logging.getLogger("LambdaPrecisionUDGGenerator").propagate = True
logging.getLogger("LambdaPrecisionUDG").propagate = True
logging.getLogger("LambdaPrecisionPointsGenerator").propagate = True
logging.getLogger("LambdaPrecisionPoints").propagate = True

from src.graph_generator.points.generator import RandomPointsGenerator


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


def test_density_from_metadata() -> None:
    """ Tests if the density of the graph reconstructed from metadata matches the original density.

    This is done by generating a graph with a specified number of points and minimum distance, then reconstructing the points from the graph's metadata and checking if the density matches. The test runs multiple iterations to ensure consistency and reliability of the density calculation.

    Raises:
        AssertionError: If the original density does not match the reconstructed density within a specified tolerance.
    """

    lambda_precision_points_generator = RandomPointsGenerator(point_number=300, min_dist=0.037)
    for _ in range(20):
        generator = LambdaPrecisionUDGGenerator(lambda_precision_points_generator, radius=0.083, logger=root)
        graph = generator.generate_graph(connected=True)
        original_density = generator._lpp.get_density()

        # build UDG and reconstruct points from metadata
        reconstructed_pp = graph.get_lambda_precision_points()
        reconstructed_density = reconstructed_pp.get_density()

        # densities must match
        root.info(f"Original density: {original_density}, Reconstructed density: {reconstructed_density}")
        assert pytest.approx(original_density, rel=1e-3) == reconstructed_density


def test_lambda_precision_udg_generator():
    for _ in range(20):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.037), radius=0.083,
                                                logger=logger)
        graph = generator.generate_graph(connected=True)
        # graph.draw_random_geometric_graph(filepath="test_output")

        print(f"Average degree: {graph.average_degree()}")
        print(
            f"Average degree, after reducing it: {utils.reduce_avg_degree(graph, 4, preserve_bridges=False).average_degree()}")
        print(f"Average degree: {graph.average_degree()}")
        print(
            f"Average degree w/o bridges: {utils.reduce_avg_degree(graph, 3, preserve_bridges=True).average_degree()}")


def test_old_generator_seeds():
    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=100, min_dist=0.086932),
                                                radius=0.121875)
        graph = generator.generate_graph()

        print(f"Is connected: {nx.is_connected(graph.graph)}")
        print(f"Average degree: {graph.average_degree()}")
        print(f"Coverage: {graph.get_lambda_precision_points().get_density()}")
    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.048588),
                                                radius=0.065625)
        graph = generator.generate_graph()

        print(f"Is connected: {nx.is_connected(graph.graph)}")
        print(f"Average degree: {graph.average_degree()}")
        print(f"Coverage: {graph.get_lambda_precision_points().get_density()}")


def test_connecting_graphs():
    generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.037), radius=0.0649)
    graph = generator.generate_graph()

    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test.png")
    print(f"Is connected: {nx.is_connected(graph)}")
    if not nx.is_connected(graph):
        utils.connect_components(graph, strategy="largest")
    print(f"Is connected: {nx.is_connected(graph)}")
    assert nx.is_connected(graph)
    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test1.png")

    print(f"Bridges: {list(nx.bridges(graph))}")
    if list(nx.bridges(graph)):
        utils.augment_bridges_knn(graph, strategy="smallest")
    graph_depiction.draw_graph_with_segmented_nodes(graph, mean_types=5, max_bandwidth=100,
                                                    save_path="../test_output/test2.png")

    print(f"Bridges: {list(nx.bridges(graph))}")
    assert nx.is_connected(graph)
    assert len(list(nx.bridges(graph))) == 0
