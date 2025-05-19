import networkx as nx

from src.graph_generator.graphs.generator import LambdaPrecisionUDGGenerator
from src.graph_generator.points.generator import RandomPointsGenerator
# from src.graph_generator.graphs.lambda_precision_udg2 import LambdaPrecisionUDG
import src.graph_generator.graphs.utils as utils


def test_lambda_precision_udg_generator():
    for _ in range(20):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.037), radius=0.083)
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

    print(f"Is connected: {nx.is_connected(graph)}")
    if not nx.is_connected(graph):
        utils.connect_components(graph)
    print(f"Is connected: {nx.is_connected(graph)}")
    assert nx.is_connected(graph)

    print(f"Bridges: {list(nx.bridges(graph))}")
    if list(nx.bridges(graph)):
        utils.augment_bridges_knn(graph)

    print(f"Bridges: {list(nx.bridges(graph))}")
    assert nx.is_connected(graph)
    assert len(list(nx.bridges(graph))) == 0
