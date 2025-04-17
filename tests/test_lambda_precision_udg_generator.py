from src.graph_generation.lambda_precision_udg_generator import LambdaPrecisionUDGGenerator
from src.graph_generation.random_points_generator import RandomPointsGenerator
import networkx as nx


def test_lambda_precision_udg_generator():
    for _ in range(20):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.0486), radius=0.0849)
        graph = generator.generate_graph()
        # graph.draw_random_geometric_graph(filepath="test_output")

        print(f"Average degree: {graph.average_degree()}")
        print(f"Average degree: {graph.reduce_avg_degree(3).average_degree()}")
        print(f"Average degree: {graph.average_degree()}")
        print(f"Average degree w/o bridges: {graph.reduce_avg_degree(3, bridges=False).average_degree()}")


def test_old_generator_seeds():
    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=100, min_dist=0.086932),
                                                radius=0.121875)
        graph = generator.generate_graph()

        print(f"Is connected: {nx.is_connected(graph.graph)}")
        print(f"Average degree: {graph.average_degree()}")
        print(f"Coverage: {graph.lambda_precision_points.get_density()}")
    for _ in range(10):
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.048588),
                                                radius=0.065625)
        graph = generator.generate_graph()

        print(f"Is connected: {nx.is_connected(graph.graph)}")
        print(f"Average degree: {graph.average_degree()}")
        print(f"Coverage: {graph.lambda_precision_points.get_density()}")


def test_connecting_graphs():
    generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(point_number=300, min_dist=0.0486), radius=0.0649)
    graph = generator.generate_graph()

    print(f"Is connected: {nx.is_connected(graph.graph)}")

    if not nx.is_connected(graph.graph):
        graph.connect_graph_components()

    print(f"Is connected: {nx.is_connected(graph.graph)}")

    print(f"Bridges: {list(nx.bridges(graph.graph))}")

    if list(nx.bridges(graph.graph)):
        graph.augment_bridges_knn()

    print(f"Bridges: {list(nx.bridges(graph.graph))}")

# Perform assertions to check the expected behavior
# assert len(points) == 300
# assert sum(map(sum, np.where(graph_generation.field > 1, 1, graph_generation.field))) == 300
# print(points[0:50])
# assert all(isinstance(point, tuple) for point in points)
