from src.graph_generator.graphs.lambda_precision_udg import LambdaPrecisionUDG
from src.graph_generator.points.generator import RandomPointsGenerator
from src.graph_generator.points.lambda_precision_points import LambdaPrecisionPoints

import networkx as nx
import numpy as np

from src.graph_generator.seeds.database import UDGGeneratorSeedDB
from src.graph_generator.seeds.seed import UDGGeneratorSeed


class UDGSeedGenerator:
    """ Represents a generator for creating uniform disk graphs (UDGs) based on specified parameters and characteristics.

    The `UDGSeedGenerator` class provides functionality for generating seeds for UDG graphs based on input parameters like average degrees, node numbers, and coverage bounds. These seeds can then be used to create graphs with specific properties for further simulations or experiments.

    Attributes:
        sample_size (int): Number of samples to generate for determining graph properties like coverage and density.
    """

    def __init__(self, sample_size: int = 10) -> None:
        """ Represents a class that initialises and stores a sample size value.

        This class provides a mechanism to set a sample size with an optional default and retain it as an instance attribute.

        Args:
            sample_size: The size of the sample used for configuration or any operational purposes within the class. Defaults to 10.
        """

        self.sample_size = sample_size

    def generate_seeds(
            self,
            avg_degs: list[float],
            coverage_bound: (float, float) = (0.9, 0.95),
            node_numbers: [int] = [i for i in range(20, 320, 20)],
            padding: bool = True
    ) -> list["UDGGeneratorSeed"]:
        """ Generate seeds for random geometric graphs based on various parameters and constraints.

        This method generates a list of UDGGeneratorSeed objects by iteratively refining the parameters such as minimum distance between points, coverage, and average degree. It utilises helper functions to approach value ranges, determine coverage, average degree, and generate point distributions for a series of node counts. This process ensures that the generated seeds meet specific coverage and average degree criteria.

        Args:
            avg_degs: List of target average degrees for graph nodes.
            coverage_bound: Lower and upper bounds for the target coverage, default is [0.9, 0.95].
            node_numbers: List of node counts for which seeds will be generated.
            padding: Whether to pad the coverage range for convergence, default is True.

        Returns:
            List of UDGGeneratorSeed objects representing the configuration and parameters for generated random geometric graphs.
        """

        def approach_value_range(input: dict[str, float], output: dict[str, float], target: float) -> None:
            """ A class responsible for generating seeds for Unit Disk Graph (UDG) generators. The seeds are calculated based on provided average degrees, coverage bounds, node numbers, and other operational parameters.

            The primary purpose of the class is to produce seeds that can be used in further simulation or analysis of Unit Disk Graphs, ensuring adaptability and performance using precise input constraints.

            Args:
                input: dict with keys "lower", "upper", "result"
                output: dict with key "result"
                target: target value for the output
            """

            if output["result"] < target:
                input["lower"] = input["result"]
            else:
                input["upper"] = input["result"]
            input["result"] = (input["lower"] + input["upper"]) / 2.0

        def determine_coverage(min_dist: dict, coverage: dict, node_number: int, padding: bool = True) -> None:
            """ Determines the coverage of a graph based on the minimum distance and node number.

            This function iteratively refines the minimum distance and coverage values until they fall within specified bounds. It generates point distributions and calculates their density to ensure the generated graphs meet the desired coverage criteria.

            Args:
                min_dist: dict with keys "lower", "upper", "result"
                coverage: dict with key "result"
                node_number: number of points in the point distribution
                padding: whether to pad the coverage range for convergence, default is True
            """

            print(f'{node_number}:\tmin_dist: {min_dist["result"]:.6f},\t\t coverage: {coverage["result"]:.6f}')
            # sd = 1.0
            result_min_dist = 0.0
            coverage_padding = (coverage_bound[1] - coverage_bound[0]) * 0.25 if padding else 0
            while not coverage_bound[0] <= coverage["result"] <= coverage_bound[1] - 2 * coverage_padding:
                generator = RandomPointsGenerator(point_number=node_number, min_dist=min_dist["result"])
                point_sets = list(filter(None, generator.generate_points_parallel(self.sample_size)))
                for points in point_sets:
                    print(f"density: {points.get_density()}, points: {len(points.points)}")
                if len(point_sets) < 1:
                    min_dist["upper"] = min_dist["result"]
                    min_dist["result"] = (min_dist["lower"] + min_dist["upper"]) / 2.0
                    continue
                if len(point_sets) < self.sample_size / 3.0:
                    continue
                coverage["result"] = np.mean([points.get_density() for points in point_sets])
                # sd = np.std([points.get_density() for points in point_sets])
                print(f'{node_number}:\tmin_dist: {min_dist["result"]:.6f},\
                        \t\t coverage: {coverage["result"]:.6f},\
                        \t\t avg_points: {np.mean([len(points.points) for points in point_sets if points]):.6f}')
                result_min_dist = min_dist["result"]
                approach_value_range(min_dist, coverage, np.mean(coverage_bound))
                print(
                    f'{coverage_bound[0] + coverage_padding} <= {coverage["result"]} <= {coverage_bound[1] - coverage_padding}')
                # if generator_seeds:
                #   if 0.8 * min_dist["result"] < generator_seeds[-1].min_dist:
                #       coverage = {"result": 1.0}  #       min_dist["lower"] = 0.0
                #       determine_coverage(min_dist, coverage, node_number)
            min_dist["result"] = result_min_dist

        def determine_avg_deg(
                radius: dict[str, float],
                min_dist: dict[str, float],
                point_sets: list[LambdaPrecisionPoints],
                node_number: int,
                generator_seeds: ["UDGGeneratorSeed"],
                avg_deg_margin: float = 0.25
        ) -> None:
            """ Determines the average degree of a graph based on the radius and minimum distance.

            This function iteratively refines the radius and average degree values until they fall within specified bounds. It generates point distributions and calculates their average degree to ensure the generated graphs meet the desired average degree criteria.

            Args:
                radius: dict with keys "lower", "upper", "result"
                min_dist: dict with key "result"
                point_sets: list of point distributions
                node_number: number of points in the point distribution
                generator_seeds: list of UDGGeneratorSeed objects
                avg_deg_margin: margin for average degree, default is 0.25

            TODO Coverage and Min Dist are potentially wrongly combined - old and new results
            mixed
            TODO Probably the same here!
            """

            for avg_deg in avg_degs:
                radius["upper"] = 0.6
                degree = {"result": 100}
                graphs = []
                result_radius = 0.0
                while not avg_deg <= degree["result"] < avg_deg + avg_deg_margin:
                    graphs = [LambdaPrecisionUDG(nx.random_geometric_graph(node_number, radius["result"], pos={
                        i: (points.get_lambda_precision_points()[i][0], points.get_lambda_precision_points()[i][1],) for
                        i in range(node_number)}), points, radius["result"], ) for points in point_sets]
                    degree["result"] = np.mean([graph.average_degree() for graph in graphs])
                    result_radius = radius["result"]
                    approach_value_range(input=radius, output=degree, target=avg_deg + 0.5 * avg_deg_margin)

                print(f'{coverage_bound[0]} <= {coverage["result"]} <= {coverage_bound[1]}')
                print(f'{avg_deg} <= {degree["result"]} <= {avg_deg + 0.25}')
                print(f'Connected: {sum([nx.is_connected(graph.graph) for graph in graphs])}')
                generator_seeds.append(
                    UDGGeneratorSeed(node_number, min_dist["result"], result_radius, coverage_bound,
                                     [avg_deg, avg_deg + 0.25],
                                     sum(nx.is_connected(graph.graph) for graph in graphs) / len(graphs),
                                     self.sample_size, graphs))

        def generate_point_distributions(
                min_dist: dict[str, float],
                node_number: int,
                point_sets: list[LambdaPrecisionPoints] = []
        ) -> list[LambdaPrecisionPoints]:
            """ Generates point distributions for a given node number and min_dist

            Args:
                min_dist: Dictionary with keys "lower", "upper", "result"
                node_number: Number of points in the point distribution
                point_sets: List of point distributions

            Returns:
                point_sets: List of point distributions
            """

            generator = RandomPointsGenerator(point_number=node_number, min_dist=min_dist["result"])
            while len(point_sets) < self.sample_size:
                list(map(point_sets.append, list(
                    filter(None, generator.generate_points_parallel(self.sample_size - len(point_sets)))
                )))
            # print(f"{len(point_sets[0].get_lambda_precision_points())} = {node_number}")
            return point_sets

        generator_seeds = []
        min_dist = {"upper": 0.25, "lower": 0.0, "result": 0.125, }
        for node_number in node_numbers:
            coverage = {"result": 1.0}
            min_dist["lower"] = 0.0
            # print(f"node_number: {node_number}")
            determine_coverage(min_dist, coverage, node_number, padding=padding)
            point_sets = []
            generate_point_distributions(min_dist, node_number, point_sets)
            determine_avg_deg(min_dist=min_dist, radius={"upper": 0.6, "lower": 0.0, "result": 0.3},
                              point_sets=point_sets, node_number=node_number, generator_seeds=generator_seeds)
        return generator_seeds


if __name__ == "__main__":
    generator = UDGSeedGenerator(sample_size=3)
    seeds = generator.generate_seeds(avg_degs=[3, 4], node_numbers=list(range(20, 60, 20)))
    # list(map(lambda seed: seed.generate_graphs(3), seeds))
    for seed in seeds:
        seed.generate_graphs(sample_size=3, connected=True)
    seed_db = UDGGeneratorSeedDB(*seeds)
    seed_db.serialize(f"test_output/{id(seed_db)}")
    print(seed_db.latex_table())
    seed_db = UDGGeneratorSeedDB.deserialize(f"test_output/{id(seed_db)}")
    print("Seeds:")
    for seed in seed_db.seeds:
        print(seed)
