import logging

import networkx as nx
import numpy as np

from lambdaprecisionudggenerator.graph_generator.graphs.lambda_precision_udg import (
    LambdaPrecisionUDG,
)
from lambdaprecisionudggenerator.graph_generator.points.generator import RandomPointsGenerator
from lambdaprecisionudggenerator.graph_generator.points.lambda_precision_points import (
    LambdaPrecisionPoints,
)
from lambdaprecisionudggenerator.graph_generator.seeds.seed import GeneratorSeed


class SeedGenerator:
    """Represents a generator for creating uniform disk graphs (UDGs) based on specified parameters and characteristics.

    The `UDGSeedGenerator` class provides functionality for generating seeds for UDG graphs based on input parameters like average degrees, node numbers, and coverage bounds. These seeds can then be used to create graphs with specific properties for further simulations or experiments.

    Attributes:
        sample_size (int): Number of samples to generate for determining graph properties like coverage and density.
        logger (logging.Logger): Logger instance for logging information, warnings, or errors during the generation process.
    """

    def __init__(self, sample_size: int = 10, logger: logging.Logger | None = None) -> None:
        """Represents a class that initialises and stores a sample size value.

        This class provides a mechanism to set a sample size with an optional default and retain it as an instance attribute.

        Args:
            sample_size: The size of the sample used for configuration or any operational purposes within the class. Defaults to 10.
            logger: An optional logger instance for logging information, warnings, or errors. Defaults to a logger named after the current module if not provided.
        """

        self.logger = logger or logging.getLogger(__name__)
        self.sample_size = sample_size

    @staticmethod
    def _approach_value_range(input_values: dict[str, float], result: float, target: float) -> None:
        """A function that adjusts the input range based on the output result compared to a target value.

        This function modifies the input dictionary by adjusting its "lower" and "upper" bounds based on the comparison of the output's "result" with a target value. The "result" in the input dictionary is then recalculated as the average of the updated "lower" and "upper" bounds.

        Args:
            input_values: dict with keys "lower", "upper", "result"
            result: computed using the input values, representing the output of a function or process
            target: target value for the output
        """

        if result < target:
            input_values["lower"] = input_values["result"]
        else:
            input_values["upper"] = input_values["result"]
        input_values["result"] = (input_values["lower"] + input_values["upper"]) / 2.0

    def _determine_coverage(
        self,
        min_dist: dict[str, float],
        coverage_bound: tuple[float, float],
        node_number: int,
        padding: bool = True,
    ) -> float:
        coverage_padding = (coverage_bound[1] - coverage_bound[0]) * 0.25 if padding else 0
        target_min = coverage_bound[0]
        target_max = coverage_bound[1] - 2 * coverage_padding

        while True:
            point_sets = RandomPointsGenerator(
                point_number=node_number, min_dist=min_dist["result"]
            ).generate_points_parallel(self.sample_size)

            if not point_sets:
                min_dist["upper"] = min_dist["result"]
                min_dist["result"] = (min_dist["lower"] + min_dist["upper"]) / 2.0
                continue

            densities = [points.get_density() for points in point_sets]
            coverage = float(np.mean(densities))

            self.logger.debug(f"{coverage_bound[0]} <= {coverage} <= {coverage_bound[1]}")

            if target_min <= coverage <= target_max:
                return min_dist["result"]

            self._approach_value_range(min_dist, coverage, float(np.mean(coverage_bound)))

    def _determine_avg_deg(
        self,
        point_sets: list[LambdaPrecisionPoints],
        target_average_degree: float,
        target_degree_margin: float = 0.125,
    ) -> tuple[float, list[LambdaPrecisionUDG]]:
        radius = {"upper": 0.6, "lower": 0.0, "result": 0.3}
        target_degree = target_average_degree + target_degree_margin

        while True:
            graphs = [LambdaPrecisionUDG(points, radius["result"]) for points in point_sets]
            average_degree = float(np.mean([graph.average_degree() for graph in graphs]))

            if target_average_degree <= average_degree < target_degree:
                return radius["result"], graphs

            self.logger.debug(
                f"{target_average_degree} <= {average_degree} <= {target_average_degree + 2 * target_degree_margin}"
                f"Connected: {sum(nx.is_connected(graph) for graph in graphs)}"
            )

            self._approach_value_range(radius, average_degree, target_degree)

    def generate_seeds(
        self,
        avg_degs: list[float],
        coverage_bound: tuple[float, float] = (0.9, 0.95),
        node_numbers: list[int] | None = None,
        padding: bool = True,
    ) -> list[GeneratorSeed]:
        """Generate seeds for random geometric graphs based on various parameters and constraints.

        This method generates a list of UDGGeneratorSeed objects by iteratively refining the parameters such as minimum distance between points, coverage, and average degree. It utilises helper functions to approach value ranges, determine coverage, average degree, and generate point distributions for a series of node counts. This process ensures that the generated seeds meet specific coverage and average degree criteria.

        Args:
            avg_degs: List of target average degrees for graph nodes.
            coverage_bound: Lower and upper bounds for the target coverage, default is [0.9, 0.95].
            node_numbers: List of node counts for which seeds will be generated.
            padding: Whether to pad the coverage range for convergence, default is True.

        Returns:
            List of UDGGeneratorSeed objects representing the configuration and parameters for generated random geometric graphs.
        """

        node_numbers = node_numbers or list(range(20, 320, 20))
        seeds = []
        min_dist_config = {"upper": 0.25, "lower": 0.0, "result": 0.125}

        for node_number in node_numbers:
            min_dist = self._determine_coverage(
                min_dist_config.copy(), coverage_bound, node_number, padding
            )

            point_sets = RandomPointsGenerator(node_number, min_dist).generate_points_parallel(
                self.sample_size
            )

            for avg_deg in avg_degs:
                radius, graphs = self._determine_avg_deg(point_sets, avg_deg)

                seeds.append(
                    GeneratorSeed(
                        node_number=node_number,
                        min_distance=min_dist,
                        radius=radius,
                        coverage_bound=coverage_bound,
                        avg_deg_bound=(avg_deg, avg_deg + 0.25),
                        probability_connected=sum(nx.is_connected(graph) for graph in graphs)
                        / float(len(graphs)),
                        sample_size=self.sample_size,
                        graphs=graphs,
                    )
                )
        return seeds
