import numpy as np
from joblib import Parallel, delayed
import math
import os

from src.graph_generator.points.lambda_precision_points import LambdaPrecisionPoints


class RandomPointsGenerator:
    """
    Generates sets of random points that satisfy minimum distance constraints.
    """

    def __init__(self, point_number: int, min_dist: float, field_size: int = 1000):
        """
        Initialise a RandomPointsGenerator.

        Args:
            point_number: Target number of points to generate
            min_dist: Minimum distance between points (in [0,1] range)
            field_size: Size of the discrete field (default 1000)
        """

        if not 0 <= min_dist <= 1:
            raise ValueError("Distances have to be between zero and one")

        self.point_number = point_number
        # Use ceiling to ensure we never go below the requested minimum distance
        self.min_dist = math.ceil(min_dist * field_size)
        self.field_size = field_size

    def generate_points(
            self, generate_image_options: dict[str, any] = {}
    ) -> LambdaPrecisionPoints | None:
        """
        Generates a set of random points respecting the minimum distance constraint.

        Args:
            generate_image_options: Dictionary with options for generating images:
                - 'interval': Interval for generating images (0 = no images)
                - 'output_path': Path to save images

        Returns:
            LambdaPrecisionPoints object or None if generation fails
        """

        generate_image_interval = generate_image_options.get("interval", 0)
        output_path = generate_image_options.get("output_path", "test_output")

        # Create output directory if needed
        if generate_image_interval > 0:
            os.makedirs(output_path, exist_ok=True)

        # Initialise empty point field
        field = np.zeros((self.field_size, self.field_size), dtype=int)
        lpp = LambdaPrecisionPoints([], self.min_dist, field)

        for i in range(self.point_number):
            if not lpp.add_random_point():
                print(f"Field is full after {i} iterations")
                if i < self.point_number * 0.5:  # If we couldn't add even half the points
                    return None
                else:
                    break  # Return what we have if we got at least half the points

            if generate_image_interval and i % generate_image_interval == 0:
                lpp.generate_image(iteration=i, output_path=output_path)

        return lpp

    def generate_points_parallel(
            self, number: int, prefer: str = None
    ) -> list[LambdaPrecisionPoints]:
        """
        Generates multiple point sets in parallel.

        Args:
            number: Number of point sets to generate
            prefer: Argument for joblib about the preferred way to parallelise

        Returns:
            List of generated LambdaPrecisionPoints objects (successful generations only)
        """

        # Filter out None results (failed generations)
        return list(
            filter(
                None,
                Parallel(n_jobs=-1, prefer=prefer)(
                    delayed(self.generate_points)() for _ in range(number)
                )
            )
        )
