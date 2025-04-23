import random as rnd
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import math
import os


class LambdaPrecisionPoints:
    """
    Manages a set of points with minimum distance constraints.

    This class maintains points in a discretised field and ensures the minimum
    distance constraint is respected for all points.
    """

    def __init__(self, points: list[tuple[int, int]], min_dist: int, field: np.ndarray):
        """
        Initialise a LambdaPrecisionPoints object.

        Args:
            points: List of discrete point coordinates
            min_dist: Minimum distance between points in discrete units
            field: 2D numpy array representing the discretised field
        """

        self.points = points
        self.min_dist = min_dist
        self.field = field

    def get_lambda_precision_points(self) -> np.ndarray:
        """
        Returns normalised points in [0,1] × [0,1] range.

        Returns:
            Normalized point coordinates as numpy array
        """

        return np.array(self.points) / float(len(self.field))

    def get_min_dist(self) -> float:
        """
        Returns normalized minimum distance.

        Returns:
            Minimum distance in [0,1] range
        """

        return self.min_dist / float(len(self.field))

    def add_random_point(self) -> bool:
        """
        Adds a random point that respects the minimum distance constraint.
        Updates the field to mark occupied regions.

        Returns:
            True if a point was added, False if the field is full
        """

        try:
            # Find available positions (where field is 0)
            available_positions = np.where(self.field == 0)
            if len(available_positions[0]) == 0:
                return False

            # Choose a random available position
            idx = rnd.randint(0, len(available_positions[0]) - 1)
            center = (available_positions[0][idx], available_positions[1][idx])
            self.points.append(center)

            # Mark region around point as occupied using vectorized operations
            field_size = len(self.field)

            # Create meshgrid for efficient distance calculation
            x_min = max(0, center[0] - self.min_dist - 1)
            x_max = min(field_size, center[0] + self.min_dist + 1)
            y_min = max(0, center[1] - self.min_dist - 1)
            y_max = min(field_size, center[1] + self.min_dist + 1)

            x_range = np.arange(x_min, x_max, dtype=int)
            y_range = np.arange(y_min, y_max, dtype=int)
            xx, yy = np.meshgrid(x_range, y_range)

            # Calculate squared distances
            dist_squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2

            # Find points within min_dist
            mask = dist_squared <= self.min_dist ** 2

            # Update field
            self.field[xx[mask], yy[mask]] += 1

            return True
        except (ValueError, IndexError):
            return False

    def get_density(self) -> float:
        """
        Computes density of the point distribution as the fraction of field marked as occupied.

        Returns:
            Density value between 0 and 1
        """

        return float(np.mean(np.where(self.field > 0, 1, 0)))

    def generate_image(self, iteration: int, output_path: str) -> None:
        """
        Generates an image of the point distribution and saves it as a JPEG file.

        Args:
            iteration: Iteration number for the filename
            output_path: Path to save the image to
        """

        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        filename = f"image_{len(self.points)}_points_{len(self.field)}_size_{self.min_dist}_distance_iteration_{iteration}.jpg"
        filepath = os.path.join(output_path, filename)

        plt.figure(figsize=(8, 8))
        plt.clf()
        plt.imshow(
            np.where(self.field > 0, 1, 0),
            cmap="binary_r",
        )
        plt.axis("off")
        plt.savefig(filepath, format="jpeg")
        plt.close()

    def serialize(self, path: str) -> None:
        """
        Serialises the object to a file.

        Args:
            path: Path to save the serialised object
        """

        import pickle

        with open(f"{path}/{id(self)}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(cls, filepath: str) -> "LambdaPrecisionPoints":
        """
        Deserialises an object from a file.

        Args:
            filepath: Path to the serialised object

        Returns:
            Deserialized LambdaPrecisionPoints object
        """

        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)


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
