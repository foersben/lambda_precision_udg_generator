import logging
import random as rnd
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


class LambdaPrecisionPoints:
    """Manages a set of points with minimum distance constraints.

    This class maintains points in a discretised field and ensures the minimum
    distance constraint is respected for all points.

    Attributes:
        points (list[tuple[int, int]]): List of discrete point coordinates
        min_distance (int): Minimum distance between points in discrete units
        field (np.ndarray): 2D numpy array representing the discretised field
    """

    def __init__(
        self,
        points: list[tuple[int, int]],
        min_distance: int,
        field: np.ndarray,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise a LambdaPrecisionPoints object.

        Args:
            points: List of discrete point coordinates
            min_distance: Minimum distance between points in discrete units
            field: 2D numpy array representing the discretised field
            logger: Optional logger instance
        """

        self.logger = logger or logging.getLogger(__name__)

        self.points = points
        self.min_distance = min_distance
        self.field = field

    def __len__(self) -> int:
        """Return the number of points.

        Returns:
            Number of points in the collection
        """

        return len(self.points)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Enable iteration over points.

        Returns:
            Iterator over the collection of points
        """

        return iter(self.points)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        """Retrieve a specific point by index.

        Args:
            idx: The index of the point to retrieve

        Returns:
            Point coordinates as (x, y) tuple
        """

        return self.points[idx]

    def get_metadata(self) -> dict[str, int]:
        """Return metadata about the object.

        Returns:
            Dictionary containing 'min_distance' and 'field_size'
        """

        return {"min_distance": self.min_distance, "field_size": len(self.field)}

    def get_lambda_precision_points(self) -> np.ndarray:
        """
        Returns normalised points in [0,1] × [0,1] range.

        Returns:
            Normalized point coordinates as numpy array
        """

        return np.array(self.points, dtype=np.float32) / float(len(self.field))

    def get_min_dist(self) -> float:
        """
        Returns normalized minimum distance.

        Returns:
            Minimum distance in [0,1] range
        """

        return self.min_distance / float(len(self.field))

    def add_random_point(self) -> bool:
        """Adds a random point that respects the minimum distance constraint. Updates the field to mark occupied regions.

        Returns:
            True if a point was added, False if the field is full
        """

        try:
            available_positions = np.where(self.field == 0)
            if len(available_positions[0]) == 0:
                return False

            idx = rnd.randint(0, len(available_positions[0]) - 1)
            center = (available_positions[0][idx], available_positions[1][idx])
            self.points.append(center)
            self._update_field_for_point(center)  # Use helper method
            return True
        except (ValueError, IndexError):
            return False

    # def add_random_point(self) -> bool:
    #     """ Adds a random point that respects the minimum distance constraint.
    #     Updates the field to mark occupied regions.

    #     Returns:
    #         True if a point was added, False if the field is full
    #     """

    #     try:
    #         # Find available positions (where field is 0)
    #         available_positions = np.where(self.field == 0)
    #         if len(available_positions[0]) == 0:
    #             return False

    #         # Choose a random available position
    #         idx = rnd.randint(0, len(available_positions[0]) - 1)
    #         center = (available_positions[0][idx], available_positions[1][idx])
    #         self.points.append(center)

    #         # Mark region around point as occupied using vectorized operations
    #         field_size = len(self.field)

    #         # Create meshgrid for efficient distance calculation
    #         x_min = max(0, center[0] - self.min_distance - 1)
    #         x_max = min(field_size, center[0] + self.min_distance + 1)
    #         y_min = max(0, center[1] - self.min_distance - 1)
    #         y_max = min(field_size, center[1] + self.min_distance + 1)

    #         x_range = np.arange(x_min, x_max, dtype=int)
    #         y_range = np.arange(y_min, y_max, dtype=int)
    #         xx, yy = np.meshgrid(x_range, y_range)

    #         # Calculate squared distances
    #         dist_squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2

    #         # Find points within min_dist
    #         mask = dist_squared <= self.min_distance ** 2

    #         # Update field
    #         self.field[xx[mask], yy[mask]] += 1

    #         return True
    #     except (ValueError, IndexError):
    #         return False

    def get_density(self) -> float:
        """Computes density of the point distribution as the fraction of field marked as occupied.

        Returns:
            Density value between 0 and 1
        """

        return float(np.mean(np.where(self.field > 0, 1, 0)))

    def _update_field_for_point(self, center: tuple[int, int]) -> None:
        """Updates the field matrix for a specified center point within a circular region defined by the minimum distance. The method calculates a grid of points around the center, determines their distances from the center, applies a mask identifying points within the specified radius, and increments the field values at those positions.

        Args:
            center: Coordinates of the center point as a tuple of integers (x, y).
        """

        field_size = len(self.field)
        x_min = max(0, center[0] - self.min_distance - 1)
        x_max = min(field_size, center[0] + self.min_distance + 1)
        y_min = max(0, center[1] - self.min_distance - 1)
        y_max = min(field_size, center[1] + self.min_distance + 1)

        x_range = np.arange(x_min, x_max, dtype=int)
        y_range = np.arange(y_min, y_max, dtype=int)
        xx, yy = np.meshgrid(x_range, y_range)

        dist_squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        mask = dist_squared <= self.min_distance**2
        self.field[xx[mask], yy[mask]] += 1

    @classmethod
    def from_metadata(cls, graph) -> "LambdaPrecisionPoints":
        """Creates an instance of LambdaPrecisionPoints class from metadata and graph.

        This class method initialises a LambdaPrecisionPoints object by extracting relevant information from the given graph and metadata. The graph's nodes must contain positional data ("pos"), which will be scaled and converted to integer coordinates using the field size specified in metadata. A numpy field of zeros with a size indicated in metadata is also created, along with the specified minimum distance.

        Args:
            graph: A graph whose nodes hold positional data ("pos") that will be used to compute points for the instance.

        Returns:
            An instance of LambdaPrecisionPoints initialised with points, minimum distance, and a field.
        """

        metadata = graph.points_metadata
        field_size = metadata["field_size"]
        min_distance = metadata["min_distance"]
        points = [
            ((x * field_size).astype(int), (y * field_size).astype(int))
            for _, (x, y) in graph.nodes(data="pos")
        ]

        # Create field and mark occupied regions
        field = np.zeros((field_size, field_size), dtype=int)
        instance = cls(points=[], min_distance=min_distance, field=field)

        # Add points and update field
        for point in points:
            instance.points.append(point)
            instance._update_field_for_point(point)  # New helper method

        return instance

        # metadata = graph.points_metadata
        # field_size = metadata["field_size"]
        # points = [((x * field_size).astype(int), (y * field_size).astype(int)) for _, (x, y) in graph.nodes(data="pos")]
        # return cls(points=points, min_distance=metadata["min_distance"],
        #            field=np.zeros(metadata["field_size"], dtype=int))

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

        filename = f"image_{len(self.points)}_points_{len(self.field)}_size_{self.min_distance}_distance_iteration_{iteration}.jpg"
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

    def clone(self) -> "LambdaPrecisionPoints":
        """Creates a deep copy of the LambdaPrecisionPoints instance.

        Returns:
            A deep copy of the current LambdaPrecisionPoints instance.
        """

        return LambdaPrecisionPoints(
            points=deepcopy(self.points), min_distance=self.min_distance, field=deepcopy(self.field)
        )

    def to_dict(self) -> dict[str, object]:
        """Convert object to JSON-serializable dictionary.

        Returns:
            Dictionary representation with points, min_distance, and field
        """
        return {"points": self.points, "min_distance": self.min_distance, "field": self.field}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LambdaPrecisionPoints":
        """Reconstruct object from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed LambdaPrecisionPoints object
        """
        return cls(points=data["points"], min_distance=data["min_distance"], field=data["field"])

    def serialize(self, path: str | Path) -> None:
        """Serialize the object to a JSON file.

        Args:
            path: Directory path to save the JSON file
        """
        from lambdaprecisionudggenerator.utils.json_utils import save_json

        filepath = f"{path}/{id(self)}.json"
        save_json(self.to_dict(), filepath)

    @classmethod
    def deserialize(cls, filepath: str | Path) -> "LambdaPrecisionPoints":
        """Deserialize an object from a JSON file.

        Args:
            filepath: Path to the serialized JSON file

        Returns:
            Deserialized LambdaPrecisionPoints object
        """
        from lambdaprecisionudggenerator.utils.json_utils import load_json

        data = load_json(filepath)
        return cls.from_dict(data)
