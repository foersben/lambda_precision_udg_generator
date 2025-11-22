import logging

from lambdaprecisionudggenerator.utils import setup_logging

setup_logging()

import os

import numpy as np
import pytest

from lambdaprecisionudggenerator.graph_generator.points.generator import RandomPointsGenerator


@pytest.fixture(scope="function", autouse=True)
def logger(caplog: pytest.LogCaptureFixture) -> logging.Logger:
    """Fixture providing a configured logger for tests.

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
def test_random_points_generator(logger: logging.Logger) -> None:
    """Tests the functionality of the RandomPointsGenerator class by creating an instance, generating points with specified options, and performing assertions to validate the expected behaviour. Specifically, this includes verifying the number of points and ensuring all generated points are of the required type.

    Args:
        logger: Logger instance for logging debug information.
    """

    # Create an instance of RandomPointsGenerator
    generator = RandomPointsGenerator(point_number=300, min_dist=0.037, logger=logger)

    # Call the generate_points method
    generate_image_options = {"interval": 50, "output_path": "test_output"}

    # Create output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)

    lpp = generator.generate_points(generate_image_options=generate_image_options)
    if lpp is not None:
        points = lpp.get_lambda_precision_points()

        # Perform assertions to check the expected behavior
        assert len(points) == 300
        assert all(isinstance(point, np.ndarray) for point in points)


@pytest.mark.usefixtures("logger")
def test_minimum_distance_constraint(logger: logging.Logger):
    """Tests the functionality of point generation with a constraint on the minimum distance between any pair of points. Ensures that a given minimum distance is respected while generating a specific number of points using the RandomPointsGenerator class.

    This test verifies the following:
    1. The points are successfully generated.
    2. The number of points generated matches the expected count.
    3. The minimum distance between any pair of generated points satisfies the specified constraint, within an acceptable tolerance.

    Args:
        logger: Logger instance for logging debug information.

    Raises:
        AssertionError: If point generation fails, if the number of generated points does not match the required count, or if the minimum distance constraint is violated.
    """

    # Create an instance of RandomPointsGenerator with a specific min_dist
    min_dist = 0.0486
    generator = RandomPointsGenerator(point_number=300, min_dist=min_dist, logger=logger)

    # Generate points
    lpp = generator.generate_points()

    # Verify all points were placed
    assert lpp is not None, "Failed to generate points"
    points = lpp.get_lambda_precision_points()
    assert len(points) == 300, f"Expected 300 points, got {len(points)}"

    # Calculate the minimum distance between any pair of points
    from scipy.spatial import distance_matrix

    # Calculate all pairwise distances
    dist_matrix = distance_matrix(points, points)

    # Set the diagonal elements to infinity (to ignore distances of points to themselves)
    np.fill_diagonal(dist_matrix, np.inf)

    # Find the minimum distance
    min_actual_dist = np.min(dist_matrix)

    # Verify that the minimum distance constraint is respected
    # Use a small tolerance for floating-point comparison
    assert (
        min_actual_dist >= min_dist - 1e-10
    ), f"Minimum distance constraint violated: {min_actual_dist} < {min_dist}"
    logger.info(f"Minimum distance between points: {min_actual_dist}, Required minimum: {min_dist}")


@pytest.mark.usefixtures("logger")
def test_rev_eng_exp_cov(logger: logging.Logger) -> None:
    """Tests the coverage of the RandomPointsGenerator by generating points with various lambda values and node counts, and calculating the mean density over multiple iterations.

    Args:
        logger: Logger instance for logging debug information.
    """

    lambda_ = [
        0.0732,
        0.0503,
        0.0747,
        0.0525,
        0.0761,
        0.0535,
        0.0791,
        0.0543,
        0.0805,
        0.0566,
        0.0820,
        0.0589,
        0.0878,
        0.0597,
        0.0878,
        0.0617,
    ]
    node_number = [100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200]
    tuple_list = list(zip(lambda_, node_number, strict=False))
    iterations = 20
    for tuple_ in tuple_list:
        generator = RandomPointsGenerator(point_number=tuple_[1], min_dist=tuple_[0], logger=logger)
        mean_density = []
        for i in range(iterations):
            lpp = None
            while not lpp:
                lpp = generator.generate_points()
            mean_density.append(lpp.get_density())
        logger.info(
            f"(Lambda, Node Number): {tuple_!s}, Coverage: {sum(mean_density) / iterations}"
        )
