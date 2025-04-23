from src.graph_generation.random_points_generator import RandomPointsGenerator
import numpy as np
import os


def test_random_points_generator():
    # Create an instance of RandomPointsGenerator
    generator = RandomPointsGenerator(point_number=300, min_dist=0.0486)

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


def test_minimum_distance_constraint():
    """Test that all points respect the minimum distance constraint."""
    # Create an instance of RandomPointsGenerator with a specific min_dist
    min_dist = 0.0486
    generator = RandomPointsGenerator(point_number=300, min_dist=min_dist)

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
    assert min_actual_dist >= min_dist - 1e-10, f"Minimum distance constraint violated: {min_actual_dist} < {min_dist}"
    print(f"Minimum distance between points: {min_actual_dist}, Required minimum: {min_dist}")


def test_rev_eng_exp_cov():
    lambda_ = [0.0732, 0.0503, 0.0747, 0.0525, 0.0761, 0.0535, 0.0791, 0.0543, 0.0805, 0.0566, 0.0820, 0.0589, 0.0878,
               0.0597, 0.0878, 0.0617]
    node_number = [100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200]
    tuple_list = list(zip(lambda_, node_number))
    iterations = 20
    for tuple_ in tuple_list:
        generator = RandomPointsGenerator(point_number=tuple_[1], min_dist=tuple_[0])
        mean_density = []
        for i in range(iterations):
            lpp = None
            while not lpp:
                lpp = generator.generate_points()
            mean_density.append(lpp.get_density())
        print(f"(Lambda, Node Number): {str(tuple_)}, Coverage: {sum(mean_density) / iterations}")
