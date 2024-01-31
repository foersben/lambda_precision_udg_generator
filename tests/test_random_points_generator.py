from src.graph_generation.random_points_generator import RandomPointsGenerator
import numpy as np


def test_random_points_generator():
    # Create an instance of RandomPointsGenerator
    generator = RandomPointsGenerator(point_number=300, min_dist=0.0486)

    # Call the generate_points method
    generate_image_options = {"interval": 50, "output_path": "test_output"}
    lpp = generator.generate_points(generate_image_options=generate_image_options)
    if lpp is not None:
        points = lpp.get_lambda_precision_points()

        # Perform assertions to check the expected behavior
        assert len(points) == 300
        assert all(isinstance(point, np.ndarray) for point in points)


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
