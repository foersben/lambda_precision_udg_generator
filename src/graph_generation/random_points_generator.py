import random as rnd
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


class LambdaPrecisionPoints:
    def __init__(self, points, min_dist, field: [int]):
        self.points = points
        self.min_dist = min_dist
        self.field = field

    def get_lambda_precision_points(self):
        return np.array(self.points) / float(len(self.field))

    def get_min_dist(self):
        return self.min_dist / float(len(self.field))

    def add_random_point(self):
        try:
            center = rnd.choice(list(zip(*np.where(self.field == 0))))
            self.points.append(center)

            """Removes point coordinate and coordinates in radius from field"""
            field_size = len(self.field)
            x_range = np.arange(
                center[0] - self.min_dist - 1, center[0] + self.min_dist + 1, dtype=int
            )
            y_range = np.arange(
                center[1] - self.min_dist - 1, center[1] + self.min_dist + 1, dtype=int
            )
            x, y = np.where(
                (x_range[:, np.newaxis] - center[0]) ** 2 + (y_range - center[1]) ** 2
                <= self.min_dist ** 2
            )
            x, y = zip(
                *[
                    (x_coord, y_coord)
                    for (x_coord, y_coord) in list(zip(x_range[x], y_range[y]))
                    if 0 <= x_coord < field_size and 0 <= y_coord < field_size
                ]
            )
            self.field[x, y] += 1
        except (ValueError, IndexError):
            raise ValueError("Field is full")

    # def get_random_point_distribution(self):
    #     """
    #     Generates a random point distribution - whatever that means
    #     """
    #     return np.array(self.points) / float(len(self.field))

    def get_density(self):
        """
        Computes density of the point distribution - whatever that means

        :return: float - density of the point distribution
        """
        # TODO improve by allowing to use a freely chosen radius
        # field = np.where(self.field > 1, 1, self.field)
        # return float(len(list(zip(*np.where(field == 1))))) / float(
        #     len(self.field) ** 2
        # )
        return np.mean(np.where(self.field > 1, 1, self.field))

    def generate_image(self, iteration: int, output_path: str):
        """
        Generates an image of the point distribution and saves it as a JPEG file

        :param iteration: int - iteration number
        :param output_path: str - path to save the image to
        """
        import os

        filename = f"image_{len(self.points)}_points_{len(self.field)}\
                _size_{self.min_dist}_distance_iteration_{iteration}.jpg"
        filepath = os.path.join(output_path, filename)

        plt.figure(1)
        plt.clf()
        plt.imshow(
            np.where(self.field > 1, 1, self.field),
            # interpolation="nearest",
            cmap="binary_r",
        )
        plt.axis("off")
        plt.savefig(filepath, format="jpeg")
        plt.close()

    def serialize(self, path: str):
        import pickle

        with open(f"{path}/{id(self)}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(cls, filepath: str):
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)


class RandomPointsGenerator:
    def __init__(self, point_number: int, min_dist: float, field_size: int = 1000):
        if 1 < min_dist < 0:
            raise ValueError("Distances have to be between zero and one")
        self.point_number = point_number
        self.min_dist = int(min_dist * field_size)
        self.field_size = field_size

    def generate_points(self, generate_image_options: dict = {}) -> LambdaPrecisionPoints:
        generate_image_interval = generate_image_options.get("interval", 0)
        output_path = generate_image_options.get("output_path", "test_output")

        lpp = LambdaPrecisionPoints(
            [], self.min_dist, np.zeros((self.field_size, self.field_size), dtype=int)
        )
        for i in range(self.point_number):
            try:
                lpp.add_random_point()
            except ValueError:
                print(f"Field is full after {i} iterations")
                return None

            if generate_image_interval and not i % generate_image_interval:
                lpp.generate_image(iteration=i, output_path=output_path)
        return lpp

    def generate_points_parallel(self, number: int, prefer=None):
        """
        Generates for a given number as many graphs in parallel
        using the Joblib library

        @param number       number of graphs to generate
        @param prefer       argument for joblib about the preferred way to parallelise
        @return             list of generated graphs
        """
        return Parallel(n_jobs=-1, prefer=prefer)(
            delayed(self.generate_points)() for _ in range(number)
        )
