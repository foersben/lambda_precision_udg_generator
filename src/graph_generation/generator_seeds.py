from .lambda_precision_udg_generator import (LambdaPrecisionUDG, LambdaPrecisionUDGGenerator, )
from .random_points_generator import RandomPointsGenerator

import networkx as nx
import numpy as np

__all__ = ["UDGGeneratorSeedDB", "UDGGeneratorSeed", "UDGSeedGenerator", ]


class UDGGeneratorSeedDB:
    def __init__(self, *seeds: "UDGGeneratorSeed"):
        self.seeds = [*seeds]

    def append(self, *seeds: "UDGGeneratorSeed"):
        self.seeds += [*seeds]

    def mean_var_local_clustering(self, filepath: str):
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        # import tikzplotlib as tpl
        from tikzplotlib import save as tikz_save

        # x_label = r'$\overline{A_{\text{coverage}}}$'
        x_label = r'placeholder'
        y_label = r'mean of variance of local clustering'
        legend = r'$|V|=${0}, $\deg_{{exp}}=${1}'

        plt.style.use("ggplot")
        for node_number in set(seed.node_number for seed in self.seeds):
            for deg in set(seed.avg_deg_bound[0] for seed in self.seeds):
                x = []
                y = []
                for seed in self.seeds:
                    if seed.node_number == node_number and seed.avg_deg_bound[0] == deg:
                        x.append(seed.coverage_bound[0])
                        y.append(np.mean(seed.variance_local_clustering()))
                        # print(seed.variange_local_clustering())
                plt.plot(x, y, label=legend.format(node_number, deg), lw=2)
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(list(set(seed.coverage_bound[0] for seed in self.seeds)))
        plt.xlim(0.85, 1)
        # plt.ylim(0, 1.1 * max(y))
        plt.grid(True)
        # plt.show()

        # tpl.save(filepath)
        tikz_save(filepath)
        plt.savefig(f'{filepath}.png', dpi=300)
        plt.close()

    def mean_var_deg_distribution(self, filepath: str):
        """
        Plots
        :return:
        """
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        # import tikzplotlib as tpl
        from tikzplotlib import save as tikz_save

        # x_label = r'$\overline{A_{\text{coverage}}}$'
        x_label = r'placeholder'
        y_label = r"mean of variance of node degree distribution"
        legend = r"$|V|=${0}, $\deg_{{exp}}=${1}"

        plt.style.use("ggplot")
        for node_number in set(seed.node_number for seed in self.seeds):
            for deg in set(seed.avg_deg_bound[0] for seed in self.seeds):
                x = []
                y = []
                for seed in self.seeds:
                    if seed.node_number == node_number and seed.avg_deg_bound[0] == deg:
                        x.append(seed.coverage_bound[0])
                        # print(np.mean(seed.variance_node_degree_distribution()))
                        y.append(np.mean(seed.variance_node_degree_distribution()))
                print(f"DEG: {deg}: {list(y)}")
                plt.plot(x, y, label=legend.format(node_number, deg), lw=2)
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(list(set(seed.coverage_bound[0] for seed in self.seeds)))
        plt.grid(True)
        # plt.show()

        # tpl.save(filepath)
        tikz_save(filepath)
        plt.savefig(f'{filepath}.png', dpi=300)
        plt.close()

    def latex_table(self, filepath: str | None = None) -> str:
        from tabulate import tabulate
        # node number, deg_exp, lambda, r_tr, mean A_cov, mean deg_avg, P_connected
        table = [[r"\raggedleft $|V|$", r"\raggedleft $\deg_\exp$", r"\raggedleft $\lambda$", r"\raggedleft $r_{tr}$",
                  r"\raggedleft $\overline{A_\text{cov}}$", r"\raggedleft $\overline{\deg_\text{avg}}$",
                  r"\raggedleft $P_{connected}$", ], ]
        data = sorted([[seed.node_number, seed.avg_deg_bound[0], seed.min_dist, seed.radius,
                        float(np.mean([graph.lambda_precision_points.get_density() for graph in seed.graphs])),
                        float(np.mean([graph.average_degree() for graph in seed.graphs])), seed.prob_connected] for seed
                       in
                       self.seeds], key=lambda x: (x[0], x[1]))
        table += data

        latex_table = tabulate(table, headers="firstrow", tablefmt="latex_raw")
        if filepath:
            with open(f"{filepath}.tex", "w") as f:
                f.write(latex_table)
        return latex_table

    def latex_table_coverage(self, filepath: str | None = None) -> str:
        from tabulate import tabulate
        # node number, deg_exp, lambda, r_tr, mean A_cov, mean deg_avg, P_connected
        table = [[r"\raggedleft $|V|$", r"\raggedleft $\deg_\exp$", r"\raggedleft $\lambda$", r"\raggedleft $r_{tr}$",
                  r"\raggedleft $\overline{A_\text{cov}}$", r"\raggedleft $\overline{\deg_\text{avg}}$",
                  r"\raggedleft $P_{connected}$", ], ]
        data = sorted([[seed.coverage_bound, seed.node_number, seed.avg_deg_bound[0], seed.min_dist, seed.radius,
                        float(np.mean([graph.lambda_precision_points.get_density() for graph in seed.graphs])),
                        float(np.mean([graph.average_degree() for graph in seed.graphs])), seed.prob_connected] for seed
                       in
                       self.seeds], key=lambda x: (x[0][0], x[1]))
        table += data

        latex_table = tabulate(table, headers="firstrow", tablefmt="latex_raw")
        if filepath:
            with open(f"{filepath}.tex", "w") as f:
                f.write(latex_table)
        return latex_table

    def serialize(self, path: str):
        import os

        if not os.path.exists(path):
            os.makedirs(path)
        for seed in self.seeds:
            seed.serialize(path=path)

    @classmethod
    def deserialize(cls, path: str):
        import os

        if not os.path.exists(path):
            print(f"Filepath {path} does not exist.")
            return
        pkls = [f for f in os.listdir(path) if f.endswith(".pkl")]

        # Update the import statement to reflect the new module name 'graph_generation'
        # Temporary fix for the import error
        return cls(*[UDGGeneratorSeed.deserialize(os.path.join(path, j)) for j in pkls])


class UDGGeneratorSeed:
    def __init__(self, node_number, min_dist, radius, coverage_bound, avg_deg_bound, prob_connected, sample_size,
                 graphs=[]):
        self.node_number = node_number
        self.min_dist = min_dist
        self.radius = radius
        self.coverage_bound = coverage_bound
        self.avg_deg_bound = avg_deg_bound
        self.prob_connected = prob_connected
        self.sample_size = sample_size
        self.graphs = graphs

    def generate_graphs(self, sample_size: int, new: bool = True, connected: bool = False, bounds: bool = False):
        """
        Generates graphs with the given properties

        :param sample_size:     number of graphs to generate
        :param new:             whether to generate new graphs or use the ones already generated
        :param connected:       whether to generate connected graphs or not
        :param bounds:          whether to take bounds into account when accepting or rejecting graphs
        """
        if new:
            self.graphs = []
        generator = LambdaPrecisionUDGGenerator(RandomPointsGenerator(self.node_number, self.min_dist), self.radius)
        while len(self.graphs) < sample_size:
            print("Generate graphs...")
            graphs = generator.generate_graphs_parallel(max(sample_size - len(self.graphs), 10), connected=connected)
            for graph in graphs:
                self.graphs.append(graph)
            print(
                f"Graph: Node Number: {self.node_number}, Lambda: {self.min_dist}, Radius: {self.radius}, Avg Degree: {str(self.avg_deg_bound)}")
            print(f"Connected: {sum([nx.is_connected(graph.graph) for graph in self.graphs])}")
            for graph in self.graphs:
                print(f"Average Deg: {self.avg_deg_bound[0]} <= {graph.average_degree()} <= {self.avg_deg_bound[1]}")
                print(
                    f"Coverage: {self.coverage_bound[0]} <= {graph.lambda_precision_points.get_density()} <= {self.coverage_bound[1]}")
            if bounds:
                self.graphs = list(filter(
                    lambda graph: self.avg_deg_bound[0] <= graph.average_degree() <= self.avg_deg_bound[1] and
                                  self.coverage_bound[0] <= graph.lambda_precision_points.get_density() <=
                                  self.coverage_bound[1],
                    self.graphs))
            print(f"Graphs successfully generated: {len(self.graphs)} / {sample_size}")
        self.graphs = self.graphs[:sample_size]
        # print(f"last created graph: {graphs[-1]}")
        # print(f"len graphs: {len(self.graphs)}")
        # print(f"graphs: {str(self.graphs)}")

    def copy(self) -> "UDGGeneratorSeed":
        return UDGGeneratorSeed(self.node_number, self.min_dist, self.radius, self.coverage_bound, self.avg_deg_bound,
                                self.prob_connected, self.sample_size)

    def probability_connected(self):
        self.prob_connected = np.mean([nx.is_connected(graph.graph) for graph in self.graphs])
        return self.prob_connected

    def get_avg_coverage(self):
        return np.mean([graph.lambda_precision_points.get_density() for graph in self.graphs])

    def get_avg_degree(self):
        return np.mean([graph.average_degree() for graph in self.graphs])

    def degree_distribution(self, sample_size: int = 0, new: bool = False) -> list[list[int]]:
        """
        Computes the node degree distribution of the graphs in the seed

        :param sample_size:     number of graphs to compute the node degree distribution for
        :param new:             whether to generate new graphs or use the ones already generated
        :return:                node degree distribution
        """
        if not sample_size:
            sample_size = self.sample_size
        self.generate_graphs(sample_size, new)
        return [list(np.array(graph.graph.degree())[:, 1]) for graph in self.graphs[:sample_size]]

    def median_node_degree_distribution(self, sample_size: int = 0) -> list[int]:
        """
        Computes the median of the node degree distribution of the graphs in the seed

        :param sample_size:     number of graphs to compute the median of the node degree distribution for
        :return:                median of the node degree distribution
        """
        return list(map(np.median, self.degree_distribution(sample_size)))

    def variance_node_degree_distribution(self, sample_size: int = 0) -> list[float]:
        """
        Computes the variance of the node degree distribution of the graphs in the seed

        :param sample_size:     number of graphs to compute the variance of the node degree distribution for
        :return:                variance of the node degree distribution
        """
        # return np.var(self.degree_distribution(sample_size))
        return list(map(np.var, self.degree_distribution(sample_size)))

    def local_clustering(self, sample_size: int = 0, new: bool = False) -> list[dict[int, float]]:
        """
        Computes the local clustering coefficient of the graphs in the seed

        :param sample_size:     number of graphs to compute the local clustering coefficient for
        :param new:             whether to generate new graphs or use the ones already generated
        :return:                local clustering coefficient
        """
        if not sample_size:
            sample_size = self.sample_size
        self.generate_graphs(sample_size, new)
        return list(nx.clustering(graph.graph) for graph in self.graphs[:sample_size])

    def variance_local_clustering(self, sample_size: int = 0) -> list[float]:
        """
        Computes the variance of the local clustering coefficient of the graphs in the seed

        :param sample_size:     number of graphs to compute the variance of the local clustering coefficient for
        :return:                variance of the local clustering coefficient
        """
        local_clustering = self.local_clustering(sample_size)
        # return list(map(np.var, local_clustering))
        # print("===================================================")
        # print("Local Clustering")
        # print("===================================================")
        # print(local_clustering)
        # print(str(local_clustering))
        return [np.var(list(clustering.values())) for clustering in local_clustering]

    def median_local_clustering(self, sample_size: int = 0) -> float:
        """
        Computes the median local clustering coefficient of the graphs in the seed

        :param sample_size:     number of graphs to compute the median local clustering coefficient for
        :return:                median local clustering coefficient
        """
        return np.median(self.local_clustering(sample_size))

    def global_clustering(self, sample_size: int = 0, new: bool = False) -> float:
        """
        Computes the global clustering coefficient of the graphs in the seed

        :param sample_size: number of graphs to compute the global clustering coefficient for
        :param new:         whether to generate new graphs or use the ones already generated
        :return:            global clustering coefficient
        """
        if not sample_size:
            sample_size = self.sample_size
        self.generate_graphs(sample_size, new)
        return np.mean([nx.transitivity(graph.graph) for graph in self.graphs[:sample_size]])

    def __str__(self) -> str:
        """
        String representation of a UDGGeneratorSeed

        :return:    string representation of a UDGGeneratorSeed
        """
        return (f"node_number: {self.node_number}\n"
                f"min_dist: {self.min_dist}\n"
                f"radius: {self.radius}\n"
                f"coverage_bound: {self.coverage_bound}\n"
                f"avg_deg_bound: {self.avg_deg_bound}\n"
                f"prob_connected: {self.prob_connected}\n"
                f"sample_size: {self.sample_size}\n"
                f"graphs: {str((len(self.graphs), self.graphs))}")

    def serialize(self, path: str):
        """
        Serialize a UDGGeneratorSeed to a file

        :param path:    path to save the serialized UDGGeneratorSeed
        """
        import pickle

        print(f"UDGGeneratorSeed.serialize {path}/{id(self)}.pkl")
        with open(f"{path}/{id(self)}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(cls, filepath: str) -> "UDGGeneratorSeed":
        """
        Deserialize a UDGGeneratorSeed from a file

        :param filepath:    filepath to the serialized UDGGeneratorSeed
        :return:            UDGGeneratorSeed
        """
        import pickle

        print(f"UDGGeneratorSeed.deserialize {filepath}")
        with open(filepath, "rb") as file:
            return pickle.load(file)


class UDGSeedGenerator:
    def __init__(self, sample_size: int = 10):
        self.sample_size = sample_size

    def generate_seeds(self, avg_degs, coverage_bound: [float] = [0.9, 0.95],
                       node_numbers: [int] = [i for i in range(20, 320, 20)], padding: bool = True):
        """
        Generates seeds for the UDGGenerator

        :param avg_degs: average degrees to generate seeds for
        :param coverage_bound: coverage bounds to generate seeds for
        :param node_numbers: node numbers to generate seeds for

        :return: graph_generation seeds to produce graphs with the given properties
        """

        def approach_value_range(input: dict, output: dict, target: float):
            """
            Approach a value range by halving the input range
            f(input) = output < target?

            :param input:   dict with keys "lower", "upper", "result"
            :param output:  dict with keys "lower", "upper", "result"
            :param target:  target value to approach
            """
            if output["result"] < target:
                input["lower"] = input["result"]
            else:
                input["upper"] = input["result"]
            input["result"] = (input["lower"] + input["upper"]) / 2.0

        def determine_coverage(min_dist: dict, coverage: dict, node_number: int, padding: bool = True):
            """

            """
            print(f'{node_number}:\tmin_dist: {min_dist["result"]:.6f},\
                    \t\t coverage: {coverage["result"]:.6f}')
            # sd = 1.0
            result_min_dist = 0.0
            coverage_padding = (coverage_bound[1] - coverage_bound[0]) * 0.25 if padding else 0
            while not coverage_bound[0] <= coverage["result"] <= coverage_bound[
                1] - 2 * coverage_padding:  # or sd > 0.025:
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

        def determine_avg_deg(radius: dict, min_dist: dict, point_sets, node_number: int,
                              generator_seeds: ["UDGGeneratorSeed"], avg_deg_margin: float = 0.25):
            """
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

        def generate_point_distributions(min_dist: dict, node_number: int, point_sets: [] = []):
            """
            Generates point distributions for a given node number and min_dist

            :param min_dist:        dict with keys "lower", "upper", "result"
            :param node_number:     number of points in the point distribution
            :param point_sets:      list of point distributions

            :return: point_sets:    list of point distributions
            """
            generator = RandomPointsGenerator(point_number=node_number, min_dist=min_dist["result"], )
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
