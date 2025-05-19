import numpy as np

from src.graph_generator.seeds.seed import UDGGeneratorSeed


class UDGGeneratorSeedDB:
    """ Handles the generation and manipulation of UDGGeneratorSeed objects for various graph configurations.

    This class provides methods for appending new seeds, generating plots related to clustering
    and degree distributions, creating LaTeX tables for data summaries, and serialising/deserialising
    seed instances for persistence. Its primary use case is to manage a collection of graph
    generator seeds and generate analytical outputs in formats suitable for visualisation or
    integration with LaTeX documents.

    Attributes:
        seeds (list[UDGGeneratorSeed]): A list that contains instances of UDGGeneratorSeed managed by this generator.
    """

    def __init__(self, *seeds: "UDGGeneratorSeed") -> None:
        """ Initialises the class with a collection of seeds provided as arguments.

        Args:
            seeds: A variable-length argument list where each entry must be an instance of UDGGeneratorSeed.
        """

        self.seeds = [*seeds]

    def append(self, *seeds: "UDGGeneratorSeed"):
        """ Appends one or multiple `UDGGeneratorSeed` instances to the existing list of seeds.

        This method takes a variable number of `UDGGeneratorSeed` instances and appends them to the `seeds` list stored within the object. It modifies the internal state of the object by extending the `seeds` list.

        Args:
            seeds: A variable number of seed instances of type `UDGGeneratorSeed` that should be added to the internal list of seeds.
        """

        self.seeds += [*seeds]

    def mean_var_local_clustering(self, filepath: str):
        """ Generates and saves a plot representing the mean variance of local clustering coefficients as a function of coverage bounds for specific parameter groups.

        The function iterates through unique sets of parameters (node numbers and average degree bounds) and computes the mean variance of local clustering coefficients for each corresponding coverage bound. It generates a plot using these values, visualising the relationship and layout based on a predefined style. The plot is then saved in both .tikz and .png formats.

        Args:
            filepath: The file path (without file extension) where the plot and TikZ file will be saved. A .tikz and .png file will be generated accordingly.
        """

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

    def mean_var_deg_distribution(self, filepath: str) -> None:
        """ Generates and saves a plot showing the relationship between the coverage bound and
        the mean variance of the node degree distribution for a specific graph/seed configuration.
        The plot is generated for various combinations of node number and average degree bounds,
        saving the visuals in TikZ/PGFPlots format and as a PNG image.

        Args:
            filepath: The path where the generated plot will be saved. Uses TikZ/PGFPlots format for LaTeX inclusion and additionally saves as a PNG image.
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
        """ Generates a LaTeX formatted table representing statistical data calculated from seeds, which include attributes
        such as node number, average degree, radius, mean coverage, and probability of being connected. The table is
        constructed with specific column headers formatted for LaTeX and can optionally be written to a file.

        Args:
            filepath: The file path (without extension) where the LaTeX formatted table should be saved. If None, the table is not written to a file.

        Returns:
            A string containing the LaTeX formatted table.
        """

        from tabulate import tabulate
        # node number, deg_exp, lambda, r_tr, mean A_cov, mean deg_avg, P_connected
        table = [[r"\raggedleft $|V|$", r"\raggedleft $\deg_\exp$", r"\raggedleft $\lambda$", r"\raggedleft $r_{tr}$",
                  r"\raggedleft $\overline{A_\text{cov}}$", r"\raggedleft $\overline{\deg_\text{avg}}$",
                  r"\raggedleft $P_{connected}$", ], ]
        data = sorted([[seed.node_number, seed.avg_deg_bound[0], seed.min_distance, seed.radius,
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
        """ Generates a LaTeX formatted table containing statistical coverage metrics and writes it to a file
        if a filepath is specified. The table includes details such as node number, degree exponent, lambda,
        radius, mean coverage, mean average degree, and probability of connectedness.

        Args:
            filepath: Optional filepath to save the generated LaTeX table to a .tex file. If None, the table will not be written to a file.

        Returns:
            A string containing the LaTeX formatted table.
        """

        from tabulate import tabulate
        # node number, deg_exp, lambda, r_tr, mean A_cov, mean deg_avg, P_connected
        table = [[r"\raggedleft $|V|$", r"\raggedleft $\deg_\exp$", r"\raggedleft $\lambda$", r"\raggedleft $r_{tr}$",
                  r"\raggedleft $\overline{A_\text{cov}}$", r"\raggedleft $\overline{\deg_\text{avg}}$",
                  r"\raggedleft $P_{connected}$", ], ]
        data = sorted([[seed.coverage_bound, seed.node_number, seed.avg_deg_bound[0], seed.min_distance, seed.radius,
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
        """
        Serialise all seeds to the specified directory path.

        This method ensures that the directory exists before serialising the seeds.
        If the directory does not exist, it will be created automatically. Each seed in the collection will then be serialised to the given path.

        Args:
            path: The file system path where the seeds will be serialised. If the directory does not exist, it will be created.
        """

        import os

        if not os.path.exists(path):
            os.makedirs(path)
        for seed in self.seeds:
            seed.serialize(path=path)

    @classmethod
    def deserialize(cls, path: str) -> "UDGGeneratorSeedDB":
        """ Deserialise a directory containing serialised objects. It reads the specified
        directory, locates all files with a `.pkl` extension, and attempts to deserialise these files. Each deserialised object is then used as an argument to instantiate a new instance of the class.

        This method ensures to verify the existence of the provided path before proceeding
        with deserialization.

        Args:
            path: Path to the directory containing the serialised `.pkl` files. Must be a valid existing directory.

        Returns:
            An instance of the class created by passing all deserialised objects as arguments, or None if the directory does not exist.
        """

        import os

        if not os.path.exists(path):
            print(f"Filepath {path} does not exist.")
            return
        pkls = [f for f in os.listdir(path) if f.endswith(".pkl")]

        # Update the import statement to reflect the new module name 'graph_generator'
        # Temporary fix for the import error
        return cls(*[UDGGeneratorSeed.deserialize(os.path.join(path, j)) for j in pkls])
