import pyomo.environ as pyo
import numpy as np

__all__ = [
    "MinErrorsResult",
    "MinIncompleteNodesResult",
    "MinVarianceResult",
    "MinSpreadResult",
    "PartitioningResultDB",
    "MinSpreadResourceResult",
]


class PartitioningResultDB:
    def __init__(self, *results):
        self.results = list(results)

    def append(self, *results):
        """
        Appends results to the result list.

        :param results:     PartitioningResult
        """

        list(map(self.results.append, results))

    def serialize(self, path, compress: bool = True):
        """
        Serializes all results in the result list.

        :param path:    path to the directory where the results should be stored
        :param compress: whether the files should be compressed
        """
        import os

        if not os.path.exists(path):
            os.makedirs(path)
        for result in self.results:
            result.serialize(path=path, compress=compress)

    @classmethod
    def deserialize(cls, path, compressed=True):
        """
        Deserializes all results in the result list.

        :param path:    path to the directory where the results are stored
        :return:        populated PartitioningResultDB
        """
        import os

        if not os.path.exists(path):
            print(f"Filepath {path} does not exist.")
            return
        if compressed:
            pkls = [f for f in os.listdir(path) if f.endswith(".pkl.xz")]
            return cls(
                *[PartitioningResult.deserialize(os.path.join(path, j)) for j in pkls]
            )
        else:
            pkls = [f for f in os.listdir(path) if f.endswith(".pkl")]
            return cls(
                *[PartitioningResult.deserialize(os.path.join(path, j)) for j in pkls]
            )

    def _median_computation_time(self, partition_size):
        """
        Returns the median computation time for each number of nodes and average degree.

        :param partition_size:  partition size
        :return:                dict with data, x_label, y_label, title, legend
        """
        data = np.array([
            [result.wallclock_time,
             len(result.lambda_precision_points.points),
             result.seed.avg_deg_bound[0]]
            for result in self.results if result.partition_size == partition_size
        ])
        # print(f"Number of results: {str(self.results)}")
        return {
            "data": [
                [
                    [[
                        int(nodes),
                        np.mean([
                            float(time)
                            for time, num_nodes, avg_deg in data
                            if avg_deg == deg and num_nodes == nodes
                        ])
                        # np.median([
                        #     float(time)
                        #     for time, num_nodes, avg_deg in data
                        #     if avg_deg == deg and num_nodes == nodes
                        # ])
                    ] for nodes in set(data[:, 1])],
                    int(deg)
                ] for deg in set(data[:, 2])
            ],  # data [[[node_number, median(time)], deg]]
            "x_label": r"number of nodes $|V|$",
            "y_label": r"mean computation time in $s$",
            # "y_label": r"median computation time in $s$",
            "title": r'{0} {1}-soft domatic partition',
            "legend": r'$\deg_{{exp}}=${0}'
        }

    def _mean_errors(self, partition_size):
        """
        Returns the mean number of errors for each number of nodes and average degree.

        :param partition_size:  partition size
        :return:                dict with data, x_label, y_label, title, legend
        """
        data = np.array([
            [len(result.lambda_precision_points.points),
             result.number_of_errors,
             result.seed.avg_deg_bound[0],
             result.opt_type]
            for result in self.results if result.partition_size == partition_size
        ])
        return {
            "data": [
                [
                    [[
                        int(nodes),
                        np.mean([
                            float(num_errors)
                            for num_nodes, num_errors, avg_deg, opt_type in data
                            if avg_deg == deg and num_nodes == nodes and opt_type == opt
                        ])
                    ] for nodes in set(data[:, 0])],
                    int(deg),
                    opt
                ] for deg in set(data[:, 2]) for opt in set(data[:, 3])
            ],  # [[[node_number, mean(num_errors)], deg, opt]]
            "x_label": r"number of nodes $|V|$",
            "y_label": r"mean number of errors",
            "title": r'{0} {1}-soft domatic partition',
            "legend": r'$\deg_{{exp}}=${0}, {1}'
        }

    def _mean_inc_nodes(self, partition_size):
        """
        Returns the mean number of incompletely covered nodes for each number of nodes and average degree.

        :param partition_size:  partition size
        :return:                dict with data, x_label, y_label, title, legend
        """
        data = np.array([
            [len(result.lambda_precision_points.points),
             result.incomplete_nodes,
             result.seed.avg_deg_bound[0],
             result.opt_type]
            for result in self.results if result.partition_size == partition_size
        ])
        # print(f"Results Inc Nodes: {str(self.results)}")
        return {
            "data": [
                [
                    [[
                        int(nodes),
                        np.mean([
                            float(inc_nodes)
                            for num_nodes, inc_nodes, avg_deg, opt_type in data
                            if avg_deg == deg and num_nodes == nodes and opt_type == opt
                        ])
                    ] for nodes in set(data[:, 0])],
                    int(deg),
                    opt
                ] for deg in set(data[:, 2]) for opt in set(data[:, 3])
            ],
            "x_label": r"number of nodes $|V|$",
            "y_label": r"mean number of incompletely covered nodes",
            "title": r'{0} {1}-soft domatic partition',
            "legend": r'$\deg_{{exp}}=${0}, {1}'
        }

    def _extract_data(self, key, partition_size):
        """
        Extracts data from the result list.

        :param key:             key to select the data extraction method
        :param partition_size:  partition size
        :return:                dict with data, x_label, y_label, title, legend
        """
        return [
            self._median_computation_time,
            self._mean_errors,
            self._mean_inc_nodes
        ][key](partition_size)

    def plot(self, opt, data_key, partition_size, filepath):
        if opt in ["opt", "max"]:
            self.plot_opt_max(data_key, partition_size, filepath)
        if opt in ["var", "spread"]:
            self.plot_var_spread(data_key, partition_size, filepath)

    def plot_opt_max(self, data_key, partition_size, filepath):
        """
        Plots the data extracted from the result list.

        :param data_key:        key to select the data extraction method
        :param partition_size:  partition size
        :param filepath:        path to the file where the plot should be stored
        :return:                dict with data, x_label, y_label, title, legend
        """
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        # import tikzplotlib as tpl
        from tikzplotlib import save as tikz_save

        data = self._extract_data(data_key, partition_size)

        coords = data["data"]  # [[[n*[node_number, median(time)]], deg]]
        plt.style.use("ggplot")
        plt.figure()
        print(f"Plot Coords: {coords}")
        if len(coords[0]) == 2:
            for deg in sorted(set([coord[1] for coord in coords])):
                label = data["legend"].format(deg)
                # print(f"Plot Label: {label}")
                coord_set = sorted([
                                       coord[0] for coord in coords if coord[1] == deg
                                   ][0], key=lambda x: x[0])
                x = [coord[0] for coord in coord_set]
                y = [coord[1] for coord in coord_set]
                plt.plot(x, y, label=label, lw=2)
        else:
            for deg, opt in sorted(
                    set([tuple([coord[1], coord[2]]) for coord in coords])
            ):
                label = data["legend"].format(deg, opt)
                # print(f"Plot Label: {label}")
                coord_set = sorted([
                                       coord[0] for coord in coords if coord[1] == deg and coord[2] == opt
                                   ][0], key=lambda x: x[0])
                x = [coord[0] for coord in coord_set]
                y = [coord[1] for coord in coord_set]
                plt.plot(x, y, label=label, lw=2)

        # plt.legend(ncol=1)
        # plt.title(data["title"].format(
        #     "optimal" if data["opt_type"] == "opt" else "maximal", partition_size))
        plt.xlabel(data["x_label"])
        plt.ylabel(data["y_label"])

        positions = list(set([coord[0] for coord in coords[0][0]]))
        plt.xticks(positions)
        plt.grid(True)
        # plt.show()

        # tpl.save(filepath)
        tikz_save(filepath)
        plt.legend()
        plt.savefig(f'{filepath}.png', dpi=300)
        plt.close()

    def _table_data_opt_max(self, miss_cov=True, inc_nodes=True, var=True):
        from collections import defaultdict

        table = [[r"\raggedleft $\deg_\text{exp}$", r"\raggedleft $n$", r"\raggedleft $|V|$", ]]
        rows = []

        index = 0
        error_index = index
        if miss_cov:
            table[0] += [
                r"\raggedleft $\overline{e_\text{miss\_cov}^\text{non\_optimal}}$",
                r"\raggedleft $\overline{e_\text{miss\_cov}^\text{optimal}}$",
            ]
            rows += [[], []]
            index += 2

        inc_index = index
        if inc_nodes:
            table[0] += [
                r"\raggedleft $\overline{e_\text{inc\_nodes}^\text{non\_optimal}}$",
                r"\raggedleft $\overline{e_\text{inc\_nodes}^{\text{optimal}}}$",
            ]
            rows += [[], []]
            index += 2

        var_index = index
        if var:
            table[0] += [r"\raggedleft $\overline{\mathit{var}}$", ]
            rows += [[]]

        if miss_cov or inc_nodes:
            table[0] += [r"\raggedleft $\text{\#opt}$"]

        data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: rows)))

        for opt_type in ["opt", "max"]:
            for result in self.results:
                if result.opt_type != opt_type:
                    continue

                # node_count = result.graph.graph.number_of_nodes()
                node_count = len(result.lambda_precision_points.points)
                deg = result.seed.avg_deg_bound[0]
                partition_size = result.partition_size

                if miss_cov:
                    error_in = error_index + 1 if result.aborted else error_index
                    data[deg][partition_size][node_count][error_in].append(result.number_of_errors)
                if inc_nodes:
                    inc_in = inc_index + 1 if result.aborted else inc_index
                    data[deg][partition_size][node_count][inc_in].append(result.incomplete_nodes)
                if var:
                    data[deg][partition_size][node_count][var_index].append(
                        np.mean(list(result.variance_per_node.values())))

        sorted_data = {
            k1: {
                k2: {
                    k3: list(filter(None, [
                        np.mean(v3[min(error_index, index)]) if miss_cov else None,
                        np.mean(v3[min(error_index + 1, index)]) if miss_cov else None,
                        np.mean(v3[min(inc_index, index)]) if inc_nodes else None,
                        np.mean(v3[min(inc_index + 1, index)]) if inc_nodes else None,
                        np.mean(v3[min(var_index, index)]) if var else None,
                        len(v3[min(0, index)]) / (
                                len(v3[min(0, index)]) + len(v3[min(1, index)])) if miss_cov or inc_nodes else None
                    ]))
                    for k3, v3 in sorted(v2.items())
                }
                for k2, v2 in sorted(v1.items())
            }
            for k1, v1 in sorted(dict(data).items())
        }

        table_data = []
        for k1, v1 in sorted(sorted_data.items()):
            for k2, v2 in sorted(v1.items()):
                for k3, v3 in sorted(v2.items()):
                    table_data.append([k1, k2, k3] + v3)
        table += table_data
        return table

    def latex_table_opt_max(self, miss_cov=True, inc_nodes=True, var=True, filepath: str = None) -> str:
        """
        TODO Generates a latex table from the result list.

        :param miss_cov:    whether the missing coverage should be included
        :param inc_nodes:   whether the incomplete nodes should be included
        :param var:         whether the variance should be included
        :param filepath:    path to the file where the latex table should be stored
        :return:            latex table
        """
        from tabulate import tabulate
        table = self._table_data_opt_max(miss_cov, inc_nodes, var)

        latex_table = tabulate(table, headers="firstrow", tablefmt="latex_raw")
        if filepath:
            with open(f"{filepath}.tex", "w") as f:
                f.write(latex_table)
        return latex_table

    def _table_data_var_spread(self, nopt=False):
        """
        Prepares and structures the data included in the result table.

        :param node_res: optional parameter with default values to cover scenarios of evaluations of outdated results
        :return: structured data for the result table
        """
        from collections import defaultdict

        table = [
            [r"\raggedleft $|V|$", r"\raggedleft $\deg_\text{exp}$", r"\raggedleft $n$",
             fr"\raggedleft $\overline{{var_\text{{opt}}}}$",
             fr"\raggedleft $\overline{{var_\text{{nopt}}}}$",
             fr"\raggedleft $\overline{{spr_\text{{opt}}}}$",
             fr"\raggedleft $\overline{{spr_\text{{nopt}}}}$",
             r"\raggedleft $\overline{z_P}$",
             r"\raggedleft $\overline{z_D}$",
             r"\raggedleft $\overline{MIPGap}$",
             r"\raggedleft $\text{\#n}$",
             r"\raggedleft $\text{\#opt}$"] if nopt else
            [r"\raggedleft $|V|$", r"\raggedleft $\deg_\text{exp}$", r"\raggedleft $n$",
             fr"\raggedleft $\overline{{var}}$",
             fr"\raggedleft $\overline{{spr}}$",
             r"\raggedleft $\overline{z_{{P}}}$",
             r"\raggedleft $\overline{z_{{D}}}$",
             r"\raggedleft $\overline{MIPGap}$",
             r"\raggedleft $\text{\#n}$",
             r"\raggedleft $\text{\#opt}$"]
        ]

        data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[] for _ in range(len(table[0]) - 4)])))

        for result in self.results:
            node_count = len(result.lambda_precision_points.points)
            deg = result.seed.avg_deg_bound[0]
            partition_size = result.partition_size

            var_index = 1 if result.aborted else 0 if nopt else 0
            spr_index = 2 if result.aborted else 3 if nopt else 1

            data[node_count][deg][partition_size][var_index].append(
                np.mean(list(result.spread_per_node.values())))
            data[node_count][deg][partition_size][spr_index].append(
                np.mean(list(result.spread_per_node.values())))
            data[node_count][deg][partition_size][4 if nopt else 2].append(
                result.ubound if hasattr(result, "ubound") else float("nan"))  # lbound
            data[node_count][deg][partition_size][5 if nopt else 3].append(
                result.lbound if hasattr(result, "lbound") else float("nan"))  # ubound
            data[node_count][deg][partition_size][6 if nopt else 4].append(
                result.mipgap if hasattr(result, "mipgap") else float("nan"))  # mipgap
            data[node_count][deg][partition_size][7 if nopt else 5].append(
                np.var([sum(1 for (_, j) in result.partitioning if j == i) for i in range(result.partition_size)]))

        data = dict(data)

        sorted_data = {
            k1: {
                k2: {
                    k3: [np.mean(v3[i]) for i in range(len(table[0]) - 4)] + [len(v3[0]) / (len(v3[0]) + len(v3[1]))]
                    for k3, v3 in sorted(v2.items())
                }
                for k2, v2 in sorted(v1.items())
            }
            for k1, v1 in sorted(data.items())
        }

        table_data = []
        for k1, v1 in sorted(sorted_data.items()):
            for k2, v2 in sorted(v1.items()):
                for k3, v3 in sorted(v2.items()):
                    table_data.append([k1, k2, k3] + v3)

        table += table_data
        return table

    def latex_table_var_spread(self, filepath: str = None, nopt: bool = False) -> str:
        """
        TODO Generates a latex table from the result list.

        :param filepath:    path to the file where the latex table should be stored
        :param nopt:        whether between optimal and non optimal results should be differentiated
        :return:            latex table
        """
        from tabulate import tabulate

        table = self._table_data_var_spread(nopt=nopt)

        latex_table = tabulate(table, headers="firstrow", tablefmt="latex_raw")
        if filepath:
            with open(f"{filepath}.tex", "w") as f:
                f.write(latex_table)
        return latex_table

    def _table_data_spread_resources(self, node_res=(1.0, 1.0, 1.0)):
        """
        Prepares and structures the data included in the result table.

        :param node_res: optional parameter with default values to cover scenarios of evaluations of outdated results
        :return: structured data for the result table
        """
        from collections import defaultdict
        table = [[r"\raggedleft $\deg_\text{exp}$", r"\raggedleft $n$", r"\raggedleft $|V|$",
                  fr"\raggedleft $\overline{{{self.results[0].opt_type}_\text{{node_neighbourhood}}^\text{{optimal}}}}$",
                  fr"\raggedleft $\overline{{{self.results[0].opt_type}_\text{{node_neighbourhood}}^\text{{non\_optimal}}}}$",
                  r"\raggedleft $\overline{z_P}$", r"\raggedleft $\overline{z_D}$", r"\raggedleft $\overline{MIPGap}$",
                  r"\raggedleft $\overline{k_\text{res}}$",
                  r"\raggedleft $\text{\#opt}$"]]

        data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[], [], [], [], [], []])))

        for result in self.results:
            node_count = len(result.lambda_precision_points.points)
            deg = result.seed.avg_deg_bound[0]
            partition_size = result.partition_size

            residue_per_node = ([result.node_res, ]) * result.graph.graph.number_of_nodes()

            # Compute the remaining resources per node
            for node, mean_index in result.partitioning:
                residue_per_node[node] = [res - mean for res, mean in
                                          zip(residue_per_node[node], result.sm_perf_cost[mean_index - 1])]
            residue_per_node = [[round(res, 1) for res in residue_per_res] for residue_per_res in residue_per_node]

            var_index = 1 if result.aborted else 0
            data[deg][partition_size][node_count][var_index].append(
                np.mean(list(result.spread_per_node.values())))
            data[deg][partition_size][node_count][2].append(result.ubound)  # upper bound - primal objective for min
            data[deg][partition_size][node_count][3].append(result.lbound)  # lower bound
            data[deg][partition_size][node_count][4].append(result.mipgap)  # mipgap
            data[deg][partition_size][node_count][5].append(sum(sum(x) for x in residue_per_node))

        data = dict(data)

        sorted_data = {
            k1: {
                k2: {
                    k3: [np.mean(v3[0]), np.mean(v3[1]), np.mean(v3[2]), np.mean(v3[3]), np.mean(v3[4]), np.mean(v3[5]),
                         len(v3[0]) / (len(v3[0]) + len(v3[1]))]
                    for k3, v3 in sorted(v2.items())
                }
                for k2, v2 in sorted(v1.items())
            }
            for k1, v1 in sorted(data.items())
        }

        table_data = []
        for k1, v1 in sorted(sorted_data.items()):
            for k2, v2 in sorted(v1.items()):
                for k3, v3 in sorted(v2.items()):
                    table_data.append([k1, k2, k3] + v3)

        table += table_data
        return table

    def latex_table_spread_resources(self, filepath: str = None) -> str:
        """
        TODO Generates a latex table from the result list.

        :param filepath:    path to the file where the latex table should be stored
        :return:            latex table
        """
        from tabulate import tabulate

        table = self._table_data_var_spread()

        latex_table = tabulate(table, headers="firstrow", tablefmt="latex_raw")
        if filepath:
            with open(f"{filepath}.tex", "w") as f:
                f.write(latex_table)
        return latex_table

    def get_mean_incomplete_nodes(self):
        sorted_results = sorted(self.results, key=lambda result: (hash(result.seed), result.partition_size))
        return [result.incomplete_nodes for result in sorted_results]

    def get_mean_miss_cov(self):
        sorted_results = sorted(self.results, key=lambda result: (hash(result.seed), result.partition_size))
        return [(result.incomplete_nodes, hash(result.seed)) for result in sorted_results]

    def compare_mean_incomplete_nodes(self, db):
        # node_number = 0
        # self_data = [row[5] for row in self._table_data_opt_max()[1:]]
        # db_data = [row[5] for row in db._table_data_opt_max()[1:]]
        # zip_data = list(zip(self_data, db_data))
        # print(f"Self: {str(zip_data)}")
        # result = [np.abs(a - b) / node_number * 100 for a, b in zip_data]
        # print(f"Result: {result}")
        # print(f"Mean: {np.mean(result)}")
        results = []
        for i, row in enumerate(self._table_data_opt_max()[1:]):
            row_db = db._table_data_opt_max()[1:][i]
            node_number = row_db[2]
            results.append(np.abs(row[5] - row_db[5]) / node_number)
        result = np.mean(results) * 100
        print(f"Mean deviation incomplete nodes: {result}")

    def compare_mean_miss_cov(self, db):
        # partition_size = 0
        # node_number = 0
        # self_data = [row[3] for row in self._table_data_opt_max()[1:]]
        # db_data = [row[3] for row in db._table_data_opt_max()[1:]]
        # zip_data = list(zip(self_data, db_data))
        # print(f"Self: {str(zip_data)}")
        # result = [np.abs(a - b) / (partition_size * node_number) * 100 for a, b in zip_data]
        # print(f"Result: {result}")
        # print(f"Mean: {np.mean(result)}")

        results = []
        for i, row in enumerate(self._table_data_opt_max()[1:]):
            row_db = db._table_data_opt_max()[1:][i]
            partition_size = row_db[1]
            node_number = row_db[2]
            results.append(np.abs(row[3] - row_db[3]) / (partition_size * node_number))
        result = np.mean(results) * 100
        print(f"Mean deviation missing coverages: {result}")
        # print(result)


class PartitioningResult:
    def __init__(self, graph, result, model, partition_size, opt_type, seed):
        """
        Stores the result of a partitioning computation.

        :param graph:               graph to be partitioned
        :param result:              result of the partitioning computation
        :param partition_size:      partition size
        :param opt_type:            'opt' or 'max'
        :param seed:                seed used to generate the graph
        """
        self.graph = graph
        self.lambda_precision_points = graph.lambda_precision_points
        # self.radius = graph.radius
        # self.result = result
        # self.model = model
        self.partition_size = partition_size
        self.opt_type = opt_type  # 'opt' or 'max'
        # seed.graphs = []
        self.seed = seed.copy()
        self.wallclock_time = result.solver.wallclock_time
        self.aborted = result.solver.status == pyo.SolverStatus.aborted
        self.graph_id = hash(graph)

        # varobject = getattr(model, 'x')
        self.partitioning = [i for i in model.x if model.x[i].value == 1]
        print(f"Partitioning: {self.partitioning}")

        # Retrieve the lower bound
        self.lbound = result.Problem.lower_bound

        # Retrieve the upper bound
        self.ubound = result.Problem.upper_bound

        if self.ubound and self.ubound != 0:  # Check to avoid division by zero
            self.mipgap = abs(self.ubound - self.lbound) / abs(self.ubound)
        else:
            self.mipgap = float('inf')

    def graph(self):
        from src.graph_generation.lambda_precision_udg_generator import LambdaPrecisionUDG
        return LambdaPrecisionUDG(None, self.lambda_precision_points, self.seed.radius)

    def __str__(self):
        return f"PartitioningResult:\n\
            Partition size: {self.partition_size},\n\
            Wallclock time: {self.wallclock_time},\n\
            Aborted: {self.aborted}"

    def serialize(self, path, compress=True):
        """
        Serializes the result.

        :param path:    path to the directory where the result should be stored
        :param compress: whether the file should be compressed
        :return:        dict with data, x_label, y_label, title, legend
        """
        import pickle

        print(f"{path}/{id(self)}.pkl")

        if compress:
            import lzma
            with lzma.open(f"{path}/{id(self)}.pkl.xz", "wb") as f:
                pickle.dump(self, f)
        else:
            with open(f"{path}/{id(self)}.pkl", "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def deserialize(cls, filepath, compressed=True):
        """
        Deserializes the result.

        :param filepath:    path to the directory where the result is stored
        :param compressed:  whether the file is compressed
        :return:            PartitioningResult
        """
        import pickle

        if compressed:
            import lzma
            with lzma.open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)


class MinErrorsResult(PartitioningResult):
    def __init__(self, graph, result, model, partition_size, opt_type, seed):
        super().__init__(graph, result, partition_size, opt_type, seed)

        varobject = getattr(model, 'y')
        y = [i for i in varobject if varobject[i].value == 0]
        # number of errors overall
        self.number_of_errors = len(y)
        # number of incompletely covered nodes
        self.incomplete_nodes = len(set([i[0] for i in y]))
        print(f"Partitioning: {self.partitioning}")

        self.variance_per_node = {
            v: 1 / model.dom_num * sum(((model.node_degrees[v])
                                        / model.dom_num - sum(model.x[w, i].value
                                                              for w in [neighbours for neighbours in model.Nodes if
                                                                        model.links[v, neighbours] > 0])) ** 2
                                       for i in model.PartSize) for v in model.Nodes
        }

    def __str__(self):
        return f"PartitioningResult:\n\
            Partition size: {self.partition_size},\n\
            Errors: {self.number_of_errors},\n\
            Incomplete nodes: {self.incomplete_nodes},\n\
            Wallclock time: {self.wallclock_time},\n\
            Aborted: {self.aborted}"


class MinIncompleteNodesResult(MinErrorsResult):
    def __init__(self, graph, result, model, partition_size, opt_type, seed):
        super().__init__(graph, result, model, partition_size, opt_type, seed)


class MinVarianceResult(PartitioningResult):
    def __init__(self, graph, result, model, partition_size, opt_type, seed):
        super().__init__(graph, result, model, partition_size, opt_type, seed)

        self.objective = pyo.value(model.objective)

        neighbours = {v: [neighbours for neighbours in model.Nodes if model.links[v, neighbours] > 0] for v in
                      model.Nodes}
        self.variance_per_node = {v: 1 / model.part_size * sum(
            (len(neighbours[v]) / model.part_size - sum(model.x[w, i].value for w in neighbours[v])) ** 2 for i in
            model.PartSize) for v in model.Nodes}
        self.spread_per_node = {
            v: max(sum(model.x[w, i].value for w in neighbours[v]) for i in model.PartSize) - min(
                sum(model.x[w, i].value for w in neighbours[v]) for i in model.PartSize) for v in model.Nodes}

    def __str__(self):
        return f"PartitioningResult:\n\
                    Partition size: {self.partition_size},\n\
                    Objective (sum of variance): {self.objective},\n\
                    Sum of spread: {sum(self.spread_per_node.values())},\n\
                    Variance per node: {self.variance_per_node},\n\
                    Wallclock time: {self.wallclock_time},\n\
                    Aborted: {self.aborted}"


class MinSpreadResult(MinVarianceResult):
    def __init__(self, graph, result, model, partition_size, opt_type, seed):
        super().__init__(graph, result, model, partition_size, opt_type, seed)

    def __str__(self):
        return f"PartitioningResult:\n\
                    Partition size: {self.partition_size},\n\
                    Objective (sum of spread): {self.objective},\n\
                    Sum of variance: {sum(self.variance_per_node.values())},\n\
                    Variance per node: {str(self.variance_per_node)},\n\
                    Wallclock time: {self.wallclock_time},\n\
                    Aborted: {self.aborted}"


class MinSpreadResourceResult(MinSpreadResult):
    def __init__(self, graph, result, model, partition_size, opt_type, seed, sm_perf_cost, packings=None,
                 packings_matrix=None):
        super().__init__(graph, result, model, partition_size, opt_type, seed)
        self.sm_perf_cost = sm_perf_cost
        self.packings = packings
        self.packings_matrix = packings_matrix
        self.spread_per_node = {v: model.xh[v].value - model.xl[v].value for v in model.Nodes}

    def __str__(self):
        return f"PartitioningResult:\n\
            Partition size: {self.partition_size},\n\
            Objective (sum of spread): {sum(self.spread_per_node.values())},\n\
            Sum of variance: {sum(self.variance_per_node.values())},\n\
            Variance per node: {str(self.variance_per_node)},\n\
            Wallclock time: {self.wallclock_time},\n\
            Aborted: {self.aborted}"
