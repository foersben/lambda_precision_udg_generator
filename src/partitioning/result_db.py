from collections import defaultdict
from typing import Callable, TypedDict, Any

import numpy as np
import logging
import os
from enum import IntEnum, auto
from tabulate import tabulate
from src.partitioning.result import BaseResult

logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class DataKey(IntEnum):
    """ Represents enumeration for various data keys.

    This class is an enumeration that provides a set of categorised constants for various data-related purposes.

    Attributes:
        COMPUTATION_TIME: Represents the key for computation time data.
        ERRORS: Represents the key for the count of missing means in a nodes neighbourhood for complete coverage.
        INCOMPLETE_NODES: Represents the key for the count of incomplete nodes.
        VARIANCE: Represents the key for variance.
        SPREAD: Represents the key for spread.
    """

    COMPUTATION_TIME = auto()
    ERRORS = auto()
    INCOMPLETE_NODES = auto()
    VARIANCE = auto()
    SPREAD = auto()


DataDict = TypedDict(
    "DataDict",
    {
        "data": list[Any],
        "x_label": str,
        "y_label": str,
        "title": str,
        "legend": str
    })


class ResultDB:
    """ Manages a database of partitioning/assignment results, supports serialisation, deserialisation, and analysis methods.

    This class provides functionality for storing, manipulating, and analysing the results of partitioning/assignment processes. It includes methods for appending results, serialising them to disk, deserialising them back, and executing various analysis operations with extensive logging. This tool is intended for applications working with partitioning/assignment algorithms where analysis and persistence of results are required.

    Attributes:
        results (list[BaseResult]): Internal storage for PartitioningBaseResult objects.
    """

    _EXTRACTION: dict[DataKey, str] = {
        DataKey.COMPUTATION_TIME: "_compute_time_data",
        DataKey.ERRORS: "_errors_data",
        DataKey.INCOMPLETE_NODES: "_incomplete_nodes_data",
        DataKey.VARIANCE: "_variance_data",
        DataKey.SPREAD: "_spread_data",
    }

    def __init__(self, *results: BaseResult) -> None:
        """ Initialises the ResultDB instance with multiple PartitioningBaseResult objects.

        Args:
            results: The results to be added to the result database.
        """

        self.results = list(results)
        logger.info(f"Created result DB with {len(self.results)} results")

    def append(self, *results: BaseResult) -> None:
        """ Appends one or more PartitioningBaseResult objects to the internal storage.

        This method adds the given results to an internal list responsible for tracking PartitioningBaseResult instances. It logs the number of results added and provides the updated total count of results.

        Args:
            results: One or more PartitioningBaseResult objects to be added to the list.
        """

        self.results.extend(results)
        logger.info(f"Added {len(results)} results, total now {len(self.results)}")

    def serialize(self, path: str, compress: bool = True) -> None:
        """ Serialise all results to files

        Args:
            path: Directory to save files
            compress: Whether to use compression
        """
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Serializing {len(self.results)} results to {path}")

            for result in self.results:
                result.serialize(path, compress)

            logger.info(f"Successfully serialized all results")
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise

    @classmethod
    def deserialize(cls, path: str, compressed: bool = True) -> 'ResultDB':
        """ Deserialise database from directory

        Args:
            path: Directory containing result files
            compressed: Whether files are compressed

        Returns:
            Populated result database
        """

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory {path} not found")

            logger.info(f"Deserializing results from {path}")

            # Find result files
            if compressed:
                files = [f for f in os.listdir(path) if f.endswith(".pkl.xz")]
            else:
                files = [f for f in os.listdir(path) if f.endswith(".pkl")]

            results = []
            for filename in files:
                filepath = os.path.join(path, filename)
                result = BaseResult.deserialize(filepath, compressed)
                results.append(result)

            logger.info(f"Deserialized {len(results)} results")
            return cls(*results)
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise

    def _extract_data(self, key: DataKey, partition_size: int) -> DataDict:
        """ Extracts data using the specified key and partition size.

        This private method retrieves data based on the given `key` by determining the appropriate extraction method and delegating the task to the respective extractor function. The `partition_size` parameter determines the size or limit of data to be extracted. If the `key` is not found within the predefined extraction mapping, a `ValueError` is raised.

        Args:
            key: The identifier used to determine the appropriate extraction method.
            partition_size: The size or limit of the data to be extracted.

        Returns:
            The extracted data in the form of a DataDict structure.

        Raises:
            ValueError: If the specified `key` is invalid or not found in the extraction mapping.
        """

        try:
            method_name = self._EXTRACTION[key]
        except KeyError:
            raise ValueError(f"Invalid data key: {key}") from None

        extractor = getattr(self, method_name)
        return extractor(partition_size)

    def _compute_time_data(self, partition_size: int) -> dict[str, Any]:
        """ Computes and returns time-related data filtered by a specific partition size from the results. This method collects relevant metrics from the computation, such as wallclock time, graph node count, and average degree bound, for further analysis and visualisation.

        The function logs a warning and returns an empty dictionary if no matching data is found for the given partition size.

        Args:
            partition_size: The size of the partition to filter the result for.

        Returns:
            A dictionary containing grouped computation time data, indexed by descriptive labels.
        """

        filtered = [
            (len(result.graph.nodes),  # x_value: node count
             result.wallclock_time,  # y_value: computation time
             result.seed.avg_deg_bound[0])  # group_key: degree
            for result in self.results
            if result.partition_size == partition_size
        ]
        logger.warning(f"Filtered {len(filtered)} results for partition size {partition_size}")
        return self._group_data(
            filtered,
            r"mean computation time in $s$",
            lambda deg: f'$\\deg_{{exp}}={deg}$'
        )

    # def _compute_time_data(self, partition_size: int) -> dict[str, Any]:
    # """ Computes and returns time-related data filtered by a specific partition size from the results. This method collects relevant metrics from the computation, such as wallclock time, graph node count, and average degree bound, for further analysis and visualisation.

    # The function logs a warning and returns an empty dictionary if no matching data is found for the given partition size.

    # Args:
    #     partition_size: The size of the partition to filter the result for.

    # Returns:
    #     A dictionary containing grouped computation time data, indexed by descriptive labels.
    # """

    #     filtered = [(result.wallclock_time, len(result.graph.nodes), result.seed.avg_deg_bound[0]) for result in
    #                 self.results if result.partition_size == partition_size]
    #     logger.warning(f"Filtered {len(filtered)} results for partition size {partition_size}")

    #     return self._group_data(filtered, r"mean computation time in $s$", lambda deg: f'$\\deg_{{exp}}={deg}$')

    def _errors_data(self, partition_size: int) -> dict[str, Any]:
        """ Filters and processes error data for a specific partition size and groups the data for analysis and visualisation purposes. The function extracts nodes, number of errors, degree bounds, and optimisation type for results meeting the specified partition size. If no data matches the filter criteria, a warning is logged and an empty dictionary is returned.

        Args:
            partition_size: The size of the partitions for results filtering.

        Returns:
            A dictionary containing grouped error data labelled by the mean number of errors and other distinguishing factors.
        """

        filtered = [(len(result.graph.nodes), result.calculate_errors(), result.seed.avg_deg_bound[0], result.opt_type)
                    for result in self.results if result.partition_size == partition_size]

        logger.warning(f"Filtered {len(filtered)} results for partition size {partition_size}")
        return self._group_data(
            filtered,
            r"mean number of errors",
            lambda deg, opt: f'$\\deg_{{exp}}={deg}$, {opt}'
        )

    def _incomplete_nodes_data(self, partition_size: int) -> dict[str, Any]:
        """ Extracts and groups data related to incomplete nodes from the results based on the provided partition size.

        The method filters the results for entries with a specific partition size and ensures they include data on incomplete nodes. If no matching data is found, a warning is logged, and an empty dictionary is returned. For valid entries, it processes and organises the data by applying a grouping function and formatting the results.

        Args:
            partition_size: The specific size of the partition used to filter the results.

        Returns:
            A dictionary containing grouped and processed data on incomplete nodes categorised by certain criteria based on degrees and optimisation types, or an empty dictionary if no relevant data is available.
        """

        filtered = [
            (len(result.graph.nodes),
             result.calculate_incomplete_nodes(),
             result.seed.avg_deg_bound[0],
             result.opt_type)
            for result in self.results if result.partition_size == partition_size
        ]

        return self._group_data(
            filtered,
            r"mean number of incompletely covered nodes",
            lambda deg, opt: f'$\\deg_{{exp}}={deg}$, {opt}'
        )

    def _variance_data(self, partition_size: int) -> dict[str, Any]:
        """ Computes and filters variance data based on the given partition size and aggregates the results into a structured dictionary format. The filtered results include only those that match the specified partition size.

        Args:
            partition_size: The size of the partition to filter results by.

        Returns:
            A dictionary containing grouped variance data, aggregated into a specific format, or an empty dictionary if no matching results are found.
        """

        filtered = []
        for result in self.results:
            if result.partition_size == partition_size:
                filtered.append((len(result.graph.nodes), result.calculate_variance(), result.seed.avg_deg_bound[0],
                                 result.opt_type))

        return self._group_data(
            filtered,
            r"mean variance per node",
            lambda deg, opt: f'$\\deg_{{exp}}={deg}$, {opt}'
        )

    def _spread_data(self, partition_size: int) -> dict:
        """ Filters and groups data based on the given partition size and the mean spread.

        The method iterates through the results and selects entries matching the specified partition size. The filtered data is then grouped using a helper method to create a structured dictionary for further analysis.

        Args:
            partition_size: The specific size of the partition to filter the results.

        Returns:
            A dictionary grouping the filtered data by the given criteria, or an empty dictionary if no matching data is found.
        """

        filtered = []
        for result in self.results:
            if result.partition_size == partition_size:
                filtered.append(
                    (len(result.graph.nodes), result.calculate_spread(), result.seed.avg_deg_bound[0], result.opt_type))

        return self._group_data(
            filtered,
            r"mean spread per node",
            lambda deg, opt: f'$\\deg_{{exp}}={deg}$, {opt}'
        )

    # def _group_data(self, filtered_data: list[tuple[int, float, int, str]], y_label: # str,
    #                 legend_formatter: Callable) -> dict[str, Any]:
    #     """ Groups filtered data for plotting.

    #     Args:
    #         filtered_data: A list of tuples, where each tuple contains (x_value, y_value, group_key).
    #         y_label: The label for the y-axis of the plot.
    #         legend_formatter: A function to format the legend label for each group.

    #     Returns:
    #         A dictionary formatted for the plotting function.
    #     """
    #     if not filtered_data:
    #         logger.warning("No data available for plotting.")
    #         return {"data": [], "x_label": "", "y_label": "", "title": "", "legend": ""}

    #     data_array = np.array(filtered_data)
    #     grouped_data = defaultdict(lambda: defaultdict(list))

    #     # Group data by the third element (deg) and then by the first (nodes)
    #     for row in data_array:
    #         nodes, value, deg = row[0], row[1], row[2]
    #         grouped_data[deg][nodes].append(value)

    #     # Calculate the mean for each group
    #     processed_data = []
    #     for deg, node_data in grouped_data.items():
    #         coords = sorted([
    #             [node, np.mean(values)] for node, values in node_data.items()
    #         ])
    #         processed_data.append([coords, deg])

    #     return {
    #         "data": processed_data,
    #         "x_label": r"number of nodes $|V|$",
    #         "y_label": y_label,
    #         "title": r'{0} {1}-soft domatic partition',
    #         "legend": legend_formatter
    #     }

    def _group_data(self, filtered_data: list[tuple], y_label: str,
                    legend_formatter: Callable) -> dict[str, Any]:
        """ Groups filtered data for plotting.

        Args:
            filtered_data: A list of tuples, where each tuple contains (x_value, y_value, group_key1, group_key2, ...).
            y_label: The label for the y-axis of the plot.
            legend_formatter: A function to format the legend label for each group.

        Returns:
            A dictionary formatted for the plotting function.
        """

        if not filtered_data:
            logger.warning("No data available for plotting.")
            return {"data": [], "x_label": "", "y_label": "", "title": "", "legend": ""}

        # Group data by all keys except the first two (x_value, y_value)
        grouped_data = defaultdict(lambda: defaultdict(list))
        for row in filtered_data:
            nodes = row[0]
            value = row[1]
            group_key = tuple(row[2:])
            grouped_data[group_key][nodes].append(value)

        # Calculate the mean for each group and sort by group key
        processed_data = []
        for group_key, node_data in grouped_data.items():
            coords = sorted([
                [node, np.mean(values)] for node, values in node_data.items()
            ], key=lambda x: x[0])  # Sort by node count
            processed_data.append([coords, group_key])

        # Sort groups by first element of group key (degree)
        processed_data.sort(key=lambda x: x[1][0] if isinstance(x[1], tuple) and len(x[1]) > 0 else x[1])

        return {
            "data": processed_data,
            "x_label": r"number of nodes $|V|$",
            "y_label": y_label,
            "title": r'{0} {1}-soft domatic partition',
            "legend": legend_formatter
        }

    def plot(self, data_key: DataKey, partition_size: int, filepath: str = None) -> None:
        """ Generates and saves a plot based on given data key and partition size.

        The function extracts the data corresponding to the `data_key` while considering the specified `partition_size`. If the data extraction produces no results, the plotting process is terminated with a logged message. Otherwise, the plot is generated and saved to the provided `filepath`. Logs corresponding details or errors throughout the process.

        Args:
            data_key: The key identifying the data to be extracted for plotting.
            partition_size: The partition size used to filter or process the data.
            filepath: The file path where the plot will be saved.
        """

        logger.info(f"Generating plot for key {data_key}, size {partition_size}")

        data: DataDict = self._extract_data(data_key, partition_size)
        if not data:
            logger.error("No data available for plotting")
            return

        try:
            self._generate_plot(data, filepath)
            logger.info(f"Saved plot to {filepath}.*")
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

    def _generate_plot(self, data: DataDict, filepath: str = None):
        """ Generates a plot from the provided data and saves it in multiple formats including .tex, .png, and .pdf. This function is designed to work in a headless environment and utilises the 'Agg' backend for matplotlib. It plots multiple data series with provided labels, applies a specific style, and includes comprehensive elements such as a grid, title, axis labels, and legend.

        Args:
            data: A DataDict containing the data series and metadata for the plot.
            filepath: The base path for saving the plot files. If None, the plot is displayed interactively.
        """

        import matplotlib
        matplotlib.use('Agg')  # Use 'Agg' backend for headless environments

        import matplotlib.pyplot as plt
        from tikzplotlib import save as tikz_save

        plt.figure(figsize=(10, 6))
        plt.style.use("ggplot")

        all_x_values = set()

        # Plot each series
        # for series_data, legend_label in data["data"]:
        #     if not series_data:
        #         continue
        #     coord_set = sorted(series_data, key=lambda p: p[0])
        #     x = [coord[0] for coord in coord_set]
        #     y = [coord[1] for coord in coord_set]
        #     all_x_values.update(x)
        #     plt.plot(x, y, label=legend_label, lw=2, marker='o', linestyle='-')

        # Plot each series
        for series_data, group_key in data["data"]:
            if not series_data:
                continue
            coord_set = sorted(series_data, key=lambda p: p[0])
            x = [coord[0] for coord in coord_set]
            y = [coord[1] for coord in coord_set]
            all_x_values.update(x)

            # Handle different group key types
            if isinstance(group_key, tuple):
                legend_label = data["legend"](*group_key)
            else:
                legend_label = data["legend"](group_key)

            plt.plot(x, y, label=legend_label, lw=2, marker='o', linestyle='-')

        plt.xlabel(data["x_label"])
        plt.ylabel(data["y_label"])
        # plt.title(data["title"]) # Title formatting seems to be missing arguments
        plt.grid(True)
        plt.legend()
        if all_x_values:
            plt.xticks(sorted(list(all_x_values)))

        # Save in multiple formats
        if filepath:
            try:
                tikz_save(f"{filepath}.tex")
                plt.savefig(f'{filepath}.png', dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {filepath}.tex and {filepath}.png")
            except Exception as e:
                logger.error(f"Error saving plot to {filepath}: {e}")
        # plt.show(block=True)
        plt.close()

    def _table_data(
            self,
            eval_data: list[DataKey],
            eval_method: str = "mean",
            distinguish_optimality: bool = False
    ) -> list[list[Any]]:
        """ Generates a table body based on chosen information to evaluate (`eval_data`) and some default information like the computation time and percentage of optimal and non-optimal results grouped by deg, partition size and node count in this order.

        This private method constructs a table data structure from the provided evaluation data and method. It supports distinguishing optimality if specified. The resulting table is structured as a list of lists, where each inner list represents a row in the table.

        Args:
            eval_data: List of DataKey values representing the evaluation data.
            eval_method: Method to use for evaluation (default: "mean").
            distinguish_optimality: Whether to distinguish between optimal and non-optimal results (default: False).

        Returns:
            A list of lists representing the table data.
        """

        # Mapping from DataKey to extraction method
        data_extractors = {
            DataKey.ERRORS: lambda r: r.calculate_errors(),
            DataKey.INCOMPLETE_NODES: lambda r: r.calculate_incomplete_nodes(),
            DataKey.VARIANCE: lambda r: r.calculate_variance(),
            DataKey.SPREAD: lambda r: r.calculate_spread(),
            DataKey.COMPUTATION_TIME: lambda r: r.wallclock_time
        }

        # Statistical function mapping
        stat_func = np.mean if eval_method == "mean" else np.median

        # Create header
        header = [
            r"\raggedleft $\deg_\text{exp}$",
            r"\raggedleft $n$",
            r"\raggedleft $|V|$"
        ]

        # Add metric headers
        metric_names = {
            DataKey.ERRORS: r"$e_{\text{miss\_cov}}$",
            DataKey.INCOMPLETE_NODES: r"$e_{\text{inc\_nodes}}$",
            DataKey.VARIANCE: r"$\overline{\mathit{var}}$",
            DataKey.SPREAD: r"$\overline{\mathit{spread}}$",
            DataKey.COMPUTATION_TIME: r"$t_{\text{comp}}$"
        }

        for key in eval_data:
            if distinguish_optimality:
                header.append(metric_names[key] + r"$_{\text{opt}}$")
                header.append(metric_names[key] + r"$_{\text{nopt}}$")
            else:
                header.append(metric_names[key])

        header.append(r"$f_{\text{opt}}$")

        # Group data by (deg, partition_size, node_count)
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for result in self.results:
            deg = result.seed.avg_deg_bound[0]
            part_size = result.partition_size
            node_count = len(result.graph.nodes)

            metrics = {}
            for key in eval_data:
                metrics[key] = data_extractors[key](result)

            grouped[deg][part_size][node_count].append((metrics, result.aborted))

        # Build table rows
        table = [header]
        for deg in sorted(grouped.keys()):
            for part_size in sorted(grouped[deg].keys()):
                for node_count in sorted(grouped[deg][part_size].keys()):
                    results = grouped[deg][part_size][node_count]
                    row = [deg, part_size, node_count]

                    # Separate optimal and non-optimal results
                    optimal = [m for m, aborted in results if not aborted]
                    non_optimal = [m for m, aborted in results if aborted]

                    # Calculate metrics
                    for key in eval_data:
                        if distinguish_optimality:
                            # Optimal results
                            if optimal:
                                val = stat_func([m[key] for m in optimal])
                                row.append(f"{val:.4f}" if isinstance(val, float) else val)
                            else:
                                row.append(None)

                            # Non-optimal results
                            if non_optimal:
                                val = stat_func([m[key] for m in non_optimal])
                                row.append(f"{val:.4f}" if isinstance(val, float) else val)
                            else:
                                row.append(None)
                        else:
                            # Combined results
                            val = stat_func([m[key] for m, _ in results])
                            row.append(f"{val:.4f}" if isinstance(val, float) else val)

                    # Add fraction of optimal runs
                    total = len(results)
                    optimal_count = len(optimal)
                    row.append(f"{(optimal_count / total if total > 0 else 0):.4f}")

                    table.append(row)
        return table

    def get_latex_table(
            self,
            eval_data: list[DataKey],
            eval_method: str = "mean",
            distinguish_optimality: bool = False,
            filepath: str = None
    ) -> str:
        """ Generates a LaTeX formatted table from evaluation data and optionally saves it to a file.

        This method creates a LaTeX table using the provided evaluation data and method. It can optionally mark optimal values distinctly if specified. The table is generated using the `tabulate` function with the format set to "latex_raw". If a valid file path is provided, the LaTeX table is written to a file with a ".tex" extension.

        Args:
            eval_data: List of data keys used for the evaluation
            eval_method: Evaluation method to use (default: "mean")
            distinguish_optimality: Whether to indicate optimal values distinctly (default: False)
            filepath: Optional path to save the generated LaTeX table (default: None)

        Returns:
            LaTeX string representing the evaluation table
        """

        table = self._table_data(eval_data=eval_data, eval_method=eval_method,
                                 distinguish_optimality=distinguish_optimality)
        logger.info(f"Generated table with {len(table)} rows")

        latex_table = tabulate(table, headers="firstrow", tablefmt="latex_raw")
        if filepath:
            with open(f"{filepath}.tex", "w") as f:
                f.write(latex_table)
        return latex_table

    # ... (keep existing analysis methods with logging added)
    # Example method with logging:

    # Other analysis methods (_mean_errors, _mean_inc_nodes, etc.)
    # should follow the same pattern with added logging

    # def _eval_computation_time(self, partition_size: int, eval: str = "median") -> dict[str, Any]:
    #     """ Calculate the median/mean computation time for a specific partition size.

    #     This method extracts relevant data from the results stored in the instance, including wallclock time, the # number of nodes, and the average degree bound. The processed data is then formatted into a specific structure for # further use.

    #     Args:
    #         partition_size: The partition size for which the median/mean computation time is calculated.
    #         eval: The evaluation method to use for computation time, default is "median".

    #     Returns:
    #         dict: A dictionary containing processed data and plot metadata, including:
    #             - 'data': The processed data for visualization.
    #             - 'x_label': Label for the x-axis.
    #             - 'y_label': Label for the y-axis.
    #             - 'title': Title for the plot.
    #             - 'legend': Legend information for the plot.

    #     Raises:
    #         Exception: If an error occurs during the calculation or data processing,
    #         it logs the error and returns an empty dictionary.
    #     """

    #     logger.info(f"Calculating {eval} computation time for partition size {partition_size}")

    #     try:
    #         # Extract relevant data for the specified partition size
    #         data = np.array([
    #             [result.wallclock_time, len(result.graph.nodes), result.seed.avg_deg_bound[0]]
    #             for result in self.results if result.partition_size == partition_size
    #         ])

    #         eval_method = np.median if eval == "median" else np.mean if eval == "mean" else None

    #         # Process the data to compute median computation times
    #         processed_data = [
    #             [
    #                 [
    #                     int(nodes),  # Number of nodes
    #                     eval_method([
    #                         float(time)
    #                         for time, num_nodes, avg_deg in data
    #                         if avg_deg == deg and num_nodes == nodes
    #                     ])
    #                 ] for nodes in set(data[:, 1])  # Unique number of nodes
    #             ] for deg in set(data[:, 2])  # Unique average degrees
    #         ]  # processed_data: list[list[list[int, float], float]]

    #         # Prepare the return dictionary with metadata for visualisation
    #         return {
    #             "data": processed_data,
    #             "x_label": r"number of nodes $|V|$",
    #             "y_label": r"median computation time in $s$",
    #             "title": r'{0} {1}-soft domatic partition',
    #             "legend": r'$\deg_{{exp}}=${0}'
    #         }

    #     except Exception as e:
    #         logger.error(f"Error calculating computation time: {e}")

    #     return {}

    # Add to result_db.py after the existing code
