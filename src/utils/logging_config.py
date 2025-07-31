import logging
import sys

LOGGING_LEVEL_GLOBAL = logging.CRITICAL  # Default logging level
LOGGING_LEVEL_PROJECT = logging.DEBUG  # Default logging level
LOGGING_FORMAT = "[%(levelname)s] %(name)s: %(message)s"  # Default logging format


def setup_logging():
    """Global logging configuration for all modules and processes"""
    root = logging.getLogger()
    root.setLevel(LOGGING_LEVEL_GLOBAL)

    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    root.addHandler(handler)

    # Enable propagation for key modules
    modules = [
        "src.graph_generator.points.generator",
        "src.graph_generator.points.lambda_precision_points",
        "src.graph_generator.graphs.lambda_precision_udg2",
        "src.graph_generator.graphs.generator",
        "src.graph_generator.seeds.generator",
        "src.graph_generator.seeds.seed",
        "src.graph_generator.seeds.database",
        "src.partitioning.partitioning",
        "src.partitioning.result",
        "src.partitioning.result2.BaseResult",
        "src.partitioning.result2.OptSoftDomaticPartitionResult",
        "src.partitioning.result2.MaxSoftDomaticPartitionResult",
        "src.partitioning.result2.MinVarianceResult",
        "src.partitioning.result2.MinSpreadResult",
        "src.partitioning.result2.MinSpreadResourceResult",
        "src.partitioning.partitioning2",
        "src.partitioning.result_db.ResultDB"
    ]

    for module in modules:
        logger = logging.getLogger(module)
        logger.propagate = True
        logger.setLevel(LOGGING_LEVEL_PROJECT)
