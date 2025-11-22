"""JSON serialization utilities for custom objects.

This module provides custom JSON encoders and decoders to handle:
- NumPy arrays and data types
- NetworkX graphs
- Complex nested structures
- Compressed JSON files
"""

import gzip
import json
from pathlib import Path

import networkx as nx
import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays, tuples, and NetworkX graphs."""

    def default(self, obj: object) -> dict[str, object] | int | float | bool | str:
        """Convert non-serializable objects to JSON-serializable formats.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable representation of the object
        """
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }

        # Handle numpy integer types
        if isinstance(
            obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
        ):
            return int(obj)

        # Handle numpy float types
        if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        # Handle numpy bool
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Handle tuples
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "data": list(obj)}

        # Handle sets
        if isinstance(obj, set):
            return {"__type__": "set", "data": list(obj)}

        # Handle NetworkX graphs
        if isinstance(obj, nx.Graph):
            return {"__type__": "networkx_graph", "data": nx.node_link_data(obj)}

        # Fall back to default encoder
        return super().default(obj)


def custom_json_decoder(obj: dict[str, object] | list[object] | object) -> object:
    """Custom JSON decoder that reconstructs encoded objects.

    Args:
        obj: Dictionary potentially containing encoded objects

    Returns:
        Decoded object (numpy array, tuple, set, or networkx graph if encoded)
    """
    # Process dictionaries
    if isinstance(obj, dict):
        if "__type__" in obj:
            obj_type = obj["__type__"]

            # Decode numpy arrays
            if obj_type == "ndarray":
                return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

            # Decode tuples
            elif obj_type == "tuple":
                # Recursively decode nested structures in tuple
                return tuple(custom_json_decoder(item) for item in obj["data"])

            # Decode sets
            elif obj_type == "set":
                return set(custom_json_decoder(item) for item in obj["data"])

            # Decode NetworkX graphs
            elif obj_type == "networkx_graph":
                return nx.node_link_graph(obj["data"])

        # Recursively process nested dictionaries
        return {k: custom_json_decoder(v) for k, v in obj.items()}

    # Process lists
    elif isinstance(obj, list):
        return [custom_json_decoder(item) for item in obj]

    # Return primitive types as-is
    return obj


def save_json(obj: object, filepath: str | Path, compress: bool = False, indent: int = 2) -> None:
    """Save an object to a JSON file.

    Args:
        obj: Object to serialize
        filepath: Path to save the JSON file
        compress: Whether to use gzip compression
        indent: Number of spaces for indentation (default: 2)
    """
    filepath = Path(filepath)
    json_str = json.dumps(obj, cls=CustomJSONEncoder, indent=indent)

    if compress:
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            f.write(json_str)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)


def load_json(filepath: str | Path, compressed: bool = False) -> object:
    """Load an object from a JSON file.

    Args:
        filepath: Path to the JSON file
        compressed: Whether the file is gzip compressed

    Returns:
        Deserialized object
    """
    filepath = Path(filepath)

    if compressed:
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            json_str = f.read()
    else:
        with open(filepath, encoding="utf-8") as f:
            json_str = f.read()

    return json.loads(json_str, object_hook=custom_json_decoder)


def to_json_serializable(obj: object, indent: int | None = None) -> str:
    """Convert an object to a JSON string.

    Args:
        obj: Object to serialize
        indent: Number of spaces for indentation (None for compact)

    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=CustomJSONEncoder, indent=indent)


def from_json_serializable(json_str: str) -> object:
    """Reconstruct an object from a JSON string.

    Args:
        json_str: JSON string to deserialize

    Returns:
        Reconstructed object
    """
    return json.loads(json_str, object_hook=custom_json_decoder)
