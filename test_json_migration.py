"""Test script to verify JSON serialization/deserialization works correctly."""

import os
import sys
import tempfile

import numpy as np

# Add src to path
sys.path.insert(0, "/home/pilot/projects/lambda_precision_udg_generator/src")

from lambdaprecisionudggenerator.graph_generator.graphs.lambda_precision_udg import (
    LambdaPrecisionUDG,
)
from lambdaprecisionudggenerator.graph_generator.points.lambda_precision_points import (
    LambdaPrecisionPoints,
)
from lambdaprecisionudggenerator.graph_generator.seeds.seed import GeneratorSeed
from lambdaprecisionudggenerator.utils.json_utils import load_json, save_json


def test_json_utils():
    """Test basic JSON utilities."""
    print("Testing JSON utilities...")

    # Test numpy array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_json({"array": arr}, temp_path)
        loaded = load_json(temp_path)
        assert np.array_equal(loaded["array"], arr)
        print("✓ NumPy array serialization works")
    finally:
        os.unlink(temp_path)

    print("✓ JSON utilities work correctly")
    print()


def test_lambda_precision_points():
    """Test LambdaPrecisionPoints serialization."""
    print("Testing LambdaPrecisionPoints...")

    points = [(10, 20), (30, 40), (50, 60)]
    field = np.zeros((100, 100), dtype=int)
    lpp = LambdaPrecisionPoints(points=points, min_distance=5, field=field)

    with tempfile.TemporaryDirectory() as tmpdir:
        lpp.serialize(tmpdir)
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".json")

        filepath = os.path.join(tmpdir, files[0])
        loaded_lpp = LambdaPrecisionPoints.deserialize(filepath)

        # Points may be lists instead of tuples after JSON round-trip, but that's functionally equivalent
        assert len(loaded_lpp.points) == len(points)
        for loaded_pt, orig_pt in zip(loaded_lpp.points, points, strict=False):
            assert list(loaded_pt) == list(orig_pt)
        assert loaded_lpp.min_distance == 5
        assert np.array_equal(loaded_lpp.field, field)
        print("✓ LambdaPrecisionPoints serialization works")

    print()


def test_lambda_precision_udg():
    """Test LambdaPrecisionUDG serialization."""
    print("Testing LambdaPrecisionUDG...")

    points = [(10, 20), (30, 40), (50, 60)]
    field = np.zeros((100, 100), dtype=int)
    lpp = LambdaPrecisionPoints(points=points, min_distance=5, field=field)

    udg = LambdaPrecisionUDG(points=lpp, radius=0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        udg.serialize(tmpdir)
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".json")

        filepath = os.path.join(tmpdir, files[0])
        loaded_udg = LambdaPrecisionUDG.deserialize(filepath)

        assert loaded_udg.radius == 0.3
        assert loaded_udg.number_of_nodes() == udg.number_of_nodes()
        assert loaded_udg.number_of_edges() == udg.number_of_edges()
        print("✓ LambdaPrecisionUDG serialization works")

    print()


def test_generator_seed():
    """Test GeneratorSeed metadata serialization."""
    print("Testing GeneratorSeed...")

    seed = GeneratorSeed(
        node_number=10,
        min_distance=0.1,
        radius=0.3,
        coverage_bound=(0.5, 1.0),
        avg_deg_bound=(2.0, 5.0),
        probability_connected=0.95,
        sample_size=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        seed.serialize(tmpdir)
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".json")

        filepath = os.path.join(tmpdir, files[0])
        loaded_seed = GeneratorSeed.deserialize(filepath)

        assert loaded_seed.node_number == seed.node_number
        assert loaded_seed.min_distance == seed.min_distance
        assert loaded_seed.radius == seed.radius
        # Tuples become lists in JSON, but that's functionally equivalent for these bounds
        assert list(loaded_seed.coverage_bound) == list(seed.coverage_bound)
        assert list(loaded_seed.avg_deg_bound) == list(seed.avg_deg_bound)
        print("✓ GeneratorSeed serialization works")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("JSON Serialization Migration Tests")
    print("=" * 60)
    print()

    try:
        test_json_utils()
        test_lambda_precision_points()
        test_lambda_precision_udg()
        test_generator_seed()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
