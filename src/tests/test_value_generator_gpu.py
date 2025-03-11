#!/usr/bin/env python3
"""
Unit tests for GPU-accelerated value generation functions.
"""

import unittest
import numpy as np
from unittest.mock import patch

from utils.value_generator_gpu import (
    generate_value_distribution_gpu,
    add_value_clusters_gpu,
)
from utils.gpu_utils import is_gpu_available

# Check if GPU backends are available
try:
    import importlib.util

    spec = importlib.util.find_spec("numba")
    if spec is not None:
        from numba import cuda

        CUDA_AVAILABLE = cuda.is_available()
    else:
        CUDA_AVAILABLE = False
except ImportError:
    CUDA_AVAILABLE = False

try:
    import importlib.util

    spec = importlib.util.find_spec("cupy")
    CUPY_AVAILABLE = spec is not None
except ImportError:
    CUPY_AVAILABLE = False


class TestValueGeneratorGPU(unittest.TestCase):
    """Tests for GPU-accelerated value generation functions."""

    def setUp(self):
        """Set up test data."""
        # Create test grids
        np.random.seed(42)
        self.small_grid = np.random.randint(0, 2, (20, 20))
        self.medium_grid = np.random.randint(0, 2, (50, 50))
        self.noise_grid = np.random.random((50, 50))

    def test_generate_value_distribution_cpu(self):
        """Test value distribution generation with CPU backend."""
        # Generate values
        value_grid = generate_value_distribution_gpu(
            self.small_grid, self.noise_grid[:20, :20], backend="cpu"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(value_grid, np.ndarray)
        self.assertEqual(value_grid.shape, self.small_grid.shape)

        # Verify values are only assigned to non-zero cells
        np.testing.assert_array_equal(value_grid > 0, self.small_grid > 0)

        mask = self.small_grid > 0
        if np.any(mask):
            # Verify minimum value constraint
            min_value = 1
            self.assertTrue(np.all(value_grid[mask] >= min_value))

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_generate_value_distribution_cupy(self):
        """Test value distribution generation with CuPy backend."""
        # Generate values
        value_grid = generate_value_distribution_gpu(
            self.small_grid, self.noise_grid[:20, :20], backend="cupy"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(value_grid, np.ndarray)
        self.assertEqual(value_grid.shape, self.small_grid.shape)

        # Verify values are only assigned to non-zero cells
        np.testing.assert_array_equal(value_grid > 0, self.small_grid > 0)

        mask = self.small_grid > 0
        if np.any(mask):
            # Verify minimum value constraint
            min_value = 1
            self.assertTrue(np.all(value_grid[mask] >= min_value))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_generate_value_distribution_cuda(self):
        """Test value distribution generation with CUDA backend."""
        # Generate values
        value_grid = generate_value_distribution_gpu(
            self.small_grid, self.noise_grid[:20, :20], backend="cuda"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(value_grid, np.ndarray)
        self.assertEqual(value_grid.shape, self.small_grid.shape)

        # Verify values are only assigned to non-zero cells
        np.testing.assert_array_equal(value_grid > 0, self.small_grid > 0)

        mask = self.small_grid > 0
        if np.any(mask):
            # Verify minimum value constraint
            min_value = 1
            self.assertTrue(np.all(value_grid[mask] >= min_value))

    def test_generate_value_distribution_auto(self):
        """Test value distribution generation with auto backend selection."""
        # Generate values
        value_grid = generate_value_distribution_gpu(
            self.small_grid, self.noise_grid[:20, :20], backend="auto"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(value_grid, np.ndarray)
        self.assertEqual(value_grid.shape, self.small_grid.shape)

        # Verify values are only assigned to non-zero cells
        np.testing.assert_array_equal(value_grid > 0, self.small_grid > 0)

    def test_add_value_clusters_cpu(self):
        """Test adding value clusters with CPU backend."""
        # Create a value grid
        value_grid = np.zeros((30, 30), dtype=np.int32)
        value_grid[10:20, 10:20] = 5  # Add a block of values

        # Add clusters
        result = add_value_clusters_gpu(
            value_grid, num_clusters=3, cluster_radius=5, backend="cpu"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, value_grid.shape)

        # Verify clusters were added (some values should be higher than original)
        self.assertTrue(np.any(result > value_grid))

        # Verify values are only modified where original values exist
        np.testing.assert_array_equal(result > 0, value_grid > 0)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_add_value_clusters_cupy(self):
        """Test adding value clusters with CuPy backend."""
        # Create a value grid
        value_grid = np.zeros((30, 30), dtype=np.int32)
        value_grid[10:20, 10:20] = 5  # Add a block of values

        # Add clusters
        result = add_value_clusters_gpu(
            value_grid, num_clusters=3, cluster_radius=5, backend="cupy"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, value_grid.shape)

        # Verify clusters were added (some values should be higher than original)
        self.assertTrue(np.any(result > value_grid))

        # Verify values are only modified where original values exist
        np.testing.assert_array_equal(result > 0, value_grid > 0)

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_add_value_clusters_cuda(self):
        """Test adding value clusters with CUDA backend."""
        # Create a value grid
        value_grid = np.zeros((30, 30), dtype=np.int32)
        value_grid[10:20, 10:20] = 5  # Add a block of values

        # Add clusters
        result = add_value_clusters_gpu(
            value_grid, num_clusters=3, cluster_radius=5, backend="cuda"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, value_grid.shape)

        # Verify clusters were added (some values should be higher than original)
        self.assertTrue(np.any(result > value_grid))

        # Verify values are only modified where original values exist
        np.testing.assert_array_equal(result > 0, value_grid > 0)

    def test_add_value_clusters_auto(self):
        """Test adding value clusters with auto backend selection."""
        # Create a value grid
        value_grid = np.zeros((30, 30), dtype=np.int32)
        value_grid[10:20, 10:20] = 5  # Add a block of values

        # Add clusters
        result = add_value_clusters_gpu(
            value_grid, num_clusters=3, cluster_radius=5, backend="auto"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, value_grid.shape)

        # Verify clusters were added (some values should be higher than original)
        self.assertTrue(np.any(result > value_grid))

    def test_value_distribution_consistency(self):
        """Test that GPU and CPU implementations give similar results."""
        # Skip if no GPU is available
        if not is_gpu_available():
            self.skipTest("No GPU available for consistency test")

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Create test data
        test_grid = np.random.randint(0, 2, (30, 30))
        noise_grid = np.random.random((30, 30))

        # Run with CPU backend
        cpu_result = generate_value_distribution_gpu(
            test_grid, noise_grid, backend="cpu"
        )

        # Run with auto backend (which should select GPU if available)
        gpu_result = generate_value_distribution_gpu(
            test_grid, noise_grid, backend="auto"
        )

        # Check that results are similar
        # We can't expect exact equality due to floating point differences
        # between CPU and GPU implementations
        mask = test_grid > 0
        if np.any(mask):
            # Calculate relative difference where values exist
            rel_diff = np.abs(cpu_result[mask] - gpu_result[mask]) / np.maximum(
                1, cpu_result[mask]
            )
            # Allow for small differences (less than 5%)
            self.assertTrue(np.all(rel_diff < 0.05))

    def test_value_clusters_consistency(self):
        """Test that GPU and CPU implementations give similar results for clustering."""
        # Skip if no GPU is available
        if not is_gpu_available():
            self.skipTest("No GPU available for consistency test")

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Create test data
        value_grid = np.zeros((30, 30), dtype=np.int32)
        value_grid[10:20, 10:20] = 5  # Add a block of values

        # Run with CPU backend
        cpu_result = add_value_clusters_gpu(
            value_grid, num_clusters=3, cluster_radius=5, backend="cpu"
        )

        # Run with auto backend (which should select GPU if available)
        gpu_result = add_value_clusters_gpu(
            value_grid, num_clusters=3, cluster_radius=5, backend="auto"
        )

        # Check that results have similar characteristics
        # We can't expect exact equality due to random cluster placement
        # and floating point differences between CPU and GPU implementations

        # Both should have the same shape
        self.assertEqual(cpu_result.shape, gpu_result.shape)

        # Both should have values only where original grid had values
        np.testing.assert_array_equal(cpu_result > 0, value_grid > 0)
        np.testing.assert_array_equal(gpu_result > 0, value_grid > 0)

        # Both should have increased some values
        self.assertTrue(np.any(cpu_result > value_grid))
        self.assertTrue(np.any(gpu_result > value_grid))

    @patch("src.utils.value_generator_gpu.CUDA_AVAILABLE", False)
    @patch("src.utils.value_generator_gpu.CUPY_AVAILABLE", False)
    def test_fallback_when_gpu_unavailable_value_distribution(self):
        """Test fallback to CPU when GPU is unavailable for value distribution."""
        with patch(
            "src.utils.value_generator_gpu.generate_value_distribution"
        ) as mock_cpu_fn:
            mock_cpu_fn.return_value = np.zeros((10, 10))

            # This should use the CPU implementation
            generate_value_distribution_gpu(
                np.ones((10, 10)), np.ones((10, 10)), backend="auto"
            )

            # Verify that the CPU function was called
            mock_cpu_fn.assert_called_once()

    @patch("src.utils.value_generator_gpu.CUDA_AVAILABLE", False)
    @patch("src.utils.value_generator_gpu.CUPY_AVAILABLE", False)
    def test_fallback_when_gpu_unavailable_value_clusters(self):
        """Test fallback to CPU when GPU is unavailable for value clusters."""
        with patch("src.utils.value_generator_gpu.add_value_clusters") as mock_cpu_fn:
            mock_cpu_fn.return_value = np.zeros((10, 10))

            # This should use the CPU implementation
            add_value_clusters_gpu(np.ones((10, 10)), backend="auto")

            # Verify that the CPU function was called
            mock_cpu_fn.assert_called_once()

    def test_empty_grid_value_distribution(self):
        """Test value distribution with an empty grid."""
        # Create an empty grid
        empty_grid = np.zeros((20, 20))
        noise_grid = np.random.random((20, 20))

        # Generate values
        result = generate_value_distribution_gpu(empty_grid, noise_grid, backend="auto")

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, empty_grid.shape)

        # Verify no values were assigned (all zeros)
        self.assertTrue(np.all(result == 0))

    def test_empty_grid_value_clusters(self):
        """Test adding value clusters to an empty grid."""
        # Create an empty grid
        empty_grid = np.zeros((20, 20))

        # Add clusters
        result = add_value_clusters_gpu(
            empty_grid, num_clusters=3, cluster_radius=5, backend="auto"
        )

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, empty_grid.shape)

        # Verify no values were assigned (all zeros)
        self.assertTrue(np.all(result == 0))


if __name__ == "__main__":
    unittest.main()
