#!/usr/bin/env python3
"""
Unit tests for the GPU utilities module.

These tests verify the functionality of the GPU acceleration utilities,
with appropriate fallbacks for systems without GPU support.
"""

import unittest
import numpy as np
from unittest.mock import patch

# Import the module to test
from utils.gpu_utils import (
    is_gpu_available,
    get_available_backends,
    to_gpu,
    to_cpu,
    apply_cellular_automaton_gpu,
    apply_noise_generation_gpu,
    CUDA_AVAILABLE,
    CUPY_AVAILABLE,
)


class TestGPUUtils(unittest.TestCase):
    """Test cases for GPU utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample grids for testing
        self.small_grid = np.zeros((10, 10), dtype=np.int8)
        self.small_grid[4:7, 4:7] = 1  # Create a small square

        self.medium_grid = np.zeros((50, 50), dtype=np.int8)
        self.medium_grid[20:30, 20:30] = 1  # Create a medium square

        # Glider pattern for testing cellular automaton
        self.glider_grid = np.zeros((20, 20), dtype=np.int8)
        self.glider_grid[1, 2] = 1
        self.glider_grid[2, 3] = 1
        self.glider_grid[3, 1:4] = 1

    def test_is_gpu_available(self):
        """Test GPU availability detection."""
        # This just verifies the function runs without error
        result = is_gpu_available()
        self.assertIsInstance(result, bool)

    def test_get_available_backends(self):
        """Test backend detection."""
        backends = get_available_backends()
        self.assertIsInstance(backends, list)
        self.assertTrue(len(backends) > 0)  # At least CPU should be available

    def test_array_transfer(self):
        """Test array transfer between CPU and GPU."""
        # Create a test array
        test_array = np.random.random((10, 10))

        # Transfer to GPU
        gpu_array = to_gpu(test_array)

        # Transfer back to CPU
        cpu_array = to_cpu(gpu_array)

        # Verify data integrity
        np.testing.assert_array_almost_equal(test_array, cpu_array)

    def test_cellular_automaton_cpu(self):
        """Test cellular automaton with CPU backend."""
        # Run cellular automaton for one iteration
        result = apply_cellular_automaton_gpu(
            self.glider_grid, backend="cpu", iterations=1
        )

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.glider_grid.shape)

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_cellular_automaton_cuda(self):
        """Test cellular automaton with CUDA backend."""
        # Run cellular automaton for one iteration
        result = apply_cellular_automaton_gpu(
            self.glider_grid, backend="cuda", iterations=1
        )

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.glider_grid.shape)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_cellular_automaton_cupy(self):
        """Test cellular automaton with CuPy backend."""
        # Run cellular automaton for one iteration
        result = apply_cellular_automaton_gpu(
            self.glider_grid, backend="cupy", iterations=1
        )

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.glider_grid.shape)

    def test_cellular_automaton_auto(self):
        """Test cellular automaton with auto backend selection."""
        # Run cellular automaton for one iteration
        result = apply_cellular_automaton_gpu(
            self.glider_grid, backend="auto", iterations=1
        )

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.glider_grid.shape)

    def test_noise_generation_cpu(self):
        """Test noise generation with CPU backend."""
        # Generate noise
        noise = apply_noise_generation_gpu(width=32, height=32, backend="cpu")

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(noise.shape, (32, 32))

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_noise_generation_cupy(self):
        """Test noise generation with CuPy backend."""
        # Generate noise
        noise = apply_noise_generation_gpu(width=32, height=32, backend="cupy")

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(noise.shape, (32, 32))

    def test_noise_generation_auto(self):
        """Test noise generation with auto backend selection."""
        # Generate noise
        noise = apply_noise_generation_gpu(width=32, height=32, backend="auto")

        # Verify the result is a numpy array with expected shape
        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(noise.shape, (32, 32))

    def test_cellular_automaton_consistency(self):
        """Test that GPU and CPU implementations give similar results."""
        # Skip if no GPU is available
        if not is_gpu_available():
            self.skipTest("No GPU available for consistency test")

        # Run with CPU backend
        cpu_result = apply_cellular_automaton_gpu(
            self.small_grid, backend="cpu", iterations=3
        )

        # Run with auto backend (which should select GPU if available)
        gpu_result = apply_cellular_automaton_gpu(
            self.small_grid, backend="auto", iterations=3
        )

        # Check that results are similar (may not be identical due to floating point)
        similarity = np.sum(cpu_result == gpu_result) / cpu_result.size
        self.assertGreater(similarity, 0.9)  # At least 90% similar

    @patch("src.utils.gpu_utils.CUDA_AVAILABLE", False)
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", False)
    def test_fallback_when_gpu_unavailable(self):
        """Test fallback to CPU when GPU is unavailable."""
        # Create a mock function to be called when GPU is unavailable
        with patch(
            "src.utils.cellular_automaton_utils.apply_cellular_automaton"
        ) as mock_cpu_fn:
            mock_cpu_fn.return_value = np.zeros((10, 10))

            # This should use the CPU implementation
            apply_cellular_automaton_gpu(self.small_grid, backend="auto")

            # Verify that the CPU function was called
            mock_cpu_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
