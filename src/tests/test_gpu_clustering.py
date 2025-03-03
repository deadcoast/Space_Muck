#!/usr/bin/env python3
"""
Unit tests for GPU-based clustering algorithms.
"""

import unittest
import numpy as np
from pathlib import Path
import logging

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import scikit-learn, but don't fail if it's not available
try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.gpu_utils import (
    apply_kmeans_clustering_gpu,
    apply_dbscan_clustering_gpu,
    is_gpu_available,
    get_available_backends,
)


class TestGPUClustering(unittest.TestCase):
    """Tests for GPU-based clustering algorithms."""

    def setUp(self):
        """Set up test data."""
        # Create a dataset with 3 clear clusters
        np.random.seed(42)

        # Cluster 1
        cluster1 = np.random.randn(100, 2) * 0.5 + np.array([5, 5])

        # Cluster 2
        cluster2 = np.random.randn(100, 2) * 0.5 + np.array([0, 0])

        # Cluster 3
        cluster3 = np.random.randn(100, 2) * 0.5 + np.array([5, 0])

        # Combine clusters
        self.data = np.vstack([cluster1, cluster2, cluster3])

        # Add some noise points
        noise = np.random.uniform(-2, 7, (20, 2))
        self.data_with_noise = np.vstack([self.data, noise])

        # Create output directory for plots
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def test_kmeans_clustering_cpu(self):
        """Test K-means clustering using CPU backend."""
        centroids, labels = apply_kmeans_clustering_gpu(
            self.data, n_clusters=3, backend="cpu", seed=42
        )

        # Check that we have 3 centroids
        self.assertEqual(centroids.shape[0], 3)

        # Check that all points are assigned to a cluster
        self.assertEqual(labels.shape[0], self.data.shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))

        # Check that all clusters have points
        for i in range(3):
            self.assertTrue(np.sum(labels == i) > 0)

        # Visualize results
        self._plot_kmeans_results(self.data, labels, centroids, "kmeans_cpu")

    def test_kmeans_clustering_gpu(self):
        """Test K-means clustering using GPU backend if available."""
        if not is_gpu_available():
            self.skipTest("GPU not available for testing")

        # Get the first available GPU backend
        backend = get_available_backends()[0]
        if backend == "cpu":
            self.skipTest("No GPU backend available")

        centroids, labels = apply_kmeans_clustering_gpu(
            self.data, n_clusters=3, backend=backend, seed=42
        )

        # Check that we have 3 centroids
        self.assertEqual(centroids.shape[0], 3)

        # Check that all points are assigned to a cluster
        self.assertEqual(labels.shape[0], self.data.shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))

        # Check that all clusters have points
        for i in range(3):
            self.assertTrue(np.sum(labels == i) > 0)

        # Visualize results
        self._plot_kmeans_results(self.data, labels, centroids, f"kmeans_{backend}")

    def test_dbscan_clustering_cpu(self):
        """Test DBSCAN clustering using CPU backend."""
        if not SKLEARN_AVAILABLE:
            self.skipTest("scikit-learn not available for DBSCAN testing")

        labels = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend="cpu"
        )

        # Check that we have some clusters and some noise
        unique_labels = np.unique(labels)
        self.assertTrue(-1 in unique_labels)  # Noise should be present
        self.assertTrue(len(unique_labels) > 1)  # Should have at least one cluster

        # Visualize results
        self._plot_dbscan_results(self.data_with_noise, labels, "dbscan_cpu")

    def test_dbscan_clustering_gpu(self):
        """Test DBSCAN clustering using GPU backend if available."""
        if not is_gpu_available():
            self.skipTest("GPU not available for testing")

        # Get the first available GPU backend
        backend = get_available_backends()[0]
        if backend == "cpu":
            self.skipTest("No GPU backend available")

        labels = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend=backend
        )

        # Check that we have some clusters and some noise
        unique_labels = np.unique(labels)
        self.assertTrue(-1 in unique_labels)  # Noise should be present
        self.assertTrue(len(unique_labels) > 1)  # Should have at least one cluster

        # Visualize results
        self._plot_dbscan_results(self.data_with_noise, labels, f"dbscan_{backend}")

    def test_kmeans_consistency(self):
        """Test that K-means results are consistent across backends."""
        if not is_gpu_available():
            self.skipTest("GPU not available for testing")

        # Get available backends
        backends = get_available_backends()
        if len(backends) <= 1 or backends[0] == "cpu":
            self.skipTest("Multiple GPU backends not available")

        # Run K-means with CPU backend
        centroids_cpu, labels_cpu = apply_kmeans_clustering_gpu(
            self.data, n_clusters=3, backend="cpu", seed=42
        )

        # Run K-means with GPU backend
        gpu_backend = backends[0]
        centroids_gpu, labels_gpu = apply_kmeans_clustering_gpu(
            self.data, n_clusters=3, backend=gpu_backend, seed=42
        )

        # Check that cluster assignments are mostly consistent
        # Note: Due to floating-point differences and randomness,
        # the exact labels might differ, but the clustering should be similar

        # Count points in each cluster
        cpu_counts = [np.sum(labels_cpu == i) for i in range(3)]
        gpu_counts = [np.sum(labels_gpu == i) for i in range(3)]

        # Sort counts to compare
        cpu_counts.sort()
        gpu_counts.sort()

        # Check that cluster sizes are similar
        for cpu_count, gpu_count in zip(cpu_counts, gpu_counts):
            self.assertAlmostEqual(cpu_count, gpu_count, delta=20)

    def test_dbscan_consistency(self):
        """Test that DBSCAN results are consistent across backends."""
        if not SKLEARN_AVAILABLE:
            self.skipTest("scikit-learn not available for DBSCAN testing")

        if not is_gpu_available():
            self.skipTest("GPU not available for testing")

        # Get available backends
        backends = get_available_backends()
        if len(backends) <= 1 or backends[0] == "cpu":
            self.skipTest("Multiple GPU backends not available")

        # Run DBSCAN with CPU backend
        labels_cpu = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend="cpu"
        )

        # Run DBSCAN with GPU backend
        gpu_backend = backends[0]
        labels_gpu = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend=gpu_backend
        )

        # Count number of clusters and noise points
        n_clusters_cpu = len(np.unique(labels_cpu)) - (1 if -1 in labels_cpu else 0)
        n_clusters_gpu = len(np.unique(labels_gpu)) - (1 if -1 in labels_gpu else 0)

        n_noise_cpu = np.sum(labels_cpu == -1)
        n_noise_gpu = np.sum(labels_gpu == -1)

        # Check that number of clusters is similar
        self.assertAlmostEqual(n_clusters_cpu, n_clusters_gpu, delta=1)

        # Check that number of noise points is similar
        self.assertAlmostEqual(n_noise_cpu, n_noise_gpu, delta=10)

    def _plot_kmeans_results(self, data, labels, centroids, filename):
        """Helper to visualize K-means results."""
        if not MATPLOTLIB_AVAILABLE:
            return

        plt.figure(figsize=(10, 8))

        # Plot points colored by cluster
        for i in range(len(np.unique(labels))):
            plt.scatter(
                data[labels == i, 0],
                data[labels == i, 1],
                s=50,
                alpha=0.7,
                label=f"Cluster {i}",
            )

        # Plot centroids
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=200,
            marker="X",
            c="black",
            label="Centroids",
        )

        plt.title(f"K-means Clustering ({filename})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f"{filename}.png")
        plt.close()

    def _plot_dbscan_results(self, data, labels, filename):
        """Helper to visualize DBSCAN results."""
        if not MATPLOTLIB_AVAILABLE:
            return

        plt.figure(figsize=(10, 8))

        # Get unique labels
        unique_labels = np.unique(labels)

        # Plot noise points
        if -1 in unique_labels:
            plt.scatter(
                data[labels == -1, 0],
                data[labels == -1, 1],
                s=50,
                alpha=0.5,
                c="gray",
                label="Noise",
            )

        # Plot clusters
        for label in unique_labels:
            if label == -1:
                continue

            plt.scatter(
                data[labels == label, 0],
                data[labels == label, 1],
                s=50,
                alpha=0.7,
                label=f"Cluster {label}",
            )

        plt.title(f"DBSCAN Clustering ({filename})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f"{filename}.png")
        plt.close()


if __name__ == "__main__":
    unittest.main()
