#!/usr/bin/env python3
"""
Unit tests for GPU-based clustering algorithms.

# Note on Sourcery warnings:
# This test file intentionally contains loops and conditionals that are necessary for testing:
# - Loops are required for verifying clustering patterns, checking points in each cluster,
#   and validating results across different parameter combinations
# - Conditionals are required for hardware compatibility testing (GPU availability),
#   checking backend availability, and handling platform-specific test requirements
# sourcery skip: no-loop-in-tests, no-conditionals-in-tests
"""

# Standard library imports
import logging
from logging import getLogger

# Local application imports
from pathlib import Path

# Third-party library imports
import numpy as np

# Import GPU utilities
from utils.gpu_utils import (
    apply_dbscan_clustering_gpu,
    apply_kmeans_clustering_gpu,
    get_available_backends,
    importlib,
    is_gpu_available,
    unittest,
)

# Configure logging
logger = getLogger(__name__)

logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Use direct imports for visualization if available
# but make visualization optional
MATHPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available - skipping visualizations")

# Check for scikit-learn using importlib.util.find_spec
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
if not SKLEARN_AVAILABLE:
    logger.warning("scikit-learn not available - some tests will be skipped")


class TestGPUClustering(unittest.TestCase):
    """Tests for GPU-based clustering algorithms."""

    def setUp(self):
        """Set up test data."""
        # Create a dataset with 3 clear clusters
        rng = np.random.default_rng(42)

        # Cluster 1
        cluster1 = rng.normal(0, 0.5, (100, 2)) + np.array([5, 5])

        # Cluster 2
        cluster2 = rng.normal(0, 0.5, (100, 2)) + np.array([0, 0])

        # Cluster 3
        cluster3 = rng.normal(0, 0.5, (100, 2)) + np.array([5, 0])

        # Combine clusters
        self.data = np.vstack([cluster1, cluster2, cluster3])

        # Add some noise points
        noise = rng.uniform(-2, 7, (20, 2))
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
            self.assertGreater(np.sum(labels == i) > 0)

        # Visualize results
        self._plot_kmeans_results(self.data, labels, centroids, "kmeans_cpu")

    def test_kmeans_clustering_gpu(self):
        """Test K-means clustering using GPU backend if available."""
        if not is_gpu_available():
            logger.warning("GPU not available for testing")
            self.skipTest("GPU not available for testing")

        # Get the first available GPU backend
        backend = get_available_backends()[0]
        if backend == "cpu":
            logger.warning(f"No GPU backend available, only found: {backend}")
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
            self.assertGreater(np.sum(labels == i), 0)

        # Visualize results
        self._plot_kmeans_results(self.data, labels, centroids, f"kmeans_{backend}")

    def test_dbscan_clustering_cpu(self):
        """Test DBSCAN clustering using CPU backend."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for DBSCAN testing")
            self.skipTest("scikit-learn not available for DBSCAN testing")

        labels = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend="cpu"
        )

        # Check that we have some clusters and some noise
        unique_labels = np.unique(labels)
        self.assertGreater(-1 in unique_labels, 0)  # Noise should be present
        self.assertGreater(len(unique_labels), 1)  # Should have at least one cluster

        # Visualize results
        self._plot_dbscan_results(self.data_with_noise, labels, "dbscan_cpu")

    def test_dbscan_clustering_gpu(self):
        """Test DBSCAN clustering using GPU backend if available."""
        if not is_gpu_available():
            logger.warning("GPU not available for testing")
            self.skipTest("GPU not available for testing")

        # Get the first available GPU backend
        backend = get_available_backends()[0]
        if backend == "cpu":
            logger.warning(f"No GPU backend available, only found: {backend}")
            self.skipTest("No GPU backend available")

        labels = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend=backend
        )

        # Check that we have some clusters and some noise
        unique_labels = np.unique(labels)
        self.assertIn(-1, unique_labels)  # Noise should be present
        self.assertGreater(len(unique_labels), 1)  # Should have at least one cluster

        # Visualize results
        self._plot_dbscan_results(self.data_with_noise, labels, f"dbscan_{backend}")

    def test_kmeans_consistency(self):
        """Test that K-means results are consistent across backends."""
        # Check if GPU is available using the actual implementation
        if not is_gpu_available():
            logger.info("GPU not available - running CPU-only consistency test")

            # Even without GPU, we can test consistency of the CPU backend
            # with different random seeds
            centroids1, labels1 = apply_kmeans_clustering_gpu(
                self.data, n_clusters=3, backend="cpu", seed=42
            )

            centroids2, labels2 = apply_kmeans_clustering_gpu(
                self.data, n_clusters=3, backend="cpu", seed=84
            )

            # Instead of comparing labels directly (which can vary due to random initialization),
            # we'll verify that the clustering quality is similar
            # by measuring inertia (within-cluster sum of squares)
            inertia1 = self._calculate_inertia(self.data, labels1, centroids1)
            inertia2 = self._calculate_inertia(self.data, labels2, centroids2)

            # The inertias should be relatively close
            relative_diff = abs(inertia1 - inertia2) / max(inertia1, inertia2)
            self.assertLess(
                relative_diff, 0.5
            )  # Allow up to 50% difference due to initialization

            return

        # Get available backends for GPU testing
        backends = get_available_backends()
        gpu_backends = [b for b in backends if b != "cpu"]

        if not gpu_backends:
            logger.info("No GPU backends available - running CPU-only test")
            self.skipTest("No GPU backends available")
            return

        # Run K-means with CPU backend
        centroids_cpu, labels_cpu = apply_kmeans_clustering_gpu(
            self.data, n_clusters=3, backend="cpu", seed=42
        )

        # Run K-means with GPU backend
        gpu_backend = gpu_backends[0]
        logger.info(f"Testing with GPU backend: {gpu_backend}")
        centroids_gpu, labels_gpu = apply_kmeans_clustering_gpu(
            self.data, n_clusters=3, backend=gpu_backend, seed=42
        )

        # Calculate inertia for both results
        inertia_cpu = self._calculate_inertia(self.data, labels_cpu, centroids_cpu)
        inertia_gpu = self._calculate_inertia(self.data, labels_gpu, centroids_gpu)

        # The clustering quality (measured by inertia) should be comparable
        # Allow for some difference due to implementation variations
        logger.info(f"CPU inertia: {inertia_cpu}, GPU inertia: {inertia_gpu}")
        relative_diff = abs(inertia_cpu - inertia_gpu) / max(inertia_cpu, inertia_gpu)
        self.assertLess(relative_diff, 0.5)  # Allow up to 50% difference

        # Count points in each cluster
        cpu_counts = [np.sum(labels_cpu == i) for i in range(3)]
        gpu_counts = [np.sum(labels_gpu == i) for i in range(3)]

        # Sort counts to compare
        cpu_counts.sort()
        gpu_counts.sort()

        # Check that cluster sizes are similar, with a more lenient delta
        for cpu_count, gpu_count in zip(cpu_counts, gpu_counts):
            # Allow for up to 20% difference in cluster sizes
            max_delta = max(cpu_count, gpu_count) * 0.2
            self.assertLessEqual(abs(cpu_count - gpu_count), max_delta)

    def test_dbscan_consistency(self):
        """Test that DBSCAN results are consistent across backends."""
        # Check scikit-learn availability
        if not SKLEARN_AVAILABLE:
            logger.info("scikit-learn not available - skipping DBSCAN consistency test")
            self.skipTest("scikit-learn not available for DBSCAN testing")

        # Check if GPU is available
        if not is_gpu_available():
            logger.info("GPU not available - running CPU-only DBSCAN consistency test")

            # Even without GPU, we can test consistency of the CPU backend
            # with different parameters
            labels1 = apply_dbscan_clustering_gpu(
                self.data_with_noise, eps=0.8, min_samples=5, backend="cpu"
            )

            # Run with slightly different parameters
            labels2 = apply_dbscan_clustering_gpu(
                self.data_with_noise, eps=0.9, min_samples=5, backend="cpu"
            )

            # Count clusters in both runs
            n_clusters1 = len(np.unique(labels1)) - (1 if -1 in labels1 else 0)
            n_clusters2 = len(np.unique(labels2)) - (1 if -1 in labels2 else 0)

            # Verify both runs found some clusters
            self.assertGreater(n_clusters1, 0, "First CPU run found no clusters")
            self.assertGreater(n_clusters2, 0, "Second CPU run found no clusters")

            # The difference in cluster count should be reasonable
            self.assertLessEqual(abs(n_clusters1 - n_clusters2), 2)

            return

        # Get available backends for GPU testing
        backends = get_available_backends()
        gpu_backends = [b for b in backends if b != "cpu"]

        if not gpu_backends:
            logger.info("No GPU backends available - skipping GPU vs CPU comparison")
            self.skipTest("No GPU backends available")
            return

        # Run DBSCAN with CPU backend
        labels_cpu = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend="cpu"
        )

        # Run DBSCAN with GPU backend
        gpu_backend = gpu_backends[0]
        logger.info(f"Testing with GPU backend: {gpu_backend}")
        labels_gpu = apply_dbscan_clustering_gpu(
            self.data_with_noise, eps=0.8, min_samples=5, backend=gpu_backend
        )

        # Count number of clusters and noise points
        n_clusters_cpu = len(np.unique(labels_cpu)) - (1 if -1 in labels_cpu else 0)
        n_clusters_gpu = len(np.unique(labels_gpu)) - (1 if -1 in labels_gpu else 0)

        n_noise_cpu = np.sum(labels_cpu == -1)
        n_noise_gpu = np.sum(labels_gpu == -1)

        logger.info(f"CPU clusters: {n_clusters_cpu}, noise points: {n_noise_cpu}")
        logger.info(f"GPU clusters: {n_clusters_gpu}, noise points: {n_noise_gpu}")

        # Check that number of clusters is similar with more flexibility
        # Different implementations might identify slightly different clusters
        self.assertLessEqual(abs(n_clusters_cpu - n_clusters_gpu), 2)

        # Check that number of noise points is similar with more tolerance
        # Allow for up to 20% difference in noise points
        max_noise_diff = max(n_noise_cpu, n_noise_gpu) * 0.2
        self.assertLessEqual(abs(n_noise_cpu - n_noise_gpu), max_noise_diff)

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

    def _calculate_inertia(self, data, labels, centroids):
        """Calculate the sum of squared distances from points to their centroids.

        This is a measure of clustering quality - lower is better.

        Args:
            data: Data points
            labels: Cluster assignments
            centroids: Cluster centers

        Returns:
            float: Inertia (sum of squared distances)
        """
        inertia = 0.0
        for i in range(len(centroids)):
            # Get points in this cluster
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                continue

            # Calculate distance from each point to centroid
            distances = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
            inertia += np.sum(distances)

        return inertia


if __name__ == "__main__":
    unittest.main()
