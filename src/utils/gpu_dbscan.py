#!/usr/bin/env python3
"""
GPU-accelerated DBSCAN clustering for Space Muck.

This module provides GPU-accelerated implementations of DBSCAN clustering.
"""

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING

import numpy as np

# Check for optional dependencies
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
# Safe check for MPS availability
MPS_AVAILABLE = False
if importlib.util.find_spec("torch") is not None:
    with contextlib.suppress(ImportError):
        import torch

        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            MPS_AVAILABLE = torch.backends.mps.is_available()
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

# For type checking only
if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        import torch
        from sklearn.cluster import DBSCAN
# Import optional dependencies at runtime
if TORCH_AVAILABLE:
    try:
        import torch
    except ImportError:
        TORCH_AVAILABLE = False
        MPS_AVAILABLE = False

if SKLEARN_AVAILABLE:
    try:
        # Import DBSCAN only at runtime if not already imported for type checking
        if not TYPE_CHECKING:
            from sklearn.cluster import DBSCAN
    except ImportError:
        SKLEARN_AVAILABLE = False

# Define constant for duplicate warning message
SCIKIT_DBSCAN_WARNING = (
    "scikit-learn not available. DBSCAN requires scikit-learn or GPU."
)


def _apply_dbscan_cpu(
    data: np.ndarray, eps: float, min_samples: int, n_samples: int
) -> np.ndarray:
    """
    Apply DBSCAN clustering using scikit-learn CPU implementation.

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        n_samples: Number of samples

    Returns:
        np.ndarray: Cluster labels
    """
    if not SKLEARN_AVAILABLE:
        logging.warning(SCIKIT_DBSCAN_WARNING)
        return np.zeros(n_samples, dtype=np.int32) - 1

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(data)
    except Exception as e:
        logging.warning(f"DBSCAN CPU implementation failed: {e}")
        return np.zeros(n_samples, dtype=np.int32) - 1


def _apply_dbscan_mps(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Apply DBSCAN clustering using MPS (Metal Performance Shaders).

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN

    Returns:
        np.ndarray: Cluster labels
    """
    if not TORCH_AVAILABLE or not MPS_AVAILABLE:
        return _apply_dbscan_cpu(data, eps, min_samples, data.shape[0])

    try:
        return _create_mps_transfer(data, eps, min_samples)
    except Exception as e:
        logging.warning(f"DBSCAN MPS implementation failed: {e}")
        return _apply_dbscan_cpu(data, eps, min_samples, data.shape[0])


def _identify_core_samples(torch_data, eps, min_samples, n_samples, device):
    """Identify core samples in the dataset using chunked processing.

    Args:
        torch_data: Tensor data on MPS device
        eps: Epsilon distance parameter
        min_samples: Minimum samples for core point
        n_samples: Number of samples
        device: MPS device

    Returns:
        Tuple containing core_samples and initial labels
    """
    max_chunk_size = 1000
    labels = torch.zeros(n_samples, dtype=torch.int32, device=device) - 1
    core_samples = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for i in range(0, n_samples, max_chunk_size):
        end_idx = min(i + max_chunk_size, n_samples)
        chunk = torch_data[i:end_idx]

        # Compute distances and identify neighbors
        distances = torch.cdist(chunk, torch_data)
        neighbors = (distances <= eps).sum(dim=1)

        # Mark core samples in this chunk
        chunk_core = neighbors >= min_samples
        core_samples[i:end_idx] = chunk_core

        # Assign preliminary IDs to core points
        labels[i:end_idx] = torch.where(
            chunk_core,
            torch.arange(i, end_idx, device=device),
            torch.tensor(-1, device=device),
        )

    return core_samples, labels


def _find_connected_components(torch_data, core_samples, eps, n_samples):
    """Find connected components among core samples using union-find.

    Args:
        torch_data: Tensor data on MPS device
        core_samples: Boolean tensor indicating core samples
        eps: Epsilon distance parameter
        n_samples: Number of samples

    Returns:
        Cluster mapping for connected components
    """
    max_chunk_size = 1000
    cluster_map = torch.arange(n_samples, device=torch_data.device)

    for i in range(0, n_samples, max_chunk_size):
        end_idx = min(i + max_chunk_size, n_samples)
        if not torch.any(core_samples[i:end_idx]):
            continue

        # Check if there are any core samples in this chunk
        chunk_core = core_samples[i:end_idx]

        # Only compute distances for core points in this chunk
        core_idx = torch.where(chunk_core)[0] + i
        if len(core_idx) == 0:
            continue

        core_chunk = torch_data[core_idx]

        # Find connections between points
        distances = torch.cdist(core_chunk, torch_data)
        connections = distances <= eps

        # Merge connected components
        for k, point_idx in enumerate(core_idx):
            connected = torch.where(connections[k])[0]
            for j in connected:
                if core_samples[j]:
                    # Union operation: map to minimum cluster ID
                    min_id = torch.min(cluster_map[point_idx], cluster_map[j])
                    cluster_map[point_idx] = min_id
                    cluster_map[j] = min_id

    # Propagate cluster labels through the mapping
    changed = True
    while changed:
        new_map = torch.min(cluster_map, cluster_map[cluster_map])
        changed = not torch.all(new_map == cluster_map)
        cluster_map = new_map

    return cluster_map


def _assign_final_labels(cluster_map, core_samples, labels, torch_data, eps):
    """Assign final cluster labels to all points.

    Args:
        cluster_map: Cluster mapping for core samples
        core_samples: Boolean tensor indicating core samples
        labels: Initial labels tensor
        torch_data: Tensor data on MPS device
        eps: Epsilon distance parameter

    Returns:
        Final cluster labels
    """
    final_labels = torch.zeros_like(labels)
    cluster_ids = torch.unique(cluster_map[core_samples], dim=0)

    # Assign cluster IDs to core samples and their connected points
    for i, cluster_id in enumerate(cluster_ids):
        final_labels[cluster_map == cluster_id] = i

    # Assign non-core points to nearest core cluster
    n_samples = torch_data.shape[0]
    for i in range(n_samples):
        if not core_samples[i]:
            distances = torch.cdist(torch_data[i:i + 1], torch_data[core_samples])
            if torch.any(distances <= eps):
                nearest_core = torch.argmin(distances[0], dim=0)
                core_points = torch.where(core_samples)[0]
                final_labels[i] = final_labels[core_points[nearest_core]]
            else:
                final_labels[i] = -1

    return final_labels


def _create_mps_transfer(data, eps, min_samples):
    """Apply DBSCAN clustering using MPS (Metal Performance Shaders).

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN

    Returns:
        np.ndarray: Cluster labels
    """
    # Create MPS device and transfer data
    device = torch.device("mps")
    torch_data = torch.tensor(data, dtype=torch.float32, device=device)
    n_samples = data.shape[0]

    # Step 1: Identify core samples
    core_samples, labels = _identify_core_samples(
        torch_data, eps, min_samples, n_samples, device
    )

    # Step 2: Find connected components
    cluster_map = _find_connected_components(torch_data, core_samples, eps, n_samples)

    # Step 3: Assign final cluster labels
    final_labels = _assign_final_labels(
        cluster_map, core_samples, labels, torch_data, eps
    )

    # Transfer results back to CPU
    return final_labels.cpu().numpy()


def apply_dbscan_clustering_gpu(
    data: np.ndarray, eps: float = 0.5, min_samples: int = 5, backend: str = "auto"
) -> np.ndarray:
    """
    Apply DBSCAN clustering using GPU acceleration if available.

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        backend: GPU backend to use ('mps', 'auto')

    Returns:
        np.ndarray: Cluster labels
    """
    # Validate inputs
    if data is None or data.size == 0:
        return np.array([], dtype=np.int32)

    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(data.shape)}D")

    if backend == "auto":
        backend = "mps" if MPS_AVAILABLE else "cpu"
    # Apply selected implementation
    if backend == "mps" and MPS_AVAILABLE:
        return _apply_dbscan_mps(data, eps, min_samples)
    else:
        return _apply_dbscan_cpu(data, eps, min_samples, data.shape[0])
