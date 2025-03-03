#!/usr/bin/env python3
"""
Comprehensive GPU Benchmarking Script for Space Muck.

This script provides a unified framework for benchmarking all GPU-accelerated operations
across different backends, grid sizes, and configurations. It generates detailed
performance metrics and visualizations to help optimize GPU acceleration usage.
"""

import os
import sys
import time
import logging
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GPU utilities
from src.utils.gpu_utils import (
    is_gpu_available,
    get_available_backends,
    to_gpu,
    to_cpu,
    apply_cellular_automaton_gpu,
    apply_noise_generation_gpu,
    apply_kmeans_clustering_gpu,
    apply_dbscan_clustering_gpu,
    CUDA_AVAILABLE,
    CUPY_AVAILABLE,
    MPS_AVAILABLE,
    METALGPU_AVAILABLE,
)

# Import CPU implementations for comparison
from src.utils.cellular_automaton_utils import apply_cellular_automaton
from src.utils.value_generator import (
    generate_value_distribution,
    add_value_clusters,
)
from src.utils.value_generator_gpu import (
    generate_value_distribution_gpu,
    add_value_clusters_gpu,
)

# Import BaseGenerator for integration testing
from src.entities.base_generator import BaseGenerator
