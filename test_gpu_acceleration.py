import os
import sys
import numpy as np
import platform

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Import the GPU utilities
    from src.utils.gpu_utils import (
        is_gpu_available,
        get_available_backends,
        to_gpu,
        to_cpu,
        CUDA_AVAILABLE,
        CUPY_AVAILABLE,
        MPS_AVAILABLE,
        METALGPU_AVAILABLE,
    )

    # Print system information
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python version: {platform.python_version()}")

    # Check if GPU is available
    gpu_available = is_gpu_available()
    print(f"GPU available: {gpu_available}")

    # Print available backends
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    print(f"MPS available: {MPS_AVAILABLE}")
    print(f"metalgpu available: {METALGPU_AVAILABLE}")

    if gpu_available:
        # Get available backends
        backends = get_available_backends()
        print(f"Available backends: {backends}")

        # Test data transfer
        test_array = np.random.random((100, 100))
        print(f"Original array shape: {test_array.shape}")

        # Transfer to GPU
        gpu_array = to_gpu(test_array, backend=backends[0])
        print(f"GPU array type: {type(gpu_array)}")

        # Transfer back to CPU
        cpu_array = to_cpu(gpu_array)
        print(f"CPU array shape: {cpu_array.shape}")

        # Check if the data is preserved
        if np.allclose(test_array, cpu_array):
            print("Data transfer successful: original and transferred arrays match")
        else:
            print("Data transfer error: arrays do not match")
    else:
        print("GPU acceleration is not available")

except ImportError as e:
    print(f"Error importing GPU utilities: {e}")
except Exception as e:
    print(f"Error testing GPU acceleration: {e}")

# Try to import and initialize BaseGenerator
try:
    # Fix import paths
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from src.entities.base_generator import BaseGenerator

    # Create a BaseGenerator instance with GPU acceleration
    bg = BaseGenerator(width=128, height=128, seed=42, use_gpu=True)

    # Print GPU-related attributes
    print("\nBaseGenerator GPU settings:")
    print(f"use_gpu: {bg.use_gpu}")
    print(f"gpu_available: {bg.gpu_available}")
    print(f"gpu_backend: {bg.gpu_backend}")

    # Generate multi-octave noise
    noise = bg.generate_multi_octave_noise(
        scale=0.1, octaves=[3, 5, 8], weights=[1.0, 0.5, 0.25]
    )
    print(f"Generated noise shape: {noise.shape}")
    print(f"Noise min/max values: {noise.min():.4f}/{noise.max():.4f}")

    # Test specific backends if available
    if MPS_AVAILABLE and platform.system() == "Darwin":
        print("\nTesting MPS backend specifically:")
        try:
            # Import the noise generation function
            from src.utils.gpu_utils import apply_noise_generation_gpu

            # Generate noise with MPS backend
            mps_noise = apply_noise_generation_gpu(
                width=128,
                height=128,
                scale=0.1,
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
                seed=42,
                backend="mps",
            )
            print(f"MPS noise generation successful, shape: {mps_noise.shape}")
            print(
                f"MPS noise min/max values: {mps_noise.min():.4f}/{mps_noise.max():.4f}"
            )

            # Test kmeans clustering with MPS backend
            from src.utils.gpu_utils import apply_kmeans_clustering_gpu

            # Create test data for clustering
            test_data = np.random.random((1000, 2))
            centroids, labels = apply_kmeans_clustering_gpu(
                data=test_data, n_clusters=5, backend="mps"
            )
            print(
                f"MPS kmeans clustering successful, centroids shape: {centroids.shape}"
            )
            print(f"Unique cluster labels: {np.unique(labels)}")

        except Exception as e:
            print(f"Error testing MPS backend: {e}")

except ImportError as e:
    print(f"\nError importing BaseGenerator: {e}")
except Exception as e:
    print(f"\nError testing BaseGenerator: {e}")

# Print summary
print("\nGPU Acceleration Test Summary:")
print(f"GPU available: {is_gpu_available()}")
print(f"Available backends: {get_available_backends()}")
if platform.system() == "Darwin":
    print(f"macOS MPS support: {'Available' if MPS_AVAILABLE else 'Not available'}")
    print(
        f"macOS metalgpu support: {'Available' if METALGPU_AVAILABLE else 'Not available'}"
    )
