#!/usr/bin/env python3
"""
GPU-accelerated noise generation for Space Muck.

This module provides GPU-accelerated implementations of noise generation algorithms
used in procedural generation, with fallback mechanisms for systems without GPU support.
"""

# Standard library imports
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import numpy as np

# Local imports
from utils.gpu_utils import (
    is_gpu_available,
    get_available_backends,
    to_gpu,
    to_cpu,
    apply_noise_generation_gpu,
)
from utils.noise_generator import NoiseGenerator, get_noise_generator

# Optional dependencies
try:
    import numba
    from numba import cuda, float32, int32

    NUMBA_AVAILABLE = True
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPUNoiseGenerator(NoiseGenerator):
    """
    GPU-accelerated noise generator implementation.

    This class provides GPU-accelerated implementations of noise generation algorithms
    with automatic fallback to CPU when GPU is not available.
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize the GPU noise generator.

        Args:
            backend: GPU backend to use ('cuda', 'cupy', 'auto')
        """
        self.backend = backend

        # Determine the best available backend
        if self.backend == "auto":
            available_backends = get_available_backends()
            if "cuda" in available_backends:
                self.backend = "cuda"
            elif "cupy" in available_backends:
                self.backend = "cupy"
            else:
                self.backend = "cpu"

        # Get fallback CPU noise generator
        self.cpu_generator = get_noise_generator()

        logging.info(f"GPUNoiseGenerator initialized with backend: {self.backend}")

    def generate_noise(
        self,
        width: int,
        height: int,
        scale: float = 0.1,
        octaves: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a 2D noise array using GPU acceleration if available.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            scale: Scale of the noise (higher values = more zoomed out)
            octaves: Number of octaves for the noise
            seed: Random seed for reproducibility

        Returns:
            A 2D numpy array of noise values between 0 and 1
        """
        if self.backend == "cpu" or not is_gpu_available():
            # Use CPU implementation
            return self.cpu_generator.generate_noise(
                width=width, height=height, scale=scale, octaves=octaves, seed=seed
            )
        try:
            return apply_noise_generation_gpu(
                width=width,
                height=height,
                scale=scale,
                octaves=octaves,
                seed=seed,
                backend=self.backend,
            )
        except Exception as e:
            logging.warning(
                f"GPU noise generation failed: {str(e)}. Falling back to CPU."
            )
            return self.cpu_generator.generate_noise(
                width=width, height=height, scale=scale, octaves=octaves, seed=seed
            )

    def generate_multi_octave_noise(
        self,
        width: int,
        height: int,
        scale: float = 0.1,
        octaves: List[int] = None,
        weights: List[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate multi-octave noise using GPU acceleration if available.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            scale: Base scale of the noise
            octaves: List of octave values
            weights: List of weights for each octave
            seed: Random seed for reproducibility

        Returns:
            A 2D numpy array of noise values between 0 and 1
        """
        # Default octaves and weights
        if octaves is None:
            octaves = [1, 2, 4, 8]

        if weights is None:
            weights = [1.0, 0.5, 0.25, 0.125]

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.backend == "cpu" or not is_gpu_available():
            # Use CPU implementation
            return self.cpu_generator.generate_multi_octave_noise(
                width=width,
                height=height,
                scale=scale,
                octaves=octaves,
                weights=weights,
                seed=seed,
            )
        try:
            # Generate noise for each octave and combine
            if self.backend == "cupy" and CUPY_AVAILABLE:
                # CuPy implementation
                result = cp.zeros((height, width), dtype=cp.float32)

                for i, (octave, weight) in enumerate(
                    zip(octaves, normalized_weights)
                ):
                    octave_scale = scale * (2**i)
                    octave_seed = seed + i if seed is not None else None

                    # Generate noise for this octave
                    octave_noise = apply_noise_generation_gpu(
                        width=width,
                        height=height,
                        scale=octave_scale,
                        octaves=octave,
                        seed=octave_seed,
                        backend="cupy",
                    )

                    # Add weighted noise to result
                    result += cp.asarray(octave_noise) * weight

                # Ensure values are in [0, 1]
                result = cp.clip(result, 0, 1)
                return to_cpu(result)

            elif self.backend == "cuda" and CUDA_AVAILABLE:
                # CUDA implementation
                result = np.zeros((height, width), dtype=np.float32)

                for i, (octave, weight) in enumerate(
                    zip(octaves, normalized_weights)
                ):
                    octave_scale = scale * (2**i)
                    octave_seed = seed + i if seed is not None else None

                    # Generate noise for this octave
                    octave_noise = apply_noise_generation_gpu(
                        width=width,
                        height=height,
                        scale=octave_scale,
                        octaves=octave,
                        seed=octave_seed,
                        backend="cuda",
                    )

                    # Add weighted noise to result
                    result += octave_noise * weight

                # Ensure values are in [0, 1]
                return np.clip(result, 0, 1)

            else:
                # Fallback to CPU
                raise ValueError(f"Unsupported backend: {self.backend}")

        except Exception as e:
            logging.warning(
                f"GPU multi-octave noise generation failed: {str(e)}. Falling back to CPU."
            )
            return self.cpu_generator.generate_multi_octave_noise(
                width=width,
                height=height,
                scale=scale,
                octaves=octaves,
                weights=weights,
                seed=seed,
            )


class FractalNoiseGenerator(GPUNoiseGenerator):
    """
    GPU-accelerated fractal noise generator.

    This class extends the GPUNoiseGenerator to provide fractal noise generation
    with domain warping and other advanced techniques.
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize the fractal noise generator.

        Args:
            backend: GPU backend to use ('cuda', 'cupy', 'auto')
        """
        super().__init__(backend=backend)

    def generate_domain_warped_noise(
        self,
        width: int,
        height: int,
        scale: float = 0.1,
        warp_scale: float = 0.5,
        warp_octaves: int = 2,
        octaves: int = 4,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate domain-warped noise for more organic patterns.

        Domain warping uses one noise function to distort the input coordinates
        of another noise function, creating more organic and interesting patterns.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            scale: Scale of the base noise
            warp_scale: Scale of the warping noise
            warp_octaves: Number of octaves for the warping noise
            octaves: Number of octaves for the base noise
            seed: Random seed for reproducibility

        Returns:
            A 2D numpy array of noise values between 0 and 1
        """
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            warp_seed = seed + 1000  # Different seed for warp noise
        else:
            warp_seed = None

        # Generate warp noise for x and y coordinates
        warp_x = self.generate_noise(
            width=width,
            height=height,
            scale=warp_scale,
            octaves=warp_octaves,
            seed=warp_seed,
        )

        warp_y = self.generate_noise(
            width=width,
            height=height,
            scale=warp_scale,
            octaves=warp_octaves,
            seed=warp_seed + 1 if warp_seed is not None else None,
        )

        # Scale the warp noise to create displacement
        warp_strength = 10.0 * scale
        warp_x = (warp_x - 0.5) * warp_strength
        warp_y = (warp_y - 0.5) * warp_strength

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Apply warping
        warped_x = x_coords + warp_x * width
        warped_y = y_coords + warp_y * height

        # Normalize coordinates to [0, 1] range for noise generation
        warped_x = warped_x / width
        warped_y = warped_y / height

        # Generate base noise using warped coordinates
        if self.backend != "cpu" and is_gpu_available():
            try:
                if self.backend == "cupy" and CUPY_AVAILABLE:
                    # CuPy implementation
                    result = cp.zeros((height, width), dtype=cp.float32)
                    warped_x_gpu = cp.asarray(warped_x)
                    warped_y_gpu = cp.asarray(warped_y)

                    # Generate a random permutation table for noise
                    perm_size = 256
                    # Generate noise for each octave
                    for i in range(octaves):
                        octave_scale = scale * (2**i)
                        octave_weight = 1.0 / (2**i)
                        octave_seed = seed + i + 2000 if seed is not None else None

                        # Custom noise generation using warped coordinates
                        # This is a simplified version - in practice, you'd use a more sophisticated approach
                        noise_values = cp.zeros((height, width), dtype=cp.float32)

                        perm = cp.asarray(
                            np.random.permutation(perm_size), dtype=cp.int32
                        )

                        # This is just a placeholder - actual implementation would be more complex
                        noise_values = cp.sin(
                            warped_x_gpu * octave_scale * 10
                        ) * cp.cos(warped_y_gpu * octave_scale * 10)
                        noise_values = (noise_values + 1) / 2  # Normalize to [0, 1]

                        result += noise_values * octave_weight

                    # Ensure values are in [0, 1]
                    result = cp.clip(result, 0, 1)
                    return to_cpu(result)
                else:
                    # Fall back to CPU implementation for now
                    raise NotImplementedError(
                        "Only CuPy backend is currently implemented for domain warping"
                    )
            except Exception as e:
                logging.warning(
                    f"GPU domain warping failed: {str(e)}. Falling back to CPU."
                )

        # CPU implementation (fallback)
        result = np.zeros((height, width), dtype=np.float32)

        # Generate noise for each octave
        for i in range(octaves):
            octave_scale = scale * (2**i)
            octave_weight = 1.0 / (2**i)
            octave_seed = seed + i + 2000 if seed is not None else None

            # Create a new noise generator for this octave
            octave_noise = self.cpu_generator.generate_noise(
                width=width,
                height=height,
                scale=octave_scale,
                octaves=1,
                seed=octave_seed,
            )

            # Sample the noise using warped coordinates
            warped_noise = np.zeros((height, width), dtype=np.float32)
            for y in range(height):
                for x in range(width):
                    # Get warped coordinates
                    wx = int(warped_x[y, x] * width) % width
                    wy = int(warped_y[y, x] * height) % height

                    # Sample noise at warped position
                    warped_noise[y, x] = octave_noise[wy, wx]

            # Add weighted noise to result
            result += warped_noise * octave_weight

        # Ensure values are in [0, 1]
        return np.clip(result, 0, 1)


def get_gpu_noise_generator(backend: str = "auto") -> NoiseGenerator:
    """
    Get a GPU-accelerated noise generator if available, otherwise fall back to CPU.

    Args:
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        A NoiseGenerator implementation
    """
    if is_gpu_available():
        return GPUNoiseGenerator(backend=backend)
    logging.info("No GPU acceleration available, using CPU noise generator")
    return get_noise_generator()


def get_fractal_noise_generator(backend: str = "auto") -> NoiseGenerator:
    """
    Get a fractal noise generator with domain warping capabilities.

    Args:
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        A NoiseGenerator implementation with domain warping capabilities
    """
    if is_gpu_available():
        return FractalNoiseGenerator(backend=backend)
    logging.info("No GPU acceleration available, using CPU noise generator")
    return get_noise_generator()
