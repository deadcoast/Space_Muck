#!/usr/bin/env python3
"""
GPU-accelerated noise generation for Space Muck.

This module provides GPU-accelerated implementations of noise generation algorithms
used in procedural generation, with fallback mechanisms for systems without GPU support.
"""

# Standard library imports
import itertools
import logging
import random
from typing import List, Optional, Tuple

# Third-party library imports
import numpy as np

# Local application imports
from .gpu_utils import (  # Standard library imports; Third-party library imports; Local imports
    NoiseGenerator,
    apply_noise_generation_gpu,
    get_available_backends,
    get_noise_generator,
    is_gpu_available,
    to_cpu,
)

# Optional dependencies
try:
    from numba import cuda

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

    @staticmethod
    def _normalize_parameters(
        octaves: List[int] = None,
        weights: List[float] = None,
        seed: Optional[int] = None,
    ) -> tuple:
        """Normalize input parameters for noise generation."""
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

        return octaves, normalized_weights

    @staticmethod
    def _generate_octave_noise(
        width: int,
        height: int,
        scale: float,
        octave: int,
        octave_index: int,
        seed: Optional[int],
        backend: str,
    ) -> np.ndarray:
        """Generate noise for a single octave with the given parameters."""
        octave_scale = scale * (2**octave_index)
        octave_seed = seed + octave_index if seed is not None else None

        return apply_noise_generation_gpu(
            width=width,
            height=height,
            scale=octave_scale,
            octaves=octave,
            seed=octave_seed,
            backend=backend,
        )

    def _combine_octaves_cupy(
        self,
        width: int,
        height: int,
        scale: float,
        octaves: List[int],
        normalized_weights: List[float],
        seed: Optional[int],
    ) -> np.ndarray:
        """Combine multiple octaves using CuPy backend."""
        result = cp.zeros((height, width), dtype=cp.float32)

        for i, (octave, weight) in enumerate(zip(octaves, normalized_weights)):
            octave_noise = self._generate_octave_noise(
                width, height, scale, octave, i, seed, "cupy"
            )

            # Add weighted noise to result
            result += cp.asarray(octave_noise) * weight

        # Ensure values are in [0, 1]
        result = cp.clip(result, 0, 1)
        return to_cpu(result)

    def _combine_octaves_cuda(
        self,
        width: int,
        height: int,
        scale: float,
        octaves: List[int],
        normalized_weights: List[float],
        seed: Optional[int],
    ) -> np.ndarray:
        """Combine multiple octaves using CUDA backend."""
        result = np.zeros((height, width), dtype=np.float32)

        for i, (octave, weight) in enumerate(zip(octaves, normalized_weights)):
            octave_noise = self._generate_octave_noise(
                width, height, scale, octave, i, seed, "cuda"
            )

            # Add weighted noise to result
            result += octave_noise * weight

        # Ensure values are in [0, 1]
        return np.clip(result, 0, 1)

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
        # Normalize parameters
        octaves, normalized_weights = self._normalize_parameters(octaves, weights, seed)

        # Use CPU if backend is set to CPU or GPU is not available
        if self.backend == "cpu" or not is_gpu_available():
            return self.cpu_generator.generate_multi_octave_noise(
                width=width,
                height=height,
                scale=scale,
                octaves=octaves,
                weights=weights,
                seed=seed,
            )

        try:
            # Choose appropriate backend implementation
            if self.backend == "cupy" and CUPY_AVAILABLE:
                return self._combine_octaves_cupy(
                    width, height, scale, octaves, normalized_weights, seed
                )

            elif self.backend == "cuda" and CUDA_AVAILABLE:
                return self._combine_octaves_cuda(
                    width, height, scale, octaves, normalized_weights, seed
                )

            # Fallback to CPU for unsupported backends
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

    def _initialize_warp_noise(
        self,
        width: int,
        height: int,
        warp_scale: float,
        warp_octaves: int,
        warp_seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate warp noise for x and y coordinates.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            warp_scale: Scale of the warping noise
            warp_octaves: Number of octaves for the warping noise
            warp_seed: Seed for the warp noise

        Returns:
            Tuple of warp_x and warp_y noise arrays
        """
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

        return warp_x, warp_y

    @staticmethod
    def _create_warped_coordinates(
        width: int,
        height: int,
        warp_x: np.ndarray,
        warp_y: np.ndarray,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create warped coordinates for domain warping.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            warp_x: X-coordinate warp noise
            warp_y: Y-coordinate warp noise
            scale: Base scale factor

        Returns:
            Tuple of warped_x and warped_y coordinate arrays
        """
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

        return warped_x, warped_y

    @staticmethod
    def _generate_cupy_warped_noise(
        width: int,
        height: int,
        warped_x: np.ndarray,
        warped_y: np.ndarray,
        scale: float,
        octaves: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """
        Generate domain-warped noise using CuPy.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            warped_x: Warped X coordinates
            warped_y: Warped Y coordinates
            scale: Scale of the base noise
            octaves: Number of octaves for the base noise
            seed: Random seed

        Returns:
            A 2D numpy array of noise values
        """
        # CuPy implementation
        result = cp.zeros((height, width), dtype=cp.float32)
        warped_x_gpu = cp.asarray(warped_x)
        warped_y_gpu = cp.asarray(warped_y)

        # Generate noise for each octave
        for i in range(octaves):
            octave_scale = scale * (2**i)
            octave_weight = 1.0 / (2**i)
            # octave_seed would be used in a more complete implementation
            # but is not needed in this simplified version

            # This is a simplified placeholder implementation
            noise_values = cp.sin(warped_x_gpu * octave_scale * 10) * cp.cos(
                warped_y_gpu * octave_scale * 10
            )
            noise_values = (noise_values + 1) / 2  # Normalize to [0, 1]

            result += noise_values * octave_weight

        # Ensure values are in [0, 1]
        result = cp.clip(result, 0, 1)
        return to_cpu(result)

    def _generate_cpu_warped_noise(
        self,
        width: int,
        height: int,
        warped_x: np.ndarray,
        warped_y: np.ndarray,
        scale: float,
        octaves: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """
        Generate domain-warped noise using CPU (fallback).

        Args:
            width: Width of the noise array
            height: Height of the noise array
            warped_x: Warped X coordinates
            warped_y: Warped Y coordinates
            scale: Scale of the base noise
            octaves: Number of octaves for the base noise
            seed: Random seed

        Returns:
            A 2D numpy array of noise values
        """
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
            warped_noise = self._sample_noise_at_warped_coordinates(
                octave_noise, warped_x, warped_y, width, height
            )

            # Add weighted noise to result
            result += warped_noise * octave_weight

        # Ensure values are in [0, 1]
        return np.clip(result, 0, 1)

    @staticmethod
    def _sample_noise_at_warped_coordinates(
        noise: np.ndarray,
        warped_x: np.ndarray,
        warped_y: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Sample noise values at warped coordinates.

        Args:
            noise: Base noise array
            warped_x: Warped X coordinates
            warped_y: Warped Y coordinates
            width: Width of the noise array
            height: Height of the noise array

        Returns:
            Sampled noise array
        """
        warped_noise = np.zeros((height, width), dtype=np.float32)
        for y, x in itertools.product(range(height), range(width)):
            # Get warped coordinates
            wx = int(warped_x[y, x] * width) % width
            wy = int(warped_y[y, x] * height) % height

            # Sample noise at warped position
            warped_noise[y, x] = noise[wy, wx]

        return warped_noise

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

        # Generate warp noise and create warped coordinates
        warp_x, warp_y = self._initialize_warp_noise(
            width, height, warp_scale, warp_octaves, warp_seed
        )
        warped_x, warped_y = self._create_warped_coordinates(
            width, height, warp_x, warp_y, scale
        )

        # Generate base noise using warped coordinates
        if self.backend != "cpu" and is_gpu_available():
            try:
                if self.backend != "cupy" or not CUPY_AVAILABLE:
                    # Fall back to CPU implementation for now
                    raise NotImplementedError(
                        "Only CuPy backend is currently implemented for domain warping"
                    )

                return self._generate_cupy_warped_noise(
                    width, height, warped_x, warped_y, scale, octaves, seed
                )

            except Exception as e:
                logging.warning(
                    f"GPU domain warping failed: {str(e)}. Falling back to CPU."
                )

        # CPU implementation (fallback)
        return self._generate_cpu_warped_noise(
            width, height, warped_x, warped_y, scale, octaves, seed
        )


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
