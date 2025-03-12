#!/usr/bin/env python3
"""
Abstract base class and implementations for noise generators.

This module provides an interface for noise generators and concrete implementations
that can be injected into generator classes.
"""


# Standard library imports
import abc
import contextlib
import itertools
import random
from typing import List, Optional

# Third-party library imports
import numpy as np
from numpy.random import Generator, PCG64

# Optional dependencies
try:
    from perlin_noise import PerlinNoise

    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    print("PerlinNoise package is not available. Using fallback noise generator.")


class NoiseGenerator(abc.ABC):
    """Abstract base class for noise generators."""

    @abc.abstractmethod
    def generate_noise(
        self,
        width: int,
        height: int,
        scale: float = 0.1,
        octaves: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a 2D noise array.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            scale: Scale of the noise (higher values = more zoomed out)
            octaves: Number of octaves for the noise
            seed: Random seed for reproducibility

        Returns:
            A 2D numpy array of noise values between 0 and 1
        """
        pass

    @abc.abstractmethod
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
        Generate multi-octave noise.

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
        pass


class PerlinNoiseGenerator(NoiseGenerator):
    """Perlin noise generator implementation."""

    def __init__(self):
        """Initialize the Perlin noise generator."""
        if not PERLIN_AVAILABLE:
            raise ImportError(
                "PerlinNoise package is not available. Install it with 'pip install perlin-noise'"
            )

    def generate_noise(
        self,
        width: int,
        height: int,
        scale: float = 0.1,
        octaves: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a 2D Perlin noise array.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            scale: Scale of the noise (higher values = more zoomed out)
            octaves: Number of octaves for the noise
            seed: Random seed for reproducibility

        Returns:
            A 2D numpy array of noise values between 0 and 1
        """
        # Set seed if provided
        if seed is not None:
            random.seed(seed)

        # Create Perlin noise object
        noise = PerlinNoise(octaves=octaves, seed=seed)

        # Generate noise grid
        noise_grid = np.zeros((height, width))
        for y, x in itertools.product(range(height), range(width)):
            noise_value = noise([x * scale, y * scale])
            # Convert from [-1, 1] to [0, 1]
            noise_grid[y, x] = (noise_value + 1) / 2

        return noise_grid

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
        Generate multi-octave Perlin noise.

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
        if octaves is None:
            octaves = [1, 2, 4, 8]

        if weights is None:
            weights = [1.0, 0.5, 0.25, 0.125]

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Set seed if provided
        if seed is not None:
            random.seed(seed)

        # Create noise objects for each octave
        noise_objects = [PerlinNoise(octaves=o, seed=seed) for o in octaves]

        # Generate combined noise grid
        noise_grid = np.zeros((height, width))

        for i, (noise, weight) in enumerate(zip(noise_objects, weights)):
            octave_scale = scale * (2**i)  # Scale increases with octave
            for y, x in itertools.product(range(height), range(width)):
                noise_value = noise([x * octave_scale, y * octave_scale])
                # Convert from [-1, 1] to [0, 1] and apply weight
                noise_grid[y, x] += ((noise_value + 1) / 2) * weight

        # Ensure values are in [0, 1]
        noise_grid = np.clip(noise_grid, 0, 1)

        return noise_grid


# Optional dependencies
try:
    from scipy.ndimage import zoom

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, using fallback implementation for noise scaling.")


class SimplexNoiseGenerator(NoiseGenerator):
    """
    Simplex noise generator implementation.

    This is a fallback implementation that simulates noise when PerlinNoise is not available.
    """

    def generate_noise(
        self,
        width: int,
        height: int,
        scale: float = 0.1,
        octaves: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a 2D simplex-like noise array.

        Args:
            width: Width of the noise array
            height: Height of the noise array
            scale: Scale of the noise (higher values = more zoomed out)
            octaves: Number of octaves for the noise (ignored in this implementation)
            seed: Random seed for reproducibility

        Returns:
            A 2D numpy array of noise values between 0 and 1
        """
        # Create random number generator with seed if provided
        rng = Generator(PCG64(seed if seed is not None else 42))
        
        # Generate random noise
        base_noise = rng.random((int(height * scale * 2), int(width * scale * 2)))

        # Resize to desired dimensions using bilinear interpolation
        if SCIPY_AVAILABLE:
            # Use scipy for better interpolation if available
            noise_grid = zoom(
                base_noise,
                (height / (height * scale * 2), width / (width * scale * 2)),
                order=1,
            )
        else:
            # Simple fallback using numpy if scipy is not available
            # This is a very basic resizing that won't look as good
            h_indices = np.linspace(0, base_noise.shape[0] - 1, height).astype(int)
            w_indices = np.linspace(0, base_noise.shape[1] - 1, width).astype(int)
            noise_grid = base_noise[h_indices[:, np.newaxis], w_indices]

        # Ensure the dimensions are correct
        noise_grid = noise_grid[:height, :width]

        return np.clip(noise_grid, 0, 1)

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
        Generate multi-octave simplex-like noise.

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
        if octaves is None:
            octaves = [1, 2, 4, 8]

        if weights is None:
            weights = [1.0, 0.5, 0.25, 0.125]

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate combined noise grid
        noise_grid = np.zeros((height, width))

        for i, (octave, weight) in enumerate(zip(octaves, weights)):
            octave_scale = scale * (2**i)  # Scale increases with octave
            octave_noise = self.generate_noise(
                width, height, octave_scale, octave, seed
            )
            noise_grid += octave_noise * weight

        # Ensure values are in [0, 1]
        noise_grid = np.clip(noise_grid, 0, 1)

        return noise_grid


# Factory function to get the appropriate noise generator
def get_noise_generator() -> NoiseGenerator:
    """
    Get the appropriate noise generator based on available dependencies.

    Returns:
        A NoiseGenerator implementation
    """
    if PERLIN_AVAILABLE:
        with contextlib.suppress(ImportError):
            return PerlinNoiseGenerator()
    return SimplexNoiseGenerator()
