"""
BaseGenerator class: The base class for all procedural generation entities in the game.
"""

# Standard library imports
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party library imports
import numpy as np

# Optional dependencies
try:
    from perlin_noise import PerlinNoise
    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    print("PerlinNoise package is not available. Using fallback noise generator.")

# Local application imports
from src.entities.base_entity import BaseEntity
from src.utils.dependency_injection import inject
from src.utils.noise_generator import NoiseGenerator, get_noise_generator


@inject
class BaseGenerator(BaseEntity):
    """
    Base class for all procedural generator entities in the game.
    Provides common functionality for seed management, noise generation, and parameter handling.
    """

    def __init__(
        self,
        entity_id: Optional[str] = None,
        entity_type: str = "generator",
        seed: Optional[int] = None,
        width: int = 100,
        height: int = 100,
        color: Tuple[int, int, int] = (100, 200, 100),
        position: Optional[Tuple[int, int]] = None,
        noise_generator: Optional[NoiseGenerator] = None,
    ) -> None:
        """
        Initialize a base generator entity.

        Args:
            entity_id: Unique identifier for the entity (defaults to a UUID)
            entity_type: Type of generator (e.g., "procedural", "asteroid", "symbiote")
            seed: Random seed for reproducible generation
            width: Width of the generated area
            height: Height of the generated area
            color: RGB color tuple for visualization
            position: Initial position as (x, y) tuple
            noise_generator: Injected noise generator (defaults to auto-selected implementation)
        """
        # Call the parent class constructor
        super().__init__(
            entity_id=entity_id, entity_type=entity_type, color=color, position=position
        )

        # Generator-specific attributes
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        self.width = width
        self.height = height

        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Use the injected noise generator or get the default one
        self.noise_generator = noise_generator or get_noise_generator()

        # Cache for generated noise layers
        self._noise_cache = {}

        # Common generation parameters
        self.parameters = {
            "density": 0.3,  # Base density of generated elements
            "complexity": 0.5,  # Complexity of patterns (0-1)
            "turbulence": 0.4,  # Chaos factor for generation (0-1)
            "iterations": 3,  # Default iterations for algorithms
            "rare_chance": 0.1,  # Chance of rare elements
        }

        logging.info(
            f"Generator initialized: {self.entity_type} (ID: {self.entity_id}, Seed: {self.seed})"
        )

    def generate_noise_layer(
        self, noise_type: str = "medium", scale: float = 0.1
    ) -> np.ndarray:
        """
        Generate a noise layer using the specified noise generator.

        Args:
            noise_type: Type of noise to use ("low", "medium", "high", "detail")
            scale: Scale factor for noise (lower = more zoomed out)

        Returns:
            np.ndarray: 2D grid of noise values
        """
        # Map noise_type to octaves
        octaves_map = {"low": 3, "medium": 5, "high": 8, "detail": 12}

        if noise_type not in octaves_map:
            logging.warning(
                f"Unknown noise type '{noise_type}', using 'medium' instead"
            )
            noise_type = "medium"

        # Create a cache key
        cache_key = f"{noise_type}_{scale}_{self.width}_{self.height}_{self.seed}"

        # Check if we have this noise layer cached
        if cache_key in self._noise_cache:
            return self._noise_cache[cache_key]

        # Generate the noise layer
        octaves = octaves_map[noise_type]
        noise_grid = self.noise_generator.generate_noise(
            width=self.width,
            height=self.height,
            scale=scale,
            octaves=octaves,
            seed=self.seed + octaves,  # Use different seeds for different octaves
        )

        # Cache the result
        self._noise_cache[cache_key] = noise_grid

        return noise_grid

    def apply_cellular_automaton(
        self,
        grid: np.ndarray,
        birth_set: Optional[Set[int]] = None,
        survival_set: Optional[Set[int]] = None,
        iterations: int = 3,
        wrap: bool = True,
    ) -> np.ndarray:
        """
        Apply cellular automaton rules to a grid.

        Args:
            grid: Input grid to evolve
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            iterations: Number of iterations to perform
            wrap: Whether to wrap around grid edges

        Returns:
            np.ndarray: Evolved grid
        """
        # Use default rule sets if not provided
        if birth_set is None:
            birth_set = {3}
        if survival_set is None:
            survival_set = {2, 3}

        # Convert to binary grid
        binary_grid = (grid > 0).astype(np.int8)
        result_grid = binary_grid.copy()

        # Define neighbor offsets once
        neighbor_offsets = [
            (dx, dy)
            for dy in [-1, 0, 1]
            for dx in [-1, 0, 1]
            if dx != 0 or dy != 0
        ]

        # Apply cellular automaton for specified iterations
        for _ in range(iterations):
            new_grid = result_grid.copy()

            # Process each cell in the grid
            for y in range(self.height):
                for x in range(self.width):
                    # Count live neighbors
                    neighbors = self._count_neighbors(result_grid, x, y, neighbor_offsets, wrap)

                    # Apply rules
                    if result_grid[y, x] == 1:  # Cell is alive
                        if neighbors not in survival_set:
                            new_grid[y, x] = 0  # Cell dies
                    elif neighbors in birth_set:  # Cell is dead
                        new_grid[y, x] = 1  # Cell is born

            result_grid = new_grid

        # Preserve original values where cells are alive
        return grid * result_grid

    def _count_neighbors(
        self,
        grid: np.ndarray,
        x: int,
        y: int,
        neighbor_offsets: List[Tuple[int, int]],
        wrap: bool
    ) -> int:
        """
        Count the number of live neighbors for a cell.
        
        Args:
            grid: The grid to analyze
            x: X coordinate of the cell
            y: Y coordinate of the cell
            neighbor_offsets: List of (dx, dy) offsets for neighbors
            wrap: Whether to wrap around grid edges
            
        Returns:
            Number of live neighbors
        """
        count = 0
        
        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            
            if wrap:
                nx = nx % self.width
                ny = ny % self.height
            elif nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
                
            count += grid[ny, nx]
            
        return count
        
    def create_clusters(
        self,
        grid: np.ndarray,
        num_clusters: int = 5,
        cluster_value_multiplier: float = 2.0,
    ) -> np.ndarray:
        """
        Create clusters of higher values in the grid.

        Args:
            grid: Input grid
            num_clusters: Number of clusters to create
            cluster_value_multiplier: Multiplier for values in clusters

        Returns:
            np.ndarray: Grid with clusters
        """
        result_grid = grid.copy()

        # Find coordinates of non-zero cells
        non_zero_coords = np.argwhere(grid > 0)

        if len(non_zero_coords) < num_clusters:
            return result_grid

        # Randomly select cluster centers
        cluster_indices = np.random.choice(
            len(non_zero_coords), num_clusters, replace=False
        )
        cluster_centers = non_zero_coords[cluster_indices]

        # Create clusters around centers
        for center in cluster_centers:
            cy, cx = center
            radius = np.random.randint(3, 10)

            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    # Calculate distance from center
                    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                    if distance <= radius and grid[y, x] > 0:
                        # Apply value multiplier with falloff based on distance
                        falloff = 1 - (distance / radius)
                        result_grid[y, x] *= (
                            1 + (cluster_value_multiplier - 1) * falloff
                        )

        return result_grid

    def apply_threshold(
        self, grid: np.ndarray, threshold: float = 0.5, value: float = 1.0
    ) -> np.ndarray:
        """
        Apply a threshold to the grid.

        Args:
            grid: Input grid
            threshold: Threshold value
            value: Value to set for cells above threshold

        Returns:
            np.ndarray: Thresholded grid
        """
        return np.where(grid > threshold, value, 0)

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a generation parameter.

        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
        logging.info(f"Generator parameter set: {name} = {value}")

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a generation parameter.

        Args:
            name: Parameter name
            default: Default value if parameter doesn't exist

        Returns:
            Any: Parameter value
        """
        return self.parameters.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the generator to a dictionary for serialization.
        Extends the BaseEntity to_dict method.

        Returns:
            Dict[str, Any]: Dictionary representation of the generator
        """
        data = super().to_dict()
        data.update(
            {
                "seed": self.seed,
                "width": self.width,
                "height": self.height,
                "parameters": self.parameters,
            }
        )
        return data

    def generate_multi_octave_noise(
        self,
        scale: float = 0.1,
        octaves: Optional[list] = None,
        weights: Optional[list] = None,
    ) -> np.ndarray:
        """
        Generate multi-octave noise.

        Args:
            scale: Base scale of the noise
            octaves: List of octave values (default: [3, 5, 8])
            weights: List of weights for each octave (default: [1.0, 0.5, 0.25])

        Returns:
            np.ndarray: 2D grid of noise values
        """
        if octaves is None:
            octaves = [3, 5, 8]

        if weights is None:
            weights = [1.0, 0.5, 0.25]

        # Create a cache key
        cache_key = (
            f"multi_{scale}_{self.width}_{self.height}_{self.seed}_{octaves}_{weights}"
        )

        # Check if we have this noise layer cached
        if cache_key in self._noise_cache:
            return self._noise_cache[cache_key]

        # Generate the multi-octave noise
        noise_grid = self.noise_generator.generate_multi_octave_noise(
            width=self.width,
            height=self.height,
            scale=scale,
            octaves=octaves,
            weights=weights,
            seed=self.seed,
        )

        # Cache the result
        self._noise_cache[cache_key] = noise_grid

        return noise_grid

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseGenerator":
        """
        Create a generator from a dictionary.
        Extends the BaseEntity from_dict method.

        Args:
            data: Dictionary containing generator data

        Returns:
            BaseGenerator: New generator instance
        """
        generator = super().from_dict(data)
        generator.seed = data.get("seed", random.randint(1, 1000000))
        generator.width = data.get("width", 100)
        generator.height = data.get("height", 100)
        generator.parameters = data.get("parameters", {})

        # Reinitialize noise generators with the loaded seed
        random.seed(generator.seed)
        np.random.seed(generator.seed)
        
        # Only create PerlinNoise generators if the library is available
        if PERLIN_AVAILABLE:
            generator.noise_generators = {
                "low": PerlinNoise(octaves=3, seed=generator.seed),
                "medium": PerlinNoise(octaves=5, seed=generator.seed + 1),
                "high": PerlinNoise(octaves=8, seed=generator.seed + 2),
                "detail": PerlinNoise(octaves=12, seed=generator.seed + 3),
            }
        else:
            # Use the noise generator from the dependency injection system as fallback
            noise_gen = generator.noise_generator or get_noise_generator()
            logging.warning("Using fallback noise generator instead of PerlinNoise")
            # Create noise generators dictionary using the fallback noise generator
            generator.noise_generators = {
                "low": noise_gen,
                "medium": noise_gen,
                "high": noise_gen,
                "detail": noise_gen
            }

        return generator
