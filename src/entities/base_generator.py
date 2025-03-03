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

        # Performance optimization: Include dimensions in cache key
        # This allows reusing noise layers across different generator instances with same parameters
        cache_key = f"{noise_type}_{scale}_{self.width}_{self.height}_{self.seed}"

        # Check if we have this noise layer cached
        if cache_key in self._noise_cache:
            return self._noise_cache[cache_key]

        # Performance optimization: Use multithreading for larger grids
        octaves = octaves_map[noise_type]

        # For larger grids, use optimized generation with parallel processing
        if self.width * self.height > 10000:  # Threshold for large grids
            try:
                # Use optimized noise generation with parallel processing
                from concurrent.futures import ThreadPoolExecutor
                import math

                # Split the grid into chunks for parallel processing
                chunk_size = math.ceil(self.height / 4)  # Process in 4 chunks
                noise_grid = np.zeros((self.height, self.width), dtype=float)

                def generate_chunk(start_y, end_y):
                    chunk = self.noise_generator.generate_noise(
                        width=self.width,
                        height=end_y - start_y,
                        scale=scale,
                        octaves=octaves,
                        seed=self.seed + octaves,
                    )
                    return start_y, chunk

                # Generate chunks in parallel
                chunks = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for i in range(0, self.height, chunk_size):
                        end_y = min(i + chunk_size, self.height)
                        chunks.append(executor.submit(generate_chunk, i, end_y))

                    # Combine the chunks
                    for future in chunks:
                        start_y, chunk = future.result()
                        end_y = min(start_y + chunk.shape[0], self.height)
                        noise_grid[start_y:end_y, :] = chunk

            except Exception as e:
                # Fall back to standard generation if parallel processing fails
                logging.warning(f"Falling back to standard noise generation: {str(e)}")
                noise_grid = self.noise_generator.generate_noise(
                    width=self.width,
                    height=self.height,
                    scale=scale,
                    octaves=octaves,
                    seed=self.seed + octaves,
                )
        else:
            # Standard generation for smaller grids
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
        
        # Create cache key for this operation
        cache_key = f"ca_{hash(frozenset(birth_set))}_{hash(frozenset(survival_set))}_{iterations}_{wrap}_{hash(binary_grid.tobytes())}"
        
        # Check if we have this result cached
        if hasattr(self, "_ca_cache") and cache_key in self._ca_cache:
            return self._ca_cache[cache_key] * grid
        
        # Initialize cache if not exists
        if not hasattr(self, "_ca_cache"):
            self._ca_cache = {}

        # Performance optimization: Use scipy's convolve2d for neighbor counting if available
        try:
            from scipy import signal
            
            # Create kernel for neighbor counting
            kernel = np.ones((3, 3), dtype=np.int8)
            kernel[1, 1] = 0  # Don't count the cell itself
            
            # Apply cellular automaton for specified iterations
            for _ in range(iterations):
                # Count neighbors using convolution
                if wrap:
                    # Use 'wrap' mode for boundary conditions
                    neighbors = signal.convolve2d(result_grid, kernel, mode='same', boundary='wrap')
                else:
                    # Use 'fill' mode with zero padding for boundary conditions
                    neighbors = signal.convolve2d(result_grid, kernel, mode='same', boundary='fill')
                
                # Create masks for rule application (vectorized operations)
                alive_mask = (result_grid == 1)
                dead_mask = (result_grid == 0)
                
                # Create masks for birth and survival based on neighbor counts
                birth_mask = np.zeros_like(neighbors, dtype=bool)
                survival_mask = np.zeros_like(neighbors, dtype=bool)
                
                for n in birth_set:
                    birth_mask |= (neighbors == n)
                    
                for n in survival_set:
                    survival_mask |= (neighbors == n)
                
                # Apply rules using vectorized operations
                new_grid = np.zeros_like(result_grid)
                new_grid[alive_mask & survival_mask] = 1  # Cells that survive
                new_grid[dead_mask & birth_mask] = 1      # Cells that are born
                
                result_grid = new_grid
                
        except ImportError:
            # Fall back to original implementation if scipy is not available
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
        
        # Cache the result
        self._ca_cache[cache_key] = result_grid

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
        # Create cache key for this operation
        cache_key = f"clusters_{num_clusters}_{cluster_value_multiplier}_{hash(grid.tobytes())}"

        # Check if we have this result cached
        if hasattr(self, "_cluster_cache") and cache_key in self._cluster_cache:
            return self._cluster_cache[cache_key]

        # Initialize cache if not exists
        if not hasattr(self, "_cluster_cache"):
            self._cluster_cache = {}

        result_grid = grid.copy()

        # Find coordinates of non-zero cells
        non_zero_coords = np.argwhere(grid > 0)

        if len(non_zero_coords) < num_clusters:
            self._cluster_cache[cache_key] = result_grid
            return result_grid

        # Randomly select cluster centers
        cluster_indices = np.random.choice(
            len(non_zero_coords), num_clusters, replace=False
        )
        cluster_centers = non_zero_coords[cluster_indices]

        # Performance optimization: Use vectorized operations with meshgrid
        try:
            # Create clusters around centers using vectorized operations
            y_indices, x_indices = np.indices((self.height, self.width))

            # Only process cells that have values
            valid_mask = grid > 0

            # For each cluster center
            for center in cluster_centers:
                cy, cx = center
                radius = np.random.randint(3, 10)

                # Calculate distances for all points at once
                distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)

                # Create mask for points within radius
                radius_mask = distances <= radius

                # Combine with valid mask (non-zero cells)
                combined_mask = radius_mask & valid_mask

                # Calculate falloff for all affected points
                falloff = 1 - (distances[combined_mask] / radius)

                # Apply multiplier with falloff
                multiplier = 1 + (cluster_value_multiplier - 1) * falloff
                result_grid[combined_mask] *= multiplier

        except Exception as e:
            # Fall back to original implementation for very large grids or if vectorized approach fails
            logging.warning(f"Falling back to non-vectorized cluster creation: {str(e)}")

            # Create clusters around centers (original approach)
            for center in cluster_centers:
                cy, cx = center
                radius = np.random.randint(3, 10)

                # Optimize by only iterating over the area within the radius
                y_min = max(0, cy - radius)
                y_max = min(self.height, cy + radius + 1)
                x_min = max(0, cx - radius)
                x_max = min(self.width, cx + radius + 1)

                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        # Calculate distance from center
                        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                        if distance <= radius and grid[y, x] > 0:
                            # Apply value multiplier with falloff based on distance
                            falloff = 1 - (distance / radius)
                            result_grid[y, x] *= (
                                1 + (cluster_value_multiplier - 1) * falloff
                            )

        # Cache the result
        self._cluster_cache[cache_key] = result_grid
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
        # Create cache key for this operation
        cache_key = f"threshold_{threshold}_{value}_{hash(grid.tobytes())}"
        
        # Check if we have this result cached
        if hasattr(self, "_threshold_cache") and cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
            
        # Initialize cache if not exists
        if not hasattr(self, "_threshold_cache"):
            self._threshold_cache = {}
        
        # Performance optimization: Use in-place operations where possible
        # This is already using vectorized operations, but we can optimize memory usage
        result = np.zeros_like(grid)
        mask = grid > threshold
        result[mask] = value
        
        # Cache the result
        self._threshold_cache[cache_key] = result
        return result

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
