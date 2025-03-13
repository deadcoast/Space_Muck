"""
BaseGenerator class: The base class for all procedural generation entities in the game.
"""

# Standard library imports
import logging
import random
import time

# Third-party library imports
import numpy as np

# Local application imports
from entities.base_entity import BaseEntity
from typing import Any, Dict, List, Optional, Set, Tuple
from utils.dependency_injection import inject
from utils.noise_generator import NoiseGenerator, get_noise_generator
import contextlib

# Standard library imports

# Third-party library imports

# Optional dependencies
try:
    from perlin_noise import PerlinNoise

    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    print("PerlinNoise package is not available. Using fallback noise generator.")

# GPU acceleration dependencies
try:
    from utils.gpu_utils import (
        is_gpu_available,
        get_available_backends,
        to_gpu,
        to_cpu,
    )

    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    logging.warning("GPU utilities not available. GPU acceleration will be disabled.")

# Local application imports


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
        use_gpu: bool = True,
        gpu_backend: str = "auto",
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
            use_gpu: Whether to use GPU acceleration when available
            gpu_backend: GPU backend to use ('cuda', 'cupy', 'auto')
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

        # GPU acceleration configuration
        self.use_gpu = use_gpu
        self.gpu_backend = gpu_backend

        # Check GPU availability
        self.gpu_available = False
        self.available_backends = ["cpu"]

        if use_gpu and GPU_UTILS_AVAILABLE:
            try:
                self.gpu_available = is_gpu_available()
                self.available_backends = get_available_backends()

                if self.gpu_available:
                    logging.info(
                        f"GPU acceleration enabled for generator {self.entity_id} "
                        f"with available backends: {', '.join(self.available_backends)}"
                    )
                else:
                    logging.info(
                        f"GPU acceleration requested but no GPU available for generator {self.entity_id}. "
                        f"Falling back to CPU implementation."
                    )
            except Exception as e:
                logging.warning(
                    f"Error detecting GPU capabilities: {str(e)}. "
                    f"Falling back to CPU implementation."
                )
                self.gpu_available = False
                self.use_gpu = False
        elif not use_gpu:
            logging.info(f"GPU acceleration disabled for generator {self.entity_id}")
        else:
            logging.warning(
                f"GPU utilities not available for generator {self.entity_id}. "
                f"Falling back to CPU implementation."
            )
            self.use_gpu = False

        logging.info(
            f"Generator initialized: {self.entity_type} (ID: {self.entity_id}, Seed: {self.seed})"
        )

    def generate_noise_layer(
        self, noise_type: str = "medium", scale: float = 0.1
    ) -> np.ndarray:
        """
        Generate a noise layer using the specified noise generator.
        Uses GPU acceleration when available.

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

        # Get octaves for the specified noise type
        octaves = octaves_map[noise_type]

        # Start timing
        start_time = time.time()

        # Try GPU acceleration if enabled and available
        if self.use_gpu and self.gpu_available and GPU_UTILS_AVAILABLE:
            try:
                logging.info(
                    f"Using GPU acceleration for noise generation (type: {noise_type}, "
                    f"dimensions: {self.width}x{self.height})"
                )

                # Use the GPU-accelerated noise generation function
                from utils.gpu_utils import apply_noise_generation_gpu

                noise_grid = apply_noise_generation_gpu(
                    width=self.width,
                    height=self.height,
                    scale=scale,
                    octaves=octaves,
                    persistence=0.5,  # Default persistence
                    lacunarity=2.0,  # Default lacunarity
                    seed=self.seed + octaves,
                    backend=self.gpu_backend,
                )

                gpu_time = time.time() - start_time
                logging.info(
                    f"GPU noise generation completed in {gpu_time:.4f} seconds"
                )

                # Cache the result
                self._noise_cache[cache_key] = noise_grid
                return noise_grid

            except Exception as e:
                logging.warning(
                    f"GPU noise generation failed: {str(e)}. Falling back to CPU implementation."
                )
                # Continue with CPU implementation

        # For larger grids, use optimized generation with parallel processing
        if self.width * self.height > 10000:  # Threshold for large grids
            try:
                logging.info(
                    f"Using parallel CPU processing for noise generation (type: {noise_type}, "
                    f"dimensions: {self.width}x{self.height})"
                )

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
                logging.warning(
                    f"Parallel noise generation failed: {str(e)}. Falling back to standard generation."
                )
                noise_grid = self.noise_generator.generate_noise(
                    width=self.width,
                    height=self.height,
                    scale=scale,
                    octaves=octaves,
                    seed=self.seed + octaves,
                )
        else:
            # Standard generation for smaller grids
            logging.info(
                f"Using standard CPU processing for noise generation (type: {noise_type}, "
                f"dimensions: {self.width}x{self.height})"
            )

            noise_grid = self.noise_generator.generate_noise(
                width=self.width,
                height=self.height,
                scale=scale,
                octaves=octaves,
                seed=self.seed + octaves,  # Use different seeds for different octaves
            )

        # Log CPU processing time
        cpu_time = time.time() - start_time
        logging.info(f"CPU noise generation completed in {cpu_time:.4f} seconds")

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

        This method evolves a grid according to cellular automaton rules, similar to
        Conway's Game of Life. The implementation uses optimized algorithms when possible
        and falls back to a manual implementation when necessary.

        Args:
            grid: Input grid to evolve. Non-zero values are considered "alive".
            birth_set: Set of neighbor counts that cause cell birth. Default is {3}.
            survival_set: Set of neighbor counts that allow cell survival. Default is {2, 3}.
            iterations: Number of iterations to perform. Higher values create more evolved patterns.
            wrap: Whether to wrap around grid edges (True) or treat edges as empty (False).

        Returns:
            np.ndarray: Evolved grid with original values preserved where cells are alive.
        """
        # Prepare the grid and parameters for cellular automaton processing
        binary_grid, prepared_birth_set, prepared_survival_set = (
            self._prepare_cellular_automaton_grid(grid, birth_set, survival_set)
        )

        # Generate a cache key for this operation
        cache_key = self._get_ca_cache_key(
            binary_grid, prepared_birth_set, prepared_survival_set, iterations, wrap
        )

        # Check if we have this result cached
        cached_result = self._get_cached_ca_result(cache_key, grid)
        if cached_result is not None:
            return cached_result

        # Process the cellular automaton
        result_grid = binary_grid.copy()

        # Try GPU acceleration first if enabled
        if self.use_gpu and self.gpu_available and GPU_UTILS_AVAILABLE:
            try:
                from utils.gpu_utils import apply_cellular_automaton_gpu

                logging.info(
                    f"Using GPU-accelerated cellular automaton with backend: {self.gpu_backend}"
                )
                start_time = time.time()

                # Transfer grid to GPU
                gpu_binary_grid = to_gpu(binary_grid, backend=self.gpu_backend)

                # Apply cellular automaton using GPU
                gpu_result_grid = apply_cellular_automaton_gpu(
                    grid=gpu_binary_grid,
                    birth_set=prepared_birth_set,
                    survival_set=prepared_survival_set,
                    iterations=iterations,
                    wrap=wrap,
                    backend=self.gpu_backend,
                )

                # Transfer result back to CPU
                result_grid = to_cpu(gpu_result_grid)

                end_time = time.time()
                logging.info(
                    f"GPU cellular automaton completed in {end_time - start_time:.4f} seconds"
                )

                # Cache the result
                self._cache_ca_result(cache_key, result_grid)

                # Preserve original values where cells are alive
                return grid * result_grid

            except Exception as e:
                logging.warning(
                    f"Error using GPU-accelerated cellular automaton: {str(e)}. "
                    f"Falling back to CPU implementation."
                )
                # Reset result_grid for CPU processing
                result_grid = binary_grid.copy()

        # Try optimized CPU implementation if GPU failed or is not available
        try:
            from utils.cellular_automaton_utils import (
                apply_cellular_automaton_optimized,
            )

            logging.info("Using optimized CPU cellular automaton from utils module")
            start_time = time.time()

            for _ in range(iterations):
                result_grid = apply_cellular_automaton_optimized(
                    result_grid, prepared_birth_set, prepared_survival_set
                )

            end_time = time.time()
            logging.info(
                f"CPU cellular automaton completed in {end_time - start_time:.4f} seconds"
            )

        except Exception as e:
            logging.warning(
                f"Error using optimized cellular automaton: {str(e)}. Falling back to internal implementation."
            )
            # Fall back to internal implementation
            result_grid = binary_grid.copy()
            start_time = time.time()

            for _ in range(iterations):
                result_grid = self._process_cellular_automaton_iteration(
                    result_grid, prepared_birth_set, prepared_survival_set, wrap
                )

            end_time = time.time()
            logging.info(
                f"Internal cellular automaton completed in {end_time - start_time:.4f} seconds"
            )

        # Cache the result
        self._cache_ca_result(cache_key, result_grid)

        # Preserve original values where cells are alive
        return grid * result_grid

    def _prepare_cellular_automaton_grid(
        self,
        grid: np.ndarray,
        birth_set: Optional[Set[int]],
        survival_set: Optional[Set[int]],
    ) -> Tuple[np.ndarray, Set[int], Set[int]]:
        """
        Prepare the grid and parameters for cellular automaton processing.

        Args:
            grid: Input grid to evolve
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival

        Returns:
            Tuple containing:
                - Binary grid (np.ndarray): Grid converted to binary (0 or 1)
                - Birth set (Set[int]): Finalized birth rule set
                - Survival set (Set[int]): Finalized survival rule set
        """
        # Use default rule sets if not provided
        if birth_set is None:
            birth_set = {3}  # Default birth rule (standard Conway's Game of Life)
        if survival_set is None:
            survival_set = {
                2,
                3,
            }  # Default survival rule (standard Conway's Game of Life)

        # Convert to binary grid (0 or 1)
        binary_grid = (grid > 0).astype(np.int8)

        return binary_grid, birth_set, survival_set

    def _get_ca_cache_key(
        self,
        binary_grid: np.ndarray,
        birth_set: Set[int],
        survival_set: Set[int],
        iterations: int,
        wrap: bool,
    ) -> str:
        """
        Generate a unique cache key for a cellular automaton operation.

        Args:
            binary_grid: Binary grid to evolve
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            iterations: Number of iterations to perform
            wrap: Whether to wrap around grid edges

        Returns:
            str: Unique cache key for this operation
        """
        # Create a deterministic hash of the grid and parameters
        # Use frozenset to make sets hashable
        return f"ca_{hash(frozenset(birth_set))}_{hash(frozenset(survival_set))}_{iterations}_{wrap}_{hash(binary_grid.tobytes())}"

    def _get_cached_ca_result(
        self,
        cache_key: str,
        original_grid: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached cellular automaton result if available.

        Args:
            cache_key: Cache key for the operation
            original_grid: Original input grid

        Returns:
            np.ndarray or None: Cached result if available, None otherwise
        """
        # Initialize cache if not exists
        if not hasattr(self, "_ca_cache"):
            self._ca_cache = {}
            return None

        # Check if we have this result cached
        if cache_key in self._ca_cache:
            # Apply the cached binary result to the original grid to preserve values
            return self._ca_cache[cache_key] * original_grid

        return None

    def _cache_ca_result(
        self,
        cache_key: str,
        result_grid: np.ndarray,
    ) -> None:
        """
        Cache a cellular automaton result for future use.

        Args:
            cache_key: Cache key for the operation
            result_grid: Result grid to cache
        """
        # Initialize cache if not exists
        if not hasattr(self, "_ca_cache"):
            self._ca_cache = {}

        # Manage cache size to prevent memory issues
        if len(self._ca_cache) > 100:  # Limit cache to 100 entries
            # Remove a random entry to keep memory usage bounded
            # In a more sophisticated implementation, we could use LRU cache
            with contextlib.suppress(StopIteration, KeyError):
                random_key = next(iter(self._ca_cache))
                del self._ca_cache[random_key]
        # Cache the result
        self._ca_cache[cache_key] = result_grid

    def _process_cellular_automaton_iteration(
        self,
        grid: np.ndarray,
        birth_set: Set[int],
        survival_set: Set[int],
        wrap: bool,
    ) -> np.ndarray:
        """
        Process a single iteration of the cellular automaton.

        Args:
            grid: Current state of the grid
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            wrap: Whether to wrap around grid edges

        Returns:
            np.ndarray: New state of the grid after one iteration
        """
        # For large grids, use parallel processing
        # Use a threshold that can be adjusted for benchmarking
        self._parallel_ca_threshold = getattr(self, "_parallel_ca_threshold", 40000)
        if (
            grid.size > self._parallel_ca_threshold
        ):  # Threshold for parallel processing (e.g., 200x200 grid)
            try:
                return self._apply_cellular_automaton_parallel(
                    grid, birth_set, survival_set, wrap
                )
            except Exception as e:
                logging.warning(
                    f"Parallel processing failed: {str(e)}. Falling back to sequential processing."
                )
                # Continue with sequential processing

        # For smaller grids or if parallel processing failed, try to use the optimized scipy implementation
        try:
            return self._apply_cellular_automaton_scipy(
                grid, birth_set, survival_set, wrap
            )
        except Exception as e:
            # Log the error and fall back to manual implementation
            logging.warning(f"Falling back to manual cellular automaton: {str(e)}")
            return self._apply_cellular_automaton_manual(
                grid, birth_set, survival_set, wrap
            )

    def _apply_cellular_automaton_parallel(
        self,
        grid: np.ndarray,
        birth_set: Set[int],
        survival_set: Set[int],
        wrap: bool,
    ) -> np.ndarray:
        """
        Apply cellular automaton rules using parallel processing for large grids.

        Args:
            grid: Current state of the grid
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            wrap: Whether to wrap around grid edges

        Returns:
            np.ndarray: New state of the grid after one iteration
        """
        from concurrent.futures import ProcessPoolExecutor
        import math

        logging.info(
            f"Using parallel processing for cellular automaton on {grid.shape} grid"
        )

        # Determine number of workers based on CPU count
        import multiprocessing

        num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers max

        # Calculate chunk size based on grid height
        chunk_size = math.ceil(grid.shape[0] / num_workers)

        # Create a new grid to store the result
        new_grid = np.zeros_like(grid)

        # Define a function to process a chunk of the grid
        def process_chunk(start_row, end_row):
            # Need to include one extra row on each side for neighbor calculations
            # Handle the boundary conditions based on wrap parameter
            if wrap:
                # For wrap=True, we need to handle the edge cases specially
                if start_row == 0:
                    # Include the last row for wrapping
                    process_start = start_row
                    input_chunk = np.vstack(
                        (grid[-1:], grid[process_start : end_row + 1])
                    )
                else:
                    process_start = start_row - 1
                    input_chunk = grid[process_start : end_row + 1]

                if end_row >= grid.shape[0]:
                    # Include the first row for wrapping
                    input_chunk = np.vstack((input_chunk, grid[:1]))
                else:
                    input_chunk = np.vstack((input_chunk, grid[end_row : end_row + 1]))
            else:
                # For wrap=False, we just need to ensure we don't go out of bounds
                process_start = max(0, start_row - 1)
                process_end = min(grid.shape[0], end_row + 1)
                input_chunk = grid[process_start:process_end]

            # Process the chunk using the appropriate method
            try:
                # Check if scipy is available using importlib
                import importlib.util

                if importlib.util.find_spec("scipy") is None:
                    raise ImportError("scipy not available")
                # Use scipy implementation for the chunk
                from scipy import signal

                # Create kernel for neighbor counting
                kernel = np.ones((3, 3), dtype=np.int8)
                kernel[1, 1] = 0  # Don't count the cell itself

                # Count neighbors using convolution
                boundary_mode = "wrap" if wrap else "fill"
                neighbors = signal.convolve2d(
                    input_chunk, kernel, mode="same", boundary=boundary_mode
                )

                # Create the neighbor masks for birth and survival rules
                birth_mask = np.zeros_like(neighbors, dtype=bool)
                survival_mask = np.zeros_like(neighbors, dtype=bool)

                # Vectorized approach to create masks
                for n in birth_set:
                    birth_mask |= neighbors == n

                for n in survival_set:
                    survival_mask |= neighbors == n

                # Apply rules using vectorized operations
                alive_mask = input_chunk == 1
                dead_mask = input_chunk == 0

                result_chunk = np.zeros_like(input_chunk)
                result_chunk[alive_mask & survival_mask] = 1  # Cells that survive
                result_chunk[dead_mask & birth_mask] = 1  # Cells that are born

            except ImportError:
                # Fall back to manual implementation for the chunk
                result_chunk = np.zeros_like(input_chunk)

                # Define neighbor offsets once for efficiency
                neighbor_offsets = [
                    (dx, dy)
                    for dy in [-1, 0, 1]
                    for dx in [-1, 0, 1]
                    if dx != 0 or dy != 0
                ]

                height, width = input_chunk.shape

                for y in range(height):
                    for x in range(width):
                        # Count neighbors
                        neighbor_count = 0
                        for dx, dy in neighbor_offsets:
                            if wrap:
                                # Wrap around edges
                                nx, ny = (x + dx) % width, (y + dy) % height
                            else:
                                # Don't wrap, check boundaries
                                nx, ny = x + dx, y + dy
                                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                                    continue

                            if input_chunk[ny, nx] > 0:
                                neighbor_count += 1

                        # Apply rules
                        if input_chunk[y, x] > 0:  # Cell is alive
                            if neighbor_count in survival_set:
                                result_chunk[y, x] = 1  # Cell survives
                        else:  # Cell is dead
                            if neighbor_count in birth_set:
                                result_chunk[y, x] = 1  # Cell is born

            # Return the processed chunk along with its position information
            # If we added padding rows, remove them from the result
            if wrap:
                if start_row == 0 and end_row >= grid.shape[0]:
                    # Both first and last rows were padded
                    result = result_chunk[1:-1]
                elif start_row == 0:
                    # Only first row was padded
                    result = result_chunk[1:]
                elif end_row >= grid.shape[0]:
                    # Only last row was padded
                    result = result_chunk[:-1]
                else:
                    # Both edges were padded
                    result = result_chunk[1:-1]
            else:
                # For non-wrap, we just need to match the original chunk size
                result = result_chunk[
                    process_start
                    - start_row : process_start
                    - start_row
                    + (end_row - start_row)
                ]

            return start_row, end_row, result

        # Process chunks in parallel
        futures = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, grid.shape[0], chunk_size):
                end_row = min(i + chunk_size, grid.shape[0])
                futures.append(executor.submit(process_chunk, i, end_row))

            # Collect results and assemble the final grid
            for future in futures:
                start_row, end_row, chunk_result = future.result()
                new_grid[start_row:end_row] = chunk_result

        return new_grid

    def _apply_cellular_automaton_scipy(
        self,
        grid: np.ndarray,
        birth_set: Set[int],
        survival_set: Set[int],
        wrap: bool,
    ) -> np.ndarray:
        """
        Apply cellular automaton rules using scipy's optimized convolution.

        Args:
            grid: Current state of the grid
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            wrap: Whether to wrap around grid edges

        Returns:
            np.ndarray: New state of the grid after one iteration
        """
        from scipy import signal

        # Create kernel for neighbor counting
        kernel = np.ones((3, 3), dtype=np.int8)
        kernel[1, 1] = 0  # Don't count the cell itself

        # Count neighbors using convolution
        boundary_mode = "wrap" if wrap else "fill"
        neighbors = signal.convolve2d(grid, kernel, mode="same", boundary=boundary_mode)

        # Create the neighbor masks for birth and survival rules
        birth_mask, survival_mask = self._create_neighbor_masks(
            neighbors, birth_set, survival_set
        )

        # Apply rules using vectorized operations
        alive_mask = grid == 1
        dead_mask = grid == 0

        new_grid = np.zeros_like(grid)
        new_grid[alive_mask & survival_mask] = 1  # Cells that survive
        new_grid[dead_mask & birth_mask] = 1  # Cells that are born

        return new_grid

    def _create_neighbor_masks(
        self,
        neighbors: np.ndarray,
        birth_set: Set[int],
        survival_set: Set[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create boolean masks for cells that should be born or survive.

        Args:
            neighbors: Grid of neighbor counts
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival

        Returns:
            Tuple containing:
                - Birth mask (np.ndarray): Boolean mask for cells that should be born
                - Survival mask (np.ndarray): Boolean mask for cells that should survive
        """
        # Create masks for birth and survival based on neighbor counts
        birth_mask = np.zeros_like(neighbors, dtype=bool)
        survival_mask = np.zeros_like(neighbors, dtype=bool)

        # Vectorized approach to create masks
        for n in birth_set:
            birth_mask |= neighbors == n

        for n in survival_set:
            survival_mask |= neighbors == n

        return birth_mask, survival_mask

    def _apply_cellular_automaton_manual(
        self,
        grid: np.ndarray,
        birth_set: Set[int],
        survival_set: Set[int],
        wrap: bool,
    ) -> np.ndarray:
        """
        Apply cellular automaton rules using a manual implementation.
        This is a fallback when scipy is not available.

        Args:
            grid: Current state of the grid
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            wrap: Whether to wrap around grid edges

        Returns:
            np.ndarray: New state of the grid after one iteration
        """
        # Define neighbor offsets once for efficiency
        neighbor_offsets = [
            (dx, dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if dx != 0 or dy != 0
        ]

        new_grid = grid.copy()

        # Process each cell in the grid
        for y in range(self.height):
            for x in range(self.width):
                # Count live neighbors
                neighbors = self._count_neighbors(grid, x, y, neighbor_offsets, wrap)

                # Apply rules
                new_grid[y, x] = self._apply_rule_to_cell(
                    grid[y, x], neighbors, birth_set, survival_set
                )

        return new_grid

    def _apply_rule_to_cell(
        self,
        cell_state: int,
        neighbor_count: int,
        birth_set: Set[int],
        survival_set: Set[int],
    ) -> int:
        """
        Apply cellular automaton rules to a single cell.

        Args:
            cell_state: Current state of the cell (0 or 1)
            neighbor_count: Number of live neighbors
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival

        Returns:
            int: New state of the cell (0 or 1)
        """
        if cell_state == 1:  # Cell is alive
            return 1 if neighbor_count in survival_set else 0
        else:  # Cell is dead
            return 1 if neighbor_count in birth_set else 0

    def _count_neighbors(
        self,
        grid: np.ndarray,
        x: int,
        y: int,
        neighbor_offsets: List[Tuple[int, int]],
        wrap: bool,
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

        This method identifies areas in the grid to enhance with higher values,
        creating "hot spots" or clusters. The implementation uses vectorized
        operations when possible and falls back to a manual implementation when necessary.

        Args:
            grid: Input grid with values to enhance
            num_clusters: Number of clusters to create
            cluster_value_multiplier: Multiplier for values in clusters (1.0 = no change)

        Returns:
            np.ndarray: Grid with enhanced value clusters
        """
        # Validate parameters
        if num_clusters <= 0:
            logging.warning(
                "Number of clusters must be positive, using default value of 5"
            )
            num_clusters = 5

        if cluster_value_multiplier <= 0:
            logging.warning(
                "Cluster value multiplier must be positive, using default value of 2.0"
            )
            cluster_value_multiplier = 2.0

        # Generate a cache key for this operation
        cache_key = self._get_cluster_cache_key(
            grid, num_clusters, cluster_value_multiplier
        )

        # Check if we have this result cached
        cached_result = self._get_cached_cluster_result(cache_key)
        if cached_result is not None:
            return cached_result

        # Try GPU acceleration first if enabled
        if self.use_gpu and self.gpu_available and GPU_UTILS_AVAILABLE:
            try:
                return self._gpu_acceleration_handler(
                    grid, num_clusters, cache_key, cluster_value_multiplier
                )
            except Exception as e:
                logging.warning(
                    f"Error using GPU-accelerated clustering: {str(e)}. "
                    f"Falling back to CPU implementation."
                )
                # Continue with CPU implementation

        # Try optimized CPU implementation if GPU failed or is not available
        try:
            from utils.value_generator import add_value_clusters

            logging.info("Using optimized clustering from value_generator module")
            start_time = time.time()

            # Calculate a reasonable cluster radius based on grid dimensions
            cluster_radius = int(min(grid.shape) * 0.1)

            # Use the add_value_clusters function from value_generator
            result_grid = add_value_clusters(
                grid,
                num_clusters=num_clusters,
                cluster_radius=cluster_radius,
                cluster_value_multiplier=cluster_value_multiplier,
            )

            end_time = time.time()
            logging.info(
                f"CPU optimized clustering completed in {end_time - start_time:.4f} seconds"
            )

            # Cache the result
            self._cache_cluster_result(cache_key, result_grid)
            return result_grid

        except Exception as e:
            logging.warning(
                f"Error using value_generator.add_value_clusters: {str(e)}. Falling back to internal implementation."
            )
            # Continue with internal implementation

        # Prepare the grid for clustering
        start_time = time.time()
        result_grid, non_zero_coords = self._prepare_cluster_grid(grid)

        # If we don't have enough non-zero cells for the requested clusters, return the original grid
        if len(non_zero_coords) < num_clusters:
            self._cache_cluster_result(cache_key, result_grid)
            return result_grid

        # Select cluster centers from non-zero coordinates
        cluster_centers = self._select_cluster_centers(non_zero_coords, num_clusters)

        # Apply clustering using either vectorized or manual approach
        try:
            # Try vectorized approach first (faster for most cases)
            result_grid = self._apply_vectorized_clustering(
                grid, result_grid, cluster_centers, cluster_value_multiplier
            )
        except Exception as e:
            # Fall back to manual approach if vectorized fails
            logging.warning(f"Falling back to manual clustering: {str(e)}")
            result_grid = self._apply_manual_clustering(
                grid, result_grid, cluster_centers, cluster_value_multiplier
            )

        end_time = time.time()
        logging.info(
            f"Internal clustering completed in {end_time - start_time:.4f} seconds"
        )

        # Cache the result
        self._cache_cluster_result(cache_key, result_grid)
        return result_grid

    def _gpu_acceleration_handler(
        self, grid, num_clusters, cache_key, cluster_value_multiplier
    ):
        from utils.gpu_utils import apply_kmeans_clustering_gpu

        logging.info(
            f"Using GPU-accelerated clustering with backend: {self.gpu_backend}"
        )
        start_time = time.time()

        # Prepare data for clustering
        non_zero_coords = np.argwhere(grid > 0)

        # If we don't have enough non-zero cells for the requested clusters, return the original grid
        if len(non_zero_coords) < num_clusters:
            result_grid = grid.copy()
            self._cache_cluster_result(cache_key, result_grid)
            return result_grid

        # Transfer data to GPU for processing
        gpu_coords = to_gpu(non_zero_coords, backend=self.gpu_backend)

        # Apply K-means clustering to find optimal cluster centers
        cluster_centers, _ = apply_kmeans_clustering_gpu(
            data=gpu_coords, n_clusters=num_clusters, backend=self.gpu_backend
        )

        # Transfer cluster centers back to CPU
        cluster_centers = to_cpu(cluster_centers)

        # Create a copy of the grid to modify
        result_grid = grid.copy()

        # Calculate a reasonable cluster radius based on grid dimensions
        cluster_radius = int(min(grid.shape) * 0.1)

        # Apply clusters using vectorized operations
        y_indices, x_indices = np.indices((self.height, self.width))
        valid_mask = grid > 0

        for center in cluster_centers:
            cy, cx = center
            radius = cluster_radius

            # Calculate distances for all points at once
            distances = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

            # Create mask for points within radius
            radius_mask = distances <= radius

            # Combine with valid mask (non-zero cells)
            combined_mask = radius_mask & valid_mask

            # Calculate falloff for all affected points
            falloff = 1 - (distances[combined_mask] / radius)

            result_grid[combined_mask] *= 1 + (cluster_value_multiplier - 1) * falloff

        end_time = time.time()
        logging.info(f"GPU clustering completed in {end_time - start_time:.4f} seconds")

        # Cache the result
        self._cache_cluster_result(cache_key, result_grid)
        return result_grid

    def _prepare_cluster_grid(
        self,
        grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the grid for clustering operations.

        Args:
            grid: Input grid

        Returns:
            Tuple containing:
                - result_grid (np.ndarray): Copy of the input grid
                - non_zero_coords (np.ndarray): Coordinates of non-zero cells
        """
        # Create a copy of the grid to avoid modifying the original
        result_grid = grid.copy()

        # Find coordinates of non-zero cells for cluster center selection
        non_zero_coords = np.argwhere(grid > 0)

        return result_grid, non_zero_coords

    def _get_cluster_cache_key(
        self,
        grid: np.ndarray,
        num_clusters: int,
        cluster_value_multiplier: float,
    ) -> str:
        """
        Generate a unique cache key for a clustering operation.

        Args:
            grid: Input grid
            num_clusters: Number of clusters
            cluster_value_multiplier: Multiplier for values in clusters

        Returns:
            str: Unique cache key for this operation
        """
        # Create a deterministic hash of the grid and parameters
        return (
            f"clusters_{num_clusters}_{cluster_value_multiplier}_{hash(grid.tobytes())}"
        )

    def _get_cached_cluster_result(
        self,
        cache_key: str,
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached clustering result if available.

        Args:
            cache_key: Cache key for the operation

        Returns:
            np.ndarray or None: Cached result if available, None otherwise
        """
        # Initialize cache if not exists
        if not hasattr(self, "_cluster_cache"):
            self._cluster_cache = {}
            return None

        # Check if we have this result cached
        if cache_key in self._cluster_cache:
            return self._cluster_cache[cache_key]

        return None

    def _cache_cluster_result(
        self,
        cache_key: str,
        result_grid: np.ndarray,
    ) -> None:
        """
        Cache a clustering result for future use.

        Args:
            cache_key: Cache key for the operation
            result_grid: Result grid to cache
        """
        # Initialize cache if not exists
        if not hasattr(self, "_cluster_cache"):
            self._cluster_cache = {}

        # Manage cache size to prevent memory issues
        if len(self._cluster_cache) > 100:  # Limit cache to 100 entries
            # Remove a random entry to keep memory usage bounded
            with contextlib.suppress(StopIteration, KeyError):
                random_key = next(iter(self._cluster_cache))
                del self._cluster_cache[random_key]
        # Cache the result
        self._cluster_cache[cache_key] = result_grid

    def _select_cluster_centers(
        self,
        non_zero_coords: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        """
        Select random cluster centers from non-zero coordinates.

        Args:
            non_zero_coords: Coordinates of non-zero cells
            num_clusters: Number of clusters to create

        Returns:
            np.ndarray: Selected cluster centers
        """
        # Randomly select cluster centers without replacement
        cluster_indices = np.random.choice(
            len(non_zero_coords), num_clusters, replace=False
        )
        return non_zero_coords[cluster_indices]

    def _apply_vectorized_clustering(
        self,
        grid: np.ndarray,
        result_grid: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_value_multiplier: float,
    ) -> np.ndarray:
        """
        Apply clustering using vectorized operations for better performance.

        Args:
            grid: Original input grid
            result_grid: Grid to modify with clusters
            cluster_centers: Centers for the clusters
            cluster_value_multiplier: Multiplier for values in clusters

        Returns:
            np.ndarray: Grid with clusters applied
        """
        # For large grids, use parallel processing
        # Use a threshold that can be adjusted for benchmarking
        self._parallel_clustering_threshold = getattr(
            self, "_parallel_clustering_threshold", 40000
        )
        if (
            grid.size > self._parallel_clustering_threshold and len(cluster_centers) > 3
        ):  # Threshold for parallel processing
            try:
                return self._apply_parallel_clustering(
                    grid, result_grid, cluster_centers, cluster_value_multiplier
                )
            except Exception as e:
                logging.warning(
                    f"Parallel clustering failed: {str(e)}. Falling back to sequential vectorized clustering."
                )
                # Continue with sequential vectorized processing

        # Create indices for the entire grid
        y_indices, x_indices = np.indices((self.height, self.width))

        # Only process cells that have values
        valid_mask = grid > 0

        # For each cluster center
        for center in cluster_centers:
            cy, cx = center
            radius = np.random.randint(3, 10)

            # Calculate distances for all points at once
            distances = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

            # Create mask for points within radius
            radius_mask = distances <= radius

            # Combine with valid mask (non-zero cells)
            combined_mask = radius_mask & valid_mask

            # Calculate falloff for all affected points
            falloff = 1 - (distances[combined_mask] / radius)

            # Apply multiplier with falloff
            multiplier = 1 + (cluster_value_multiplier - 1) * falloff
            result_grid[combined_mask] *= multiplier

        return result_grid

    def _apply_parallel_clustering(
        self,
        grid: np.ndarray,
        result_grid: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_value_multiplier: float,
    ) -> np.ndarray:
        """
        Apply clustering using parallel processing for large grids.

        Args:
            grid: Original input grid
            result_grid: Grid to modify with clusters
            cluster_centers: Centers for the clusters
            cluster_value_multiplier: Multiplier for values in clusters

        Returns:
            np.ndarray: Grid with clusters applied
        """
        from concurrent.futures import ProcessPoolExecutor
        import math
        import multiprocessing

        logging.info(
            f"Using parallel processing for clustering on {grid.shape} grid with {len(cluster_centers)} clusters"
        )

        # Create a copy of the result grid to avoid modifying the original during parallel processing
        parallel_result = result_grid.copy()

        # Determine number of workers based on CPU count and number of clusters
        num_workers = min(
            multiprocessing.cpu_count(), 8, len(cluster_centers)
        )  # Limit to 8 workers max

        # If we have very few clusters, it's not worth parallelizing
        if num_workers <= 1 or len(cluster_centers) <= 2:
            return self._apply_vectorized_clustering(
                grid, result_grid, cluster_centers, cluster_value_multiplier
            )

        # Divide the cluster centers among workers
        clusters_per_worker = math.ceil(len(cluster_centers) / num_workers)

        # Define a function to process a subset of clusters
        def process_clusters(worker_clusters, worker_seed):
            # Set a unique seed for this worker to ensure reproducibility
            np.random.seed(worker_seed)

            # Create a local multiplier grid (initialized to 1s)
            local_result = np.ones_like(grid)

            # Create indices for the entire grid (only once per worker)
            y_indices, x_indices = np.indices((self.height, self.width))

            # Only process cells that have values
            valid_mask = grid > 0

            # Process each cluster center assigned to this worker
            for center in worker_clusters:
                cy, cx = center
                radius = np.random.randint(3, 10)

                # Calculate distances for all points at once
                distances = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

                # Create mask for points within radius
                radius_mask = distances <= radius

                # Combine with valid mask (non-zero cells)
                combined_mask = radius_mask & valid_mask

                # Calculate falloff for all affected points
                falloff = 1 - (distances[combined_mask] / radius)

                # Apply multiplier with falloff
                multiplier = 1 + (cluster_value_multiplier - 1) * falloff
                local_result[combined_mask] *= multiplier

            # Return the local result for this worker's clusters
            return local_result

        # Split the cluster centers into chunks for each worker
        cluster_chunks = []
        for i in range(0, len(cluster_centers), clusters_per_worker):
            end_idx = min(i + clusters_per_worker, len(cluster_centers))
            cluster_chunks.append(cluster_centers[i:end_idx])

        # Process clusters in parallel
        futures = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, chunk in enumerate(cluster_chunks):
                # Use a different seed for each worker based on the main seed
                worker_seed = self.seed + i + 1
                futures.append(executor.submit(process_clusters, chunk, worker_seed))

            # Combine the results from all workers
            for future in futures:
                worker_result = future.result()
                # Multiply the worker's result into the final result
                parallel_result *= worker_result

        return parallel_result

    def _apply_manual_clustering(
        self,
        grid: np.ndarray,
        result_grid: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_value_multiplier: float,
    ) -> np.ndarray:
        """
        Apply clustering using a manual loop-based approach.
        This is a fallback when vectorized operations fail.

        Args:
            grid: Original input grid
            result_grid: Grid to modify with clusters
            cluster_centers: Centers for the clusters
            cluster_value_multiplier: Multiplier for values in clusters

        Returns:
            np.ndarray: Grid with clusters applied
        """
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
                "use_gpu": self.use_gpu,
                "gpu_backend": self.gpu_backend,
                "gpu_available": self.gpu_available,
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
        Uses GPU acceleration when available.

        Args:
            scale: Base scale of the noise
            octaves: List of octave values (default: [3, 5, 8])
            weights: List of weights for each octave (default: [1.0, 0.5, 0.25])

        Returns:
            np.ndarray: 2D grid of noise values
        """
        # Set default values if not provided
        octaves = octaves or [3, 5, 8]
        weights = weights or [1.0, 0.5, 0.25]

        # Create a cache key
        cache_key = (
            f"multi_{scale}_{self.width}_{self.height}_{self.seed}_{octaves}_{weights}"
        )

        # Check if we have this noise layer cached
        if cache_key in self._noise_cache:
            return self._noise_cache[cache_key]

        # Start timing
        start_time = time.time()

        # Try GPU acceleration if available
        if self.use_gpu and self.gpu_available and GPU_UTILS_AVAILABLE:
            noise_grid = self._generate_noise_with_gpu(
                scale, octaves, weights, start_time, cache_key
            )
            if noise_grid is not None:
                return noise_grid

        # Use CPU implementation if GPU failed or not available
        return self._generate_noise_with_cpu(
            scale, octaves, weights, start_time, cache_key
        )

    def _generate_noise_with_gpu(self, scale, octaves, weights, start_time, cache_key):
        """Generate multi-octave noise using GPU acceleration.

        Args:
            scale: Base scale of the noise
            octaves: List of octave values
            weights: List of weights for each octave
            start_time: Time when generation started
            cache_key: Cache key for storing the result

        Returns:
            np.ndarray or None: Noise grid if successful, None if GPU generation failed
        """
        try:
            logging.info(
                f"Using GPU acceleration for multi-octave noise generation "
                f"(dimensions: {self.width}x{self.height})"
            )

            # Generate combined noise grid
            noise_grid = np.zeros((self.height, self.width))

            # Normalize weights
            total_weight = sum(weights)
            norm_weights = [w / total_weight for w in weights]

            # Transfer the empty grid to GPU for accumulation
            gpu_noise_grid = to_gpu(noise_grid, backend=self.gpu_backend)

            # Generate and accumulate noise for each octave
            noise_grid = self._generate_and_accumulate_octaves(
                scale, octaves, norm_weights, gpu_noise_grid
            )

            # Ensure values are in [0, 1]
            noise_grid = self._clip_noise_values(noise_grid, gpu_noise_grid)

            return self._cpu_acceleration_validator(
                start_time,
                "GPU multi-octave noise generation completed in ",
                noise_grid,
                cache_key,
            )
        except Exception as e:
            logging.warning(
                f"GPU multi-octave noise generation failed: {str(e)}. "
                f"Falling back to CPU implementation."
            )
            return None

    def _generate_and_accumulate_octaves(
        self, scale, octaves, norm_weights, gpu_noise_grid
    ):
        """Generate noise for each octave and accumulate the weighted results.

        Args:
            scale: Base scale of the noise
            octaves: List of octave values
            norm_weights: Normalized weights for each octave
            gpu_noise_grid: Initial GPU grid for accumulation

        Returns:
            np.ndarray: Accumulated noise grid
        """
        from utils.gpu_utils import apply_noise_generation_gpu

        noise_grid = np.zeros((self.height, self.width))

        for i, (octave, weight) in enumerate(zip(octaves, norm_weights)):
            # Generate noise for this octave using GPU
            octave_scale = scale * (2**i)  # Scale increases with octave
            gpu_octave_noise = apply_noise_generation_gpu(
                width=self.width,
                height=self.height,
                scale=octave_scale,
                octaves=octave,  # Use the octave value as the number of octaves
                persistence=0.5,  # Default persistence
                lacunarity=2.0,  # Default lacunarity
                seed=self.seed + i,  # Different seed for each octave
                backend=self.gpu_backend,
            )

            # Scale the noise by weight while still on GPU
            try:
                # Try to perform the weighted addition on the GPU
                weighted_noise = (
                    to_gpu(weight, backend=self.gpu_backend) * gpu_octave_noise
                )
                gpu_noise_grid += weighted_noise
            except Exception as e:
                # If GPU operation fails, fall back to CPU
                logging.debug(
                    f"GPU weighted addition failed: {str(e)}. Using CPU fallback."
                )
                octave_noise = to_cpu(gpu_octave_noise)
                noise_grid += octave_noise * weight

        # Transfer the final result back to CPU
        try:
            return to_cpu(gpu_noise_grid)
        except Exception as e:
            logging.debug(
                f"GPU to CPU transfer failed: {str(e)}. Using existing CPU grid."
            )
            return noise_grid

    def _clip_noise_values(self, noise_grid, gpu_noise_grid):
        """Ensure noise values are clipped to the range [0, 1].

        Args:
            noise_grid: CPU noise grid
            gpu_noise_grid: GPU noise grid (if available)

        Returns:
            np.ndarray: Clipped noise grid
        """
        try:
            # Try to perform clipping on GPU if the data is still on GPU
            if gpu_noise_grid is not None:
                clip_bounds = to_gpu(np.array([0, 1]), backend=self.gpu_backend)[:2]
                return to_cpu(np.clip(gpu_noise_grid, clip_bounds[0], clip_bounds[1]))
            else:
                return np.clip(noise_grid, 0, 1)
        except Exception as e:
            # Fall back to CPU clipping if GPU operation fails
            logging.debug(f"GPU clipping failed: {str(e)}. Using CPU fallback.")
            return np.clip(noise_grid, 0, 1)

    def _generate_noise_with_cpu(self, scale, octaves, weights, start_time, cache_key):
        """Generate multi-octave noise using CPU implementation.

        Args:
            scale: Base scale of the noise
            octaves: List of octave values
            weights: List of weights for each octave
            start_time: Time when generation started
            cache_key: Cache key for storing the result

        Returns:
            np.ndarray: Generated noise grid
        """
        logging.info(
            f"Using CPU implementation for multi-octave noise generation "
            f"(dimensions: {self.width}x{self.height})"
        )

        # Generate the multi-octave noise using the CPU implementation
        noise_grid = self.noise_generator.generate_multi_octave_noise(
            width=self.width,
            height=self.height,
            scale=scale,
            octaves=octaves,
            weights=weights,
            seed=self.seed,
        )

        return self._cpu_acceleration_validator(
            start_time,
            "CPU multi-octave noise generation completed in ",
            noise_grid,
            cache_key,
        )

    def _cpu_acceleration_validator(self, start_time, arg1, noise_grid, cache_key):
        gpu_time = time.time() - start_time
        logging.info(f"{arg1}{gpu_time:.4f} seconds")

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

        # Load GPU-related attributes
        generator.use_gpu = data.get("use_gpu", True)
        generator.gpu_backend = data.get("gpu_backend", "auto")
        generator.gpu_available = data.get("gpu_available", False)
        generator.available_backends = data.get("available_backends", ["cpu"])

        # Re-check GPU availability if use_gpu is True
        if generator.use_gpu and GPU_UTILS_AVAILABLE:
            try:
                generator.gpu_available = is_gpu_available()
                generator.available_backends = get_available_backends()

                if generator.gpu_available:
                    logging.info(
                        f"GPU acceleration enabled for loaded generator {generator.entity_id} "
                        f"with available backends: {', '.join(generator.available_backends)}"
                    )
                else:
                    logging.info(
                        f"GPU acceleration requested but no GPU available for loaded generator {generator.entity_id}. "
                        f"Falling back to CPU implementation."
                    )
            except Exception as e:
                logging.warning(
                    f"Error detecting GPU capabilities during loading: {str(e)}. "
                    f"Falling back to CPU implementation."
                )
                generator.gpu_available = False
                generator.use_gpu = False

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
                "detail": noise_gen,
            }

        return generator
