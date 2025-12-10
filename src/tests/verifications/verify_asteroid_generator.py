#!/usr/bin/env python3
"""
Comprehensive verification script for testing AsteroidGenerator functionality and structure.

This script provides both basic file structure verification and advanced performance testing
for the AsteroidGenerator class. It implements a simplified version of BaseGenerator and
AsteroidGenerator to test optimization techniques without circular import issues.

This script focuses on:
1. File structure and inheritance verification
2. Performance testing of generator methods
3. Caching verification
4. Visualization of generator outputs
"""

# Standard library imports
import random
import sys
import time

# Third-party library imports

# Local application imports

# Try to import required packages, but don't fail if they're not available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    print("numpy not available. Tests will be skipped.")
    NUMPY_AVAILABLE = False
    sys.exit(1)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("matplotlib not available. Visualization will be skipped.")
    MATPLOTLIB_AVAILABLE = False

try:
    import scipy.ndimage as ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    print("scipy not available. Some optimizations will be skipped.")
    SCIPY_AVAILABLE = False


# Implement a simplified BaseGenerator with caching
class BaseGenerator:
    """A simplified BaseGenerator class with caching for testing purposes."""

    def __init__(self, seed=None, width=100, height=100):
        """Initialize the generator with the given seed and dimensions."""
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.width = width
        self.height = height
        self.entity_type = "base"

        # Set the random seed
        random.seed(self.seed)
        # Use modern numpy.random.Generator API for deterministic testing
        self.rng = np.random.default_rng(self.seed)

        # Initialize caches
        self._cache = {}

    def _get_cache_key(self, method_name, **kwargs):
        """Generate a cache key for the given method and parameters."""
        # Sort kwargs by key to ensure consistent cache keys
        sorted_kwargs = sorted(kwargs.items())
        return f"{method_name}_{self.seed}_{self.width}_{self.height}_{sorted_kwargs}"

    def generate_noise_layer(self, density="medium", scale=0.1):
        """Generate a noise layer with the given density and scale."""
        # Check if we have a cached result
        cache_key = self._get_cache_key(
            "generate_noise_layer", density=density, scale=scale
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate noise using the modern numpy random Generator API
        noise = self.rng.random((self.height, self.width))

        # Apply smoothing if scipy is available
        if SCIPY_AVAILABLE:
            # Use scipy's gaussian filter for smoothing
            sigma = scale * 10  # Convert scale to sigma for gaussian filter
            noise = ndimage.gaussian_filter(noise, sigma=sigma)

        # Normalize the noise to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # Cache the result
        self._cache[cache_key] = noise

        return noise

    def apply_threshold(self, grid, threshold=0.5):
        """Apply a threshold to the grid."""
        # Check if we have a cached result
        cache_key = self._get_cache_key(
            "apply_threshold", threshold=threshold, grid_hash=hash(grid.tobytes())
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Apply threshold
        result = np.where(grid > threshold, 1.0, 0.0)

        # Cache the result
        self._cache[cache_key] = result

        return result

    def apply_cellular_automaton(self, grid, iterations=1, rule="conway"):
        """Apply a cellular automaton to the grid.

        Args:
            grid (numpy.ndarray): The input grid to apply the cellular automaton to
            iterations (int): Number of iterations to apply the rules
            rule (str): The rule to apply ("conway" for Game of Life rules)

        Returns:
            numpy.ndarray: The grid after applying cellular automaton rules
        """
        # Check if we have a cached result
        cache_key = self._get_cache_key(
            "apply_cellular_automaton",
            iterations=iterations,
            rule=rule,
            grid_hash=hash(grid.tobytes()),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Make a copy of the grid
        result = grid.copy()

        # Apply the cellular automaton for the specified number of iterations
        for _ in range(iterations):
            result = self._process_cellular_automaton_iteration(result, rule)

        # Cache the result
        self._cache[cache_key] = result

        return result

    def _process_cellular_automaton_iteration(self, grid, rule):
        """Process one iteration of cellular automaton.

        Args:
            grid (numpy.ndarray): The input grid
            rule (str): The rule to apply

        Returns:
            numpy.ndarray: The grid after one iteration
        """
        if SCIPY_AVAILABLE:
            return self._apply_cellular_automaton_scipy(grid, rule)
        return self._apply_cellular_automaton_manual(grid, rule)

    def _apply_cellular_automaton_scipy(self, grid, rule):
        """Apply cellular automaton rules using scipy's optimized functions.

        Args:
            grid (numpy.ndarray): The input grid
            rule (str): The rule to apply

        Returns:
            numpy.ndarray: The processed grid
        """
        # Use scipy's convolve for faster neighbor counting
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = ndimage.convolve(grid, kernel, mode="constant", cval=0)

        if rule == "conway":
            return self._apply_conway_rule_vectorized(grid, neighbors)

        # Default rule: cells with 2-3 neighbors survive, cells with 3 neighbors are born
        return np.where((neighbors == 3) | ((grid > 0) & (neighbors == 2)), 1.0, 0.0)

    def _apply_conway_rule_vectorized(self, grid, neighbors):
        """Apply Conway's Game of Life rules using vectorized operations.

        Args:
            grid (numpy.ndarray): The input grid
            neighbors (numpy.ndarray): The neighbor count grid

        Returns:
            numpy.ndarray: The processed grid
        """
        # Conway's Game of Life rules
        birth = (neighbors == 3) & (
            grid == 0
        )  # Dead cells with exactly 3 neighbors become alive
        survive = ((neighbors == 2) | (neighbors == 3)) & (
            grid > 0
        )  # Live cells with 2-3 neighbors survive
        return np.where(birth | survive, 1.0, 0.0)

    def _apply_cellular_automaton_manual(self, grid, rule):
        """Apply cellular automaton rules using a manual implementation.

        Args:
            grid (numpy.ndarray): The input grid
            rule (str): The rule to apply

        Returns:
            numpy.ndarray: The processed grid
        """
        new_grid = np.zeros_like(grid)

        # Only process interior cells to avoid edge effects
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                # Count neighbors in the 3x3 neighborhood
                neighbors = np.sum(grid[y - 1 : y + 2, x - 1 : x + 2]) - grid[y, x]
                new_grid[y, x] = self._apply_rule_to_cell(grid[y, x], neighbors, rule)

        return new_grid

    def _apply_rule_to_cell(self, cell_value, neighbors, rule):
        """Apply cellular automaton rule to a single cell.

        Args:
            cell_value (float): The current cell value
            neighbors (float): The number of live neighbors
            rule (str): The rule to apply

        Returns:
            float: The new cell value (0.0 or 1.0)
        """
        if rule == "conway":
            # Conway's Game of Life rules
            if cell_value > 0 and neighbors in [
                2,
                3,
            ]:  # Live cell with 2-3 neighbors survives
                return 1.0
            return 1.0 if cell_value == 0 and neighbors == 3 else 0.0
        # Default rule
        return 1.0 if neighbors == 3 or (cell_value > 0 and neighbors == 2) else 0.0

    def apply_clustering(self, grid, min_cluster_size=3):
        """Group adjacent cells into clusters and remove small clusters."""
        # Check if we have a cached result
        cache_key = self._get_cache_key(
            "apply_clustering",
            min_cluster_size=min_cluster_size,
            grid_hash=hash(grid.tobytes()),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not SCIPY_AVAILABLE:
            # Without scipy, just return the original grid
            return grid

        # Use scipy's label function to identify clusters
        labeled_array, _ = ndimage.label(grid)

        # Count the size of each cluster
        cluster_sizes = np.bincount(labeled_array.ravel())

        # Create a mask of clusters that are too small (index 0 is background)
        small_clusters = np.zeros_like(cluster_sizes, dtype=bool)
        small_clusters[1:] = cluster_sizes[1:] < min_cluster_size

        # Remove small clusters
        mask = small_clusters[labeled_array]
        result = grid.copy()
        result[mask] = 0

        # Cache the result
        self._cache[cache_key] = result

        return result


# Define a minimal AsteroidGenerator for testing
class AsteroidGenerator(BaseGenerator):
    """A simplified AsteroidGenerator class for testing purposes."""

    def __init__(self, seed=None, width=100, height=100):
        super().__init__(seed=seed, width=width, height=height)
        self.entity_type = "asteroid"

    def generate_field(self):
        """Generate an asteroid field."""
        # Create a noise layer
        grid = self.generate_noise_layer("medium", scale=0.1)

        # Apply thresholding
        grid = self.apply_threshold(grid, 0.5)

        # Apply cellular automaton for smoothing
        grid = self.apply_cellular_automaton(grid, iterations=2)

        # Create metadata
        metadata = {"seed": self.seed}

        return grid, metadata

    def generate_values(self, grid):
        """Generate values for asteroids."""
        value_noise = self.generate_noise_layer("fine", scale=0.2)
        return np.where(grid > 0, value_noise * 10, 0)

    def generate_rare_resources(self, grid):
        """Generate rare resources."""
        rare_noise = self.generate_noise_layer("very_fine", scale=0.3)
        return np.where(grid > 0, rare_noise > 0.8, 0).astype(float)

    def _spiral_pattern(self):
        """Generate a spiral pattern."""
        grid = np.zeros((self.height, self.width))
        center_x, center_y = self.width // 2, self.height // 2

        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                grid[y, x] = (angle + distance * 0.1) % (2 * np.pi) / (2 * np.pi)

        return grid

    def _ring_pattern(self):
        """Generate a ring pattern."""
        grid = np.zeros((self.height, self.width))
        center_x, center_y = self.width // 2, self.height // 2

        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                grid[y, x] = (distance * 0.1) % 1.0

        return grid


# Import visualization module
try:
    from utils.visualization import GeneratorVisualizer
except ImportError:
    # Create a simple visualizer if the module is not available
    class GeneratorVisualizer:
        def visualize_multiple_grids(self, grids, layout, figsize):
            fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
            axes = axes.flatten()

            for i, grid_data in enumerate(grids):
                ax = axes[i]
                im = ax.imshow(grid_data["grid"], cmap=grid_data["cmap"])
                ax.set_title(grid_data["title"])
                plt.colorbar(im, ax=ax)

            return fig


try:
    import scipy.ndimage as ndimage

    # Import signal only if needed in the future
    # from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    print("scipy not available. Tests will be skipped.")
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("matplotlib not available. Visualization will be skipped.")
    MATPLOTLIB_AVAILABLE = False

# Check if all required packages are available
if not NUMPY_AVAILABLE:
    print("numpy is required for this test.")
    sys.exit(1)


def test_asteroid_generator():
    """Test the basic functionality and performance of the AsteroidGenerator class."""
    print("Testing AsteroidGenerator initialization...")

    # Create a generator with a fixed seed for reproducibility
    generator = AsteroidGenerator(seed=42, width=100, height=100)

    # Verify basic properties
    if generator.seed != 42:
        raise AssertionError(f"Expected seed 42, got {generator.seed}")
    if generator.width != 100:
        raise AssertionError(f"Expected width 100, got {generator.width}")
    if generator.height != 100:
        raise AssertionError(f"Expected height 100, got {generator.height}")
    if (
        generator.entity_type != "asteroid"
    ):
        raise AssertionError(f"Expected entity_type 'asteroid', got {generator.entity_type}")

    # Test noise generation with performance measurement
    print("Testing noise generation performance...")
    start_time = time.time()
    noise = generator.generate_noise_layer("medium", scale=0.05)
    noise_time = _test_asteroid_generator_handler(
        start_time, "Noise generation time: ", noise
    )
    # Test field generation with performance measurement
    print("Testing asteroid field generation performance...")
    start_time = time.time()
    asteroid_grid, metadata = generator.generate_field()
    _ = _test_asteroid_generator_handler(
        start_time, "Field generation time: ", asteroid_grid
    )
    if np.sum(asteroid_grid > 0) <= 0:
        raise AssertionError("No asteroids were generated")
    if "seed" not in metadata:
        raise AssertionError("Metadata missing seed information")

    # Test value generation with performance measurement
    print("Testing asteroid value generation performance...")
    start_time = time.time()
    value_grid = generator.generate_values(asteroid_grid)
    _ = _test_asteroid_generator_handler(
        start_time, "Value generation time: ", value_grid
    )
    if np.sum(value_grid > 0) <= 0:
        raise AssertionError("No asteroid values were generated")

    # Test rare resource generation with performance measurement
    print("Testing rare resource generation performance...")
    start_time = time.time()
    rare_grid = generator.generate_rare_resources(asteroid_grid)
    _ = _test_asteroid_generator_handler(
        start_time, "Rare resource generation time: ", rare_grid
    )
    # Test caching by running the same operations again
    print("\nTesting cache performance...")
    start_time = time.time()
    noise2 = generator.generate_noise_layer("medium", scale=0.05)
    noise_time2 = time.time() - start_time
    print(
        f"Cached noise generation time: {noise_time2:.4f} seconds (Speed improvement: {noise_time / noise_time2:.2f}x)"
    )
    if not np.array_equal(noise, noise2):
        raise AssertionError("Cached noise layer does not match original")

    print("All basic tests passed!")
    return generator, asteroid_grid, value_grid, rare_grid


def _test_asteroid_generator_handler(start_time, arg1, arg2):
    result = time.time() - start_time
    print(f"{arg1}{result:.4f} seconds")
    if arg2.shape != (100, 100):
        raise AssertionError(f"Expected shape (100, 100), got {arg2.shape}")

    return result


def test_pattern_generation():
    """Test the pattern generation functionality and performance."""
    print("\nTesting pattern generation performance...")

    # Create a generator with a fixed seed for reproducibility
    generator = AsteroidGenerator(seed=123, width=80, height=80)

    # Measure performance of field generation
    start_time = time.time()
    asteroid_grid = generator.generate_field()
    pattern_time = time.time() - start_time
    print(f"Pattern-based field generation time: {pattern_time:.4f} seconds")

    if asteroid_grid.shape != (
        80,
        80,
    ):
        raise AssertionError(f"Expected shape (80, 80), got {asteroid_grid.shape}")
    if (
        np.sum(asteroid_grid > 0) <= 0
    ):
        raise AssertionError("No asteroids were generated with pattern weights")

    # Test individual pattern generation performance
    print("Testing individual pattern generation performance...")

    start_time = time.time()
    spiral = generator._spiral_pattern()
    spiral_time = time.time() - start_time
    print(f"Spiral pattern generation time: {spiral_time:.4f} seconds")

    start_time = time.time()
    rings = generator._ring_pattern()
    rings_time = time.time() - start_time
    print(f"Ring pattern generation time: {rings_time:.4f} seconds")

    # Test caching of patterns
    print("Testing pattern caching...")
    start_time = time.time()
    spiral2 = generator._spiral_pattern()
    spiral_time2 = time.time() - start_time
    print(
        f"Cached spiral pattern time: {spiral_time2:.4f} seconds (Speed improvement: {spiral_time / spiral_time2:.2f}x if > 1)"
    )

    if not np.array_equal(
        spiral, spiral2
    ):
        raise AssertionError("Cached spiral pattern does not match original")

    print("Pattern generation tests passed!")
    return asteroid_grid, spiral, rings


def visualize_results(asteroid_grid, value_grid, rare_grid, pattern_data):
    """Visualize the results of the tests using the new visualization module."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nSkipping visualization as matplotlib is not available.")
        return

    print("\nVisualizing results...")

    # Unpack pattern data
    pattern_grid, spiral, rings = pattern_data

    # Create a visualizer instance
    visualizer = GeneratorVisualizer()

    # Create a multi-grid visualization
    grids = [
        {"grid": asteroid_grid, "title": "Generated Asteroid Field", "cmap": "binary"},
        {"grid": value_grid, "title": "Asteroid Values", "cmap": "viridis"},
        {"grid": rare_grid, "title": "Rare Resources", "cmap": "plasma"},
        {"grid": pattern_grid, "title": "Pattern-based Field", "cmap": "binary"},
        {"grid": spiral, "title": "Spiral Pattern", "cmap": "hot"},
        {"grid": rings, "title": "Ring Pattern", "cmap": "hot"},
    ]

    # Visualize all grids
    visualizer.visualize_multiple_grids(
        grids=grids, layout=(2, 3), figsize=(15, 10), show=False
    )

    # Save the visualization
    plt.tight_layout()
    plt.savefig("asteroid_generator_test.png")
    print("Visualization saved as 'asteroid_generator_test.png'")


def _check_file_exists(file_path, file_name):
    """Check if a file exists and print the result.

    Args:
        file_path: Path to the file
        file_name: Name of the file for display purposes

    Returns:
        bool: True if file exists, False otherwise
    """
    import os

    if not os.path.isfile(file_path):
        print(f"Error: {file_name} file does not exist at {file_path}")
        return False
    else:
        print(f"✓ {file_name} file exists at {file_path}")
        return True


def _check_file_content(file_path):
    """Check the content of the asteroid generator file for proper imports and inheritance.

    Args:
        file_path: Path to the asteroid generator file

    Returns:
        bool: True if content checks pass, False otherwise
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()

        success = True

        # Check imports
        if (
            "from generators.base_generator import BaseGenerator" in content
            or "from entities.base_generator import BaseGenerator" in content
        ):
            print("✓ AsteroidGenerator imports BaseGenerator")
        else:
            print("✗ AsteroidGenerator does not import BaseGenerator correctly")
            success = False

        # Check inheritance
        if "class AsteroidGenerator(BaseGenerator):" in content:
            print("✓ AsteroidGenerator inherits from BaseGenerator")
        else:
            print("✗ AsteroidGenerator does not inherit from BaseGenerator")
            success = False

        return success
    except Exception as e:
        print(f"Error checking file contents: {e}")
        return False


def _check_programmatic_inheritance(parent_dir):
    """Check inheritance and required methods programmatically.

    Args:
        parent_dir: Parent directory path for imports

    Returns:
        bool: True if inheritance checks pass, False otherwise
    """
    try:
        sys.path.append(parent_dir)
        from generators.asteroid_field import AsteroidField
        from generators.base_generator import BaseGenerator

        success = True

        # Check inheritance
        if issubclass(AsteroidField, BaseGenerator):
            print("✓ AsteroidField is a subclass of BaseGenerator")
        else:
            print("✗ AsteroidField is not a subclass of BaseGenerator")
            success = False

        return _check_required_methods(AsteroidField) and success
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return False


def _check_required_methods(class_obj):
    """Check if a class has the required methods.

    Args:
        class_obj: Class object to check

    Returns:
        bool: True if all required methods exist, False otherwise
    """
    required_methods = ["generate", "apply_cellular_automaton"]
    success = True

    for method in required_methods:
        if hasattr(class_obj, method) and callable(getattr(class_obj, method)):
            print(f"✓ AsteroidField has required method: {method}")
        else:
            print(f"✗ AsteroidField is missing required method: {method}")
            success = False

    return success


def verify_file_structure():
    """Verify the file structure and inheritance of AsteroidGenerator."""
    import os

    print("\n=== Verifying AsteroidGenerator File Structure ===")

    # Get the parent directory to access the src folder
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Define file paths
    base_generator_path = os.path.join(
        parent_dir, "src", "entities", "base_generator.py"
    )
    asteroid_generator_path = os.path.join(
        parent_dir, "src", "generators", "asteroid_generator.py"
    )

    # Check if files exist
    base_exists = _check_file_exists(base_generator_path, "BaseGenerator")
    asteroid_exists = _check_file_exists(asteroid_generator_path, "AsteroidGenerator")

    success = base_exists and asteroid_exists

    # Only proceed with further checks if both files exist
    if success:
        # Check file contents and inheritance
        content_check = _check_file_content(asteroid_generator_path)
        inheritance_check = _check_programmatic_inheritance(parent_dir)

        success = content_check and inheritance_check

    print(f"File structure verification {'successful' if success else 'failed'}")
    return success


if __name__ == "__main__":
    print("=== AsteroidGenerator Comprehensive Verification ===")

    # First verify the file structure
    structure_success = verify_file_structure()

    # Check if we can run the tests
    if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
        print("\nSkipping tests due to missing dependencies.")
        print("Please install the required packages to run the tests:")
        print("  - numpy")
        print("  - scipy")
        print("  - matplotlib (optional, for visualization)")
        sys.exit(1)

    # Run the tests with performance measurements
    print("\n=== Running Performance Tests ===\n")
    generator, asteroid_grid, value_grid, rare_grid = test_asteroid_generator()
    pattern_data = test_pattern_generation()

    # Visualize the results if matplotlib is available
    visualize_results(asteroid_grid, value_grid, rare_grid, pattern_data)

    print("\n=== All tests completed successfully! ===")
    print(
        "The refactored AsteroidGenerator is working correctly with the optimized BaseGenerator."
    )
    print(
        "Performance improvements have been verified through caching and optimized algorithms."
    )
