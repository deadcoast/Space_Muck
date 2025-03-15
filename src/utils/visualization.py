"""
Visualization utilities for Space Muck generators.

This module provides visualization tools for generator outputs,
helping developers understand and debug procedural generation.
"""

# Standard library imports
import logging
import os

# Local application imports
from typing import List, Optional

# Third-party library imports
import numpy as np

# Constants
MATPLOTLIB_NOT_AVAILABLE_MSG = "Cannot visualize: matplotlib not available"

try:
    # matplotlib is imported to check availability
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Visualization features will be limited.")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image export features will be limited.")


class GeneratorVisualizer:
    """Visualization tools for generator outputs."""

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check for visualization dependencies
        self.can_visualize = MATPLOTLIB_AVAILABLE
        self.can_export = PIL_AVAILABLE

        # Default color maps
        self.default_cmaps = {
            "terrain": plt.cm.terrain if MATPLOTLIB_AVAILABLE else None,
            "heat": plt.cm.hot if MATPLOTLIB_AVAILABLE else None,
            "binary": plt.cm.binary if MATPLOTLIB_AVAILABLE else None,
            "space": self._create_space_colormap() if MATPLOTLIB_AVAILABLE else None,
        }

    def _create_space_colormap(self):
        """Create a custom space-themed colormap."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Deep space blue to bright star colors
        colors = [(0, 0, 0.1), (0.1, 0.1, 0.3), (0.5, 0.5, 0.8), (1, 1, 1)]
        return LinearSegmentedColormap.from_list("space", colors)

    def visualize_grid(
        self,
        grid: np.ndarray,
        title: str = "Generator Output",
        colormap: str = "terrain",
        show: bool = True,
        save: bool = False,
        filename: str = "generator_output.png",
    ) -> Optional:
        """
        Visualize a 2D grid from a generator.

        Args:
            grid: 2D numpy array to visualize
            title: Title for the visualization
            colormap: Colormap to use ('terrain', 'heat', 'binary', 'space')
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            filename: Filename to save the visualization

        Returns:
            matplotlib.Figure if matplotlib is available, None otherwise
        """
        if not self.can_visualize:
            logging.warning(MATPLOTLIB_NOT_AVAILABLE_MSG)
            return None

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get colormap
        cmap = self.default_cmaps.get(colormap, plt.cm.terrain)

        # Plot the grid
        im = ax.imshow(grid, cmap=cmap, interpolation="nearest")
        ax.set_title(title)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Save if requested
        if save:
            self._save_visualization_to_file(filename, "Saved visualization to ")
        # Show if requested
        if show:
            plt.show()

        return fig

    def compare_grids(
        self,
        grids: List[np.ndarray],
        titles: List[str],
        colormap: str = "terrain",
        show: bool = True,
        save: bool = False,
        filename: str = "grid_comparison.png",
    ) -> Optional:
        """
        Compare multiple grids side by side.

        Args:
            grids: List of 2D numpy arrays to compare
            titles: List of titles for each grid
            colormap: Colormap to use
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            filename: Filename to save the visualization

        Returns:
            matplotlib.Figure if matplotlib is available, None otherwise
        """
        if not self.can_visualize:
            logging.warning(MATPLOTLIB_NOT_AVAILABLE_MSG)
            return None

        if len(grids) != len(titles):
            raise ValueError("Number of grids must match number of titles")

        # Determine grid layout
        n = len(grids)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        # Get colormap
        cmap = self.default_cmaps.get(colormap, plt.cm.terrain)

        # Flatten axes for easy iteration if there are multiple
        axes = axes.flatten() if n > 1 else [axes]
        # Plot each grid
        for i, (grid, title) in enumerate(zip(grids, titles)):
            if i < len(axes):
                im = axes[i].imshow(grid, cmap=cmap, interpolation="nearest")
                axes[i].set_title(title)
                plt.colorbar(im, ax=axes[i])

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        # Save if requested
        if save:
            self._save_visualization_to_file(filename, "Saved comparison to ")
        # Show if requested
        if show:
            plt.show()

        return fig

    def _save_visualization_to_file(self, filename, message_prefix):
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"{message_prefix}{save_path}")

    def visualize_evolution(
        self,
        grids: List[np.ndarray],
        title: str = "Generator Evolution",
        colormap: str = "terrain",
        show: bool = True,
        save: bool = False,
        filename: str = "evolution.png",
        animation: bool = False,
        animation_filename: str = "evolution.gif",
    ) -> Optional:
        """
        Visualize the evolution of a grid over multiple iterations.

        Args:
            grids: List of 2D numpy arrays representing evolution steps
            title: Base title for the visualization
            colormap: Colormap to use
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            filename: Filename to save the visualization
            animation: Whether to create an animated GIF
            animation_filename: Filename for the animation

        Returns:
            matplotlib.Figure if matplotlib is available, None otherwise
        """
        if not self.can_visualize:
            logging.warning(MATPLOTLIB_NOT_AVAILABLE_MSG)
            return None

        # Create titles for each step
        titles = [f"{title} - Step {i + 1}" for i in range(len(grids))]

        # Use compare_grids for static visualization
        fig = self.compare_grids(grids, titles, colormap, show, save, filename)

        # Create animation if requested
        if animation and self.can_export and PIL_AVAILABLE:
            try:
                # Create temporary figures for each frame
                frames = []
                for i, grid in enumerate(grids):
                    temp_fig, temp_ax = plt.subplots(figsize=(8, 6))
                    cmap = self.default_cmaps.get(colormap, plt.cm.terrain)
                    temp_ax.imshow(grid, cmap=cmap, interpolation="nearest")
                    temp_ax.set_title(f"{title} - Step {i + 1}")

                    # Convert figure to image
                    temp_fig.canvas.draw()
                    image = Image.frombytes(
                        "RGB",
                        temp_fig.canvas.get_width_height(),
                        temp_fig.canvas.tostring_rgb(),
                    )
                    frames.append(image)
                    plt.close(temp_fig)

                # Save animation
                save_path = os.path.join(self.output_dir, animation_filename)
                frames[0].save(
                    save_path,
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,  # 500ms per frame
                    loop=0,  # Loop forever
                )
                logging.info(f"Saved animation to {save_path}")
            except Exception as e:
                logging.error(f"Failed to create animation: {str(e)}")

        return fig

    def export_grid_as_image(
        self,
        grid: np.ndarray,
        filename: str,
        colormap: str = "terrain",
        normalize: bool = True,
    ) -> bool:
        """
        Export a grid as an image file.

        Args:
            grid: 2D numpy array to export
            filename: Filename to save the image
            colormap: Colormap to use
            normalize: Whether to normalize the grid values to 0-255

        Returns:
            True if successful, False otherwise
        """
        if not self.can_export:
            logging.warning("Cannot export: PIL not available")
            return False

        try:
            return self._process_and_save_grid_as_image(
                normalize, grid, colormap, filename
            )
        except Exception as e:
            logging.error(f"Failed to export grid: {str(e)}")
            return False

    def _process_and_save_grid_as_image(self, normalize, grid, colormap, filename):
        # Normalize grid values if requested
        if normalize:
            min_val = np.min(grid)
            max_val = np.max(grid)
            normalized_grid = (
                (grid - min_val) * 255 / (max_val - min_val)
                if max_val > min_val
                else np.zeros_like(grid)
            )
        else:
            # Clip values to 0-255 range
            normalized_grid = np.clip(grid, 0, 255)

        # Convert to 8-bit unsigned integer
        img_data = normalized_grid.astype(np.uint8)

        # Apply colormap if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            cmap = self.default_cmaps.get(colormap, plt.cm.terrain)
            img_data = (cmap(img_data) * 255).astype(np.uint8)

        # Create and save image
        img = Image.fromarray(img_data)
        save_path = os.path.join(self.output_dir, filename)
        img.save(save_path)
        logging.info(f"Exported grid to {save_path}")
        return True

    def visualize_grid_comparison(
        self,
        grids: List[np.ndarray],
        titles: List[str],
        colormap: str = "binary",
        show: bool = True,
        save: bool = False,
        filename: str = "grid_comparison.png",
    ) -> Optional:
        """
        Visualize a comparison of multiple grids side by side.

        Args:
            grids: List of 2D numpy arrays to compare
            titles: List of titles for each grid
            colormap: Colormap to use
            show: Whether to display the visualization
            save: Whether to save the visualization to a file
            filename: Filename to save the visualization

        Returns:
            matplotlib.Figure if matplotlib is available, None otherwise
        """
        return self.compare_grids(
            grids=grids,
            titles=titles,
            colormap=colormap,
            show=show,
            save=save,
            filename=filename,
        )


def visualize_generator_output(
    generator,
    output_dir: str = "visualizations",
    show: bool = True,
    save: bool = True,
    colormap: str = "terrain",
) -> None:
    """
    Convenience function to visualize outputs from a generator.

    Args:
        generator: Generator instance with generate_* methods
        output_dir: Directory to save visualizations
        show: Whether to display the visualizations
        save: Whether to save the visualizations
        colormap: Colormap to use
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Cannot visualize: matplotlib not available")
        return

    visualizer = GeneratorVisualizer(output_dir)

    # Get generator class name for filenames
    generator_name = generator.__class__.__name__.lower()

    # Visualize noise layers if available
    if hasattr(generator, "generate_noise_layer"):
        noise_types = ["low", "medium", "high", "detail"]
        noise_grids = []
        noise_titles = []

        for noise_type in noise_types:
            try:
                grid = generator.generate_noise_layer(noise_type=noise_type)
                noise_grids.append(grid)
                noise_titles.append(f"{noise_type.capitalize()} Noise")
            except Exception as e:
                logging.warning(f"Could not generate {noise_type} noise: {str(e)}")

        if noise_grids:
            visualizer.compare_grids(
                noise_grids,
                noise_titles,
                colormap=colormap,
                show=show,
                save=save,
                filename=f"{generator_name}_noise_comparison.png",
            )

    # Visualize cellular automaton if available
    if hasattr(generator, "apply_cellular_automaton") and hasattr(
        generator, "generate_noise_layer"
    ):
        try:
            # Generate base noise
            base_grid = generator.generate_noise_layer(noise_type="medium")

            # Apply threshold to create binary grid
            if hasattr(generator, "apply_threshold"):
                binary_grid = generator.apply_threshold(base_grid, 0.5, 1.0)
            else:
                binary_grid = (base_grid > 0.5).astype(np.float32)

            # Apply cellular automaton with different iterations
            ca_grids = [binary_grid]
            ca_titles = ["Initial Grid"]

            for iterations in [1, 3, 5]:
                ca_grid = generator.apply_cellular_automaton(
                    binary_grid.copy(), iterations=iterations
                )
                ca_grids.append(ca_grid)
                ca_titles.append(f"After {iterations} Iterations")

            visualizer.compare_grids(
                ca_grids,
                ca_titles,
                colormap="binary",
                show=show,
                save=save,
                filename=f"{generator_name}_ca_evolution.png",
            )

            # Create animation
            visualizer.visualize_evolution(
                ca_grids,
                title="Cellular Automaton Evolution",
                colormap="binary",
                show=False,
                save=save,
                filename=f"{generator_name}_ca_evolution_grid.png",
                animation=True,
                animation_filename=f"{generator_name}_ca_evolution.gif",
            )
        except Exception as e:
            logging.warning(f"Could not visualize cellular automaton: {str(e)}")

    # Visualize clusters if available
    if hasattr(generator, "create_clusters") and hasattr(
        generator, "generate_noise_layer"
    ):
        try:
            # Generate base noise
            base_grid = generator.generate_noise_layer(noise_type="medium")

            # Apply threshold
            if hasattr(generator, "apply_threshold"):
                thresholded_grid = generator.apply_threshold(base_grid, 0.5, 1.0)
            else:
                thresholded_grid = (base_grid > 0.5).astype(np.float32)

            # Create clusters
            clustered_grid = generator.create_clusters(
                thresholded_grid, num_clusters=5, cluster_value_multiplier=2.0
            )

            # Visualize before and after
            visualizer.compare_grids(
                [thresholded_grid, clustered_grid],
                ["Before Clustering", "After Clustering"],
                colormap=colormap,
                show=show,
                save=save,
                filename=f"{generator_name}_clustering.png",
            )
        except Exception as e:
            logging.warning(f"Could not visualize clustering: {str(e)}")
