"""
Converter interface for Space Muck.

This module provides the user interface components for the converter management system,
including the main dashboard, detailed views, chain management and efficiency monitoring.
"""

# Standard library imports
import time

# Third-party library imports

# Local application imports
from config import COLOR_TEXT, COLOR_BG
from converters.converter_models import (
from typing import Dict, List, Optional, Tuple, Any, Callable
from src.ui.ascii_ui import (
from src.ui.draw_utils import draw_text
from src.ui.event_system import EventSystem
import pygame

    ASCIIBox,
    ASCIIPanel,
    ASCIIProgressBar,
    ASCIIButton,
    ASCIIChainVisualizer,
    draw_ascii_table,
)

    Converter,
    Recipe,
    ConversionProcess,
    ProductionChain,
    ChainStep,
    ConverterType,
    ConverterTier,
    ResourceType,
    EfficiencyFactor,
    OptimizationSuggestion,
)

class ConverterDashboard:
    """Main dashboard for converter management overview."""

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize the converter dashboard.

        Args:
            x: X position
            y: Y position
            width: Width in pixels
            height: Height in pixels
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.panel = ASCIIPanel(
            pygame.Rect(x, y, width, height),
            title="Converter Management System",
            border_style="double",
        )
        self.converters: List[Converter] = []
        self.selected_converter_id: Optional[str] = None

        # Dashboard components
        self.converter_list_box = ASCIIBox(
            x + 10, y + 40, 30, 10, title="Converters", border_style="single"
        )

        self.status_box = ASCIIBox(
            x + width - 200, y + 40, 25, 5, title="Status", border_style="single"
        )

        # Buttons
        self.details_button = ASCIIButton(
            x + 20, y + height - 40, "View Details", self.view_details
        )

        self.chain_button = ASCIIButton(
            x + 120, y + height - 40, "Manage Chains", self.manage_chains
        )

        self.efficiency_button = ASCIIButton(
            x + 220, y + height - 40, "Efficiency Monitor", self.monitor_efficiency
        )

    def set_converters(self, converters: List[Converter]) -> None:
        """
        Set the list of converters to display.

        Args:
            converters: List of converters
        """
        self.converters = converters
        self.update_converter_list()

    def update_converter_list(self) -> None:
        """Update the converter list display."""
        self.converter_list_box.content = []
        for i, converter in enumerate(self.converters):
            prefix = ">" if converter.id == self.selected_converter_id else " "
            color = (
                (100, 255, 100) if len(converter.active_processes) > 0 else COLOR_TEXT
            )
            self.converter_list_box.add_text(
                1,
                1 + i,
                f"{prefix} {converter.name} ({converter.type.value.capitalize()})",
                {"color": color},
            )

    def update_status_box(self) -> None:
        """Update the status box with current information."""
        self.status_box.content = []

        active_count = sum(len(c.active_processes) > 0 for c in self.converters)
        total_count = len(self.converters)

        self.status_box.add_text(1, 1, f"Active: {active_count}/{total_count}")

        # Count by type
        type_counts = {}
        for c in self.converters:
            type_name = c.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        for y_pos, (type_name, count) in enumerate(type_counts.items(), start=2):
            self.status_box.add_text(1, y_pos, f"{type_name.capitalize()}: {count}")

    def select_converter(self, converter_id: str) -> None:
        """
        Select a converter by ID.

        Args:
            converter_id: ID of the converter to select
        """
        self.selected_converter_id = converter_id
        self.update_converter_list()

    def view_details(self) -> None:
        """Handler for View Details button."""
        # To be implemented with event/callback system
        pass

    def manage_chains(self) -> None:
        """Handler for Manage Chains button."""
        # To be implemented with event/callback system
        pass

    def monitor_efficiency(self) -> None:
        """Handler for Efficiency Monitor button."""
        # To be implemented with event/callback system
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for the dashboard.

        Args:
            event: Pygame event

        Returns:
            bool: True if the event was handled
        """
        # Handle button events
        if self.details_button.handle_event(event):
            return True

        if self.chain_button.handle_event(event):
            return True

        if self.efficiency_button.handle_event(event):
            return True

        # Handle converter selection
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            char_width, char_height = 8, 16  # Approximate for typical font

            for i, converter in enumerate(self.converters):
                # Check if mouse is over a converter in the list
                item_rect = pygame.Rect(
                    self.converter_list_box.x + char_width,
                    self.converter_list_box.y + (i + 1) * char_height,
                    self.converter_list_box.width * char_width - 2 * char_width,
                    char_height,
                )

                if item_rect.collidepoint(mouse_pos):
                    self.select_converter(converter.id)
                    return True

        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Draw the dashboard.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Update status before drawing
        self.update_status_box()

        # Draw panel background
        self.panel.draw(surface, font)

        # Draw components
        self.converter_list_box.draw(surface, font)
        self.status_box.draw(surface, font)

        # Draw buttons
        self.details_button.draw(surface, font)
        self.chain_button.draw(surface, font)
        self.efficiency_button.draw(surface, font)

        # Draw instruction text
        draw_text(
            surface,
            "Select a converter to view details or manage chains",
            self.x + 10,
            self.y + self.height - 70,
            font=font,
            color=COLOR_TEXT,
        )

class ConverterDetailsView:
    """Detailed view of a single converter with process management."""

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize the converter details view.

        Args:
            x: X position
            y: Y position
            width: Width in pixels
            height: Height in pixels
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.panel = ASCIIPanel(
            pygame.Rect(x, y, width, height),
            title="Converter Details",
            border_style="double",
        )
        self.converter: Optional[Converter] = None

        # Details components
        self.info_box = ASCIIBox(
            x + 10, y + 40, 40, 7, title="Information", border_style="single"
        )

        self.processes_box = ASCIIBox(
            x + 10,
            y + 160,
            width - 20,
            10,
            title="Active Processes",
            border_style="single",
        )

        # Progress bars for processes
        self.progress_bars: List[Tuple[str, ASCIIProgressBar]] = []

        # Buttons
        self.start_process_button = ASCIIButton(
            x + 20, y + height - 40, "Start Process", self.start_process
        )

        self.cancel_process_button = ASCIIButton(
            x + 150, y + height - 40, "Cancel Process", self.cancel_process
        )

        self.back_button = ASCIIButton(
            x + width - 100, y + height - 40, "Back", self.go_back
        )

    def set_converter(self, converter: Converter) -> None:
        """
        Set the converter to display.

        Args:
            converter: Converter to display
        """
        self.converter = converter
        self.update_info_box()
        self.update_processes_box()

    def update_info_box(self) -> None:
        """Update the information box with current converter details."""
        if not self.converter:
            return

        self.info_box.content = []

        # Basic information
        self.info_box.add_text(1, 1, f"Name: {self.converter.name}")

        # Type info with background panel
        type_panel = draw_panel(
            pygame.Rect(0, 0, 20, 3), border_style="single", bg_color=COLOR_BG
        )
        draw_text(type_panel, 1, 1, f"Type: {self.converter.type.value.capitalize()}")

        # Tier with color coding based on ConverterTier
        tier_colors: Dict[ConverterTier, Tuple[int, int, int]] = {
            ConverterTier.BASIC: (200, 200, 200),  # Gray
            ConverterTier.STANDARD: (100, 255, 100),  # Green
            ConverterTier.ADVANCED: (100, 100, 255),  # Blue
            ConverterTier.ELITE: (255, 215, 0),  # Gold
        }
        tier_color = tier_colors.get(self.converter.tier, COLOR_TEXT)
        self.info_box.add_text(
            1,
            3,
            f"Tier: {self.converter.tier.name.capitalize()}",
            {"color": tier_color},
        )

        # Process info with ResourceType display
        if self.converter.active_processes:
            process: ConversionProcess = self.converter.active_processes[0]
            input_type: ResourceType = process.recipe.input_type
            output_type: ResourceType = process.recipe.output_type
            self.info_box.add_text(
                1, 4, f"Converting: {input_type.value} → {output_type.value}"
            )

        # Efficiency with color coding and factors
        if self.converter.base_efficiency >= 0.9:
            efficiency_color = (100, 255, 100)
        elif self.converter.base_efficiency >= 0.7:
            efficiency_color = (255, 255, 100)
        else:
            efficiency_color = (255, 100, 100)
        self.info_box.add_text(
            1,
            5,
            f"Efficiency: {self.converter.base_efficiency:.2f}",
            {"color": efficiency_color},
        )

        # Energy status
        energy_ratio = self.converter.current_energy / self.converter.energy_capacity
        if energy_ratio >= 0.7:
            energy_color = (100, 255, 100)
        elif energy_ratio >= 0.3:
            energy_color = (255, 255, 100)
        else:
            energy_color = (255, 100, 100)
        self.info_box.add_text(
            1,
            6,
            f"Energy: {self.converter.current_energy:.1f}/{self.converter.energy_capacity:.1f}",
            {"color": energy_color},
        )

        # Display current efficiency factors if available
        if (
            hasattr(self.converter, "efficiency_factors")
            and self.converter.efficiency_factors
        ):
            factor: EfficiencyFactor = self.converter.efficiency_factors[0]
            self.info_box.add_text(
                1, 7, f"Active Factor: {factor.name} ({factor.value:+.2f})"
            )

        # Show optimization suggestion if available
        if (
            hasattr(self.converter, "optimization_suggestions")
            and self.converter.optimization_suggestions
        ):
            suggestion: OptimizationSuggestion = (
                self.converter.optimization_suggestions[0]
            )
            self.info_box.add_text(
                1,
                8,
                f"Suggestion: {suggestion.description}",
                {"color": (255, 215, 0)},  # Gold color for suggestions
            )

        # Add chain step information if available
        if hasattr(self.converter, "chain_step"):
            step: ChainStep = self.converter.chain_step
            self.info_box.add_text(
                1, 9, f"Chain Step: {step.position} in {step.chain_id}"
            )

        # Add callback for efficiency monitoring
        self.optimize_callback: Callable[[], Any] = lambda: self.monitor_efficiency()

    def update_processes_box(self) -> None:
        """Update the processes box with current process information."""
        if not self.converter:
            return

        self.processes_box.content = []
        self.progress_bars = []

        if not self.converter.active_processes:
            self.processes_box.add_text(1, 1, "No active processes")
            return

        # Headers
        self.processes_box.add_text(1, 1, "Recipe")
        self.processes_box.add_text(20, 1, "Progress")
        self.processes_box.add_text(40, 1, "Time")

        # Process entries
        for i, process in enumerate(self.converter.active_processes):
            y_pos = i + 2

            # Recipe name
            self.processes_box.add_text(1, y_pos, process.recipe.name[:18])

            # Progress bar
            progress_bar = ASCIIProgressBar(
                self.processes_box.x + 20 * 8,  # Approximate char width
                self.processes_box.y + y_pos * 16,  # Approximate char height
                15,  # Width in characters
                process.progress,
            )
            self.progress_bars.append((process.id, progress_bar))

            # Time left (placeholder)
            time_left = "--:--"
            self.processes_box.add_text(40, y_pos, time_left)

    def start_process(self) -> None:
        """Handler for Start Process button."""
        # To be implemented with event/callback system
        pass

    def cancel_process(self) -> None:
        """Handler for Cancel Process button."""
        # To be implemented with event/callback system
        pass

    def go_back(self) -> None:
        """Handler for Back button."""
        # To be implemented with event/callback system
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for the details view.

        Args:
            event: Pygame event

        Returns:
            bool: True if the event was handled
        """
        # Handle button events
        if self.start_process_button.handle_event(event):
            return True

        if self.cancel_process_button.handle_event(event):
            return True

        return bool(self.back_button.handle_event(event))

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Draw the details view.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Update components before drawing
        if self.converter:
            self.update_info_box()
            self.update_processes_box()

        # Draw panel background
        self.panel.draw(surface, font)

        # Draw components
        self.info_box.draw(surface, font)
        self.processes_box.draw(surface, font)

        # Draw progress bars
        for _, progress_bar in self.progress_bars:
            progress_bar.draw(surface, font)

        # Draw buttons
        self.start_process_button.draw(surface, font)
        self.cancel_process_button.draw(surface, font)
        self.back_button.draw(surface, font)

class ChainManagementInterface:
    """Interface for creating and managing production chains."""

    # Color mapping for converter types
    TYPE_COLORS = {
        ConverterType.BASIC: (200, 200, 200),  # Gray
        ConverterType.ADVANCED: (100, 100, 255),  # Blue
        ConverterType.SPECIALIZED: (255, 215, 0),  # Gold
        ConverterType.INDUSTRIAL: (255, 100, 255),  # Purple
    }

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize the chain management interface.

        Args:
            x: X position
            y: Y position
            width: Width in pixels
            height: Height in pixels
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.panel = ASCIIPanel(
            pygame.Rect(x, y, width, height),
            title="Production Chain Management",
            border_style="double",
        )

        # Chain management components
        self.chains_box = ASCIIBox(
            x + 10, y + 40, 30, 8, title="Production Chains", border_style="single"
        )

        # Chain visualization
        self.chain_visualizer = ASCIIChainVisualizer(
            x + width - 350,
            y + 40,
            42,
            15,
            title="Chain Visualization",
            border_style="single",
        )

        # Chain details box below visualizer
        self.chain_details_box = ASCIIBox(
            x + width - 350,
            y + 250,  # Positioned below visualizer
            42,
            8,
            title="Chain Details",
            border_style="single",
        )

        # Buttons
        self.create_chain_button = ASCIIButton(
            x + 20, y + height - 40, "Create Chain", self.create_chain
        )

        self.edit_chain_button = ASCIIButton(
            x + 130, y + height - 40, "Edit Chain", self.edit_chain
        )

        self.delete_chain_button = ASCIIButton(
            x + 220, y + height - 40, "Delete Chain", self.delete_chain
        )

        self.back_button = ASCIIButton(
            x + width - 100, y + height - 40, "Back", self.go_back
        )

        # Data
        self.chains: List[ProductionChain] = []
        self.selected_chain_id: Optional[str] = None
        self.converters: List[Converter] = []
        self.recipes: List[Recipe] = []

    def set_chains(self, chains: List[ProductionChain]) -> None:
        """
        Set the list of chains to display.

        Args:
            chains: List of production chains
        """
        self.chains = chains
        self.update_chains_box()

    def set_data(
        self,
        chains: List[ProductionChain],
        converters: List[Converter],
        recipes: List[Recipe],
    ) -> None:
        """
        Set all data for chain management.

        Args:
            chains: List of production chains
            converters: List of available converters
            recipes: List of available recipes
        """
        self.chains = chains
        self.converters = converters
        self.recipes = recipes
        self.update_chains_box()

    def update_chains_box(self) -> None:
        """Update the chains box with current chains list."""
        self.chains_box.content = []

        if not self.chains:
            self.chains_box.add_text(1, 1, "No chains created")
            return

        for i, chain in enumerate(self.chains):
            prefix = ">" if chain.id == self.selected_chain_id else " "
            color = (100, 255, 100) if chain.active else COLOR_TEXT
            self.chains_box.add_text(
                1, 1 + i, f"{prefix} {chain.name}", {"color": color}
            )

    def update_chain_details(self) -> None:
        """Update the chain details and visualization with selected chain information."""
        self.chain_details_box.content = []

        if not self.selected_chain_id:
            self.chain_details_box.add_text(1, 1, "No chain selected")
            self.chain_visualizer.set_chain([], [])
            return

        # Find selected chain
        selected_chain = next(
            (c for c in self.chains if c.id == self.selected_chain_id), None
        )
        if not selected_chain:
            return

        # Basic info with color-coded status
        self.chain_details_box.add_text(1, 1, f"Name: {selected_chain.name}")
        status = "Active" if selected_chain.active else "Inactive"
        status_color = (100, 255, 100) if selected_chain.active else (255, 100, 100)
        self.chain_details_box.add_text(
            1, 2, f"Status: {status}", {"color": status_color}
        )

        # Prepare chain visualization data
        converters = []
        connections = []
        flow_rates = {}

        # Convert chain steps to visualization format
        for i, step in enumerate(selected_chain.steps):
            converter = next(
                (c for c in self.converters if c.id == step.converter_id), None
            )
            recipe = next((r for r in self.recipes if r.id == step.recipe_id), None)

            if converter and recipe:
                # Add converter node
                converter_info = {
                    "name": converter.name,
                    "type": converter.type.value,
                    "tier": getattr(converter, "tier", 1),
                    "efficiency": getattr(converter, "efficiency", 100.0),
                    "rate": getattr(step, "rate", 1.0),
                    "queue_size": len(getattr(converter, "active_processes", [])),
                }
                converters.append(converter_info)

                # Add connection to previous step
                if i > 0:
                    connections.append((i - 1, i))
                    # Add flow rate if available
                    if hasattr(step, "flow_rate"):
                        flow_rates[(i - 1, i)] = step.flow_rate

        # Update chain visualization
        self.chain_visualizer.set_chain(converters, connections, flow_rates)

        # Display converter type distribution
        y_pos = 4
        self.chain_details_box.add_text(1, y_pos, "Converter Types:")
        y_pos += 1

        # Group converters by type and display distribution
        type_counts: Dict[ConverterType, int] = {}
        for step in selected_chain.steps:
            if converter := next(
                (c for c in self.converters if c.id == step.converter_id), None
            ):
                conv_type = converter.type
                type_counts[conv_type] = type_counts.get(conv_type, 0) + 1

        # Display type counts with colors
        for conv_type, count in type_counts.items():
            color = self.TYPE_COLORS.get(conv_type, COLOR_TEXT)
            self.chain_details_box.add_text(
                3,
                y_pos,
                f"{conv_type.value.capitalize()}: {count}",
                {"color": color},
            )
            y_pos += 1

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to fit within a given width.

        Args:
            text: Text to wrap
            width: Maximum line width

        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}" if current_line else word
            if len(test_line) <= width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def select_chain(self, chain_id: str) -> None:
        """
        Select a chain by ID.

        Args:
            chain_id: ID of the chain to select
        """
        self.selected_chain_id = chain_id
        self.update_chains_box()
        self.update_chain_details()

    def create_chain(self) -> None:
        """Handler for Create Chain button."""
        # To be implemented with event/callback system
        pass

    def edit_chain(self) -> None:
        """Handler for Edit Chain button."""
        # To be implemented with event/callback system
        pass

    def delete_chain(self) -> None:
        """Handler for Delete Chain button."""
        # To be implemented with event/callback system
        pass

    def go_back(self) -> None:
        """Handler for Back button."""
        # To be implemented with event/callback system
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for the chain management interface.

        Args:
            event: Pygame event

        Returns:
            bool: True if the event was handled
        """
        # Handle button events
        if self.create_chain_button.handle_event(event):
            return True

        if self.edit_chain_button.handle_event(event):
            return True

        if self.delete_chain_button.handle_event(event):
            return True

        if self.back_button.handle_event(event):
            return True

        # Handle chain selection
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            char_width, char_height = 8, 16  # Approximate for typical font

            for i, chain in enumerate(self.chains):
                # Check if mouse is over a chain in the list
                item_rect = pygame.Rect(
                    self.chains_box.x + char_width,
                    self.chains_box.y + (i + 1) * char_height,
                    self.chains_box.width * char_width - 2 * char_width,
                    char_height,
                )

                if item_rect.collidepoint(mouse_pos):
                    self.select_chain(chain.id)
                    return True

        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Draw the chain management interface.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Update components before drawing
        self.update_chain_details()

        # Draw panel background
        self.panel.draw(surface, font)

        # Draw components
        self.chains_box.draw(surface, font)
        self.chain_visualizer.draw(surface, font)
        self.chain_details_box.draw(surface, font)

        # Draw buttons
        self.create_chain_button.draw(surface, font)
        self.edit_chain_button.draw(surface, font)
        self.delete_chain_button.draw(surface, font)
        self.back_button.draw(surface, font)

class EfficiencyMonitor:
    """Interface for monitoring and improving converter efficiency."""

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize the efficiency monitor.

        Args:
            x: X position
            y: Y position
            width: Width in pixels
            height: Height in pixels
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.panel = ASCIIPanel(
            pygame.Rect(x, y, width, height),
            title="Efficiency Monitor",
            border_style="double",
        )

        # Efficiency components
        self.efficiency_table_x = x + 10
        self.efficiency_table_y = y + 40

        self.suggestions_box = ASCIIBox(
            x + width - 350,
            y + 40,
            42,
            15,
            title="Optimization Suggestions",
            border_style="single",
        )

        # Buttons
        self.optimize_button = ASCIIButton(
            x + 20, y + height - 40, "Apply Optimization", self.apply_optimization
        )

        self.analyze_button = ASCIIButton(
            x + 170, y + height - 40, "Analyze System", self.analyze_system
        )

        self.back_button = ASCIIButton(
            x + width - 100, y + height - 40, "Back", self.go_back
        )

        # Data
        self.converters: List[Converter] = []
        self.efficiency_factors: List[EfficiencyFactor] = []
        self.suggestions: List[OptimizationSuggestion] = []
        self.selected_suggestion_index: int = -1

    def set_data(
        self,
        converters: List[Converter],
        factors: List[EfficiencyFactor],
        suggestions: List[OptimizationSuggestion],
    ) -> None:
        """
        Set data for the efficiency monitor.

        Args:
            converters: List of converters
            factors: List of efficiency factors
            suggestions: List of optimization suggestions
        """
        self.converters = converters
        self.efficiency_factors = factors
        self.suggestions = suggestions
        self.selected_suggestion_index = -1

    def update_suggestions_box(self) -> None:
        """Update the suggestions box with current suggestions and efficiency factors."""
        self.suggestions_box.content = []
        y_pos = 1

        # Display current efficiency factors if available
        y_pos = self._display_efficiency_factors(y_pos)
        
        # Display optimization suggestions
        if not self.suggestions:
            self.suggestions_box.add_text(
                1, y_pos, "No optimization suggestions available"
            )
            return

        self.suggestions_box.add_text(
            1, y_pos, "Available Optimizations:", {"color": (200, 200, 200)}
        )
        y_pos += 1

        for i, suggestion in enumerate(self.suggestions):
            y_pos = self._display_suggestion(i, suggestion, y_pos)
    
    def _display_efficiency_factors(self, y_pos: int) -> int:
        """
        Display efficiency factors in the suggestions box.
        
        Args:
            y_pos: Current vertical position
            
        Returns:
            int: Updated vertical position
        """
        if not self.efficiency_factors:
            return y_pos
            
        self.suggestions_box.add_text(
            1, y_pos, "Current Efficiency Factors:", {"color": (200, 200, 200)}
        )
        y_pos += 1

        for factor in self.efficiency_factors:
            factor_color = self._get_factor_color(factor.value)
            self.suggestions_box.add_text(
                3,
                y_pos,
                f"{factor.name}: {factor.value:+.2f}",
                {"color": factor_color},
            )
            y_pos += 1
            
        return y_pos + 1  # Add spacing
    
    def _get_factor_color(self, factor_value: float) -> Tuple[int, int, int]:
        """
        Get color based on factor value.
        
        Args:
            factor_value: The efficiency factor value
            
        Returns:
            Tuple[int, int, int]: RGB color tuple
        """
        if factor_value > 0.1:  # Green for positive
            return (100, 255, 100)
        elif factor_value < -0.1:  # Red for negative
            return (255, 100, 100)
        else:  # Gray for minimal impact
            return (200, 200, 200)
    
    def _display_suggestion(self, index: int, suggestion: OptimizationSuggestion, y_pos: int) -> int:
        """
        Display a single optimization suggestion.
        
        Args:
            index: Suggestion index
            suggestion: The suggestion to display
            y_pos: Current vertical position
            
        Returns:
            int: Updated vertical position
        """
        prefix = ">" if index == self.selected_suggestion_index else " "
        gain_color = self._get_gain_color(suggestion.potential_gain)

        # Format suggestion with truncated description
        self.suggestions_box.add_text(
            1,
            y_pos,
            f"{prefix} {suggestion.description[:38]}",
            {"color": gain_color},
        )

        # Show potential gain with color coding
        self.suggestions_box.add_text(
            3,
            y_pos + 1,
            f"Gain: +{suggestion.potential_gain:.2f}",
            {"color": gain_color},
        )

        # Show cost if available
        if suggestion.cost is not None:
            cost_color = self._get_cost_color(suggestion.cost)
            self.suggestions_box.add_text(
                20, y_pos + 1, f"Cost: {suggestion.cost}", {"color": cost_color}
            )

        return y_pos + 3  # Space for next suggestion
    
    def _get_gain_color(self, gain: float) -> Tuple[int, int, int]:
        """
        Get color based on potential gain value.
        
        Args:
            gain: The potential gain value
            
        Returns:
            Tuple[int, int, int]: RGB color tuple
        """
        if gain > 0.5:  # Gold for high impact
            return (255, 215, 0)
        elif gain > 0.2:  # Green for medium impact
            return (100, 255, 100)
        else:  # Gray for low impact
            return (200, 200, 200)
    
    def _get_cost_color(self, cost: float) -> Tuple[int, int, int]:
        """
        Get color based on cost value.
        
        Args:
            cost: The cost value
            
        Returns:
            Tuple[int, int, int]: RGB color tuple
        """
        if cost > 1000:  # Red for high cost
            return (255, 100, 100)
        elif cost > 500:  # Yellow for medium cost
            return (255, 255, 100)
        else:  # Green for low cost
            return (100, 255, 100)

    def select_suggestion(self, index: int) -> None:
        """
        Select a suggestion by index.

        Args:
            index: Index of the suggestion to select
        """
        if 0 <= index < len(self.suggestions):
            self.selected_suggestion_index = index

    def apply_optimization(self) -> None:
        """Handler for Apply Optimization button."""
        # To be implemented with event/callback system
        pass

    def analyze_system(self) -> None:
        """Handler for Analyze System button."""
        # To be implemented with event/callback system
        pass

    def go_back(self) -> None:
        """Handler for Back button."""
        # To be implemented with event/callback system
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for the efficiency monitor.

        Args:
            event: Pygame event

        Returns:
            bool: True if the event was handled
        """
        # Handle button events
        if self.optimize_button.handle_event(event):
            return True

        if self.analyze_button.handle_event(event):
            return True

        if self.back_button.handle_event(event):
            return True

        # Handle suggestion selection
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            char_width, char_height = 8, 16  # Approximate for typical font

            for i, _ in enumerate(self.suggestions):
                # Calculate the position of each suggestion in the list
                # Each suggestion takes up 2 lines
                y_pos = i * 2 + 3

                item_rect = pygame.Rect(
                    self.suggestions_box.x + char_width,
                    self.suggestions_box.y + y_pos * char_height,
                    self.suggestions_box.width * char_width - 2 * char_width,
                    char_height * 2,
                )

                if item_rect.collidepoint(mouse_pos):
                    self.select_suggestion(i)
                    return True

        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Draw the efficiency monitor.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Draw panel background
        self.panel.draw(surface, font)

        # Draw efficiency table
        if self.converters and self.efficiency_factors:
            headers = ["Converter", "Type", "Base", "Current", "Max"]

            rows = []
            for converter in self.converters:
                # Calculate current and maximum possible efficiency
                current = converter.get_overall_efficiency()
                max_possible = current * 1.5  # Placeholder for max possible

                rows.append(
                    [
                        converter.name,
                        converter.type.value.capitalize(),
                        f"{converter.base_efficiency:.2f}",
                        f"{current:.2f}",
                        f"{max_possible:.2f}",
                    ]
                )

            draw_ascii_table(
                surface,
                self.efficiency_table_x,
                self.efficiency_table_y,
                headers,
                rows,
                font=font,
                border_style="single",
            )

        # Update and draw suggestions box
        self.update_suggestions_box()
        self.suggestions_box.draw(surface, font)

        # Draw buttons
        self.optimize_button.draw(surface, font)
        self.analyze_button.draw(surface, font)
        self.back_button.draw(surface, font)

class ConverterInterface:
    """Main interface for the converter management system."""

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the converter interface.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Interface state
        self.current_view = (
            "dashboard"  # "dashboard", "details", "chains", "efficiency"
        )
        self.last_update = time.time()
        self.update_interval = 1.0  # seconds

        # Create event system
        self.event_system = EventSystem()
        self._setup_event_handlers()

        # Create component interfaces
        interface_width = min(screen_width - 40, 800)
        interface_height = min(screen_height - 40, 600)
        x = (screen_width - interface_width) // 2
        y = (screen_height - interface_height) // 2

        self.dashboard = ConverterDashboard(x, y, interface_width, interface_height)
        self.details_view = ConverterDetailsView(
            x, y, interface_width, interface_height
        )
        self.chain_interface = ChainManagementInterface(
            x, y, interface_width, interface_height
        )
        self.efficiency_monitor = EfficiencyMonitor(
            x, y, interface_width, interface_height
        )

        # Data
        self.converters: List[Converter] = []
        self.recipes: List[Recipe] = []
        self.chains: List[ProductionChain] = []
        self.efficiency_factors: List[EfficiencyFactor] = []
        self.optimization_suggestions: List[OptimizationSuggestion] = []

        # Setup view transitions
        self._setup_callbacks()

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for metrics and notifications."""
        # Converter events
        self.event_system.subscribe(
            EventType.CONVERTER_SELECTED, self._handle_converter_selected
        )
        self.event_system.subscribe(
            EventType.PROCESS_STARTED, self._handle_process_started
        )
        self.event_system.subscribe(
            EventType.PROCESS_COMPLETED, self._handle_process_completed
        )

        # Chain events
        self.event_system.subscribe(EventType.CHAIN_CREATED, self._handle_chain_created)
        self.event_system.subscribe(
            EventType.CHAIN_MODIFIED, self._handle_chain_modified
        )
        self.event_system.subscribe(EventType.CHAIN_DELETED, self._handle_chain_deleted)

        # Efficiency events
        self.event_system.subscribe(
            EventType.EFFICIENCY_UPDATED, self._handle_efficiency_updated
        )

        # Metric handlers
        self.event_system.subscribe_metric(
            MetricType.THROUGHPUT, self._handle_throughput_update
        )
        self.event_system.subscribe_metric(
            MetricType.ENERGY_USAGE, self._handle_energy_update
        )
        self.event_system.subscribe_metric(
            MetricType.UTILIZATION, self._handle_utilization_update
        )

    def _handle_converter_selected(self, event: EventData) -> None:
        """Handle converter selection events."""
        if converter_id := event.data.get("converter_id"):
            self._show_details_view()
            self.details_view.set_converter(
                next(c for c in self.converters if c.id == converter_id)
            )

    def _handle_process_started(self, event: EventData) -> None:
        """Handle process start events."""
        self._update_converter_metrics_from_event(event)

    def _handle_process_completed(self, event: EventData) -> None:
        """Handle process completion events."""
        self._update_converter_metrics_from_event(event)

    def _update_converter_metrics_from_event(self, event):
        converter_id = event.data.get("converter_id")
        process_id = event.data.get("process_id")
        if converter_id and process_id:
            self._update_converter_metrics(converter_id)

    def _handle_chain_created(self, event: EventData) -> None:
        """Handle chain creation events."""
        # Update metrics regardless of chain_id since we update all chains
        self._update_chain_metrics()

    def _handle_chain_modified(self, event: EventData) -> None:
        """Handle chain modification events."""
        # Update metrics regardless of chain_id since we update all chains
        self._update_chain_metrics()

    def _handle_chain_deleted(self, event: EventData) -> None:
        """Handle chain deletion events."""
        # Update metrics regardless of chain_id since we update all chains
        self._update_chain_metrics()

    def _handle_efficiency_updated(self, event: EventData) -> None:
        """Handle efficiency update events."""
        if converter_id := event.data.get("converter_id"):
            self._update_efficiency_metrics(converter_id)

    def _handle_throughput_update(self, metric: MetricData) -> None:
        """Handle throughput metric updates."""
        if self.current_view == "dashboard":
            self.dashboard.update_status_box()

    def _handle_energy_update(self, metric: MetricData) -> None:
        """Handle energy usage metric updates."""
        if self.current_view == "details":
            self.details_view.update_info_box()

    def _handle_utilization_update(self, metric: MetricData) -> None:
        """Handle utilization metric updates."""
        if self.current_view == "efficiency":
            self.efficiency_monitor.update_suggestions_box()

    def _update_converter_metrics(self, converter_id: str) -> None:
        """Update metrics for a specific converter."""
        converter = next((c for c in self.converters if c.id == converter_id), None)
        if not converter:
            return

        # Update throughput
        throughput = sum(p.output_rate for p in converter.active_processes)
        self.event_system.update_metric(
            MetricType.THROUGHPUT, throughput, "units/min", f"converter_{converter_id}"
        )

        # Update energy usage
        energy_usage = sum(p.energy_consumption for p in converter.active_processes)
        self.event_system.update_metric(
            MetricType.ENERGY_USAGE, energy_usage, "kW", f"converter_{converter_id}"
        )

        # Update utilization
        utilization = len(converter.active_processes) / converter.max_processes
        self.event_system.update_metric(
            MetricType.UTILIZATION, utilization, "%", f"converter_{converter_id}"
        )

    def _update_chain_metrics(self) -> None:
        """Update metrics for all production chains."""
        for chain in self.chains:
            total_throughput = 0
            total_energy = 0

            for step in chain.steps:
                if step.converter and step.recipe:
                    total_throughput += step.recipe.output_rate
                    total_energy += step.recipe.energy_cost

            self.event_system.update_metric(
                MetricType.THROUGHPUT,
                total_throughput,
                "units/min",
                f"chain_{chain.id}",
            )

            self.event_system.update_metric(
                MetricType.ENERGY_USAGE, total_energy, "kW", f"chain_{chain.id}"
            )

    def _update_efficiency_metrics(self, converter_id: str) -> None:
        """Update efficiency metrics for a specific converter."""
        if converter := next(
            (c for c in self.converters if c.id == converter_id), None
        ):
            self.event_system.update_metric(
                MetricType.EFFICIENCY,
                converter.base_efficiency,
                "%",
                f"converter_{converter_id}",
            )
        else:
            return

    def _setup_callbacks(self) -> None:
        """Set up callbacks for view transitions."""
        # Dashboard callbacks
        self.dashboard.view_details = self._show_details_view
        self.dashboard.manage_chains = self._show_chain_interface
        self.dashboard.monitor_efficiency = self._show_efficiency_monitor

        # Details view callbacks
        self.details_view.go_back = self._show_dashboard

        # Chain interface callbacks
        self.chain_interface.go_back = self._show_dashboard

        # Efficiency monitor callbacks
        self.efficiency_monitor.go_back = self._show_dashboard

    def set_data(
        self,
        converters: List[Converter],
        recipes: List[Recipe],
        chains: List[ProductionChain],
        efficiency_factors: Optional[List[EfficiencyFactor]] = None,
        optimization_suggestions: Optional[List[OptimizationSuggestion]] = None,
    ) -> None:
        """
        Set data for the interface.

        Args:
            converters: List of converters
            recipes: List of recipes
            chains: List of production chains
            efficiency_factors: Optional list of efficiency factors
            optimization_suggestions: Optional list of optimization suggestions
        """
        self.converters = converters
        self.recipes = recipes
        self.chains = chains
        self.efficiency_factors = efficiency_factors or []
        self.optimization_suggestions = optimization_suggestions or []

        # Update components with data
        self.dashboard.set_converters(converters)
        self.chain_interface.set_data(chains, converters, recipes)
        self.efficiency_monitor.set_data(
            converters, self.efficiency_factors, self.optimization_suggestions
        )

    def _show_dashboard(self) -> None:
        """Show the main dashboard view."""
        self.current_view = "dashboard"

    def _show_details_view(self) -> None:
        """Show the converter details view."""
        if self.dashboard.selected_converter_id:
            if selected_converter := next(
                (
                    c
                    for c in self.converters
                    if c.id == self.dashboard.selected_converter_id
                ),
                None,
            ):
                self.details_view.set_converter(selected_converter)
                self.current_view = "details"

    def _show_chain_interface(self) -> None:
        """Show the chain management interface."""
        self.current_view = "chains"

    def _show_efficiency_monitor(self) -> None:
        """Show the efficiency monitor."""
        self.current_view = "efficiency"

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for the converter interface.

        Args:
            event: Pygame event

        Returns:
            bool: True if the event was handled
        """
        # Delegate to the current active view
        if self.current_view == "dashboard":
            return self.dashboard.handle_event(event)
        elif self.current_view == "details":
            return self.details_view.handle_event(event)
        elif self.current_view == "chains":
            return self.chain_interface.handle_event(event)
        elif self.current_view == "efficiency":
            return self.efficiency_monitor.handle_event(event)

        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Draw the converter interface.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self._update_metrics()
            self.last_update = current_time

        # Draw the current active view
        if self.current_view == "dashboard":
            self.dashboard.draw(surface, font)
        elif self.current_view == "details":
            self.details_view.draw(surface, font)
        elif self.current_view == "chains":
            self.chain_interface.draw(surface, font)
        elif self.current_view == "efficiency":
            self.efficiency_monitor.draw(surface, font)

        # Draw common elements
        draw_text(
            surface,
            f"Converter Management - {self.current_view.capitalize()}",
            10,
            10,
            font=font,
            color=COLOR_TEXT,
        )

    def _update_metrics(self) -> None:
        """Update all metrics periodically."""
        # Update converter metrics
        for converter in self.converters:
            self._update_converter_metrics(converter.id)

        # Update chain metrics
        self._update_chain_metrics()

        # Update efficiency metrics
        for converter in self.converters:
            self._update_efficiency_metrics(converter.id)
