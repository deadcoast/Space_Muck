"""
Demo script showcasing Space Muck's ASCII UI components.
"""

# Standard library imports
import math
import os
import random
import sys
import time

# Third-party library imports

# Local application imports
from config import COLOR_BG
from ui.ascii_ui import (
from ui.game_screen import ASCIIGameScreen
from ui.minimap_panel import ASCIIMinimapPanel
import pygame

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    ASCIIBox,
    ASCIIPanel,
    ASCIIProgressBar,
    ASCIIButton,
    ASCIIMetricsPanel,
    ASCIIChainVisualizer,
    ASCIIRecipePanel,
    ASCIIEfficiencyMonitor,
    UIStyle,
)

class ASCIIUIDemo:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Space Muck ASCII UI Demo")
        self.font = pygame.font.Font(None, 32)
        self.clock = pygame.time.Clock()
        self.running = True

        # State management
        self.selected_recipe_idx = None
        self.selected_converter_idx = None
        self.show_help = False
        self.paused = False
        self.high_contrast = False

        # Animation state
        self.animation_time = 0.0
        self.pulse_state = 0.0

        # Demo components
        self.setup_components()

        # Demo data
        self.setup_demo_data()

        # Help text
        self.help_text = [
            "Keyboard Controls:",
            "H - Toggle Help",
            "P - Pause/Resume",
            "C - Toggle High Contrast",
            "↑/↓ - Select Recipe",
            "←/→ - Select Converter",
            "Space - Start Selected Recipe",
            "R - Reset Demo",
            "Q - Quit",
        ]

    def setup_components(self):
        # Define panel styles for different sections
        styles = {
            "basic": UIStyle.MECHANICAL,
            "quantum": UIStyle.QUANTUM,
            "symbiotic": UIStyle.SYMBIOTIC,
        }

        # Create the game screen as the main container
        self.game_screen = ASCIIGameScreen(
            pygame.Rect(0, 0, 1200, 800),
            title="Space Muck Game",
            style=styles["mechanical"],
        )

        # Create a minimap panel
        self.minimap = ASCIIMinimapPanel(
            pygame.Rect(900, 50, 250, 250),
            title="Navigation Map",
            style=styles["symbiotic"],
        )

        # Add the minimap to the game screen
        self.game_screen.add_panel(self.minimap)

        # Basic components demo (left side)
        self.basic_panel = ASCIIPanel(
            pygame.Rect(20, 20, 350, 760),
            title="Basic Components",
            style=styles["basic"],
        )

        # Basic interactive elements
        self.demo_box = ASCIIBox(40, 60, 30, 8, "Demo Box")
        self.demo_progress = ASCIIProgressBar(40, 200, 25)
        self.demo_button = ASCIIButton(
            40, 250, "Click Me!", self.on_button_click, style=styles["basic"]
        )

        # Advanced components demo (right side)
        self.advanced_panel = ASCIIPanel(
            pygame.Rect(390, 20, 790, 760),
            title="Advanced Components",
            style=styles["quantum"],
        )

        # Metrics panel with quantum style
        self.metrics_panel = ASCIIMetricsPanel(
            pygame.Rect(410, 60, 350, 200),
            "Production Metrics",
            style=styles["quantum"],
            converter_type="QUANTUM",
        )

        # Chain visualizer with symbiotic style
        self.chain_viz = ASCIIChainVisualizer(
            pygame.Rect(410, 280, 350, 200),
            "Chain Visualization",
            style=styles["symbiotic"],
        )

        # Recipe panel with mechanical style
        self.recipe_panel = ASCIIRecipePanel(
            pygame.Rect(780, 60, 380, 200), "Available Recipes", style=styles["basic"]
        )

        # Efficiency monitor with quantum style
        self.efficiency_monitor = ASCIIEfficiencyMonitor(
            pygame.Rect(780, 280, 380, 200),
            "Efficiency Monitor",
            style=styles["quantum"],
        )

    def setup_demo_data(self):
        # Demo text for box
        self.demo_box.add_text(2, 2, "Hello Space Muck!")
        self.demo_box.add_text(2, 4, "ASCII UI Demo")

        # Demo metrics
        self.metrics = {
            "throughput": 45.5,
            "energy_usage": 120.0,
            "queue_size": 5,
            "utilization": 0.85,
            "uptime": 12.5,
            "efficiency": 0.92,
        }

        # Demo chain with enhanced data
        self.chain_data = {
            "converters": [
                {
                    "id": "1",
                    "name": "Quantum Extractor",
                    "type": "QUANTUM",
                    "efficiency": 0.92,
                    "status": "active",
                    "output_type": "energy",
                    "uptime": 3600,
                    "queue_size": 3,
                },
                {
                    "id": "2",
                    "name": "Matter Synthesizer",
                    "type": "SYMBIOTIC",
                    "efficiency": 0.88,
                    "status": "active",
                    "output_type": "matter",
                    "uptime": 1800,
                    "queue_size": 2,
                },
                {
                    "id": "3",
                    "name": "Data Processor",
                    "type": "MECHANICAL",
                    "efficiency": 0.95,
                    "status": "active",
                    "output_type": "data",
                    "uptime": 7200,
                    "queue_size": 5,
                },
                {
                    "id": "4",
                    "name": "Fluid Manipulator",
                    "type": "SYMBIOTIC",
                    "efficiency": 0.91,
                    "status": "active",
                    "output_type": "fluid",
                    "uptime": 900,
                    "queue_size": 1,
                },
            ],
            "connections": [(0, 1), (1, 2), (2, 3)],
            "flow_rates": {(0, 1): 1500.5, (1, 2): 850.2, (2, 3): 425.8},
        }

        # Demo recipes with enhanced details
        self.recipes = [
            {
                "name": "Quantum Stabilizer",
                "inputs": ["Energy Matrix", "Quantum Dust"],
                "outputs": ["Stable Quantum Core"],
                "efficiency": 0.94,
                "type": "QUANTUM",
                "duration": 60,
                "energy_cost": 250,
            },
            {
                "name": "Bio-Neural Network",
                "inputs": ["Synaptic Fluid", "Neural Matrix"],
                "outputs": ["Bio-Processor"],
                "efficiency": 0.89,
                "type": "SYMBIOTIC",
                "duration": 45,
                "energy_cost": 180,
            },
            {
                "name": "Data Crystal",
                "inputs": ["Raw Data", "Crystal Matrix"],
                "outputs": ["Encoded Crystal"],
                "efficiency": 0.92,
                "type": "MECHANICAL",
                "duration": 30,
                "energy_cost": 120,
            },
        ]

    def on_button_click(self):
        print("Button clicked!")
        self.demo_box.start_animation()

    def update(self):
        if self.paused:
            return

        current_time = time.time()
        dt = current_time - self.animation_time
        self.animation_time = current_time

        # Update pulse effect for selected items
        self.pulse_state = (self.pulse_state + dt * 3) % (2 * 3.14159)
        pulse_alpha = int(128 + 127 * math.sin(self.pulse_state))

        # Update progress bar with pulse effect
        progress = (pygame.time.get_ticks() % 3000) / 3000
        self.demo_progress.set_progress(progress)

        # Update metrics with realistic fluctuations
        self.metrics.update(
            {
                "throughput": 45.5 + random.uniform(-2, 2),
                "energy_usage": 120.0 + random.uniform(-5, 5),
                "queue_size": max(
                    0, self.metrics["queue_size"] + random.randint(-1, 1)
                ),
                "utilization": min(
                    1.0,
                    max(0.0, self.metrics["utilization"] + random.uniform(-0.02, 0.02)),
                ),
                "uptime": self.metrics["uptime"] + dt,  # Add elapsed time
                "efficiency": min(
                    1.0,
                    max(0.0, self.metrics["efficiency"] + random.uniform(-0.01, 0.01)),
                ),
            }
        )
        self.metrics_panel.update_metrics(self.metrics)

        # Update chain data with dynamic flow rates and efficiencies
        for i, conv in enumerate(self.chain_data["converters"]):
            # Apply visual effects and handle active converters
            conv["highlight_alpha"] = (
                pulse_alpha if i == self.selected_converter_idx else 0
            )

            if conv["status"] == "active":
                # Update active converter stats
                conv["efficiency"] = min(
                    1.0, max(0.0, conv["efficiency"] + random.uniform(-0.01, 0.01))
                )
                conv["uptime"] += dt

                # Process queue and update flow rates
                if conv["queue_size"] > 0 and random.random() < 0.1:
                    conv["queue_size"] -= 1
                    # Boost connected flow rates
                    for connection in self.chain_data["connections"]:
                        if connection[0] == i:
                            self.chain_data["flow_rates"][connection] *= 1.2

                # Check recipe completion
                if (
                    "start_time" in conv
                    and time.time() - conv["start_time"] > conv["recipe_duration"]
                ):
                    conv["status"] = "idle"
                    conv["current_recipe"] = None
                    self.metrics["utilization"] = max(
                        0.0, self.metrics["utilization"] - 0.1
                    )

        # Update flow rates with realistic fluctuations
        for connection in self.chain_data["connections"]:
            if current_rate := self.chain_data["flow_rates"].get(connection, 0):
                # Smoother rate changes
                target_rate = current_rate * (1 + random.uniform(-0.05, 0.05))
                self.chain_data["flow_rates"][connection] = current_rate + (
                    target_rate - current_rate
                ) * min(1.0, dt * 2)

        # Update efficiency monitor with trend analysis
        if active_converters := [
            conv for conv in self.chain_data["converters"] if conv["status"] == "active"
        ]:
            self.efficiency_monitor.update_efficiency(
                sum(conv["efficiency"] for conv in active_converters)
                / len(active_converters)
            )

        # Update recipe panel animations
        if self.selected_recipe_idx is not None:
            self.recipe_panel.highlight_alpha = pulse_alpha

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keyboard_event(event)

            # Only handle component events if not paused
            if not self.paused:
                self.demo_button.handle_event(event)
                self.recipe_panel.handle_event(event)

    def handle_keyboard_event(self, event):
        if event.key == pygame.K_h:
            self.show_help = not self.show_help
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_c:
            self.high_contrast = not self.high_contrast
        elif event.key == pygame.K_q:
            self.running = False
        elif event.key == pygame.K_r:
            self.setup_demo_data()  # Reset demo data

        # Handle recipe selection
        elif event.key in (pygame.K_UP, pygame.K_DOWN) and not self.paused:
            direction = -1 if event.key == pygame.K_UP else 1
            if self.selected_recipe_idx is None:
                self.selected_recipe_idx = 0
            else:
                self.selected_recipe_idx = (self.selected_recipe_idx + direction) % len(
                    self.recipes
                )

        # Handle converter selection
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and not self.paused:
            direction = -1 if event.key == pygame.K_LEFT else 1
            if self.selected_converter_idx is None:
                self.selected_converter_idx = 0
            else:
                self.selected_converter_idx = (
                    self.selected_converter_idx + direction
                ) % len(self.chain_data["converters"])

        # Start selected recipe
        elif event.key == pygame.K_SPACE and not self.paused:
            if (
                self.selected_recipe_idx is not None
                and self.selected_converter_idx is not None
            ):
                self.start_recipe(self.selected_recipe_idx, self.selected_converter_idx)

    def start_recipe(self, recipe_idx, converter_idx):
        recipe = self.recipes[recipe_idx]
        converter = self.chain_data["converters"][converter_idx]

        # Check recipe compatibility
        if recipe["type"].lower() != converter["type"].lower():
            # Visual feedback for incompatible recipe
            converter["error_flash"] = 1.0
            return

        # Update converter state
        converter["status"] = "active"
        converter["current_recipe"] = recipe["name"]
        converter["efficiency"] = recipe["efficiency"] * random.uniform(
            0.95, 1.05
        )  # Add some variance
        converter["start_time"] = time.time()
        converter["recipe_duration"] = recipe["duration"]
        converter["energy_cost"] = recipe["energy_cost"]

        # Update metrics
        self.metrics["queue_size"] += len(recipe["inputs"])
        self.metrics["utilization"] = min(1.0, self.metrics["utilization"] + 0.1)
        self.metrics["energy_usage"] += recipe["energy_cost"]

        # Visual feedback
        converter["success_flash"] = 1.0

        # Update connected converters
        for connection in self.chain_data["connections"]:
            if connection[0] == converter_idx:
                # Increase flow rate for outputs
                self.chain_data["flow_rates"][connection] = (
                    self.chain_data["flow_rates"].get(connection, 0)
                    + recipe["efficiency"] * 100
                )

    def draw_tooltip(self, lines: list, x: int, y: int, align: str = "left") -> int:
        """Draw a tooltip with the given lines at the specified position.
        Returns the final y position after drawing all lines."""
        for line in lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            if align == "left":
                text_rect = text_surface.get_rect(left=x, top=y)
            else:  # right align
                text_rect = text_surface.get_rect(right=x, top=y)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), text_rect.inflate(20, 10))
            self.screen.blit(text_surface, text_rect)
            y += 30
        return y

    def get_converter_tooltip_lines(self, conv: dict) -> list:
        """Get the tooltip lines for a converter."""
        lines = [
            f"Status: {conv['status'].title()}",
            f"Efficiency: {conv['efficiency']:.1%}",
            f"Uptime: {conv['uptime']/3600:.1f}h",
            f"Queue: {conv['queue_size']} items",
        ]
        if conv.get("current_recipe"):
            lines.extend(
                [
                    f"Recipe: {conv['current_recipe']}",
                    f"Time Left: {conv['recipe_duration'] - (time.time() - conv['start_time']):.1f}s",
                ]
            )
        return lines

    def get_recipe_tooltip_lines(self, recipe: dict) -> list:
        """Get the tooltip lines for a recipe."""
        return [
            f"Type: {recipe['type']}",
            f"Duration: {recipe['duration']}s",
            f"Energy Cost: {recipe['energy_cost']}kW",
            "Inputs: " + ", ".join(recipe["inputs"]),
            "Output: " + ", ".join(recipe["outputs"]),
        ]

    def draw(self):
        # Apply high contrast if enabled
        bg_color = (0, 0, 0) if self.high_contrast else COLOR_BG
        self.screen.fill(bg_color)

        # Draw basic components
        self.basic_panel.draw(self.screen, self.font)
        self.demo_box.draw(self.screen, self.font)
        self.demo_progress.draw(self.screen, self.font)
        self.demo_button.draw(self.screen, self.font)

        # Draw advanced components
        self.advanced_panel.draw(self.screen, self.font)
        self.metrics_panel.draw(self.screen, self.font)

        # Update and draw chain visualizer with selection highlight
        converters = self.chain_data["converters"]
        if self.selected_converter_idx is not None:
            converters[self.selected_converter_idx]["selected"] = True
            # Draw tooltip for selected converter
            conv = converters[self.selected_converter_idx]
            self.draw_tooltip(
                self.get_converter_tooltip_lines(conv),
                self.screen.get_width() - 20,
                60,
                "right",
            )

        self.chain_viz.set_chain(
            converters, self.chain_data["connections"], self.chain_data["flow_rates"]
        )
        self.chain_viz.draw(self.screen, self.font)

        # Update and draw recipe panel with selection
        if self.selected_recipe_idx is not None:
            self.recipe_panel.selected_idx = self.selected_recipe_idx
            # Draw recipe details tooltip
            recipe = self.recipes[self.selected_recipe_idx]
            self.draw_tooltip(
                self.get_recipe_tooltip_lines(recipe),
                20,
                self.screen.get_height() - 180,
            )

        self.recipe_panel.set_recipes(self.recipes)
        self.recipe_panel.draw(self.screen, self.font)

        self.efficiency_monitor.draw(self.screen, self.font)

        # Draw help overlay if enabled
        if self.show_help:
            self.draw_help_overlay()

        if status_text := [
            text
            for condition, text in [
                (self.paused, "PAUSED - Press P to Resume"),
                (self.high_contrast, "High Contrast Mode - Press C to Toggle"),
            ]
            if condition
        ]:
            self.draw_tooltip(status_text, self.screen.get_width() // 2, 30, "right")

        pygame.display.flip()

    def draw_help_overlay(self):
        # Draw semi-transparent overlay
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Draw help text
        y = 100
        for line in self.help_text:
            text_surface = self.font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.screen.get_width() // 2, y))
            self.screen.blit(text_surface, text_rect)
            y += 40

    def run(self):
        try:
            while self.running:
                self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(60)
        except KeyboardInterrupt:
            self.running = False
        finally:
            pygame.quit()

if __name__ == "__main__":
    demo = ASCIIUIDemo()
    demo.run()
