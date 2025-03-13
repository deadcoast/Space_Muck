"""
Demo for the ASCIIResourceDisplay component.

This demo shows the ASCIIResourceDisplay component in action, with simulated resource data
that changes over time to demonstrate the component's functionality.
"""

# Standard library imports
import os
import random
import sys
import time

# Third-party library imports

# Local application imports
from systems.resource_manager import ResourceType, ResourceState
from ui.ui_element.ascii_resource_display import ASCIIResourceDisplay
import pygame

# Add the src directory to the path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def initialize_resources():
    """Initialize and return resource data."""
    return {
        ResourceType.ENERGY: {
            "amount": 75.0,
            "capacity": 100.0,
            "state": ResourceState.STABLE,
        },
        ResourceType.MATTER: {
            "amount": 50.0,
            "capacity": 100.0,
            "state": ResourceState.STABLE,
        },
        ResourceType.FLUID: {
            "amount": 25.0,
            "capacity": 100.0,
            "state": ResourceState.DEPLETING,
        },
        ResourceType.DATA: {
            "amount": 10.0,
            "capacity": 100.0,
            "state": ResourceState.CRITICAL,
        },
    }


def initialize_critical_stats():
    """Initialize and return critical stats."""
    return {
        "shield": 80.0,
        "hull": 95.0,
        "power_output": 85.0,
        "power_usage": 65.0,
        "efficiency": 76.5,
    }


def update_resource_amounts(resources):
    """Update resource amounts with random changes."""
    resources[ResourceType.ENERGY]["amount"] = max(
        0, min(100, resources[ResourceType.ENERGY]["amount"] + random.uniform(-5, 5))
    )
    resources[ResourceType.MATTER]["amount"] = max(
        0, min(100, resources[ResourceType.MATTER]["amount"] + random.uniform(-3, 3))
    )
    resources[ResourceType.FLUID]["amount"] = max(
        0, min(100, resources[ResourceType.FLUID]["amount"] + random.uniform(-2, 1))
    )
    resources[ResourceType.DATA]["amount"] = max(
        0, min(100, resources[ResourceType.DATA]["amount"] + random.uniform(-1, 2))
    )


def update_resource_states(resources):
    """Update resource states based on amounts."""
    for resource_type in resources:
        amount = resources[resource_type]["amount"]
        capacity = resources[resource_type]["capacity"]
        ratio = amount / capacity

        if ratio <= 0.1:
            resources[resource_type]["state"] = ResourceState.CRITICAL
        elif ratio <= 0.25:
            resources[resource_type]["state"] = ResourceState.DEPLETING
        elif ratio >= 0.9:
            resources[resource_type]["state"] = ResourceState.GROWING
        else:
            resources[resource_type]["state"] = ResourceState.STABLE


def update_critical_stats(critical_stats):
    """Update critical stats with random changes."""
    critical_stats["shield"] = max(
        0, min(100, critical_stats["shield"] + random.uniform(-2, 2))
    )
    critical_stats["hull"] = max(
        0, min(100, critical_stats["hull"] + random.uniform(-1, 1))
    )
    critical_stats["power_output"] = max(
        0, min(100, critical_stats["power_output"] + random.uniform(-3, 3))
    )
    critical_stats["power_usage"] = max(
        0, min(100, critical_stats["power_usage"] + random.uniform(-2, 2))
    )
    critical_stats["efficiency"] = max(
        0, min(100, critical_stats["efficiency"] + random.uniform(-1, 1))
    )


def draw_instructions(screen, font):
    """Draw instruction text on the screen."""
    instructions = [
        "ASCIIResourceDisplay Demo",
        "Press ESC to exit",
        "",
        "Resources update every second with random changes",
        "Resource states change based on resource levels:",
        "  - CRITICAL: <= 10% of capacity",
        "  - DEPLETING: <= 25% of capacity",
        "  - GROWING: >= 90% of capacity",
        "  - STABLE: otherwise",
    ]

    for i, line in enumerate(instructions):
        text_surface = font.render(line, True, (200, 200, 200))
        screen.blit(text_surface, (50, 470 + i * 20))


def handle_events():
    """Handle pygame events and return whether the game should continue running."""
    return not any(
        (
            event.type != pygame.QUIT
            and event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
            or event.type == pygame.QUIT
        )
        for event in pygame.event.get()
    )


def main():
    """Run the ASCIIResourceDisplay demo."""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ASCIIResourceDisplay Demo")

    # Create a font
    font = pygame.font.SysFont("monospace", 16)

    # Create the resource display
    resource_display = ASCIIResourceDisplay(
        pygame.Rect(50, 50, 700, 400),
        "Ship Resources",
        # You can change the style here to see different appearances
    )

    # Initialize data
    resources = initialize_resources()
    critical_stats = initialize_critical_stats()

    # Update the resource display with initial data
    resource_display.update_resources(resources)
    resource_display.update_critical_stats(critical_stats)

    # Main loop
    running = True
    clock = pygame.time.Clock()
    last_update = time.time()

    while running:
        # Handle events
        running = handle_events()

        # Update resources every second with simulated changes
        current_time = time.time()
        if current_time - last_update >= 1.0:
            update_resource_amounts(resources)
            update_resource_states(resources)
            update_critical_stats(critical_stats)

            # Update the resource display
            resource_display.update_resources(resources)
            resource_display.update_critical_stats(critical_stats)

            last_update = current_time

        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the resource display
        resource_display.draw(screen, font)

        # Draw instructions
        draw_instructions(screen, font)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
