"""
Demo for the ASCIIShipStatusPanel component.

This demo shows the ASCIIShipStatusPanel component in action, with simulated ship data
to demonstrate the component's display capabilities for ship health, energy, and system statuses.
"""

# Standard library imports
import os
import random
import sys
import time

# Add the src directory to the path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame

# Local application imports
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_element.ascii_ship_status_panel import ASCIIShipStatusPanel

# Third-party library imports


def initialize_ship_panel():
    """Create and initialize the ship status panel with default values."""
    ship_panel = ASCIIShipStatusPanel(
        pygame.Rect(50, 50, 700, 450), "Starship Status", UIStyle.MECHANICAL
    )

    # Set initial ship data
    ship_panel.update_ship_info(2, "S.S. Muckraker")
    ship_panel.update_hull_shield(85, 100, 70, 100, 1.5)
    ship_panel.update_combat_stats(15, 1.2, 120, 7.5, 12.0, 8.0)
    ship_panel.update_power(70, 100, 80, 65)

    # Initialize system statuses
    update_initial_systems(ship_panel)
    
    return ship_panel


def update_initial_systems(ship_panel):
    """Set up initial system statuses for the ship panel."""
    systems_data = {
        "engines": {"status": "online", "efficiency": 95, "power_usage": 15},
        "weapons": {"status": "online", "efficiency": 90, "power_usage": 20},
        "shields": {"status": "online", "efficiency": 85, "power_usage": 25},
        "sensors": {"status": "online", "efficiency": 100, "power_usage": 5},
        "life_support": {"status": "online", "efficiency": 100, "power_usage": 10},
    }
    
    for system_name, system_data in systems_data.items():
        ship_panel.update_system_status(system_name, system_data)


def handle_damage_event(ship_panel):
    """Simulate taking damage when spacebar is pressed."""
    current_hull = ship_panel.current_hull
    current_shield = ship_panel.current_shield

    # Damage shields first, then hull
    damage = random.randint(10, 25)
    if current_shield > 0:
        new_shield = max(0, current_shield - damage)
        absorbed = current_shield - new_shield
        hull_damage = max(0, damage - absorbed)
        new_hull = max(0, current_hull - hull_damage)
    else:
        new_shield = 0
        new_hull = max(0, current_hull - damage)

    ship_panel.update_hull_shield(
        new_hull,
        ship_panel.max_hull,
        new_shield,
        ship_panel.max_shield,
        ship_panel.shield_recharge_rate,
    )

    # Possibly damage a system
    maybe_damage_random_system(ship_panel)


def maybe_damage_random_system(ship_panel, damage_chance=0.3):
    """Randomly damage a ship system with the specified probability."""
    if random.random() < damage_chance:
        systems = list(ship_panel.systems.keys())
        damaged_system = random.choice(systems)
        current_efficiency = ship_panel.systems[damaged_system]["efficiency"]
        new_efficiency = max(30, current_efficiency - random.randint(10, 30))

        status = determine_system_status(new_efficiency)

        ship_panel.update_system_status(
            damaged_system,
            {"status": status, "efficiency": new_efficiency},
        )


def determine_system_status(efficiency):
    """Determine the status of a system based on its efficiency."""
    if efficiency < 30:
        return "offline"
    elif efficiency < 70:
        return "damaged"
    else:
        return "online"


def handle_events(ship_panel):
    """Handle pygame events and return whether the simulation should continue."""
    for event in pygame.event.get():
        # Check for quit events
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return False
        # Handle damage simulation via spacebar
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            handle_damage_event(ship_panel)
    return True


def update_shield_recharge(ship_panel, delta_time):
    """Simulate shield recharge if not at maximum capacity."""
    current_shield = ship_panel.current_shield
    max_shield = ship_panel.max_shield
    if current_shield < max_shield:
        # Only recharge if at least one system is online
        can_recharge = any(
            system["status"] == "online"
            for system in ship_panel.systems.values()
        )

        if can_recharge:
            recharge_rate = ship_panel.shield_recharge_rate

            new_shield = min(
                max_shield, current_shield + recharge_rate * delta_time
            )
            ship_panel.update_hull_shield(
                ship_panel.current_hull,
                ship_panel.max_hull,
                new_shield,
                max_shield,
                recharge_rate,
            )


def update_power_systems(ship_panel, delta_time):
    """Simulate power fluctuations and update ship power levels."""
    # Simulate power fluctuations
    power_gen = ship_panel.power_generation + random.uniform(-2, 2)
    power_usage = sum(
        system["power_usage"] for system in ship_panel.systems.values()
    )
    power_usage += random.uniform(-1, 1)  # Small random fluctuation

    current_power = ship_panel.current_power
    max_power = ship_panel.max_power

    # Update power based on generation vs usage
    power_delta = (power_gen - power_usage) * delta_time
    new_power = max(0, min(max_power, current_power + power_delta))

    ship_panel.update_power(new_power, max_power, power_gen, power_usage)


def update_random_system_efficiency(ship_panel):
    """Randomly update the efficiency of one ship system."""
    # Randomly select a system to update
    systems = list(ship_panel.systems.keys())
    system_to_update = random.choice(systems)
    current_system = ship_panel.systems[system_to_update]

    # Determine if efficiency should increase or decrease
    current_efficiency = current_system["efficiency"]
    change_direction = determine_efficiency_change_direction(current_efficiency)
    
    # Calculate new efficiency
    efficiency_change = random.randint(1, 10) * change_direction
    new_efficiency = max(10, min(100, current_efficiency + efficiency_change))

    # Determine status based on efficiency
    status = determine_system_status(new_efficiency)

    # Update the system
    ship_panel.update_system_status(
        system_to_update, {"status": status, "efficiency": new_efficiency}
    )


def determine_efficiency_change_direction(current_efficiency):
    """Determine the direction of efficiency change based on current value."""
    if current_efficiency < 50:
        # More likely to increase if efficiency is low
        return 1 if random.random() < 0.8 else -1
    elif current_efficiency > 90:
        # More likely to decrease if efficiency is high
        return -1 if random.random() < 0.8 else 1
    else:
        # Equal chance either way
        return 1 if random.random() < 0.5 else -1


def update_ship_simulation(ship_panel, delta_time, system_damage_timer):
    """Update all aspects of the ship simulation."""
    # Update shield recharge
    update_shield_recharge(ship_panel, delta_time)
    
    # Update power systems
    update_power_systems(ship_panel, delta_time)
    
    # Every 5 seconds, simulate a system efficiency change
    if system_damage_timer + delta_time >= 5.0:
        update_random_system_efficiency(ship_panel)


def draw_instructions(screen, font):
    """Draw instruction text at the bottom of the screen."""
    instructions = [
        "ASCIIShipStatusPanel Demo",
        "Press ESC to exit",
        "",
        "Controls:",
        "- Press SPACE to simulate taking damage",
        "",
        "Simulation:",
        "- Shield recharges automatically if not at maximum",
        "- System efficiencies change randomly every 5 seconds",
        "- Power generation and usage fluctuate slightly",
    ]

    for i, line in enumerate(instructions):
        text_surface = font.render(line, True, (200, 200, 200))
        screen.blit(text_surface, (50, 520 + i * 20))


def draw_demo_screen(screen, ship_panel, font):
    """Draw the complete demo screen."""
    # Clear the screen
    screen.fill((0, 0, 0))
    
    # Draw the ship status panel
    ship_panel.draw(screen, font)
    
    # Draw instructions
    draw_instructions(screen, font)
    
    # Update the display
    pygame.display.flip()


def main():
    """Run the ASCIIShipStatusPanel demo."""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ASCIIShipStatusPanel Demo")

    # Create a font
    font = pygame.font.SysFont("monospace", 16)

    # Create and initialize the ship status panel
    ship_panel = initialize_ship_panel()

    # Main loop
    running = True
    clock = pygame.time.Clock()
    last_update = time.time()

    # Simulation variables
    system_damage_timer = 0

    while running:
        # Handle events
        running = handle_events(ship_panel)

        # Update simulation every second
        current_time = time.time()
        if current_time - last_update >= 0.1:  # Update 10 times per second for smoother animation
            delta_time = current_time - last_update
            
            # Run ship systems simulation
            update_ship_simulation(ship_panel, delta_time, system_damage_timer)
            system_damage_timer = (system_damage_timer + delta_time) % 5.0
            
            last_update = current_time

        # Draw everything
        draw_demo_screen(screen, ship_panel, font)

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
