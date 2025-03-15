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

import pygame

# Local application imports
from ui.ui_base.ascii_base import UIStyle
from ui.ui_element.ascii_ship_status_panel import ASCIIShipStatusPanel

# Third-party library imports


# Add the src directory to the path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    """Run the ASCIIShipStatusPanel demo."""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ASCIIShipStatusPanel Demo")

    # Create a font
    font = pygame.font.SysFont("monospace", 16)

    # Create the ship status panel
    ship_panel = ASCIIShipStatusPanel(
        pygame.Rect(50, 50, 700, 450), "Starship Status", UIStyle.MECHANICAL
    )

    # Set initial ship data
    ship_panel.update_ship_info(2, "S.S. Muckraker")
    ship_panel.update_hull_shield(85, 100, 70, 100, 1.5)
    ship_panel.update_combat_stats(15, 1.2, 120, 7.5, 12.0, 8.0)
    ship_panel.update_power(70, 100, 80, 65)

    # Update system statuses
    ship_panel.update_system_status(
        "engines", {"status": "online", "efficiency": 95, "power_usage": 15}
    )

    ship_panel.update_system_status(
        "weapons", {"status": "online", "efficiency": 90, "power_usage": 20}
    )

    ship_panel.update_system_status(
        "shields", {"status": "online", "efficiency": 85, "power_usage": 25}
    )

    ship_panel.update_system_status(
        "sensors", {"status": "online", "efficiency": 100, "power_usage": 5}
    )

    ship_panel.update_system_status(
        "life_support", {"status": "online", "efficiency": 100, "power_usage": 10}
    )

    # Main loop
    running = True
    clock = pygame.time.Clock()
    last_update = time.time()

    # Simulation variables
    system_damage_timer = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if (
                event.type != pygame.QUIT
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
                or event.type == pygame.QUIT
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Simulate taking damage on spacebar press
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
                if random.random() < 0.3:
                    systems = list(ship_panel.systems.keys())
                    damaged_system = random.choice(systems)
                    current_efficiency = ship_panel.systems[damaged_system][
                        "efficiency"
                    ]
                    new_efficiency = max(
                        30, current_efficiency - random.randint(10, 30)
                    )

                    status = "damaged" if new_efficiency < 70 else "online"
                    if new_efficiency < 40:
                        status = "offline"

                    ship_panel.update_system_status(
                        damaged_system,
                        {"status": status, "efficiency": new_efficiency},
                    )

        # Update simulation every second
        current_time = time.time()
        if (
            current_time - last_update >= 0.1
        ):  # Update 10 times per second for smoother animation
            delta_time = current_time - last_update

            # Simulate shield recharge if not at max
            current_shield = ship_panel.current_shield
            max_shield = ship_panel.max_shield
            recharge_rate = ship_panel.shield_recharge_rate

            if current_shield < max_shield:
                # Only recharge if at least one system is online
                can_recharge = any(
                    system["status"] == "online"
                    for system in ship_panel.systems.values()
                )

                if can_recharge:
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

            # Every 5 seconds, simulate a system efficiency change
            system_damage_timer += delta_time
            if system_damage_timer >= 5.0:
                system_damage_timer = 0

                # Randomly select a system to update
                systems = list(ship_panel.systems.keys())
                system_to_update = random.choice(systems)
                current_system = ship_panel.systems[system_to_update]

                # Determine if efficiency should increase or decrease
                current_efficiency = current_system["efficiency"]
                if current_efficiency < 50:
                    # More likely to increase if efficiency is low
                    change_direction = 1 if random.random() < 0.8 else -1
                elif current_efficiency > 90:
                    # More likely to decrease if efficiency is high
                    change_direction = -1 if random.random() < 0.8 else 1
                else:
                    # Equal chance either way
                    change_direction = 1 if random.random() < 0.5 else -1

                # Calculate new efficiency
                efficiency_change = random.randint(1, 10) * change_direction
                new_efficiency = max(
                    10, min(100, current_efficiency + efficiency_change)
                )

                # Determine status based on efficiency
                if new_efficiency < 30:
                    status = "offline"
                elif new_efficiency < 70:
                    status = "damaged"
                else:
                    status = "online"

                # Update the system
                ship_panel.update_system_status(
                    system_to_update, {"status": status, "efficiency": new_efficiency}
                )

            last_update = current_time

        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the ship status panel
        ship_panel.draw(screen, font)

        # Draw instructions
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

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
