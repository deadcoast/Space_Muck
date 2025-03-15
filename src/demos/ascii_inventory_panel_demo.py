"""
Demo for the ASCIIInventoryPanel component.

This demo shows the ASCIIInventoryPanel component in action, with simulated inventory data
to demonstrate the component's sorting, filtering, and display capabilities.
"""

# Standard library imports
import os
import random
import sys
import time

import pygame

# Local application imports
from ui.ui_base.ascii_base import UIStyle
from ui.ui_element.ascii_inventory_panel import ASCIIInventoryPanel

# Third-party library imports


# Add the src directory to the path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    """Run the ASCIIInventoryPanel demo."""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ASCIIInventoryPanel Demo")

    # Create a font
    font = pygame.font.SysFont("monospace", 16)

    # Create the inventory panel
    inventory_panel = ASCIIInventoryPanel(
        pygame.Rect(50, 50, 700, 450), "Ship Inventory", UIStyle.MECHANICAL
    )

    # Create sample inventory data
    inventory = {
        "iron_ore": 42,
        "copper_ore": 28,
        "titanium_ore": 15,
        "hydrogen_gas": 63,
        "oxygen_gas": 87,
        "carbon_material": 34,
        "silicon_material": 19,
        "laser_weapon_mk1": 2,
        "shield_generator_basic": 1,
        "engine_booster": 1,
        "repair_kit": 5,
        "medical_supplies": 8,
        "rare_artifact_alpha": 1,
        "rare_crystal_fragment": 3,
        "quest_item_distress_beacon": 1,
        "quest_item_ancient_data_core": 1,
        "fuel_cell": 12,
        "power_converter": 3,
        "navigation_module": 1,
        "scanner_upgrade": 2,
    }

    # Create sample item details
    item_details = {
        "iron_ore": {
            "type": "resource",
            "description": "Common metal ore used in basic construction",
        },
        "copper_ore": {
            "type": "resource",
            "description": "Conductive metal ore used in electronics",
        },
        "titanium_ore": {
            "type": "resource",
            "description": "Strong, lightweight metal ore for advanced construction",
        },
        "hydrogen_gas": {
            "type": "resource",
            "description": "Common fuel source for engines and reactors",
        },
        "oxygen_gas": {
            "type": "resource",
            "description": "Essential for life support systems",
        },
        "carbon_material": {
            "type": "resource",
            "description": "Versatile material for various applications",
        },
        "silicon_material": {
            "type": "resource",
            "description": "Key component in computer systems",
        },
        "laser_weapon_mk1": {
            "type": "equipment",
            "description": "Basic laser weapon system",
        },
        "shield_generator_basic": {
            "type": "equipment",
            "description": "Standard shield generator",
        },
        "engine_booster": {
            "type": "equipment",
            "description": "Increases ship speed and maneuverability",
        },
        "repair_kit": {"type": "misc", "description": "Used to repair ship damage"},
        "medical_supplies": {
            "type": "misc",
            "description": "Basic medical supplies for the crew",
        },
        "rare_artifact_alpha": {
            "type": "rare",
            "description": "Mysterious alien artifact of unknown origin",
        },
        "rare_crystal_fragment": {
            "type": "rare",
            "description": "Fragment of a rare energy crystal",
        },
        "quest_item_distress_beacon": {
            "type": "quest",
            "description": "Distress beacon from a stranded ship",
        },
        "quest_item_ancient_data_core": {
            "type": "quest",
            "description": "Ancient data core containing valuable information",
        },
        "fuel_cell": {
            "type": "resource",
            "description": "Standardized fuel cell for various systems",
        },
        "power_converter": {
            "type": "equipment",
            "description": "Converts power between different systems",
        },
        "navigation_module": {
            "type": "equipment",
            "description": "Advanced navigation system",
        },
        "scanner_upgrade": {
            "type": "equipment",
            "description": "Enhances scanner range and accuracy",
        },
    }

    # Update the inventory panel with initial data
    inventory_panel.update_inventory(inventory, 200)  # 200 is the cargo capacity
    inventory_panel.update_item_details(item_details)

    # Main loop
    running = True
    clock = pygame.time.Clock()
    last_update = time.time()

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    inventory_panel.handle_input(event.key)
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                inventory_panel.handle_mouse_event(event.type, pygame.mouse.get_pos())

        # Simulate inventory changes every 5 seconds
        current_time = time.time()
        if current_time - last_update >= 5.0:
            # Randomly modify some inventory quantities
            for item_id in random.sample(list(inventory.keys()), 3):
                change = random.randint(-5, 5)
                inventory[item_id] = max(0, inventory[item_id] + change)

                # Remove item if quantity is 0
                if inventory[item_id] == 0:
                    del inventory[item_id]

            # Occasionally add a new item
            if random.random() < 0.3:
                new_items = [
                    "platinum_ore",
                    "gold_ore",
                    "uranium_ore",
                    "helium_gas",
                    "nitrogen_gas",
                    "laser_weapon_mk2",
                    "shield_generator_advanced",
                    "rare_artifact_beta",
                    "quest_item_encrypted_message",
                ]

                new_item = random.choice(new_items)
                if new_item not in inventory:
                    inventory[new_item] = random.randint(1, 10)

                    # Add item details if needed
                    if new_item not in item_details:
                        item_type = "resource"
                        if (
                            "weapon" in new_item
                            or "shield" in new_item
                            or "generator" in new_item
                        ):
                            item_type = "equipment"
                        elif "rare" in new_item:
                            item_type = "rare"
                        elif "quest" in new_item:
                            item_type = "quest"

                        item_details[new_item] = {
                            "type": item_type,
                            "description": f"New {item_type}: {new_item.replace('_', ' ').title()}",
                        }

            # Update the inventory panel
            inventory_panel.update_inventory(inventory)
            inventory_panel.update_item_details(item_details)

            last_update = current_time

        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the inventory panel
        inventory_panel.draw(screen, font)

        # Draw instructions
        instructions = [
            "ASCIIInventoryPanel Demo",
            "Press ESC to exit",
            "",
            "Controls:",
            "- Click on buttons to sort, filter, and navigate pages",
            "- Use arrow keys to select items",
            "- Use Page Up/Down to change pages",
            "",
            "Inventory updates every 5 seconds with random changes",
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
