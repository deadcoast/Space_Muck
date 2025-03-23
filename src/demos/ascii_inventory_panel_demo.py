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
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_element.ascii_inventory_panel import ASCIIInventoryPanel

# Third-party library imports


# Add the src directory to the path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_sample_inventory():
    """Create sample inventory data for the demo."""
    return {
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


def create_sample_item_details():
    """Create sample item details for the demo."""
    return {
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


def get_potential_new_items():
    """Get a list of potential new items that can be added to inventory."""
    return [
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


def generate_random_item_details(_):
    """Generate random details for a new item."""
    item_types = ["resource", "equipment", "misc", "rare", "quest"]
    descriptions = [
        "A valuable material used in advanced construction",
        "A component for high-tech systems",
        "A rare find with unique properties",
        "An essential supply for space exploration",
        "A mysterious object with unknown capabilities",
    ]
    
    return {
        "type": random.choice(item_types),
        "description": random.choice(descriptions),
    }


def identify_item_type(item_name):
    """Determine the item type based on its name."""
    if any(keyword in item_name for keyword in ["weapon", "shield", "generator"]):
        return "equipment"
    elif "rare" in item_name:
        return "rare"
    elif "quest" in item_name:
        return "quest"
    return "resource"


def simulate_inventory_changes(inventory, item_details):
    """Simulate random changes to the inventory."""
    # Randomly modify some inventory quantities
    for item_id in random.sample(list(inventory.keys()), min(3, len(inventory))):
        change = random.randint(-5, 5)
        inventory[item_id] = max(0, inventory[item_id] + change)

        # Remove item if quantity is 0
        if inventory[item_id] == 0:
            del inventory[item_id]

    # Occasionally add a new item
    if random.random() < 0.3:
        new_items = get_potential_new_items()
        new_item = random.choice(new_items)
        
        if new_item not in inventory:
            inventory[new_item] = random.randint(1, 10)

            # Add item details if needed
            if new_item not in item_details:
                item_type = identify_item_type(new_item)
                item_details[new_item] = {
                    "type": item_type,
                    "description": f"New {item_type}: {new_item.replace('_', ' ').title()}",
                }
    
    return inventory, item_details


def handle_events(inventory_panel):
    """Handle pygame events for the demo."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            else:
                inventory_panel.handle_input(event.key)
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            inventory_panel.handle_mouse_event(event.type, pygame.mouse.get_pos())
    return True


def draw_instructions(screen, font):
    """Draw instruction text at the bottom of the screen."""
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


def draw_demo_screen(screen, inventory_panel, font):
    """Draw everything to the screen."""
    # Clear the screen
    screen.fill((0, 0, 0))
    
    # Draw inventory panel
    inventory_panel.draw(screen, font)
    
    # Draw instructions
    draw_instructions(screen, font)
    
    # Update the display
    pygame.display.flip()


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

    # Create sample inventory data and details
    inventory = create_sample_inventory()
    item_details = create_sample_item_details()

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
            # Update inventory with simulated changes
            inventory, item_details = simulate_inventory_changes(inventory, item_details)
            
            # Update panel with modified inventory
            inventory_panel.update_inventory(inventory, 200)
            inventory_panel.update_item_details(item_details)
            last_update = current_time

        # Draw everything
        draw_demo_screen(screen, inventory_panel, font)

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
