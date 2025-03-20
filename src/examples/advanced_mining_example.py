"""
Example usage of the advanced population models with the mining system.

This example demonstrates how to:
1. Create EnhancedMinerEntity instances
2. Initialize and use the MiningSystem
3. Visualize population distributions by life stage
"""

import pygame
import sys
import numpy as np

# Import required system components
from systems.mining_system import MiningSystem
from entities.enhanced_miner_entity import EnhancedMinerEntity
from environment.asteroid_field import AsteroidField
from config import COLOR_RACE_1, COLOR_RACE_2, COLOR_RACE_3

# Constants
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
CELL_SIZE = 4
FIELD_WIDTH = 200
FIELD_HEIGHT = 150


def run_advanced_mining_example():
    """Run an example simulation with advanced population models."""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Advanced Mining Population Models")
    clock = pygame.time.Clock()

    # Create asteroid field
    asteroid_field = AsteroidField(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        density=0.3
    )

    # Initialize mining system
    mining_system = MiningSystem()

    # Create enhanced miner entities with different traits
    miners = [
        EnhancedMinerEntity(
            race_id=1,
            color=COLOR_RACE_1,
            position=(40, 40),
            initial_density=0.05,
            trait="adaptive"
        ),
        EnhancedMinerEntity(
            race_id=2,
            color=COLOR_RACE_2,
            position=(100, 80),
            initial_density=0.04,
            trait="expansive"
        ),
        EnhancedMinerEntity(
            race_id=3,
            color=COLOR_RACE_3,
            position=(160, 60),
            initial_density=0.03,
            trait="selective"
        )
    ]

    # Add miners to the system
    for miner in miners:
        mining_system.add_miner(miner)

    # Simulation variables
    ticks = 0
    camera_offset = (0, 0)
    running = True

    # Main loop
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

        # Update simulation every 5 frames
        if ticks % 5 == 0:
            # Update asteroid field
            asteroid_field.update_asteroids()

            # Update mining system
            mining_system.update(asteroid_field)

        # Clear screen
        screen.fill((0, 0, 0))

        # Render asteroid field
        asteroid_field.render(screen, camera_offset, CELL_SIZE)

        # Render mining territories
        mining_system.render_mining_territories(screen, camera_offset, CELL_SIZE)

        # Render miners
        for miner in miners:
            miner.render(screen, camera_offset, CELL_SIZE)

        # Render stage distribution chart
        render_stage_distribution(screen, mining_system, 700, 50)

        # Render population graph
        render_population_graph(screen, mining_system, 700, 300)

        # Update display
        pygame.display.flip()
        clock.tick(60)
        ticks += 1

    pygame.quit()
    sys.exit()


def render_stage_distribution(
    surface: pygame.Surface, 
    mining_system: MiningSystem, 
    x: int, 
    y: int
) -> None:
    """
    Render a stacked bar chart showing population breakdown by life stage.
    
    Args:
        surface: Pygame surface to render on
        mining_system: The mining system with population data
        x, y: Top-left position for the chart
    """
    # Get stage distribution for each race
    race_colors = {
        1: COLOR_RACE_1,
        2: COLOR_RACE_2,
        3: COLOR_RACE_3
    }
    
    # Stage colors (different shade for each stage)
    stage_colors = {
        "juvenile": (150, 220, 150),  # Light green
        "worker": (80, 180, 80),      # Medium green
        "specialized": (40, 120, 40), # Dark green
        "elder": (20, 60, 20)         # Very dark green
    }
    
    # Draw title
    font = pygame.font.SysFont("Arial", 16)
    title = font.render("Population by Life Stage", True, (255, 255, 255))
    surface.blit(title, (x, y))
    
    # Chart dimensions
    bar_width = 50
    max_height = 150
    gap = 20
    
    # Draw each race's population breakdown
    for i, race_id in enumerate([1, 2, 3]):
        if race_id not in mining_system.miners:
            continue
            
        miner = mining_system.miners[race_id]
        if not hasattr(miner, "get_stage_populations"):
            continue
            
        # Get actual population counts by stage
        stage_pops = miner.get_stage_populations()
        total_pop = sum(stage_pops.values())
        
        # Draw race label
        label = font.render(f"Race {race_id}", True, race_colors[race_id])
        surface.blit(label, (x + i * (bar_width + gap), y + 25))
        
        # Draw stacked bar
        bar_x = x + i * (bar_width + gap)
        bar_y = y + 45
        
        # Scale factor
        scale = min(1.0, max_height / max(1, total_pop))
        
        # Track current height
        current_y = bar_y + max_height
        
        # Draw each stage segment (bottom-up)
        for stage in ["elder", "specialized", "worker", "juvenile"]:
            stage_pop = stage_pops.get(stage, 0)
            if stage_pop > 0:
                segment_height = int(stage_pop * scale)
                segment_rect = pygame.Rect(
                    bar_x, 
                    current_y - segment_height, 
                    bar_width, 
                    segment_height
                )
                pygame.draw.rect(surface, stage_colors[stage], segment_rect)
                pygame.draw.rect(surface, (200, 200, 200), segment_rect, 1)  # Border
                
                # Move up for next segment
                current_y -= segment_height
        
        # Draw total population
        pop_text = font.render(str(total_pop), True, (255, 255, 255))
        surface.blit(pop_text, (bar_x, bar_y + max_height + 10))


def render_population_graph(
    surface: pygame.Surface, 
    mining_system: MiningSystem, 
    x: int, 
    y: int
) -> None:
    """
    Render a line graph showing population over time.
    
    Args:
        surface: Pygame surface to render on
        mining_system: The mining system with population data
        x, y: Top-left position for the chart
    """
    # Draw title
    font = pygame.font.SysFont("Arial", 16)
    title = font.render("Population Trends", True, (255, 255, 255))
    surface.blit(title, (x, y))
    
    # Chart dimensions
    width = 250
    height = 150
    
    # Draw axes
    pygame.draw.line(surface, (200, 200, 200), (x, y + height), (x + width, y + height))
    pygame.draw.line(surface, (200, 200, 200), (x, y + height), (x, y))
    
    # Cache population history (in a real implementation, this would be stored persistently)
    # For this example, we'll use random data to simulate history
    for race_id in range(1, 4):
        if race_id not in mining_system.miners:
            continue
            
        current_pop = mining_system.miners[race_id].population
        color = getattr(mining_system.miners[race_id], "color", (255, 255, 255))
        
        # Generate fake history data (just for demonstration)
        points = []
        for i in range(20):
            # Create a curve leading to current population
            t = i / 19
            history_pop = int(current_pop * t + 50 * (1 - t) * (1 + 0.5 * np.sin(i * 0.4)))
            points.append((x + i * (width / 19), y + height - (history_pop / 2)))
        
        # Add current population point
        points.append((x + width, y + height - (current_pop / 2)))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)


if __name__ == "__main__":
    run_advanced_mining_example()
