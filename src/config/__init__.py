"""Config package initialization.

This file defines all constants for the Space Muck game.
"""

# Grid and window configuration
CELL_SIZE = 8  # Smaller cells for larger world
GRID_WIDTH = 400  # Much larger grid for extensive exploration
GRID_HEIGHT = 300
GAME_MAP_SIZE = (GRID_WIDTH, GRID_HEIGHT)  # Size of the game map
WINDOW_WIDTH = 1600  # Fixed window size
WINDOW_HEIGHT = 1200
VIEW_WIDTH = WINDOW_WIDTH // CELL_SIZE  # Visible grid cells
VIEW_HEIGHT = WINDOW_HEIGHT // CELL_SIZE
FPS = 60  # Higher FPS for smoother gameplay
MINIMAP_SIZE = 200  # Size of the minimap
MINIMAP_PADDING = 10  # Padding around minimap

# Define race colors that are used in miner_entity.py
COLOR_RACE_1 = (50, 100, 255)  # Blue race
COLOR_RACE_2 = (255, 50, 150)  # Magenta race
COLOR_RACE_3 = (255, 165, 0)  # Orange race
COLOR_RACE_4 = (0, 200, 80)  # Green race - for future expansion
COLOR_RACE_5 = (200, 50, 50)  # Red race - for future expansion

# Define entity colors
COLOR_ENTITY_DEFAULT = (200, 200, 200)
COLOR_ENTITY_FEEDING = (100, 255, 100)
COLOR_ENTITY_EXPANDING = (255, 255, 100)
COLOR_ENTITY_MIGRATING = (100, 100, 255)
COLOR_ENTITY_AGGRESSIVE = (255, 100, 100)

# Colors (RGB)
COLOR_BG = (10, 10, 15)
COLOR_GRID = (20, 20, 30)
COLOR_ASTEROID = (80, 80, 80)
COLOR_ASTEROID_RARE = (255, 215, 0)  # Gold
COLOR_ASTEROID_PRECIOUS = (0, 191, 255)  # Deep Sky Blue
COLOR_ASTEROID_ANOMALY = (148, 0, 211)  # Dark Violet
COLOR_PLAYER = (0, 255, 0)
COLOR_TEXT = (220, 220, 220)
COLOR_HIGHLIGHT = (255, 0, 0)

# Combat system settings
COMBAT_BASE_ATTACK_POWER = 10  # Base attack power for level 1 weapons
COMBAT_BASE_ATTACK_SPEED = 1.0  # Base attacks per time unit
COMBAT_BASE_WEAPON_RANGE = 5  # Base weapon range in grid units
COMBAT_BASE_CRIT_CHANCE = 0.05  # 5% base critical hit chance
COMBAT_CRIT_MULTIPLIER = 2.0  # Critical hits do double damage

COMBAT_BASE_SHIELD_STRENGTH = 50  # Base shield points
COMBAT_BASE_SHIELD_RECHARGE = 1.0  # Shield points recharged per time unit
COMBAT_BASE_HULL_STRENGTH = 100  # Base hull integrity points
COMBAT_BASE_EVASION = 0.1  # 10% base chance to evade attacks
COMBAT_BASE_ARMOR = 0.05  # 5% base damage reduction

COMBAT_WEAPON_UPGRADE_COST = [
    0,
    1500,
    4000,
    10000,
    20000,
]  # Costs for weapon levels 1-5
COMBAT_SHIELD_UPGRADE_COST = [
    0,
    2000,
    5000,
    12000,
    25000,
]  # Costs for shield levels 1-5
COMBAT_HULL_UPGRADE_COST = [0, 3000, 7000, 15000, 30000]  # Costs for hull levels 1-5

COMBAT_ENEMY_TYPES = ["pirate", "patrol", "mercenary", "elite"]  # Types of enemy ships
COMBAT_DIFFICULTY_MULTIPLIER = {  # Difficulty multipliers for enemy stats
    "easy": 0.8,
    "medium": 1.0,
    "hard": 1.3,
    "elite": 1.8,
}

# Player settings
PLAYER_START_CURRENCY = 100  # Starting currency
PLAYER_START_MINING_EFFICIENCY = 1.0  # Starting mining efficiency
PLAYER_START_MINING_RANGE = 1  # Starting mining range
PLAYER_START_MOVE_SPEED = 1  # Starting move speed
PLAYER_MAX_MINING_SHIPS = 10  # Maximum number of mining ships
PLAYER_SHIP_HEALTH = 100  # Ship health
PLAYER_SHIP_COST = 500  # Cost to build a new ship
PLAYER_SHIP_COST_SCALING = 1.2  # Cost scaling for each additional ship
