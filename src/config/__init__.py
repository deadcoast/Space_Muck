"""Config package initialization.

This file defines all constants for the Space Muck game.
"""

# Standard library imports

# Third-party library imports

# Local application imports
from .config import (

    # Grid and window configuration
    CELL_SIZE,
    GRID_WIDTH,
    GRID_HEIGHT,
    GAME_MAP_SIZE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    VIEW_WIDTH,
    VIEW_HEIGHT,
    FPS,
    MINIMAP_SIZE,
    MINIMAP_PADDING,
    # UI Colors
    COLOR_UI_BG,
    COLOR_UI_BORDER,
    COLOR_UI_HIGHLIGHT,
    COLOR_UI_TEXT,
    COLOR_UI_BUTTON,
    COLOR_UI_BUTTON_HOVER,
    COLOR_UI_BUTTON_DISABLED,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_INFO,
    # Game States
    STATE_PLAY,
    STATE_SHOP,
    STATE_MAP,
    STATE_MENU,
    STATE_PAUSE,
    STATE_GAMEOVER,
    # Game Configuration
    GAME_CONFIG,
    # Race Configuration
    RACE_INITIAL_DENSITY,
)

# Re-export all constants
__all__ = [
    # Grid and window configuration
    "CELL_SIZE",
    "GRID_WIDTH",
    "GRID_HEIGHT",
    "GAME_MAP_SIZE",
    "WINDOW_WIDTH",
    "WINDOW_HEIGHT",
    "VIEW_WIDTH",
    "VIEW_HEIGHT",
    "FPS",
    "MINIMAP_SIZE",
    "MINIMAP_PADDING",
    # UI Colors
    "COLOR_UI_BG",
    "COLOR_UI_BORDER",
    "COLOR_UI_HIGHLIGHT",
    "COLOR_UI_TEXT",
    "COLOR_UI_BUTTON",
    "COLOR_UI_BUTTON_HOVER",
    "COLOR_UI_BUTTON_DISABLED",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_ERROR",
    "COLOR_INFO",
    # Game States
    "STATE_PLAY",
    "STATE_SHOP",
    "STATE_MAP",
    "STATE_MENU",
    "STATE_PAUSE",
    "STATE_GAMEOVER",
    # Game Configuration
    "GAME_CONFIG",
    # Race Configuration
    "RACE_INITIAL_DENSITY",
    # Race colors
    "COLOR_RACE_1",
    "COLOR_RACE_2",
    "COLOR_RACE_3",
    "COLOR_RACE_4",
    "COLOR_RACE_5",
    # Entity colors
    "COLOR_ENTITY_DEFAULT",
    "COLOR_ENTITY_FEEDING",
    "COLOR_ENTITY_EXPANDING",
    "COLOR_ENTITY_MIGRATING",
    "COLOR_ENTITY_AGGRESSIVE",
    # Combat settings
    "COMBAT_BASE_ATTACK_POWER",
    "COMBAT_BASE_ATTACK_SPEED",
    "COMBAT_BASE_WEAPON_RANGE",
    "COMBAT_BASE_CRIT_CHANCE",
    "COMBAT_CRIT_MULTIPLIER",
    "COMBAT_BASE_SHIELD_STRENGTH",
    "COMBAT_BASE_SHIELD_RECHARGE",
    "COMBAT_BASE_HULL_STRENGTH",
    "COMBAT_BASE_EVASION",
    "COMBAT_BASE_ARMOR",
    "COMBAT_WEAPON_UPGRADE_COST",
    "COMBAT_SHIELD_UPGRADE_COST",
    "COMBAT_HULL_UPGRADE_COST",
    "COMBAT_ENEMY_TYPES",
    "COMBAT_DIFFICULTY_MULTIPLIER",
    # Player settings
    "PLAYER_START_CURRENCY",
    "PLAYER_START_MINING_EFFICIENCY",
    "PLAYER_START_MINING_RANGE",
    "PLAYER_START_MOVE_SPEED",
    "PLAYER_MAX_MINING_SHIPS",
    "PLAYER_SHIP_HEALTH",
    "PLAYER_SHIP_COST",
    "PLAYER_SHIP_COST_SCALING",
]

# Re-export constants for backward compatibility
CELL_SIZE = CELL_SIZE
GRID_WIDTH = GRID_WIDTH
GRID_HEIGHT = GRID_HEIGHT
GAME_MAP_SIZE = GAME_MAP_SIZE
WINDOW_WIDTH = WINDOW_WIDTH
WINDOW_HEIGHT = WINDOW_HEIGHT
VIEW_WIDTH = VIEW_WIDTH
VIEW_HEIGHT = VIEW_HEIGHT
FPS = FPS
MINIMAP_SIZE = MINIMAP_SIZE
MINIMAP_PADDING = MINIMAP_PADDING

# Re-export UI colors
COLOR_UI_BG = COLOR_UI_BG
COLOR_UI_BORDER = COLOR_UI_BORDER
COLOR_UI_HIGHLIGHT = COLOR_UI_HIGHLIGHT
COLOR_UI_TEXT = COLOR_UI_TEXT
COLOR_UI_BUTTON = COLOR_UI_BUTTON
COLOR_UI_BUTTON_HOVER = COLOR_UI_BUTTON_HOVER
COLOR_UI_BUTTON_DISABLED = COLOR_UI_BUTTON_DISABLED
COLOR_SUCCESS = COLOR_SUCCESS
COLOR_WARNING = COLOR_WARNING
COLOR_ERROR = COLOR_ERROR
COLOR_INFO = COLOR_INFO

# Re-export game states
STATE_PLAY = STATE_PLAY
STATE_SHOP = STATE_SHOP
STATE_MAP = STATE_MAP
STATE_MENU = STATE_MENU
STATE_PAUSE = STATE_PAUSE
STATE_GAMEOVER = STATE_GAMEOVER

# Re-export game config
GAME_CONFIG = GAME_CONFIG

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
