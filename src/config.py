"""
Game configuration and constants for Space Muck.

This file centralizes all game constants and configuration values to make
adjustments and balancing easier across the entire codebase.
"""

from typing import Tuple, Dict, Any, List

# Version information
VERSION: str = "1.0.0"
BUILD_DATE: str = "2025-03-01"

# Grid and window configuration
CELL_SIZE: int = 8  # Smaller cells for larger world
GRID_WIDTH: int = 400  # Much larger grid for extensive exploration
GRID_HEIGHT: int = 300
GAME_MAP_SIZE: Tuple[int, int] = (GRID_WIDTH, GRID_HEIGHT)  # Size of the game map
WINDOW_WIDTH: int = 1600  # Fixed window size
WINDOW_HEIGHT: int = 1200
VIEW_WIDTH: int = WINDOW_WIDTH // CELL_SIZE  # Visible grid cells
VIEW_HEIGHT: int = WINDOW_HEIGHT // CELL_SIZE
FPS: int = 60  # Higher FPS for smoother gameplay
MINIMAP_SIZE: int = 200  # Size of the minimap
MINIMAP_PADDING: int = 10  # Padding around minimap

# Performance settings
RENDER_DISTANCE: int = 50  # Distance from player to render
UPDATE_INTERVAL: int = 10  # Update field every N frames
MAX_ENTITIES_RENDERED: int = 5000  # Maximum number of entities to render

# Colors (RGB)
COLOR_BG: Tuple[int, int, int] = (10, 10, 15)
COLOR_GRID: Tuple[int, int, int] = (20, 20, 30)
COLOR_ASTEROID: Tuple[int, int, int] = (80, 80, 80)
COLOR_ASTEROID_RARE: Tuple[int, int, int] = (255, 215, 0)  # Gold
COLOR_ASTEROID_PRECIOUS: Tuple[int, int, int] = (0, 191, 255)  # Deep Sky Blue
COLOR_ASTEROID_ANOMALY: Tuple[int, int, int] = (148, 0, 211)  # Dark Violet
COLOR_PLAYER: Tuple[int, int, int] = (0, 255, 0)
COLOR_TEXT: Tuple[int, int, int] = (220, 220, 220)
COLOR_HIGHLIGHT: Tuple[int, int, int] = (255, 0, 0)
COLOR_SHOP_BG: Tuple[int, int, int] = (20, 20, 30)
COLOR_EVENT: Tuple[int, int, int] = (0, 255, 255)
COLOR_UI_BG: Tuple[int, int, int] = (30, 30, 40, 220)  # Semi-transparent UI background
COLOR_UI_BORDER: Tuple[int, int, int] = (100, 100, 140, 255)
COLOR_UI_HIGHLIGHT: Tuple[int, int, int] = (120, 120, 160)
COLOR_UI_TEXT: Tuple[int, int, int] = (220, 220, 220)
COLOR_UI_BUTTON: Tuple[int, int, int] = (60, 60, 80)
COLOR_UI_BUTTON_HOVER: Tuple[int, int, int] = (80, 80, 100)
COLOR_UI_BUTTON_DISABLED: Tuple[int, int, int] = (40, 40, 50)
COLOR_SUCCESS: Tuple[int, int, int] = (0, 255, 0)
COLOR_WARNING: Tuple[int, int, int] = (255, 165, 0)
COLOR_ERROR: Tuple[int, int, int] = (255, 0, 0)
COLOR_INFO: Tuple[int, int, int] = (0, 191, 255)

# Race colors with better distinction
COLOR_RACE_1: Tuple[int, int, int] = (50, 100, 255)  # Blue race
COLOR_RACE_2: Tuple[int, int, int] = (255, 50, 150)  # Magenta race
COLOR_RACE_3: Tuple[int, int, int] = (255, 165, 0)  # Orange race
COLOR_RACE_4: Tuple[int, int, int] = (0, 200, 80)  # Green race - for future expansion
COLOR_RACE_5: Tuple[int, int, int] = (200, 50, 50)  # Red race - for future expansion

# Entity behavior colors
COLOR_ENTITY_DEFAULT: Tuple[int, int, int] = (180, 180, 180)  # Default gray
COLOR_ENTITY_FEEDING: Tuple[int, int, int] = (0, 200, 0)  # Green for feeding
COLOR_ENTITY_EXPANDING: Tuple[int, int, int] = (0, 100, 255)  # Blue for expanding
COLOR_ENTITY_MIGRATING: Tuple[int, int, int] = (255, 165, 0)  # Orange for migrating
COLOR_ENTITY_AGGRESSIVE: Tuple[int, int, int] = (255, 0, 0)  # Red for aggressive

# Game States
STATE_PLAY: str = "PLAY"
STATE_SHOP: str = "SHOP"
STATE_MAP: str = "MAP"
STATE_MENU: str = "MENU"
STATE_PAUSE: str = "PAUSE"
STATE_GAMEOVER: str = "GAMEOVER"

# Asteroid field generation parameters
INITIAL_DENSITY: float = 0.3  # Initial asteroid density
RARE_THRESHOLD: float = 0.92  # Threshold for rare asteroid generation
PRECIOUS_THRESHOLD: float = 0.97  # Threshold for precious asteroid generation
ANOMALY_THRESHOLD: float = 0.995  # Threshold for anomaly generation
MIN_ASTEROID_VALUE: int = 1  # Minimum value of an asteroid
MAX_ASTEROID_VALUE: int = 100  # Maximum value of an asteroid
RARE_BONUS_MULTIPLIER: float = 2.5  # Value multiplier for rare asteroids
PRECIOUS_BONUS_MULTIPLIER: float = 5.0  # Value multiplier for precious asteroids
ANOMALY_BONUS_MULTIPLIER: float = 10.0  # Value multiplier for anomalies

# Perlin noise settings for field generation
PERLIN_OCTAVES: int = 5  # Number of octaves for Perlin noise
PERLIN_PERSISTENCE: float = 0.5  # Persistence for Perlin noise
PERLIN_LACUNARITY: float = 2.0  # Lacunarity for Perlin noise
PERLIN_SCALE: float = 100.0  # Scale for Perlin noise

# Cellular automaton parameters
CA_STEPS: int = 3  # Number of CA steps to run during initialization
DEFAULT_BIRTH_SET: set = {3}  # Default cells born with this many neighbors
DEFAULT_SURVIVAL_SET: set = {2, 3}  # Default cells survive with this many neighbors
ENERGY_DIFFUSION_RATE: float = 0.05  # Rate at which energy diffuses
ENERGY_DECAY_RATE: float = 0.01  # Rate at which energy decays

# Combat system settings
COMBAT_BASE_ATTACK_POWER: int = 10  # Base attack power for level 1 weapons
COMBAT_BASE_ATTACK_SPEED: float = 1.0  # Base attacks per time unit
COMBAT_BASE_WEAPON_RANGE: int = 5  # Base weapon range in grid units
COMBAT_BASE_CRIT_CHANCE: float = 0.05  # 5% base critical hit chance
COMBAT_CRIT_MULTIPLIER: float = 2.0  # Critical hits do double damage

COMBAT_BASE_SHIELD_STRENGTH: int = 50  # Base shield points
COMBAT_BASE_SHIELD_RECHARGE: float = 1.0  # Shield points recharged per time unit
COMBAT_BASE_HULL_STRENGTH: int = 100  # Base hull integrity points
COMBAT_BASE_EVASION: float = 0.1  # 10% base chance to evade attacks
COMBAT_BASE_ARMOR: float = 0.05  # 5% base damage reduction

COMBAT_WEAPON_UPGRADE_COST: List[int] = [
    0,
    1500,
    4000,
    10000,
    20000,
]  # Costs for weapon levels 1-5
COMBAT_SHIELD_UPGRADE_COST: List[int] = [
    0,
    2000,
    5000,
    12000,
    25000,
]  # Costs for shield levels 1-5
COMBAT_HULL_UPGRADE_COST: List[int] = [
    0,
    3000,
    7000,
    15000,
    30000,
]  # Costs for hull levels 1-5

COMBAT_ENEMY_TYPES: List[str] = [
    "pirate",
    "patrol",
    "mercenary",
    "elite",
]  # Types of enemy ships
COMBAT_DIFFICULTY_MULTIPLIER: Dict[str, float] = (
    {  # Difficulty multipliers for enemy stats
        "easy": 0.8,
        "medium": 1.0,
        "hard": 1.3,
        "elite": 1.8,
    }
)

# Player settings
PLAYER_START_CURRENCY: int = 100  # Starting currency
PLAYER_START_MINING_EFFICIENCY: float = 1.0  # Starting mining efficiency
PLAYER_START_MINING_RANGE: int = 1  # Starting mining range
PLAYER_START_MOVE_SPEED: int = 1  # Starting move speed
PLAYER_MAX_MINING_SHIPS: int = 10  # Maximum number of mining ships
PLAYER_SHIP_HEALTH: int = 100  # Ship health
PLAYER_SHIP_COST: int = 500  # Cost to build a new ship
PLAYER_SHIP_COST_SCALING: float = 1.2  # Cost scaling for each additional ship

# Symbiote race settings
RACE_INITIAL_DENSITY: float = 0.005  # Initial density of race cells
RACE_HUNGER_RATE: float = 0.01  # Rate at which race hunger increases
RACE_BASE_AGGRESSION: float = 0.2  # Base aggression level
RACE_BASE_MINING_EFFICIENCY: float = 0.5  # Base mining efficiency for races
RACE_EVOLUTION_THRESHOLD: int = 100  # Points needed for evolution
RACE_MAX_POPULATION: int = 5000  # Maximum population for a race

# Shop settings
SHOP_PANEL_WIDTH: int = 400  # Width of expanded shop panel
SHOP_COLLAPSED_WIDTH: int = 60  # Width of collapsed shop panel
SHOP_ANIMATION_SPEED: float = 0.2  # Speed of shop panel animation
SHOP_MAX_VISIBLE_ITEMS: int = 7  # Maximum visible shop items
SHOP_ITEM_HEIGHT: int = 80  # Height of shop item
SHOP_CATEGORIES: List[str] = ["ship", "field", "race", "special"]  # Shop categories

# Notification settings
NOTIFICATION_PANEL_WIDTH: int = 300  # Width of notification panel
NOTIFICATION_MAX_STORED: int = 100  # Maximum stored notifications
NOTIFICATION_MAX_VISIBLE: int = 10  # Maximum visible notifications
NOTIFICATION_HEIGHT: int = 25  # Height of each notification
NOTIFICATION_FADE_SPEED: float = 0.5  # Fade speed for notifications
NOTIFICATION_HIGHLIGHT_TIME: float = 2.0  # Time to highlight new notifications

# Sound settings
SOUND_ENABLED: bool = True  # Whether sound is enabled
SOUND_VOLUME: float = 0.7  # Sound volume (0.0 - 1.0)
MUSIC_VOLUME: float = 0.5  # Music volume (0.0 - 1.0)

# Upgrade scaling factors
UPGRADE_BASE_SCALING: float = 1.5  # Base cost scaling for upgrades
UPGRADE_SPECIAL_CHANCE: float = 0.01  # Chance of special upgrades appearing
UPGRADE_REFRESH_INTERVAL: int = 1000  # Frames between upgrade refreshes

# Mineral types and their relative values
MINERAL_TYPES: Dict[str, Dict[str, Any]] = {
    "common": {
        "value_multiplier": 1.0,
        "color": COLOR_ASTEROID,
        "rarity": 0.0,  # Base threshold
    },
    "rare": {
        "value_multiplier": RARE_BONUS_MULTIPLIER,
        "color": COLOR_ASTEROID_RARE,
        "rarity": RARE_THRESHOLD,
    },
    "precious": {
        "value_multiplier": PRECIOUS_BONUS_MULTIPLIER,
        "color": COLOR_ASTEROID_PRECIOUS,
        "rarity": PRECIOUS_THRESHOLD,
    },
    "anomaly": {
        "value_multiplier": ANOMALY_BONUS_MULTIPLIER,
        "color": COLOR_ASTEROID_ANOMALY,
        "rarity": ANOMALY_THRESHOLD,
    },
}

# Debug settings
DEBUG_MODE: bool = False  # Enable debug features
SHOW_FPS: bool = True  # Show FPS counter
SHOW_GRID: bool = False  # Show grid lines
LOG_LEVEL: str = "INFO"  # Logging level
PROFILE_PERFORMANCE: bool = False  # Profile performance
