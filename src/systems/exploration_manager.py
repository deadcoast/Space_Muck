"""
Exploration Management System: Handles exploration, discovery, and mapping.

This module provides functionality for managing exploration activities,
including region discovery, resource scanning, and exploration missions.
"""

# Standard library imports
import logging
import random

# Local application imports
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

# Third-party library imports


# Exploration States
class ExplorationState(Enum):
    UNEXPLORED = auto()
    SCANNING = auto()
    EXPLORED = auto()
    DEPLETED = auto()
    DANGEROUS = auto()


# Region Types
class RegionType(Enum):
    EMPTY = auto()
    RESOURCE_RICH = auto()
    ANOMALY = auto()
    SETTLEMENT = auto()
    HAZARD = auto()


# Discovery Categories
class DiscoveryType(Enum):
    RESOURCE = auto()
    TECHNOLOGY = auto()
    ARTIFACT = auto()
    STRUCTURE = auto()
    PHENOMENON = auto()


# Exploration Patterns
EXPLORATION_PATTERNS = {
    "spiral": "Explore in expanding spiral pattern",
    "grid": "Systematic grid-based exploration",
    "targeted": "Focus on specific regions of interest",
    "random": "Random exploration pattern",
}


@dataclass
class Region:
    """Represents an explorable region in space."""

    region_id: str
    position: Tuple[int, int]
    size: float
    type: RegionType
    state: ExplorationState = ExplorationState.UNEXPLORED
    resources: Dict[str, float] = None
    hazard_level: float = 0.0
    discoveries: List[str] = None

    def __post_init__(self):
        """Initialize optional fields."""
        self.resources = self.resources or {}
        self.discoveries = self.discoveries or []


@dataclass
class ExplorationMission:
    """Represents an active exploration mission."""

    mission_id: str
    target_region: str
    pattern: str
    duration: float
    progress: float = 0.0
    discoveries: List[str] = None
    active: bool = True

    def __post_init__(self):
        """Initialize optional fields."""
        self.discoveries = self.discoveries or []


class ExplorationManager:
    """
    Central manager for handling all exploration-related operations.
    """

    def __init__(self) -> None:
        """Initialize the exploration manager."""
        # Region tracking
        self.regions: Dict[str, Region] = {}
        self.region_grid: Dict[
            Tuple[int, int], str
        ] = {}  # Position to region_id mapping

        # Mission management
        self.active_missions: Dict[str, ExplorationMission] = {}
        self.completed_missions: Set[str] = set()

        # Discovery tracking
        self.discoveries: Dict[str, DiscoveryType] = {}
        self.discovery_locations: Dict[str, str] = {}  # Discovery to region_id mapping

        # System state
        self.active = True
        self.paused = False
        self.update_interval = 1.0  # seconds
        self.last_update = 0.0

        # Exploration settings
        self.scan_range = 10.0
        self.discovery_chance = 0.1
        self.hazard_threshold = 0.7

        logging.info("ExplorationManager initialized")

    def register_region(
        self,
        position: Tuple[int, int],
        size: float,
        region_type: RegionType = RegionType.EMPTY,
    ) -> Optional[str]:
        """
        Register a new region for exploration.

        Args:
            position: (x, y) coordinates of the region
            size: Size of the region
            region_type: Type of the region

        Returns:
            str: Region ID if successful, None if position already occupied
        """
        if position in self.region_grid:
            logging.warning(f"Region already exists at position {position}")
            return None

        region_id = f"region_{len(self.regions)}"
        region = Region(
            region_id=region_id, position=position, size=size, type=region_type
        )

        self.regions[region_id] = region
        self.region_grid[position] = region_id

        logging.info(f"Registered region {region_id} at {position}")
        return region_id

    def start_exploration(
        self, region_id: str, pattern: str = "grid", duration: float = 60.0
    ) -> Optional[str]:
        """
        Start an exploration mission in a region.

        Args:
            region_id: Target region to explore
            pattern: Exploration pattern to use
            duration: Expected duration of the mission

        Returns:
            str: Mission ID if successful, None otherwise
        """
        if region_id not in self.regions:
            logging.error(f"Region {region_id} not found")
            return None

        if pattern not in EXPLORATION_PATTERNS:
            logging.error(f"Invalid exploration pattern: {pattern}")
            return None

        region = self.regions[region_id]
        if region.state not in {ExplorationState.UNEXPLORED, ExplorationState.SCANNING}:
            logging.warning(f"Region {region_id} already explored")
            return None

        mission_id = f"mission_{len(self.active_missions)}"
        mission = ExplorationMission(
            mission_id=mission_id,
            target_region=region_id,
            pattern=pattern,
            duration=duration,
        )

        self.active_missions[mission_id] = mission
        region.state = ExplorationState.SCANNING

        logging.info(f"Started exploration mission {mission_id} in region {region_id}")
        return mission_id

    def update(self, dt: float) -> None:
        """
        Update exploration missions and process discoveries.

        Args:
            dt: Time delta since last update
        """
        if not self.active or self.paused:
            return

        self.last_update += dt
        if self.last_update < self.update_interval:
            return

        # Update active missions
        completed_missions = []
        for mission_id, mission in self.active_missions.items():
            if not mission.active:
                continue

            # Update mission progress
            progress_increment = self.last_update / mission.duration
            mission.progress = min(1.0, mission.progress + progress_increment)

            # Check for discoveries
            if random.random() < self.discovery_chance * self.last_update:
                self._process_discovery(mission)

            # Check for mission completion
            if mission.progress >= 1.0:
                self._complete_mission(mission_id)
                completed_missions.append(mission_id)

        # Clean up completed missions
        for mission_id in completed_missions:
            del self.active_missions[mission_id]

        self.last_update = 0

    def _process_discovery(self, mission: ExplorationMission) -> None:
        """
        Process a new discovery in a mission.

        Args:
            mission: Active mission that made the discovery
        """
        region = self.regions[mission.target_region]

        # Generate discovery based on region type
        discovery_type = random.choice(list(DiscoveryType))
        discovery_id = f"discovery_{len(self.discoveries)}"

        self.discoveries[discovery_id] = discovery_type
        self.discovery_locations[discovery_id] = mission.target_region
        mission.discoveries.append(discovery_id)
        region.discoveries.append(discovery_id)

        logging.info(
            f"New discovery {discovery_id} ({discovery_type}) in region {mission.target_region}"
        )

    def _complete_mission(self, mission_id: str) -> None:
        """
        Complete an exploration mission.

        Args:
            mission_id: ID of mission to complete
        """
        mission = self.active_missions[mission_id]
        region = self.regions[mission.target_region]

        # Update region state based on discoveries
        if len(region.discoveries) == 0:
            region.state = ExplorationState.DEPLETED
        elif region.hazard_level >= self.hazard_threshold:
            region.state = ExplorationState.DANGEROUS
        else:
            region.state = ExplorationState.EXPLORED

        # Archive mission
        mission.active = False
        self.completed_missions.add(mission_id)

        logging.info(
            f"Completed mission {mission_id} in region {mission.target_region}"
        )

    def get_region_state(self, region_id: str) -> Optional[ExplorationState]:
        """
        Get the current state of a region.

        Args:
            region_id: Region to check

        Returns:
            ExplorationState: Current state or None if not found
        """
        return self.regions[region_id].state if region_id in self.regions else None

    def get_discoveries(self, region_id: str) -> List[str]:
        """
        Get all discoveries in a region.

        Args:
            region_id: Region to check

        Returns:
            List[str]: List of discovery IDs
        """
        return self.regions[region_id].discoveries if region_id in self.regions else []

    def get_nearby_regions(
        self, position: Tuple[int, int], range_limit: float
    ) -> List[str]:
        """
        Get regions within range of a position.

        Args:
            position: Center position to search from
            range_limit: Maximum distance to search

        Returns:
            List[str]: List of region IDs within range
        """
        nearby = []
        x, y = position

        for (rx, ry), region_id in self.region_grid.items():
            distance = ((rx - x) ** 2 + (ry - y) ** 2) ** 0.5
            if distance <= range_limit:
                nearby.append(region_id)

        return nearby

    def pause(self) -> None:
        """Pause exploration processing."""
        self.paused = True
        logging.info("ExplorationManager paused")

    def resume(self) -> None:
        """Resume exploration processing."""
        self.paused = False
        logging.info("ExplorationManager resumed")

    def shutdown(self) -> None:
        """Shutdown the exploration manager."""
        self.active = False
        logging.info("ExplorationManager shut down")
