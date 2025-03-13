"""
BaseEntity class: The root class for all entities in the game.
"""

# Standard library imports
import logging
import uuid

# Third-party library imports

# Local application imports
from typing import Tuple, Optional, Dict, Any

class BaseEntity:
    """
    Base class for all entities in the game.
    Provides common functionality and attributes that all entities should have.
    """

    def __init__(
        self,
        entity_id: Optional[str] = None,
        entity_type: str = "generic",
        color: Tuple[int, int, int] = (255, 255, 255),
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize a base entity.

        Args:
            entity_id: Unique identifier for the entity (defaults to a UUID)
            entity_type: Type of entity (e.g., "player", "miner", "npc")
            color: RGB color tuple for visualization
            position: Initial position as (x, y) tuple
        """
        # Core entity attributes
        self.entity_id = entity_id or str(uuid.uuid4())
        self.entity_type = entity_type
        self.color = color
        self.position = position
        self.active = True
        self.created_at = 0  # Game tick when entity was created

        # Stats and properties
        self.health = 100
        self.max_health = 100
        self.level = 1
        self.tags = set()  # Set of tags for entity categorization

        logging.info(f"Entity created: {self.entity_type} (ID: {self.entity_id})")

    def update(self, game_state: Dict[str, Any]) -> None:
        """
        Update the entity state based on the current game state.
        This is a base method that should be overridden by subclasses.

        Args:
            game_state: Dictionary containing the current game state
        """
        # Base implementation does nothing
        pass

    def is_active(self) -> bool:
        """
        Check if the entity is active.

        Returns:
            bool: True if the entity is active, False otherwise
        """
        return self.active

    def deactivate(self) -> None:
        """Deactivate the entity."""
        self.active = False
        logging.info(f"Entity deactivated: {self.entity_type} (ID: {self.entity_id})")

    def activate(self) -> None:
        """Activate the entity."""
        self.active = True
        logging.info(f"Entity activated: {self.entity_type} (ID: {self.entity_id})")

    def set_position(self, x: int, y: int) -> None:
        """
        Set the entity's position.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.position = (x, y)

    def get_position(self) -> Optional[Tuple[int, int]]:
        """
        Get the entity's position.

        Returns:
            Tuple[int, int] or None: The entity's position as (x, y) or None if not set
        """
        return self.position

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the entity.

        Args:
            tag: Tag to add
        """
        self.tags.add(tag)

    def has_tag(self, tag: str) -> bool:
        """
        Check if the entity has a specific tag.

        Args:
            tag: Tag to check for

        Returns:
            bool: True if the entity has the tag, False otherwise
        """
        return tag in self.tags

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the entity
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "color": self.color,
            "position": self.position,
            "active": self.active,
            "health": self.health,
            "max_health": self.max_health,
            "level": self.level,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        """
        Create an entity from a dictionary.

        Args:
            data: Dictionary containing entity data

        Returns:
            BaseEntity: New entity instance
        """
        entity = cls(
            entity_id=data.get("entity_id"),
            entity_type=data.get("entity_type", "generic"),
            color=data.get("color", (255, 255, 255)),
            position=data.get("position"),
        )
        entity.active = data.get("active", True)
        entity.health = data.get("health", 100)
        entity.max_health = data.get("max_health", 100)
        entity.level = data.get("level", 1)
        entity.tags = set(data.get("tags", []))
        return entity
