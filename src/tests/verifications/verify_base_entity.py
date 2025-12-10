"""
Verification script for BaseEntity class.
"""

# Standard library imports
import os
import sys
from unittest.mock import MagicMock

# Local application imports
from entities.base_entity import BaseEntity

# Third-party library imports


# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the entities - moved to top of file to avoid E402 error

# Mock modules before importing entities
sys.modules["perlin_noise"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.spatial"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()

# Mock src modules
sys.modules["src.algorithms.symbiote_algorithm"] = MagicMock()
sys.modules["src.algorithms.symbiote_algorithm"].SymbioteEvolutionAlgorithm = (
    MagicMock()
)
sys.modules["src.utils.logging_setup"] = MagicMock()
sys.modules["src.utils.logging_setup"].log_exception = MagicMock()
sys.modules["src.config"] = MagicMock()
sys.modules["src.config"].COLOR_PLAYER = (0, 255, 255)

# BaseEntity already imported at the top of the file


# Create mock for MinerEntity and Player
class MockMinerEntity(BaseEntity):
    def __init__(self, race_id=1, color=(0, 255, 0), **kwargs):
        super().__init__(entity_id=str(race_id), entity_type="miner", color=color)
        self.race_id = race_id
        self.trait = "standard"


class MockPlayer(MockMinerEntity):
    def __init__(self, race_id=0, color=(0, 255, 255), **kwargs):
        super().__init__(race_id=race_id, color=color)
        self.is_player = True
        self.credits = 1000
        self.ship_level = 1
        self.trait = "adaptive"


def verify_base_entity():
    """Test the BaseEntity class."""
    print("Testing BaseEntity class...")

    # Create a basic entity
    entity = BaseEntity(entity_type="test", color=(255, 0, 0), position=(10, 20))

    # Verify basic attributes
    if (
        entity.entity_type != "test"
    ):
        raise AssertionError(f"Expected entity_type 'test', got {entity.entity_type}")
    if entity.color != (
        255,
        0,
        0,
    ):
        raise AssertionError(f"Expected color (255, 0, 0), got {entity.color}")

    # Check that position is not None before comparing
    assert entity.position is not None, "Entity position should not be None"
    if entity.position:  # Type guard for position
        if entity.position != (
            10,
            20,
        ):
            raise AssertionError(f"Expected position (10, 20), got {entity.position}")

    if entity.active is not True:
        raise AssertionError(f"Expected active True, got {entity.active}")

    # Test methods
    entity.add_tag("important")
    if not entity.has_tag("important"):
        raise AssertionError("Tag 'important' should be present")

    entity.set_position(30, 40)

    # Check that position is not None before comparing
    position = entity.get_position()
    assert position is not None, "Entity position should not be None after set_position"
    if position:  # Type guard for position
        if position != (
            30,
            40,
        ):
            raise AssertionError(f"Expected position (30, 40), got {position}")

    entity.deactivate()
    if entity.is_active():
        raise AssertionError("Entity should be inactive after deactivate()")

    entity.activate()
    if not entity.is_active():
        raise AssertionError("Entity should be active after activate()")

    # Test serialization
    data = entity.to_dict()
    if (
        data["entity_type"] != "test"
    ):
        raise AssertionError(f"Expected entity_type 'test', got {data['entity_type']}")

    new_entity = BaseEntity.from_dict(data)
    if (
        new_entity.entity_type != entity.entity_type
    ):
        raise AssertionError("Entity types should match after deserialization")

    print("BaseEntity tests passed!")
    return True


def verify_inheritance():
    """Test the inheritance hierarchy using mock classes."""
    print("Testing entity inheritance hierarchy...")

    # Create instances of each entity type
    base = BaseEntity(entity_type="base")
    miner = MockMinerEntity(race_id=1, color=(0, 255, 0))
    player = MockPlayer()

    # Verify inheritance
    assert isinstance(
        miner, BaseEntity
    ), "MockMinerEntity should be an instance of BaseEntity"
    assert isinstance(
        player, MockMinerEntity
    ), "MockPlayer should be an instance of MockMinerEntity"
    assert isinstance(
        player, BaseEntity
    ), "MockPlayer should be an instance of BaseEntity"

    # Verify entity types
    if (
        base.entity_type != "base"
    ):
        raise AssertionError(f"Expected entity_type 'base', got {base.entity_type}")
    if (
        miner.entity_type != "miner"
    ):
        raise AssertionError(f"Expected entity_type 'miner', got {miner.entity_type}")

    # Verify player-specific attributes
    if player.is_player is not True:
        raise AssertionError("Player should have is_player=True")
    if (
        player.credits != 1000
    ):
        raise AssertionError(f"Player should start with 1000 credits, got {player.credits}")
    if (
        player.trait != "adaptive"
    ):
        raise AssertionError(f"Player should have 'adaptive' trait, got {player.trait}")

    print("Inheritance tests passed!")
    return True


if __name__ == "__main__":
    try:
        verify_base_entity()
        verify_inheritance()
        print("All verification tests passed!")
    except AssertionError as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
