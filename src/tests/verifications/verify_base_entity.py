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
sys.modules[
    "src.algorithms.symbiote_algorithm"
].SymbioteEvolutionAlgorithm = MagicMock()
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
    assert entity.entity_type == "test", (
        f"Expected entity_type 'test', got {entity.entity_type}"
    )
    assert entity.color == (
        255,
        0,
        0,
    ), f"Expected color (255, 0, 0), got {entity.color}"

    # Check that position is not None before comparing
    assert entity.position is not None, "Entity position should not be None"
    if entity.position:  # Type guard for position
        assert entity.position == (
            10,
            20,
        ), f"Expected position (10, 20), got {entity.position}"

    assert entity.active is True, f"Expected active True, got {entity.active}"

    # Test methods
    entity.add_tag("important")
    assert entity.has_tag("important"), "Tag 'important' should be present"

    entity.set_position(30, 40)

    # Check that position is not None before comparing
    position = entity.get_position()
    assert position is not None, "Entity position should not be None after set_position"
    if position:  # Type guard for position
        assert position == (
            30,
            40,
        ), f"Expected position (30, 40), got {position}"

    entity.deactivate()
    assert not entity.is_active(), "Entity should be inactive after deactivate()"

    entity.activate()
    assert entity.is_active(), "Entity should be active after activate()"

    # Test serialization
    data = entity.to_dict()
    assert data["entity_type"] == "test", (
        f"Expected entity_type 'test', got {data['entity_type']}"
    )

    new_entity = BaseEntity.from_dict(data)
    assert new_entity.entity_type == entity.entity_type, (
        "Entity types should match after deserialization"
    )

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
    assert isinstance(miner, BaseEntity), (
        "MockMinerEntity should be an instance of BaseEntity"
    )
    assert isinstance(player, MockMinerEntity), (
        "MockPlayer should be an instance of MockMinerEntity"
    )
    assert isinstance(player, BaseEntity), (
        "MockPlayer should be an instance of BaseEntity"
    )

    # Verify entity types
    assert base.entity_type == "base", (
        f"Expected entity_type 'base', got {base.entity_type}"
    )
    assert miner.entity_type == "miner", (
        f"Expected entity_type 'miner', got {miner.entity_type}"
    )

    # Verify player-specific attributes
    assert player.is_player is True, "Player should have is_player=True"
    assert player.credits == 1000, (
        f"Player should start with 1000 credits, got {player.credits}"
    )
    assert player.trait == "adaptive", (
        f"Player should have 'adaptive' trait, got {player.trait}"
    )

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
