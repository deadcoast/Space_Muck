"""
Verification script for Fleet class.
"""

# Standard library imports
import os
import sys
from unittest.mock import MagicMock

# Local application imports
from entities.base_entity import BaseEntity
from entities.fleet import Fleet

# Third-party library imports


# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import entities after adding src to path

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


def verify_fleet():
    """Test the Fleet class."""
    print("Testing Fleet class...")

    # Create a basic fleet
    fleet = Fleet(
        fleet_id="fleet-001",
        owner_id="player-001",
        fleet_name="Alpha Squadron",
        color=(100, 150, 255),
        position=(50, 50),
    )

    # Verify basic attributes
    assert (
        fleet.entity_type == "fleet"
    ), f"Expected entity_type 'fleet', got {fleet.entity_type}"
    assert (
        fleet.entity_id == "fleet-001"
    ), f"Expected entity_id 'fleet-001', got {fleet.entity_id}"
    assert (
        fleet.owner_id == "player-001"
    ), f"Expected owner_id 'player-001', got {fleet.owner_id}"
    assert (
        fleet.fleet_name == "Alpha Squadron"
    ), f"Expected fleet_name 'Alpha Squadron', got {fleet.fleet_name}"
    assert fleet.color == (
        100,
        150,
        255,
    ), f"Expected color (100, 150, 255), got {fleet.color}"
    assert fleet.position == (
        50,
        50,
    ), f"Expected position (50, 50), got {fleet.position}"
    assert fleet.ships == {}, "Fleet should start with empty ships dictionary"

    # Test adding ships
    fleet.add_ship("ship-001", {"type": "fighter", "strength": 10})
    fleet.add_ship("ship-002", {"type": "cruiser", "strength": 25})

    assert len(fleet.ships) == 2, f"Expected 2 ships, got {len(fleet.ships)}"
    assert "ship-001" in fleet.ships, "ship-001 should be in the fleet"
    assert fleet.ships["ship-001"]["type"] == "fighter", "ship-001 should be a fighter"

    # Test removing ships
    result = fleet.remove_ship("ship-001")
    assert result is True, "remove_ship should return True for existing ship"
    assert "ship-001" not in fleet.ships, "ship-001 should be removed from the fleet"
    assert (
        len(fleet.ships) == 1
    ), f"Expected 1 ship after removal, got {len(fleet.ships)}"

    # Test fleet strength calculation
    assert (
        fleet.get_fleet_strength() == 25
    ), f"Expected fleet strength 25, got {fleet.get_fleet_strength()}"

    # Test setting destination and path calculation
    fleet.set_destination(60, 60)
    assert fleet.destination == (
        60,
        60,
    ), f"Expected destination (60, 60), got {fleet.destination}"
    assert fleet.is_moving is True, "Fleet should be moving after setting destination"
    assert len(fleet.path) > 0, "Fleet should have a path after setting destination"

    # Test serialization
    data = fleet.to_dict()
    assert (
        data["entity_type"] == "fleet"
    ), f"Expected entity_type 'fleet', got {data['entity_type']}"
    assert (
        data["fleet_name"] == "Alpha Squadron"
    ), f"Expected fleet_name 'Alpha Squadron', got {data['fleet_name']}"
    assert (
        data["owner_id"] == "player-001"
    ), f"Expected owner_id 'player-001', got {data['owner_id']}"

    # Test deserialization
    new_fleet = Fleet.from_dict(data)
    assert (
        new_fleet.entity_type == fleet.entity_type
    ), "Entity types should match after deserialization"
    assert (
        new_fleet.fleet_name == fleet.fleet_name
    ), "Fleet names should match after deserialization"
    assert (
        new_fleet.owner_id == fleet.owner_id
    ), "Owner IDs should match after deserialization"

    print("Fleet tests passed!")
    return True


def verify_inheritance():
    """Test that Fleet inherits from BaseEntity."""
    print("Testing Fleet inheritance...")

    # Create a fleet
    fleet = Fleet()

    # Verify inheritance
    assert isinstance(fleet, BaseEntity), "Fleet should be an instance of BaseEntity"

    # Test inherited methods
    fleet.add_tag("important")
    assert fleet.has_tag(
        "important"
    ), "Fleet should have tag 'important' (inherited method)"

    fleet.set_position(10, 20)
    assert fleet.get_position() == (
        10,
        20,
    ), f"Expected position (10, 20), got {fleet.get_position()} (inherited method)"

    fleet.deactivate()
    assert (
        not fleet.is_active()
    ), "Fleet should be inactive after deactivate() (inherited method)"

    fleet.activate()
    assert (
        fleet.is_active()
    ), "Fleet should be active after activate() (inherited method)"

    print("Fleet inheritance tests passed!")
    return True


if __name__ == "__main__":
    try:
        verify_fleet()
        verify_inheritance()
        print("All verification tests passed!")
    except AssertionError as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
