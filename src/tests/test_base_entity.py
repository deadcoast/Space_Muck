#!/usr/bin/env python3
"""
Unit tests for the BaseEntity class.
"""

import unittest
import sys
import os
import uuid
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the class to test
from entities.base_entity import BaseEntity


class TestBaseEntity(unittest.TestCase):
    """Test cases for the BaseEntity class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a basic entity for testing
        self.entity = BaseEntity(
            entity_id="test-123",
            entity_type="test_entity",
            color=(255, 0, 0),
            position=(10, 20),
        )

    def test_initialization(self):
        """Test that entity initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.entity.entity_id, "test-123")
        self.assertEqual(self.entity.entity_type, "test_entity")
        self.assertEqual(self.entity.color, (255, 0, 0))
        self.assertEqual(self.entity.position, (10, 20))
        self.assertTrue(self.entity.active)
        self.assertEqual(self.entity.health, 100)
        self.assertEqual(self.entity.max_health, 100)
        self.assertEqual(self.entity.level, 1)
        self.assertEqual(self.entity.tags, set())

        # Test with default values
        default_entity = BaseEntity()
        self.assertIsNotNone(default_entity.entity_id)
        self.assertEqual(default_entity.entity_type, "generic")
        self.assertEqual(default_entity.color, (255, 255, 255))
        self.assertIsNone(default_entity.position)
        self.assertTrue(default_entity.active)

    def test_uuid_generation(self):
        """Test that a UUID is generated when entity_id is not provided."""
        entity = BaseEntity()
        # Verify that the entity_id is a valid UUID string
        try:
            uuid_obj = uuid.UUID(entity.entity_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False

        self.assertTrue(is_valid_uuid)

    def test_activation_methods(self):
        """Test activation and deactivation methods."""
        self.assertTrue(self.entity.is_active())

        self.entity.deactivate()
        self.assertFalse(self.entity.is_active())

        self.entity.activate()
        self.assertTrue(self.entity.is_active())

    def test_position_methods(self):
        """Test position getter and setter methods."""
        # Test initial position
        self.assertEqual(self.entity.get_position(), (10, 20))

        # Test setting a new position
        self.entity.set_position(30, 40)
        self.assertEqual(self.entity.get_position(), (30, 40))

        # Create entity with no position
        entity_no_pos = BaseEntity()
        self.assertIsNone(entity_no_pos.get_position())

    def test_tag_methods(self):
        """Test tag management methods."""
        # Initially no tags
        self.assertFalse(self.entity.has_tag("important"))

        # Add a tag
        self.entity.add_tag("important")
        self.assertTrue(self.entity.has_tag("important"))

        # Add another tag
        self.entity.add_tag("test")
        self.assertTrue(self.entity.has_tag("test"))

        # Check that both tags exist
        self.assertEqual(self.entity.tags, {"important", "test"})

    def test_update_method(self):
        """Test the base update method."""
        # The base update method does nothing, so we just ensure it doesn't raise exceptions
        game_state = {"tick": 100, "entities": {}}
        try:
            self.entity.update(game_state)
            update_succeeded = True
        except Exception:
            update_succeeded = False

        self.assertTrue(update_succeeded)

    def test_serialization(self):
        """Test serialization to and from dictionary."""
        # Add some tags for testing
        self.entity.add_tag("important")
        self.entity.add_tag("test")

        # Convert to dictionary
        entity_dict = self.entity.to_dict()

        # Verify dictionary contents
        self.assertEqual(entity_dict["entity_id"], "test-123")
        self.assertEqual(entity_dict["entity_type"], "test_entity")
        self.assertEqual(entity_dict["color"], (255, 0, 0))
        self.assertEqual(entity_dict["position"], (10, 20))
        self.assertTrue(entity_dict["active"])
        self.assertEqual(entity_dict["health"], 100)
        self.assertEqual(entity_dict["max_health"], 100)
        self.assertEqual(entity_dict["level"], 1)
        self.assertCountEqual(entity_dict["tags"], ["important", "test"])

        # Create a new entity from the dictionary
        new_entity = BaseEntity.from_dict(entity_dict)

        # Verify the new entity has the same attributes
        self.assertEqual(new_entity.entity_id, "test-123")
        self.assertEqual(new_entity.entity_type, "test_entity")
        self.assertEqual(new_entity.color, (255, 0, 0))
        self.assertEqual(new_entity.position, (10, 20))
        self.assertTrue(new_entity.active)
        self.assertEqual(new_entity.health, 100)
        self.assertEqual(new_entity.max_health, 100)
        self.assertEqual(new_entity.level, 1)
        self.assertEqual(new_entity.tags, {"important", "test"})

    def test_from_dict_with_missing_values(self):
        """Test creating an entity from a dictionary with missing values."""
        # Dictionary with minimal data
        minimal_dict = {"entity_id": "minimal-123"}

        # Create entity from minimal dictionary
        entity = BaseEntity.from_dict(minimal_dict)

        # Verify default values are used for missing fields
        self.assertEqual(entity.entity_id, "minimal-123")
        self.assertEqual(entity.entity_type, "generic")
        self.assertEqual(entity.color, (255, 255, 255))
        self.assertIsNone(entity.position)
        self.assertTrue(entity.active)
        self.assertEqual(entity.health, 100)
        self.assertEqual(entity.max_health, 100)
        self.assertEqual(entity.level, 1)
        self.assertEqual(entity.tags, set())


if __name__ == "__main__":
    unittest.main()
