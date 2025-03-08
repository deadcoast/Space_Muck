#!/usr/bin/env python3

import unittest
from enhancement_system import EnhancementSystem
from event_system import EventSystem, EventType, Event

class TestShip:
    """Test class for enhancement demonstration."""
    def __init__(self):
        self.shields = 100
        self.speed = 10
        
    def move(self, distance):
        """Move the ship."""
        return self.speed * distance
        
    def take_damage(self, amount):
        """Take damage to shields."""
        self.shields -= amount
        return self.shields

class TestEnhancementSystem(unittest.TestCase):
    """Test cases for the enhancement system."""
    
    def setUp(self):
        """Set up test cases."""
        self.enhancer = EnhancementSystem()
        self.events = EventSystem()
        self.ship = TestShip()
        
    def test_enhance_method(self):
        """Test enhancing an existing method."""
        @self.enhancer.enhance_method(TestShip, 'move')
        def enhanced_move(self, original_method, distance):
            # Double the speed for testing
            self.speed *= 2
            result = original_method(distance)
            self.speed //= 2
            return result
            
        # Test the enhancement
        self.assertEqual(self.ship.move(5), 100)  # 10 * 2 * 5
        
        # Reset the ship's speed (since it was modified)
        self.ship.speed = 10
        
        # Test rollback
        enhancement_id = "TestShip_move_enhanced"
        self.enhancer.rollback_enhancement(enhancement_id)
        self.assertEqual(self.ship.move(5), 50)  # Back to original: 10 * 5
        
    def test_add_method(self):
        """Test adding a new method."""
        @self.enhancer.add_method(TestShip, 'boost_shields')
        def boost_shields(self, amount):
            self.shields += amount
            return self.shields
            
        # Test the new method
        self.assertEqual(self.ship.boost_shields(50), 150)
        
        # Create a new ship instance to avoid state from previous test
        self.ship = TestShip()
        
        # Test rollback
        enhancement_id = "TestShip_boost_shields_added"
        self.enhancer.rollback_enhancement(enhancement_id)
        self.assertFalse(hasattr(self.ship, 'boost_shields'))
        
    def test_event_handling(self):
        """Test event handling system."""
        events_received = []
        
        def test_handler(event: Event):
            events_received.append(event)
            
        # Register handlers
        self.events.register_handler(EventType.PRE_ENHANCEMENT, test_handler)
        self.events.register_handler(EventType.POST_ENHANCEMENT, test_handler)
        
        # Emit events
        self.events.pre_enhancement('test', method='move')
        self.events.post_enhancement('test', method='move', status='success')
        
        # Verify events
        self.assertEqual(len(events_received), 2)
        self.assertEqual(events_received[0].type, EventType.PRE_ENHANCEMENT)
        self.assertEqual(events_received[1].type, EventType.POST_ENHANCEMENT)
        
    def test_enhancement_tracking(self):
        """Test enhancement tracking."""
        # Add an enhancement
        @self.enhancer.enhance_method(TestShip, 'take_damage')
        def enhanced_damage(self, original_method, amount):
            # Reduce damage by 50%
            return original_method(amount // 2)
            
        # Check enhancement registry
        enhancements = self.enhancer.list_enhancements()
        self.assertEqual(len(enhancements), 1)
        
        # Verify enhancement info
        enhancement = self.enhancer.get_enhancement_info('TestShip_take_damage_enhanced')
        self.assertIsNotNone(enhancement)
        self.assertEqual(enhancement['class_name'], 'TestShip')
        self.assertEqual(enhancement['method_name'], 'take_damage')
        self.assertTrue(enhancement['active'])

if __name__ == '__main__':
    unittest.main()
