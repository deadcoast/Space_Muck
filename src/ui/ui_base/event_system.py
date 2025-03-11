"""
UI Event System for Space Muck UI components.

This module extends the main event system with UI-specific events,
such as focus changes, hover events, and UI state updates. It is designed to be
lightweight and optional for UI components while integrating with the
existing event infrastructure.
"""

import logging
import time
from typing import Dict, List, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass

from ui.event_system import EventType


# Extend the main EventType enum with UI-specific events
class UIEventType(Enum):
    """Types of UI-specific events that can be emitted."""
    
    # Focus events
    FOCUS_GAINED = auto()
    FOCUS_LOST = auto()
    
    # Mouse events
    MOUSE_ENTER = auto()
    MOUSE_LEAVE = auto()
    MOUSE_CLICK = auto()
    MOUSE_PRESS = auto()
    MOUSE_RELEASE = auto()
    
    # Keyboard events
    KEY_PRESS = auto()
    KEY_RELEASE = auto()
    
    # State events
    VISIBILITY_CHANGED = auto()
    ENABLED_CHANGED = auto()
    POSITION_CHANGED = auto()
    SIZE_CHANGED = auto()
    STYLE_CHANGED = auto()
    
    # Content events
    CONTENT_CHANGED = auto()
    SCROLL_CHANGED = auto()
    
    # Component lifecycle events
    COMPONENT_CREATED = auto()
    COMPONENT_INITIALIZED = auto()
    COMPONENT_DESTROYED = auto()
    
    @classmethod
    def to_event_type(cls, ui_event_type):
        """Map UIEventType to main EventType for integration."""
        # Direct mapping of UI events to main event types where applicable
        mapping = {
            cls.FOCUS_GAINED: EventType.CONVERTER_SELECTED,
            cls.PROCESS_STARTED: EventType.PROCESS_STARTED,
            cls.PROCESS_COMPLETED: EventType.PROCESS_COMPLETED,
            cls.PROCESS_CANCELLED: EventType.PROCESS_CANCELLED,
        }
        
        # Return mapped event type if available, otherwise use UI_EVENT
        return mapping.get(ui_event_type, EventType.METRIC_UPDATED)


@dataclass
class UIEventData:
    """Container for UI event data."""
    
    type: UIEventType
    source_id: str
    data: Dict[str, Any]
    timestamp: float = time.time()
    propagation_stopped: bool = False
    
    def stop_propagation(self) -> None:
        """Stop event propagation to parent components."""
        self.propagation_stopped = True


class UIEventSystem:
    """Specialized event system for UI components.
    
    This class provides UI-specific event handling that integrates with
    the main event system and component registry. It is designed to be
    optional for UI components.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'UIEventSystem':
        """Get the singleton instance of UIEventSystem."""
        if cls._instance is None:
            cls._instance = UIEventSystem()
        return cls._instance
    
    def __init__(self):
        """Initialize the UI event system."""
        # Import here to avoid circular imports
        from ui.event_system import EventSystem
        from ui.ui_base.component_registry import ComponentRegistry
        
        # Get references to other systems
        self._main_event_system = EventSystem()
        self._component_registry = ComponentRegistry.get_instance()
        
        # Event handlers by event type
        self._handlers: Dict[UIEventType, Dict[str, List[Callable[[UIEventData], None]]]] = {}
        
        # Component relationships for event bubbling
        self._component_parents: Dict[str, str] = {}
        
        # Global event handlers (receive all events)
        self._global_handlers: List[Callable[[UIEventData], None]] = []
        
        # Event statistics for debugging
        self._event_counts: Dict[UIEventType, int] = {event_type: 0 for event_type in UIEventType}
        
        # Components with active listeners
        self._active_components: Set[str] = set()
    
    def subscribe(
        self, 
        event_type: UIEventType, 
        component_id: str, 
        handler: Callable[[UIEventData], None]
    ) -> None:
        """
        Subscribe to a UI event type for a specific component.
        
        Args:
            event_type: Type of event to subscribe to
            component_id: ID of the component to listen for events from
            handler: Function to call when event occurs
        """
        try:
            if event_type not in self._handlers:
                self._handlers[event_type] = {}
                
            if component_id not in self._handlers[event_type]:
                self._handlers[event_type][component_id] = []
                
            self._handlers[event_type][component_id].append(handler)
            self._active_components.add(component_id)
            
        except Exception as e:
            logging.error(f"Error subscribing to UI event: {e}")
    
    def unsubscribe(
        self, 
        event_type: UIEventType, 
        component_id: str, 
        handler: Callable[[UIEventData], None]
    ) -> bool:
        """
        Unsubscribe from a UI event type for a specific component.
        
        Args:
            event_type: Type of event to unsubscribe from
            component_id: ID of the component
            handler: Handler function to remove
            
        Returns:
            True if the handler was removed, False otherwise
        """
        try:
            if (event_type in self._handlers and 
                component_id in self._handlers[event_type] and
                handler in self._handlers[event_type][component_id]):
                
                self._handlers[event_type][component_id].remove(handler)
                
                # Clean up empty lists
                if not self._handlers[event_type][component_id]:
                    del self._handlers[event_type][component_id]
                    
                # Check if component has any handlers left
                has_handlers = any(component_id in event_handlers for event_handlers in self._handlers.values())
                        
                if not has_handlers:
                    self._active_components.discard(component_id)
                    
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error unsubscribing from UI event: {e}")
            return False
    
    def subscribe_global(self, handler: Callable[[UIEventData], None]) -> None:
        """
        Subscribe to all UI events regardless of type or source.
        
        Args:
            handler: Function to call for all events
        """
        try:
            if handler not in self._global_handlers:
                self._global_handlers.append(handler)
        except Exception as e:
            logging.error(f"Error subscribing to global UI events: {e}")
    
    def unsubscribe_global(self, handler: Callable[[UIEventData], None]) -> bool:
        """
        Unsubscribe from all UI events.
        
        Args:
            handler: Handler function to remove
            
        Returns:
            True if the handler was removed, False otherwise
        """
        try:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
                return True
            return False
        except Exception as e:
            logging.error(f"Error unsubscribing from global UI events: {e}")
            return False
    
    def emit(self, event_type: UIEventType, source_id: str, data: Dict[str, Any]) -> None:
        """
        Emit a UI event.
        
        Args:
            event_type: Type of event
            source_id: ID of the component that generated the event
            data: Event data
        """
        try:
            # Create event data
            event = UIEventData(event_type, source_id, data)
            
            # Update statistics
            self._event_counts[event_type] += 1
            
            # Notify specific handlers
            self._notify_handlers(event)
            
            # Notify global handlers
            for handler in self._global_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logging.error(f"Error in global event handler: {e}")
                    
            # Bubble event to parent components if not stopped
            if not event.propagation_stopped:
                self._bubble_event(event)
                
            # Forward to main event system for integration
            # Import here to avoid circular imports
            from ui.event_system import EventSystem
            
            # Create main event data
            main_data = data.copy()
            main_data['ui_event_type'] = event_type.name
            
            # Map to appropriate main event type
            main_event_type = UIEventType.to_event_type(event_type)
            
            # Emit through main event system
            EventSystem().emit(main_event_type, source_id, main_data)
                
        except Exception as e:
            logging.error(f"Error emitting UI event: {e}")
    
    def _notify_handlers(self, event: UIEventData) -> None:
        """
        Notify handlers for a specific event.
        
        Args:
            event: Event data to pass to handlers
        """
        try:
            if event.type in self._handlers and event.source_id in self._handlers[event.type]:
                for handler in self._handlers[event.type][event.source_id]:
                    try:
                        handler(event)
                        if event.propagation_stopped:
                            break
                    except Exception as e:
                        logging.error(f"Error in UI event handler: {e}")
        except Exception as e:
            logging.error(f"Error notifying UI event handlers: {e}")
    
    def _bubble_event(self, event: UIEventData) -> None:
        """
        Bubble event to parent components.
        
        Args:
            event: Event data to bubble
        """
        try:
            current_id = event.source_id
            
            while current_id in self._component_parents and not event.propagation_stopped:
                parent_id = self._component_parents[current_id]
                
                # Create a new event with the parent as source
                parent_event = UIEventData(
                    event.type,
                    parent_id,
                    event.data,
                    event.timestamp,
                    event.propagation_stopped
                )
                
                # Notify parent's handlers
                self._notify_handlers(parent_event)
                
                # Update propagation status
                event.propagation_stopped = parent_event.propagation_stopped
                
                # Move up to next parent
                current_id = parent_id
                
        except Exception as e:
            logging.error(f"Error bubbling UI event: {e}")
    
    def set_parent_relationship(self, child_id: str, parent_id: str) -> None:
        """
        Set a parent-child relationship for event bubbling.
        
        Args:
            child_id: ID of the child component
            parent_id: ID of the parent component
        """
        try:
            # Set relationship in event system
            self._component_parents[child_id] = parent_id
            
            # Also update component registry if components are registered
            from ui.ui_base.component_registry import ComponentRegistry
            registry = ComponentRegistry.get_instance()
            
            if registry.is_registered(child_id) and registry.is_registered(parent_id):
                # Get components from registry
                child = registry.get_component(child_id)
                parent = registry.get_component(parent_id)
                
                if child and parent:
                    # Update parent-child relationship in registry
                    registry.set_parent(child_id, parent_id)
        except Exception as e:
            logging.error(f"Error setting parent relationship: {e}")
    
    def remove_parent_relationship(self, child_id: str) -> None:
        """
        Remove a parent-child relationship.
        
        Args:
            child_id: ID of the child component
        """
        try:
            if child_id in self._component_parents:
                del self._component_parents[child_id]
        except Exception as e:
            logging.error(f"Error removing parent relationship: {e}")
    
    def get_event_statistics(self) -> Dict[UIEventType, int]:
        """
        Get statistics on event counts.
        
        Returns:
            Dictionary of event types and their counts
        """
        return self._event_counts.copy()
    
    def has_listeners(self, component_id: str) -> bool:
        """
        Check if a component has any active listeners.
        
        Args:
            component_id: ID of the component to check
            
        Returns:
            True if the component has listeners, False otherwise
        """
        return component_id in self._active_components
    
    def clear_component(self, component_id: str) -> None:
        """
        Clear all handlers and relationships for a component.
        
        Args:
            component_id: ID of the component to clear
        """
        try:
            # Remove from active components
            self._active_components.discard(component_id)
            
            # Remove from handlers
            for event_type in self._handlers:
                if component_id in self._handlers[event_type]:
                    del self._handlers[event_type][component_id]
            
            # Remove parent relationship
            self.remove_parent_relationship(component_id)
            
            # Remove any child relationships
            children_to_remove = [child_id for child_id, parent_id in self._component_parents.items() 
                               if parent_id == component_id]
            
            for child_id in children_to_remove:
                self.remove_parent_relationship(child_id)
                
            # Emit component destroyed event
            data = {
                'component_id': component_id,
                'timestamp': time.time()
            }
            self.emit(UIEventType.COMPONENT_DESTROYED, component_id, data)
                
        except Exception as e:
            logging.error(f"Error clearing component: {e}")


# Example usage:
"""
# Get event system instance
event_system = UIEventSystem.get_instance()

# Define an event handler
def on_button_click(event: UIEventData) -> None:
    print(f"Button clicked: {event.source_id}")
    # Access event data
    button_id = event.data.get('button_id')
    mouse_pos = event.data.get('position')

# Subscribe to button click events
event_system.subscribe(UIEventType.MOUSE_CLICK, "button1", on_button_click)

# Emit an event when button is clicked
event_system.emit(
    UIEventType.MOUSE_CLICK,
    "button1",
    {
        'button_id': "button1",
        'position': (10, 20)
    }
)

# Set up parent-child relationship for event bubbling
event_system.set_parent_relationship("button1", "panel1")

# Unsubscribe when no longer needed
event_system.unsubscribe(UIEventType.MOUSE_CLICK, "button1", on_button_click)
"""
