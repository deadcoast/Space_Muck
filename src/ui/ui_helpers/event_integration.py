"""
Event Integration Helper.

This module provides helper functions to integrate UI components with the event system
without forcing dependencies. It follows strict principles of modularity and
optional integration.
"""

# Standard library imports
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type

# Local application imports
from src.ui.ui_base.ascii_base import UIElement
from src.ui.ui_base.component_registry import ComponentRegistry, ComponentState
from src.ui.ui_base.event_system import UIEventData, UIEventSystem, UIEventType

# Third-party library imports


def register_with_events(
    component: UIElement, component_id: Optional[str] = None
) -> str:
    """
    Register a component with both registry and event system.

    This is a convenience function that registers a component with the
    component registry, which will automatically trigger event system
    integration if the systems are connected.

    Args:
        component: The UI component to register
        component_id: Optional ID for the component. If not provided,
                     a unique ID will be generated.

    Returns:
        The ID of the registered component
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.register(component, component_id)
    except Exception as e:
        logging.error(f"Error registering component with events: {e}")
        return ""


def subscribe_to_event(
    component_id: str, event_type: UIEventType, handler: Callable[[UIEventData], None]
) -> bool:
    """
    Subscribe a component to specific UI events.

    Args:
        component_id: ID of the component to subscribe
        event_type: Type of event to subscribe to
        handler: Function to call when event occurs

    Returns:
        True if subscription was successful, False otherwise
    """
    try:
        event_system = UIEventSystem.get_instance()
        event_system.subscribe(event_type, component_id, handler)
        return True
    except Exception as e:
        logging.error(f"Error subscribing to events: {e}")
        return False


def emit_ui_event(
    source_id: str, event_type: UIEventType, data: Dict[str, Any]
) -> bool:
    """
    Emit a UI event from a component.

    Args:
        source_id: ID of the component emitting the event
        event_type: Type of event to emit
        data: Event data

    Returns:
        True if event was emitted successfully, False otherwise
    """
    try:
        # Add timestamp if not provided
        if "timestamp" not in data:
            data["timestamp"] = time.time()

        event_system = UIEventSystem.get_instance()
        event_system.emit(event_type, source_id, data)
        return True
    except Exception as e:
        logging.error(f"Error emitting UI event: {e}")
        return False


def get_component_by_id(component_id: str) -> Optional[UIElement]:
    """
    Get a component by its ID from the registry.

    Args:
        component_id: ID of the component to retrieve

    Returns:
        The component if found, None otherwise
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.get_component(component_id)
    except Exception as e:
        logging.error(f"Error getting component by ID: {e}")
        return None


def get_components_by_type(component_type: Type[UIElement]) -> List[UIElement]:
    """
    Get all components of a specific type from the registry.

    Args:
        component_type: Type of components to retrieve

    Returns:
        List of components of the specified type
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.get_components_by_type(component_type)
    except Exception as e:
        logging.error(f"Error getting components by type: {e}")
        return []


def unsubscribe_from_event(
    component_id: str, event_type: UIEventType, handler: Callable[[UIEventData], None]
) -> bool:
    """
    Unsubscribe a component from specific UI events.

    Args:
        component_id: ID of the component to unsubscribe
        event_type: Type of event to unsubscribe from
        handler: Handler function to remove

    Returns:
        True if unsubscription was successful, False otherwise
    """
    try:
        event_system = UIEventSystem.get_instance()
        return event_system.unsubscribe(event_type, component_id, handler)
    except Exception as e:
        logging.error(f"Error unsubscribing from events: {e}")
        return False


def unregister_from_events(component_id: str) -> bool:
    """
    Unregister a component from the component registry and event system.

    This will remove the component from the registry and unsubscribe it from all events.

    Args:
        component_id: ID of the component to unregister

    Returns:
        True if unregistration was successful, False otherwise
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.unregister(component_id)
    except Exception as e:
        logging.error(f"Error unregistering component from events: {e}")
        return False


def set_component_state(component_id: str, state: ComponentState) -> bool:
    """
    Set the state of a component in the registry.

    Args:
        component_id: ID of the component
        state: New state for the component

    Returns:
        True if the state was set, False otherwise
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.set_component_state(component_id, state)
    except Exception as e:
        logging.error(f"Error setting component state: {e}")
        return False


def set_parent_child_relationship(parent_id: str, child_id: str) -> bool:
    """
    Set a parent-child relationship between components.

    This relationship is used for event bubbling and component hierarchy.

    Args:
        parent_id: ID of the parent component
        child_id: ID of the child component

    Returns:
        True if the relationship was set, False otherwise
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.set_parent_child_relationship(parent_id, child_id)
    except Exception as e:
        logging.error(f"Error setting parent-child relationship: {e}")
        return False


def unregister_component(component_id: str) -> bool:
    """
    Unregister a component from the registry and event system.

    Args:
        component_id: ID of the component to unregister

    Returns:
        True if unregistration was successful, False otherwise
    """
    try:
        registry = ComponentRegistry.get_instance()
        return registry.unregister(component_id)
    except Exception as e:
        logging.error(f"Error unregistering component: {e}")
        return False


def initialize_event_integration() -> bool:
    """
    Initialize the connection between component registry and event system.

    This function ensures that the component registry and event system
    are properly connected for event propagation.

    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        from ui.ui_base.ui_system_init import initialize_ui_systems

        return initialize_ui_systems()
    except Exception as e:
        logging.error(f"Error initializing event integration: {e}")
        return False


def get_event_statistics() -> Dict[UIEventType, int]:
    """
    Get statistics on event counts.

    Returns:
        Dictionary of event types and their counts
    """
    try:
        event_system = UIEventSystem.get_instance()
        return event_system.get_event_statistics()
    except Exception as e:
        logging.error(f"Error getting event statistics: {e}")
        return {}


def is_registered_with_events(component_id: str) -> bool:
    """
    Check if a component is registered with the event system.

    Args:
        component_id: ID of the component to check

    Returns:
        True if the component is registered, False otherwise
    """
    try:
        # First check if component exists in registry
        registry = ComponentRegistry.get_instance()
        if not registry.is_registered(component_id):
            return False

        # Then check if it has any active event listeners
        event_system = UIEventSystem.get_instance()
        return event_system.has_listeners(component_id)
    except Exception as e:
        logging.debug(f"Error checking if component is registered with events: {e}")
        return False
