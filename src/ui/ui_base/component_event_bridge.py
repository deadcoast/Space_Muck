"""
Component Event Bridge.

Minimal bridge between component registry and event system that enables
UI events without modifying existing components.

This module follows strict principles of modularity and optional integration.
It connects the component registry with the event system without forcing
dependencies on either system.
"""

# Standard library imports
import logging
import time

# Third-party library imports

# Local application imports
from src.ui.ui_base.component_registry import ComponentRegistry
from src.ui.ui_base.event_system import UIEventSystem


def connect_systems() -> bool:
    """
    Connect component registry with event system.

    This function sets up the minimal connections needed between the
    component registry and event system without modifying either system.
    It follows the principle of optional integration - components can
    function without it, but gain additional capabilities when connected.

    Returns:
        True if systems were successfully connected, False otherwise
    """
    try:
        registry = ComponentRegistry.get_instance()

        # Register component lifecycle callbacks
        registry.on_component_registered = _on_component_registered
        registry.on_component_unregistered = _on_component_unregistered
        registry.on_parent_changed = _on_parent_changed

        logging.info("Successfully connected component registry and event system")
        return True
    except Exception as e:
        logging.error(f"Error connecting component registry and event system: {e}")
        return False


def _on_component_registered(component_id: str, component_type: str) -> None:
    """
    Forward component registration to event system.

    Args:
        component_id: ID of the registered component
        component_type: Type name of the registered component
    """
    try:
        # Emit component created event
        UIEventSystem.get_instance().emit(
            UIEventType.COMPONENT_CREATED,
            component_id,
            {"component_type": component_type, "timestamp": time.time()},
        )
        logging.debug(f"Component registered event emitted for {component_id}")
    except Exception as e:
        logging.error(f"Error in component registration event: {e}")


def _on_component_unregistered(component_id: str) -> None:
    """
    Clean up component from event system when unregistered.

    Args:
        component_id: ID of the unregistered component
    """
    try:
        # First clear any event handlers for this component
        UIEventSystem.get_instance().clear_component(component_id)

        # Then emit the destroyed event for any listeners
        UIEventSystem.get_instance().emit(
            UIEventType.COMPONENT_DESTROYED, component_id, {"timestamp": time.time()}
        )
        logging.debug(f"Component unregistered event emitted for {component_id}")
    except Exception as e:
        logging.error(f"Error in component unregistration event: {e}")


def _on_parent_changed(child_id: str, parent_id: str) -> None:
    """
    Update event bubbling when component hierarchy changes.

    Args:
        child_id: ID of the child component
        parent_id: ID of the parent component
    """
    try:
        # Update parent relationship in event system for event bubbling
        UIEventSystem.get_instance().set_parent_relationship(child_id, parent_id)
        logging.debug(f"Updated parent relationship: {child_id} -> {parent_id}")
    except Exception as e:
        logging.error(f"Error in parent relationship event: {e}")
