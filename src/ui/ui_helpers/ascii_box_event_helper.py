"""
Helper module for integrating ASCIIBox components with the UI event system.

This module provides utility functions for registering ASCIIBox components with
the event system and handling common events. It is designed to be optional and
maintain backward compatibility with existing code.
"""

import logging
from typing import (
    Optional,
    Any,
    Callable,
    List,
    Tuple,
)  # Removed Dict, Union as they're unused
import uuid

# Import UI components with fallback for backward compatibility
try:
    from ui.ui_element.ascii_box import ASCIIBox
    from ui.ui_base.event_system import (
        UIEventType,
        UIEventData,
    )  # Removed UIEventSystem as it's unused
    from ui.ui_base.component_registry import ComponentRegistry
    from ui.ui_helpers.event_integration import (
        register_with_events,
        unregister_from_events,
    )
except ImportError as e:
    logging.warning(f"Could not import UI event system components: {e}")
    # Define dummy types for type checking
    ASCIIBox = Any
    UIEventType = Any
    UIEventData = Any


def register_ascii_box(
    box: ASCIIBox, component_id: Optional[str] = None, register_events: bool = True
) -> str:
    """
    Register an ASCIIBox with the component registry and event system.

    Args:
        box: ASCIIBox component to register
        component_id: Optional component ID (generated if not provided)
        register_events: Whether to register with the event system

    Returns:
        Component ID of the registered box
    """
    try:
        # Generate component ID if not provided
        box_id = component_id or f"ascii_box_{uuid.uuid4().hex[:8]}"

        # Set component ID on the box
        box.component_id = box_id

        # Register with component registry and event system if requested
        if register_events:
            try:
                return register_with_events(box, box_id)
            except (ImportError, AttributeError) as e:
                logging.debug(f"Event system not available for registration: {e}")
                return box_id

        # Otherwise just return the ID
        return box_id

    except Exception as e:
        logging.error(f"Error registering ASCIIBox: {e}")
        return component_id or f"ascii_box_{uuid.uuid4().hex[:8]}"


def unregister_ascii_box(box: ASCIIBox) -> bool:
    """
    Unregister an ASCIIBox from the component registry and event system.

    Args:
        box: ASCIIBox component to unregister

    Returns:
        True if unregistered successfully, False otherwise
    """
    try:
        # Check if box has a component ID
        if not hasattr(box, "component_id") or not box.component_id:
            return False

        # Unregister from event system
        try:
            return unregister_from_events(box.component_id)
        except (ImportError, AttributeError) as e:
            logging.debug(f"Event system not available for unregistration: {e}")
            return False

    except Exception as e:
        logging.error(f"Error unregistering ASCIIBox: {e}")
        return False


def add_click_handler(box: ASCIIBox, handler: Callable[[UIEventData], None]) -> bool:
    """
    Add a click event handler to an ASCIIBox.

    Args:
        box: ASCIIBox component to add handler to
        handler: Function to call when box is clicked

    Returns:
        True if handler was added successfully, False otherwise
    """
    try:
        # Make box clickable
        box.set_clickable(True)

        # Register handler for click events
        box.register_event_handler(UIEventType.MOUSE_CLICK, handler)
        return True

    except Exception as e:
        logging.error(f"Error adding click handler: {e}")
        return False


def add_hover_handlers(
    box: ASCIIBox,
    enter_handler: Optional[Callable[[UIEventData], None]] = None,
    leave_handler: Optional[Callable[[UIEventData], None]] = None,
) -> bool:
    """
    Add hover event handlers to an ASCIIBox.

    Args:
        box: ASCIIBox component to add handlers to
        enter_handler: Function to call when mouse enters box
        leave_handler: Function to call when mouse leaves box

    Returns:
        True if handlers were added successfully, False otherwise
    """
    try:
        # Register enter handler if provided
        if enter_handler:
            box.register_event_handler(UIEventType.MOUSE_ENTER, enter_handler)

        # Register leave handler if provided
        if leave_handler:
            box.register_event_handler(UIEventType.MOUSE_LEAVE, leave_handler)

        return True
    except Exception as e:
        logging.error(f"Error adding hover handlers: {e}")
        return False


def create_interactive_box(
    x: int,
    y: int,
    width: int,
    height: int,
    title: Optional[str] = None,
    on_click: Optional[Callable[[UIEventData], None]] = None,
    on_hover_enter: Optional[Callable[[UIEventData], None]] = None,
    on_hover_leave: Optional[Callable[[UIEventData], None]] = None,
    component_id: Optional[str] = None,
    **kwargs,
) -> Tuple[ASCIIBox, str]:
    """
    Create an interactive ASCIIBox with event handlers.

    Args:
        x: X coordinate of the box
        y: Y coordinate of the box
        width: Width of the box
        height: Height of the box
        title: Optional title for the box
        on_click: Optional click handler
        on_hover_enter: Optional hover enter handler
        on_hover_leave: Optional hover leave handler
        component_id: Optional component ID
        **kwargs: Additional arguments to pass to ASCIIBox constructor

    Returns:
        Tuple of (ASCIIBox, component_id)
    """
    try:
        # Create box
        box = ASCIIBox(x, y, width, height, title, component_id=component_id, **kwargs)

        # Register with event system
        box_id = register_ascii_box(box, component_id)

        # Add event handlers
        if on_click:
            add_click_handler(box, on_click)

        if on_hover_enter or on_hover_leave:
            add_hover_handlers(box, on_hover_enter, on_hover_leave)

        return box, box_id

    except Exception as e:
        logging.error(f"Error creating interactive box: {e}")
        # Create a fallback box without event handling
        box = ASCIIBox(x, y, width, height, title, **kwargs)
        return box, component_id or f"ascii_box_{uuid.uuid4().hex[:8]}"


def handle_mouse_events(
    boxes: List[ASCIIBox],
    mouse_pos: Tuple[int, int],
    mouse_event: str,
    char_size: Tuple[int, int],
) -> List[str]:
    """
    Handle mouse events for multiple ASCIIBox components.

    Args:
        boxes: List of ASCIIBox components to handle events for
        mouse_pos: Mouse position in pixels
        mouse_event: Type of mouse event ('hover', 'click', 'press', 'release')
        char_size: Size of a character in pixels (width, height)

    Returns:
        List of component IDs that handled the event
    """
    try:
        handled_by = []

        # Process boxes in reverse order (top to bottom)
        for box in reversed(boxes):
            if box.handle_mouse_event(mouse_event, mouse_pos, char_size):
                handled_by.append(box.component_id)

                # For click events, stop after first handler
                if mouse_event in ("click", "press", "release"):
                    break

        return handled_by

    except Exception as e:
        logging.error(f"Error handling mouse events: {e}")
        return []


def get_box_by_id(component_id: str) -> Optional[ASCIIBox]:
    """
    Get an ASCIIBox by its component ID.

    Args:
        component_id: ID of the component to get

    Returns:
        ASCIIBox if found, None otherwise
    """
    try:
        # Try to get component from registry
        registry = ComponentRegistry.get_instance()
        component = registry.get_component(component_id)

        # Check if component is an ASCIIBox
        return component if isinstance(component, ASCIIBox) else None
    except (ImportError, AttributeError):
        # Registry not available
        logging.debug("Component registry not available")
        return None
    except Exception as e:
        logging.error(f"Error getting box by ID: {e}")
        return None
