# ASCIIBox Event System Integration

This document provides a comprehensive guide to integrating ASCIIBox components with the UI event system.

## Overview

The ASCIIBox event integration allows for:
- Registering ASCIIBox components with the event system
- Handling mouse events (hover, click)
- Emitting custom events
- Creating interactive UI elements

## Basic Usage

### Creating an Interactive ASCIIBox

```python
from ui.ui_element.ascii_box import ASCIIBox
from ui.ui_base.event_system import UIEventType, UIEventData
from ui.ui_helpers.ascii_box_event_helper import register_ascii_box, add_click_handler

# Create a box
box = ASCIIBox(5, 5, 20, 10, "Interactive Box")

# Register with event system
box_id = register_ascii_box(box)

# Define click handler
def on_click(event_data: UIEventData) -> None:
    print(f"Box clicked! Event data: {event_data}")
    # Update box appearance or state
    box.is_focused = True

# Add click handler
add_click_handler(box, on_click)
```

### Using the Helper Functions

The `ascii_box_event_helper` module provides several helper functions to simplify integration:

```python
from ui.ui_helpers.ascii_box_event_helper import create_interactive_box

# Create an interactive box with event handlers
def on_click(event_data):
    print("Box clicked!")

def on_hover_enter(event_data):
    print("Mouse entered box!")

def on_hover_leave(event_data):
    print("Mouse left box!")

# Create box with all handlers in one call
box, box_id = create_interactive_box(
    x=10, 
    y=10, 
    width=30, 
    height=15,
    title="Interactive Box",
    on_click=on_click,
    on_hover_enter=on_hover_enter,
    on_hover_leave=on_hover_leave
)
```

## Event Handling

### Mouse Events

ASCIIBox can handle the following mouse events:
- `MOUSE_ENTER`: Mouse cursor enters the box area
- `MOUSE_LEAVE`: Mouse cursor leaves the box area
- `MOUSE_CLICK`: Mouse click within the box area
- `MOUSE_PRESS`: Mouse button pressed within the box area
- `MOUSE_RELEASE`: Mouse button released within the box area

Example of handling mouse events for multiple boxes:

```python
from ui.ui_helpers.ascii_box_event_helper import handle_mouse_events

# List of ASCIIBox components
boxes = [box1, box2, box3]

# Handle mouse events (e.g., in your main loop)
mouse_pos = (100, 100)  # Mouse position in pixels
char_size = (8, 16)     # Character size in pixels
event_type = "click"    # Event type (hover, click, press, release)

# Process mouse events
handled_by = handle_mouse_events(boxes, mouse_pos, event_type, char_size)
```

### Custom Events

You can also emit and handle custom events:

```python
# Register custom event handler
def on_custom_event(event_data: UIEventData) -> None:
    print(f"Custom event received: {event_data.data}")

box.register_event_handler(UIEventType.CUSTOM, on_custom_event)

# Emit custom event
box.emit_event(UIEventType.CUSTOM, {
    "action": "special_action",
    "value": 42
})
```

## Integration with Component Registry

ASCIIBox components are automatically registered with the component registry when using the helper functions. This allows for:

- Retrieving components by ID
- Managing component state
- Establishing parent-child relationships

Example:

```python
from ui.ui_helpers.ascii_box_event_helper import get_box_by_id

# Get a box by its ID
box = get_box_by_id("box_123")
if box:
    # Update box properties
    box.title = "Updated Title"
```

## Best Practices

1. **Keep Event Handlers Small**: Event handlers should be small, focused functions that delegate to other functions for complex operations.

2. **Avoid Circular Dependencies**: Don't create circular dependencies between components and event handlers.

3. **Use Optional Integration**: The event system integration is designed to be optional. Components should work correctly even without event handling.

4. **Handle Exceptions**: Always handle exceptions in event handlers to prevent crashes.

5. **Clean Up Resources**: Unregister components when they are no longer needed:

```python
from ui.ui_helpers.ascii_box_event_helper import unregister_ascii_box

# Unregister when done
unregister_ascii_box(box)
```

## Example: Creating a Button

Here's a complete example of creating a button using ASCIIBox:

```python
from ui.ui_element.ascii_box import ASCIIBox
from ui.ui_helpers.ascii_box_event_helper import create_interactive_box

def create_button(x, y, width, height, label, action):
    """Create a button with the specified action."""
    
    def on_click(event_data):
        # Call the action function when clicked
        action()
    
    def on_hover_enter(event_data):
        # Show hover effect
        button.is_hovered = True
    
    def on_hover_leave(event_data):
        # Remove hover effect
        button.is_hovered = False
    
    # Create the button
    button, _ = create_interactive_box(
        x=x,
        y=y,
        width=width,
        height=height,
        title=label,
        on_click=on_click,
        on_hover_enter=on_hover_enter,
        on_hover_leave=on_hover_leave
    )
    
    return button

# Usage
def save_action():
    print("Saving data...")

save_button = create_button(10, 10, 20, 3, "Save", save_action)
```

## Testing

The `test_ascii_box_events.py` module provides comprehensive tests for the ASCIIBox event integration. Run these tests to verify that the integration is working correctly.

## See Also

- `ui/examples/ascii_box_event_demo.py`: Demonstrates various aspects of ASCIIBox event integration
- `ui/ui_helpers/event_integration.py`: General event integration helpers
- `ui/ui_base/event_system.py`: Core event system implementation
