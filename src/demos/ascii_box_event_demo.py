"""
ASCIIBox Event System Integration Demo.

This module demonstrates how to use the ASCIIBox with the UI event system.
It provides practical examples of event registration, handling, and emission.
"""

# Standard library imports
import logging
import time

# Third-party library imports

# Local application imports
from typing import List, Tuple
from ui.ui_base.event_system import UIEventType, UIEventData
from ui.ui_element.ascii_box import ASCIIBox
from ui.ui_helpers.ascii_box_event_helper import (
    register_ascii_box,
    add_click_handler,
    add_hover_handlers,
    create_interactive_box,
)


def demo_basic_event_integration() -> ASCIIBox:
    """
    Demonstrate basic event integration with ASCIIBox.

    Returns:
        An ASCIIBox with event handlers registered
    """
    # Create a basic box
    box = ASCIIBox(5, 5, 20, 5, "Event Demo")

    # Register with event system
    box_id = register_ascii_box(box)
    print(f"Registered box with ID: {box_id}")

    # Define event handlers
    def on_click(event_data: UIEventData) -> None:
        print(f"Box clicked! Event data: {event_data}")
        # Change box title to show it was clicked
        box.title = "Clicked!"
        # Emit a custom event
        box.emit_event(
            UIEventType.CUSTOM, {"action": "box_clicked", "time": time.time()}
        )

    def on_hover_enter(event_data: UIEventData) -> None:
        print(f"Mouse entered box! Event data: {event_data}")
        # Show hover effect
        box.is_hovered = True

    def on_hover_leave(event_data: UIEventData) -> None:
        print(f"Mouse left box! Event data: {event_data}")
        # Remove hover effect
        box.is_hovered = False

    # Register event handlers
    add_click_handler(box, on_click)
    add_hover_handlers(box, on_hover_enter, on_hover_leave)

    return box


def demo_interactive_box_creation() -> Tuple[ASCIIBox, str]:
    """
    Demonstrate creating an interactive box with event handlers.

    Returns:
        Tuple of (box, box_id)
    """

    # Define event handlers
    def on_click(event_data: UIEventData) -> None:
        print(f"Interactive box clicked! Event data: {event_data}")

    def on_hover_enter(event_data: UIEventData) -> None:
        print("Mouse entered interactive box!")

    def on_hover_leave(event_data: UIEventData) -> None:
        print("Mouse left interactive box!")

    # Create interactive box with event handlers
    box, box_id = create_interactive_box(
        x=30,
        y=5,
        width=25,
        height=5,
        title="Interactive Box",
        on_click=on_click,
        on_hover_enter=on_hover_enter,
        on_hover_leave=on_hover_leave,
    )

    print(f"Created interactive box with ID: {box_id}")
    return box, box_id


def demo_multiple_boxes() -> List[ASCIIBox]:
    """
    Demonstrate creating multiple interactive boxes.

    Returns:
        List of ASCIIBox instances
    """
    boxes = []

    # Create several boxes with different event handlers
    for i in range(3):
        # Define event handlers with box index
        def make_click_handler(idx):
            def handler(event_data):
                print(f"Box {idx} clicked! Event data: {event_data}")

            return handler

        def make_hover_enter_handler(idx):
            def handler(event_data):
                print(f"Mouse entered box {idx}!")

            return handler

        def make_hover_leave_handler(idx):
            def handler(event_data):
                print(f"Mouse left box {idx}!")

            return handler

        # Create box
        box, _ = create_interactive_box(
            x=5,
            y=15 + (i * 7),
            width=20,
            height=5,
            title=f"Box {i + 1}",
            on_click=make_click_handler(i),
            on_hover_enter=make_hover_enter_handler(i),
            on_hover_leave=make_hover_leave_handler(i),
        )

        boxes.append(box)

    return boxes


def demo_custom_event_handling() -> ASCIIBox:
    """
    Demonstrate custom event handling with ASCIIBox.

    Returns:
        An ASCIIBox with custom event handlers
    """
    # Create a box
    box = ASCIIBox(40, 15, 30, 10, "Custom Events")

    # Register with event system
    register_ascii_box(box)

    # Define custom event type handler
    def on_custom_event(event_data: UIEventData) -> None:
        print(f"Custom event received! Data: {event_data}")
        # Update box based on custom event
        if "message" in event_data.data:
            box.title = f"Message: {event_data.data['message']}"

    # Register custom event handler
    box.register_event_handler(UIEventType.CUSTOM, on_custom_event)

    # Define click handler that emits custom event
    def on_click(event_data: UIEventData) -> None:
        print("Emitting custom event...")
        box.emit_event(
            UIEventType.CUSTOM, {"message": "Hello from box!", "timestamp": time.time()}
        )

    # Register click handler
    add_click_handler(box, on_click)

    return box


def run_demo() -> None:
    """
    Run the full ASCIIBox event system integration demo.
    """
    print("Starting ASCIIBox Event System Integration Demo")
    print("-" * 50)

    try:
        # Create demo boxes
        basic_box = demo_basic_event_integration()
        interactive_box, _ = demo_interactive_box_creation()
        multiple_boxes = demo_multiple_boxes()
        custom_event_box = demo_custom_event_handling()

        # Collect all boxes
        all_boxes = [basic_box, interactive_box] + multiple_boxes + [custom_event_box]

        print("\nDemo boxes created successfully!")
        print(f"Created {len(all_boxes)} interactive boxes")
        print("\nTo interact with these boxes:")
        print("1. Use mouse events to hover over and click boxes")
        print("2. Watch for event messages in the console")
        print("3. Observe visual changes in the boxes")

        # Note: In a real application, you would now enter the main loop
        # and handle mouse events using handle_mouse_events()

    except Exception as e:
        logging.error(f"Error in demo: {e}")
        print(f"Demo failed with error: {e}")


if __name__ == "__main__":
    run_demo()
