"""
Component Registry for Space Muck UI.

This module provides a centralized registry for UI components, allowing
for component lifecycle management, lookup, and event propagation.
The registry is designed to be optional - components can function without it,
but gain additional capabilities when registered.
"""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from enum import Enum, auto
from typing import Dict, List, Set, Optional, TypeVar, Type
from src.ui.ui_base.ascii_base import UIElement
import weakref

# Type variable for generic component types
T = TypeVar("T", bound=UIElement)


class ComponentState(Enum):
    """Possible states of a UI component in the registry."""

    CREATED = auto()  # Component has been created but not initialized
    INITIALIZED = auto()  # Component has been initialized
    ACTIVE = auto()  # Component is active and visible
    INACTIVE = auto()  # Component exists but is not currently visible
    DISABLED = auto()  # Component exists but is disabled from interaction
    DESTROYED = auto()  # Component is marked for removal


class ComponentRegistry:
    """Registry for UI components with lifecycle management.

    This class provides a central registry for UI components, allowing for
    component lookup, state management, and event propagation. Components
    can register themselves to gain these capabilities, but registration
    is entirely optional.
    """

    # Singleton instance
    _instance = None

    @classmethod
    def get_instance(cls) -> "ComponentRegistry":
        """Get the singleton instance of ComponentRegistry."""
        if cls._instance is None:
            cls._instance = ComponentRegistry()
        return cls._instance

    def __init__(self):
        """Initialize the component registry."""
        # Main registry of components by ID
        self._components: Dict[str, UIElement] = {}

        # Registry of components by type
        self._components_by_type: Dict[Type[UIElement], List[UIElement]] = {}

        # Component states
        self._component_states: Dict[str, ComponentState] = {}

        # Component relationships (parent-child)
        self._component_children: Dict[str, Set[str]] = {}
        self._component_parent: Dict[str, Optional[str]] = {}

        # Weak references to avoid memory leaks
        self._weak_refs: Dict[str, weakref.ref] = {}

        # Event callbacks (optional)
        self.on_component_registered = None
        self.on_component_unregistered = None
        self.on_parent_changed = None

    def register(self, component: UIElement, component_id: Optional[str] = None) -> str:
        """Register a component with the registry.

        Args:
            component: The UI component to register
            component_id: Optional ID for the component. If not provided,
                         a unique ID will be generated.

        Returns:
            The ID of the registered component
        """
        try:
            # Generate ID if not provided
            if component_id is None:
                component_id = f"{component.__class__.__name__}_{id(component)}"

            # Register component
            self._components[component_id] = component

            # Register by type
            component_type = type(component)
            if component_type not in self._components_by_type:
                self._components_by_type[component_type] = []
            self._components_by_type[component_type].append(component)

            # Set initial state
            self._component_states[component_id] = ComponentState.CREATED

            # Initialize relationship tracking
            self._component_children[component_id] = set()
            self._component_parent[component_id] = None

            # Create weak reference to help with cleanup
            self._weak_refs[component_id] = weakref.ref(
                component, lambda _: self._cleanup_component(component_id)
            )

            # Notify via callback if registered
            if self.on_component_registered:
                try:
                    self.on_component_registered(
                        component_id, component.__class__.__name__
                    )
                except Exception as e:
                    logging.error(f"Error in component registered callback: {e}")

            logging.debug(f"Registered component {component_id}")
            return component_id

        except Exception as e:
            logging.error(f"Error registering component: {e}")
            return f"error_{id(component)}"

    def _remove_from_type_registry(self, component: UIElement) -> None:
        """Remove a component from the type registry.

        Args:
            component: The component to remove
        """
        component_type = type(component)
        if component_type in self._components_by_type:
            self._components_by_type[component_type].remove(component)
            # Clean up empty type entries
            if not self._components_by_type[component_type]:
                del self._components_by_type[component_type]

    def _handle_relationship_cleanup(self, component_id: str) -> None:
        """Clean up parent-child relationships for a component.

        Args:
            component_id: ID of the component being unregistered
        """
        # Orphan all children
        if component_id in self._component_children:
            for child_id in self._component_children[component_id]:
                if child_id in self._component_parent:
                    self._component_parent[child_id] = None
            del self._component_children[component_id]

        # Remove from parent's children list
        parent_id = self._component_parent.get(component_id)
        if parent_id and parent_id in self._component_children:
            self._component_children[parent_id].discard(component_id)

        # Remove from parent registry
        if component_id in self._component_parent:
            del self._component_parent[component_id]

    def _notify_unregistration(self, component_id: str) -> None:
        """Notify listeners that a component has been unregistered.

        Args:
            component_id: ID of the unregistered component
        """
        if self.on_component_unregistered:
            try:
                self.on_component_unregistered(component_id)
            except Exception as e:
                logging.error(f"Error in component unregistered callback: {e}")

    def unregister(self, component_id: str) -> bool:
        """Unregister a component from the registry.

        Args:
            component_id: ID of the component to unregister

        Returns:
            True if the component was unregistered, False otherwise
        """
        try:
            # Check if component exists
            if component_id not in self._components:
                logging.warning(f"Component {component_id} not found in registry")
                return False

            # Get component before removing
            component = self._components[component_id]

            # Remove from main registry
            del self._components[component_id]

            # Remove from type registry
            self._remove_from_type_registry(component)

            # Remove state
            if component_id in self._component_states:
                del self._component_states[component_id]

            # Notify listeners
            self._notify_unregistration(component_id)

            # Clean up relationships
            self._handle_relationship_cleanup(component_id)

            # Remove weak reference
            if component_id in self._weak_refs:
                del self._weak_refs[component_id]

            logging.debug(f"Unregistered component {component_id}")
            return True

        except Exception as e:
            logging.error(f"Error unregistering component: {e}")
            return False

    def get_component(self, component_id: str) -> Optional[UIElement]:
        """Get a component by ID.

        Args:
            component_id: ID of the component to get

        Returns:
            The component if found, None otherwise
        """
        try:
            return self._components.get(component_id)
        except Exception as e:
            logging.error(f"Error getting component: {e}")
            return None

    def get_components_by_type(self, component_type: Type[T]) -> List[T]:
        """Get all components of a specific type.

        Args:
            component_type: Type of components to get

        Returns:
            List of components of the specified type
        """
        try:
            return self._components_by_type.get(component_type, [])
        except Exception as e:
            logging.error(f"Error getting components by type: {e}")
            return []

    def set_component_state(self, component_id: str, state: ComponentState) -> bool:
        """Set the state of a component.

        Args:
            component_id: ID of the component
            state: New state for the component

        Returns:
            True if the state was set, False otherwise
        """
        try:
            if component_id not in self._components:
                logging.warning(f"Component {component_id} not found in registry")
                return False

            self._component_states[component_id] = state

            # Handle special state transitions
            # Mark DESTROYED components for unregistration in the next update cycle

            return True

        except Exception as e:
            logging.error(f"Error setting component state: {e}")
            return False

    def get_component_state(self, component_id: str) -> Optional[ComponentState]:
        """Get the state of a component.

        Args:
            component_id: ID of the component

        Returns:
            The component state if found, None otherwise
        """
        try:
            return self._component_states.get(component_id)
        except Exception as e:
            logging.error(f"Error getting component state: {e}")
            return None

    def set_parent_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Set a parent-child relationship between components.

        Args:
            parent_id: ID of the parent component
            child_id: ID of the child component

        Returns:
            True if the relationship was set, False otherwise
        """
        try:
            if parent_id not in self._components:
                logging.warning(f"Parent component {parent_id} not found in registry")
                return False

            if child_id not in self._components:
                logging.warning(f"Child component {child_id} not found in registry")
                return False

            # Remove child from previous parent
            prev_parent_id = self._component_parent.get(child_id)
            if prev_parent_id and prev_parent_id in self._component_children:
                self._component_children[prev_parent_id].discard(child_id)

            # Set new relationship
            self._component_parent[child_id] = parent_id
            if parent_id not in self._component_children:
                self._component_children[parent_id] = set()
            self._component_children[parent_id].add(child_id)

            # Notify via callback if registered
            if self.on_parent_changed:
                try:
                    self.on_parent_changed(child_id, parent_id)
                except Exception as e:
                    logging.error(f"Error in parent changed callback: {e}")

            return True

        except Exception as e:
            logging.error(f"Error setting parent-child relationship: {e}")
            return False

    def get_children(self, parent_id: str) -> List[str]:
        """Get the IDs of all children of a component.

        Args:
            parent_id: ID of the parent component

        Returns:
            List of child component IDs
        """
        try:
            return list(self._component_children.get(parent_id, set()))
        except Exception as e:
            logging.error(f"Error getting children: {e}")
            return []

    def get_parent(self, child_id: str) -> Optional[str]:
        """Get the ID of a component's parent.

        Args:
            child_id: ID of the child component

        Returns:
            ID of the parent component if found, None otherwise
        """
        try:
            return self._component_parent.get(child_id)
        except Exception as e:
            logging.error(f"Error getting parent: {e}")
            return None

    def update(self) -> None:
        """Update the registry, cleaning up destroyed components."""
        try:
            # Find components marked for destruction
            to_unregister = [
                component_id
                for component_id, state in self._component_states.items()
                if state == ComponentState.DESTROYED
            ]

            # Unregister destroyed components
            for component_id in to_unregister:
                self.unregister(component_id)

        except Exception as e:
            logging.error(f"Error updating component registry: {e}")

    def _cleanup_component(self, component_id: str) -> None:
        """Clean up a component when its weak reference is garbage collected.

        Args:
            component_id: ID of the component to clean up
        """
        try:
            if component_id in self._components:
                logging.debug(
                    f"Cleaning up component {component_id} via weak reference"
                )
                self.unregister(component_id)
        except Exception as e:
            logging.error(f"Error cleaning up component: {e}")

    def is_registered(self, component_id: str) -> bool:
        """Check if a component ID is registered in the registry.

        Args:
            component_id: ID of the component to check

        Returns:
            True if the component is registered, False otherwise
        """
        try:
            return component_id in self._components
        except Exception as e:
            logging.error(f"Error checking if component is registered: {e}")
            return False
