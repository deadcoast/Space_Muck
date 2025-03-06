#!/usr/bin/env python3
"""
Dependency Injection framework for Space Muck.

This module provides a simple dependency injection container and decorators
to facilitate better testability and decoupling of components.
"""


import contextlib
import inspect
from typing import Any, Type, get_type_hints


class DependencyContainer:
    """
    A container for managing dependencies in the application.

    This container allows registering implementations for interfaces,
    and resolving dependencies at runtime.
    """

    def __init__(self):
        """Initialize an empty dependency container."""
        self._registry = {}
        self._instance_cache = {}

    def register(
        self, interface: Type, implementation: Any, singleton: bool = False
    ) -> None:
        """
        Register an implementation for an interface.

        Args:
            interface: The interface or abstract class
            implementation: The concrete implementation or factory function
            singleton: If True, the implementation will be treated as a singleton
        """
        self._registry[interface] = {
            "implementation": implementation,
            "singleton": singleton,
        }

        # Clear the instance cache for this interface if it exists
        if interface in self._instance_cache:
            del self._instance_cache[interface]

    def resolve(self, interface: Type) -> Any:
        """
        Resolve a dependency.

        Args:
            interface: The interface to resolve

        Returns:
            An instance of the registered implementation

        Raises:
            KeyError: If no implementation is registered for the interface
        """
        if interface not in self._registry:
            raise KeyError(f"No implementation registered for {interface.__name__}")

        # Check if we have a cached instance for a singleton
        if interface in self._instance_cache:
            return self._instance_cache[interface]

        registration = self._registry[interface]
        implementation = registration["implementation"]

        # If the implementation is a class, instantiate it with dependencies
        if inspect.isclass(implementation):
            instance = self._instantiate_with_dependencies(implementation)
        # If it's a factory function, call it
        elif callable(implementation):
            instance = implementation()
        # Otherwise, it's a direct instance
        else:
            instance = implementation

        # Cache the instance if it's a singleton
        if registration["singleton"]:
            self._instance_cache[interface] = instance

        return instance

    def _instantiate_with_dependencies(self, cls: Type) -> Any:
        """
        Instantiate a class with its dependencies automatically resolved.

        Args:
            cls: The class to instantiate

        Returns:
            An instance of the class with dependencies injected
        """
        # Get constructor parameters
        signature = inspect.signature(cls.__init__)
        parameters = signature.parameters

        # Skip 'self' parameter
        parameters = list(parameters.values())[1:]

        # Get type hints to determine what dependencies to inject
        type_hints = get_type_hints(cls.__init__)

        # Prepare arguments for the constructor
        kwargs = {}

        for param in parameters:
            # Skip parameters with default values if we can't resolve them
            if param.name in type_hints and param.default is inspect.Parameter.empty:
                param_type = type_hints[param.name]
                try:
                    kwargs[param.name] = self.resolve(param_type)
                except KeyError as e:
                    # If we can't resolve a required dependency, raise an error
                    raise KeyError(
                        f"Cannot resolve dependency {param.name} of type {param_type.__name__} for {cls.__name__}"
                    ) from e

        # Instantiate the class with resolved dependencies
        return cls(**kwargs)


# Create a global container instance
container = DependencyContainer()


def inject(cls: Type) -> Type:
    """
    Decorator to inject dependencies into a class.

    This decorator modifies the class's __init__ method to automatically
    inject dependencies from the container.

    Args:
        cls: The class to decorate

    Returns:
        The decorated class
    """
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Get constructor parameters
        signature = inspect.signature(original_init)
        parameters = signature.parameters

        # Skip 'self' parameter
        parameters = list(parameters.values())[1:]

        # Get type hints to determine what dependencies to inject
        type_hints = get_type_hints(original_init)

        # Prepare arguments for the constructor
        injected_kwargs = {}

        for param in parameters:
            # If the parameter is provided in kwargs, use that
            if param.name in kwargs:
                continue

            # If the parameter has a default value and is not provided, skip it
            if (
                param.default is not inspect.Parameter.empty
                and param.name not in kwargs
            ):
                continue

            # Try to inject the dependency if it has a type hint
            if param.name in type_hints:
                param_type = type_hints[param.name]
                with contextlib.suppress(KeyError):
                    injected_kwargs[param.name] = container.resolve(param_type)
        # Update kwargs with injected dependencies
        kwargs.update(injected_kwargs)

        # Call the original __init__
        original_init(self, *args, **kwargs)

    # Replace the __init__ method
    cls.__init__ = new_init

    return cls


def provides(interface: Type, singleton: bool = False):
    """
    Decorator to register a class as an implementation of an interface.

    Args:
        interface: The interface or abstract class
        singleton: If True, the implementation will be treated as a singleton

    Returns:
        A decorator function
    """

    def decorator(cls: Type) -> Type:
        container.register(interface, cls, singleton)
        return cls

    return decorator
