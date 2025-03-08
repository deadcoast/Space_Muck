#!/usr/bin/env python3

# -----------------------------
# ENHANCEMENT SYSTEM
# -----------------------------
#
# Parent: analysis.enhancers
# Dependencies: ast, inspect, importlib, typing, logging
#
# MAP: /project_root/analysis/enhancers
# EFFECT: Provides method enhancement capabilities with state preservation
# NAMING: Enhancement[Type]

import importlib
import logging
from typing import Dict, List, Optional, Type
from functools import wraps


class EnhancementSystem:
    """System for enhancing Python classes and methods while preserving state.

    This system provides capabilities to:
    1. Enhance existing methods without modifying original code
    2. Add new methods to existing classes
    3. Preserve state during enhancements
    4. Track and validate dependencies
    5. Roll back changes if needed
    """

    def __init__(self):
        """Initialize the enhancement system."""
        self.logger = logging.getLogger(__name__)
        self.enhancements = {}  # type: Dict[str, Dict]
        self.original_methods = {}  # type: Dict[str, Dict]
        self.enhancement_registry = {}  # type: Dict[str, Dict]

    def enhance_method(
        self, target_class: Type, method_name: str, enhancement_id: str = None
    ) -> callable:
        """Decorator to enhance an existing method.

        Args:
            target_class: Class containing the method to enhance
            method_name: Name of the method to enhance
            enhancement_id: Optional unique identifier for the enhancement

        Returns:
            Decorator function

        Example:
            @enhancer.enhance_method(PlayerShip, 'move')
            def enhanced_move(self, original_method, *args, **kwargs):
                # Pre-processing
                result = original_method(*args, **kwargs)
                # Post-processing
                return result
        """

        def decorator(enhancement_func: callable) -> callable:
            # Get the original method
            original_method = getattr(target_class, method_name)

            # Store original method for potential rollback
            if target_class.__name__ not in self.original_methods:
                self.original_methods[target_class.__name__] = {}
            self.original_methods[target_class.__name__][method_name] = original_method

            @wraps(original_method)
            def wrapper(self, *args, **kwargs):
                return enhancement_func(
                    self,
                    lambda *a, **k: original_method(self, *a, **k),
                    *args,
                    **kwargs,
                )

            # Apply the enhancement
            setattr(target_class, method_name, wrapper)

            # Register the enhancement
            self._register_enhancement(
                target_class.__name__,
                method_name,
                enhancement_id or f"{target_class.__name__}_{method_name}_enhanced",
            )

            return wrapper

        return decorator

    def add_method(
        self, target_class: Type, method_name: str, enhancement_id: str = None
    ) -> callable:
        """Decorator to add a new method to an existing class.

        Args:
            target_class: Class to add the method to
            method_name: Name of the new method
            enhancement_id: Optional unique identifier for the enhancement

        Returns:
            Decorator function

        Example:
            @enhancer.add_method(PlayerShip, 'boost_shields')
            def boost_shields(self, amount):
                self.shields += amount
        """

        def decorator(new_method: callable) -> callable:
            if hasattr(target_class, method_name):
                self.logger.warning(
                    f"Method {method_name} already exists in {target_class.__name__}"
                )
                return new_method

            # Add the new method to the class
            setattr(target_class, method_name, new_method)

            # Register the enhancement
            self._register_enhancement(
                target_class.__name__,
                method_name,
                enhancement_id or f"{target_class.__name__}_{method_name}_added",
                is_new=True,
            )

            return new_method

        return decorator

    def _register_enhancement(
        self,
        class_name: str,
        method_name: str,
        enhancement_id: str,
        is_new: bool = False,
    ) -> None:
        """Register an enhancement in the system.

        Args:
            class_name: Name of the enhanced class
            method_name: Name of the enhanced method
            enhancement_id: Unique identifier for the enhancement
            is_new: Whether this is a new method (True) or enhanced method (False)
        """
        if enhancement_id in self.enhancement_registry:
            self.logger.warning(f"Enhancement ID {enhancement_id} already exists")
            return

        self.enhancement_registry[enhancement_id] = {
            "class_name": class_name,
            "method_name": method_name,
            "is_new": is_new,
            "active": True,
        }

    def rollback_enhancement(self, enhancement_id: str) -> bool:
        """Roll back a specific enhancement.

        Args:
            enhancement_id: ID of the enhancement to roll back

        Returns:
            True if rollback successful, False otherwise
        """
        if enhancement_id not in self.enhancement_registry:
            self.logger.error(f"Enhancement {enhancement_id} not found")
            return False

        enhancement = self.enhancement_registry[enhancement_id]
        if not enhancement["active"]:
            self.logger.warning(f"Enhancement {enhancement_id} already rolled back")
            return False

        try:
            class_name = enhancement["class_name"]
            method_name = enhancement["method_name"]

            target_class = self._get_target_class(class_name)

            if enhancement["is_new"]:
                # Remove added method
                delattr(target_class, method_name)
            else:
                # Restore original method
                original = self.original_methods[class_name][method_name]
                setattr(target_class, method_name, original)

            # Mark as inactive
            enhancement["active"] = False
            return True

        except Exception as e:
            self.logger.error(
                f"Error rolling back enhancement {enhancement_id}: {str(e)}"
            )
            return False

    def list_enhancements(self, active_only: bool = True) -> List[Dict]:
        """List all registered enhancements.

        Args:
            active_only: Only list active enhancements if True

        Returns:
            List of enhancement information dictionaries
        """
        return [
            {"id": enhancement_id, **info}
            for enhancement_id, info in self.enhancement_registry.items()
            if not active_only or info["active"]
        ]

    def _get_target_class(self, class_name: str) -> Type:
        """Get a class by name from the main module.

        Args:
            class_name: Name of the class to get

        Returns:
            The target class

        Raises:
            AttributeError: If class not found
        """
        module = importlib.import_module("__main__")
        return getattr(module, class_name)

    def get_enhancement_info(self, enhancement_id: str) -> Optional[Dict]:
        """Get detailed information about a specific enhancement.

        Args:
            enhancement_id: ID of the enhancement

        Returns:
            Dictionary with enhancement information or None if not found
        """
        return self.enhancement_registry.get(enhancement_id)
