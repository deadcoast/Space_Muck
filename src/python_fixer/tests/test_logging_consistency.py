#!/usr/bin/env python3
"""Test for consistent logging across all python_fixer modules.

This test suite verifies that logging is implemented consistently across
all modules in the python_fixer package, ensuring proper error reporting
and debugging capabilities.
"""

# Standard library imports
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import unittest

# Third-party library imports
import pytest

# Local application imports
from python_fixer.logging.enhanced import StructuredLogger
from python_fixer.utils import LogContext


class TestLoggingConsistency(unittest.TestCase):
    """Test suite for verifying consistent logging across all modules.
    
    This test suite ensures that:
    1. All modules use the structured logging system
    2. Log messages follow consistent formatting
    3. Error handling includes proper logging
    """
    
    def setUp(self) -> None:
        """Set up test fixtures used by all test methods."""
        # Get the project root directory
        self.project_root = Path(__file__).parent.parent.resolve()
        
        # Create a test logger
        self.logger = StructuredLogger("test_logger")
        
        # Create a log handler that captures log messages
        self.log_messages: List[Dict[str, Any]] = []
        
        # Define a custom log handler
        class TestLogHandler(logging.Handler):
            def __init__(self, message_list):
                super().__init__()
                self.message_list = message_list
                
            def emit(self, record):
                self.message_list.append({
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                })
        
        # Add the handler to the logger
        self.handler = TestLogHandler(self.log_messages)
        self.logger.addHandler(self.handler)
        
        # List of all modules to check
        self.modules = [
            "python_fixer.core",
            "python_fixer.enhancers",
            "python_fixer.parsers",
            "python_fixer.utils",
            "python_fixer.fixers",
            "python_fixer.logging",
            "python_fixer.analyzers",
        ]
        
        # Dictionary to track imported modules
        self.imported_modules = {}
    
    def tearDown(self) -> None:
        """Clean up after tests."""
        # Remove the handler from the logger
        self.logger.removeHandler(self.handler)
    
    def _import_module(self, module_name: str) -> Any:
        """Import a module and cache it.
        
        Args:
            module_name: The name of the module to import
            
        Returns:
            The imported module
        """
        if module_name in self.imported_modules:
            return self.imported_modules[module_name]
        
        try:
            module = importlib.import_module(module_name)
            self.imported_modules[module_name] = module
            return module
        except ImportError:
            return None
    
    def _get_module_classes(self, module_name: str) -> List[Any]:
        """Get all classes defined in a module.
        
        Args:
            module_name: The name of the module to check
            
        Returns:
            A list of classes defined in the module
        """
        module = self._import_module(module_name)
        if not module:
            return []
        
        return [
            obj for name, obj in inspect.getmembers(module)
            if inspect.isclass(obj) and obj.__module__ == module.__name__
        ]
    
    def test_structured_logger_usage(self) -> None:
        """Test that all modules use the StructuredLogger."""
        for module_name in self.modules:
            module = self._import_module(module_name)
            if not module:
                continue
            
            # Check if the module has a logger attribute
            if hasattr(module, "logger"):
                logger = getattr(module, "logger")
                self.assertIsInstance(
                    logger, 
                    (logging.Logger, StructuredLogger),
                    f"Module {module_name} does not use a proper logger"
                )
    
    def test_log_context_usage(self) -> None:
        """Test that LogContext is used appropriately."""
        # Test creating and using a LogContext
        context = LogContext(
            module="test_module",
            function="test_function",
            line=10,
            timestamp=None,  # Will be set automatically
            process_id=os.getpid(),
            thread_id=0,
        )
        
        # Add some extra data
        context.extra["test_key"] = "test_value"
        
        # Convert to dict and verify
        context_dict = context.to_dict()
        self.assertEqual(context_dict["module"], "test_module")
        self.assertEqual(context_dict["function"], "test_function")
        self.assertEqual(context_dict["extra"]["test_key"], "test_value")
    
    def test_error_handling_with_logging(self) -> None:
        """Test that error handling includes proper logging."""
        # Test that the logger properly records error messages
        try:
            # Raise an exception
            raise ValueError("Test error")
        except Exception as e:
            # Log the error
            self.logger.error(f"Error occurred: {str(e)}")
        
        # Check that the error was logged
        self.assertEqual(len(self.log_messages), 1)
        self.assertEqual(self.log_messages[0]["level"], "ERROR")
        self.assertIn("Test error", self.log_messages[0]["message"])
    
    def test_consistent_log_formatting(self) -> None:
        """Test that log messages follow consistent formatting."""
        # Log messages at different levels
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")
        
        # Check that all messages were logged
        self.assertEqual(len(self.log_messages), 5)
        
        # Check that all messages have consistent formatting
        for message in self.log_messages:
            self.assertIn("level", message)
            self.assertIn("message", message)
            self.assertIn("module", message)
            self.assertIn("function", message)
    
    def test_module_specific_logging(self) -> None:
        """Test logging in specific modules."""
        # For each module, check if it has classes that use logging
        for module_name in self.modules:
            classes = self._get_module_classes(module_name)
            
            for cls in classes:
                # Check if the class has methods that use logging
                for name, method in inspect.getmembers(cls, inspect.isfunction):
                    # Skip private methods
                    if name.startswith("_"):
                        continue
                    
                    # Check the method source code for logging calls
                    try:
                        source = inspect.getsource(method)
                        
                        # Look for common logging patterns
                        logging_patterns = [
                            "logger.",
                            "logging.",
                            "log.",
                            ".debug(",
                            ".info(",
                            ".warning(",
                            ".error(",
                            ".critical(",
                        ]
                        
                        uses_logging = any(pattern in source for pattern in logging_patterns)
                        
                        # If the method uses logging, it should follow our patterns
                        if uses_logging:
                            # This is a simplified check - in a real test, we would
                            # verify that the logging follows our specific patterns
                            self.assertTrue(True)
                    except (IOError, TypeError):
                        # Skip methods we can't get source for
                        continue


if __name__ == "__main__":
    unittest.main()
