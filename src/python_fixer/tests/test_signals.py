"""Tests for signal handling utilities."""

import signal
from unittest.mock import patch

import pytest

from python_fixer.core.signals import SignalManager


@pytest.fixture
def signal_manager():
    """Fixture to provide a clean SignalManager instance."""
    return SignalManager()


@pytest.fixture
def mock_logger():
    """Fixture to provide a mock logger."""
    with patch("logging.getLogger") as mock:
        yield mock.return_value


def test_signal_handler_registration(signal_manager, mock_logger):
    """Test that signal handlers are properly registered and cleaned up."""
    cleanup_called = False

    def cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    # Test handler registration
    with signal_manager.handler(cleanup):
        assert len(signal_manager._handlers) == 1
        assert signal_manager._original_handler is not None
        mock_logger.debug.assert_called_with("Registering signal handler: cleanup")

        # Simulate SIGINT
        try:
            signal_manager._handle_signal(signal.SIGINT, None)
        except SystemExit as e:
            assert e.code == 130
            assert cleanup_called
            mock_logger.warning.assert_called_with(
                "Operation cancelled by user (signal 2)"
            )

    # Test cleanup after context
    assert len(signal_manager._handlers) == 0
    assert signal_manager._original_handler is None
    mock_logger.debug.assert_called_with("Restored original SIGINT handler")


def test_nested_handlers(signal_manager, mock_logger):
    """Test that nested handlers are executed in LIFO order."""
    call_order = []

    def make_handler(name):
        def handler():
            call_order.append(name)

        return handler

    # Test nested handlers
    with signal_manager.handler(make_handler("outer")):
        mock_logger.debug.assert_called_with("Registering signal handler: handler")
        with signal_manager.handler(make_handler("inner")):
            assert len(signal_manager._handlers) == 2
            mock_logger.debug.assert_called_with("Registering signal handler: handler")

            # Simulate SIGINT
            try:
                signal_manager._handle_signal(signal.SIGINT, None)
            except SystemExit as e:
                assert e.code == 130
                # Inner handler should be called first
                assert call_order == ["inner", "outer"]
                mock_logger.warning.assert_called_with(
                    "Operation cancelled by user (signal 2)"
                )


def test_handler_error_handling(signal_manager, mock_logger):
    """Test that errors in handlers are caught and don't prevent other handlers."""
    cleanup_called = False

    def bad_handler():
        raise RuntimeError("Handler failed")

    def good_handler():
        nonlocal cleanup_called
        cleanup_called = True

    with signal_manager.handler(bad_handler):
        with signal_manager.handler(good_handler):
            # Enable debug logging to test exception logging
            mock_logger.isEnabledFor.return_value = True

            # Simulate SIGINT
            try:
                signal_manager._handle_signal(signal.SIGINT, None)
            except SystemExit as e:
                assert e.code == 130
                assert cleanup_called  # Good handler should still be called

                # Verify error logging
                mock_logger.error.assert_called_with(
                    "Error in signal handler: Handler failed"
                )
                mock_logger.exception.assert_called_with("Signal handler traceback:")


def test_handler_without_cleanup(signal_manager, mock_logger):
    """Test that handler works correctly without a cleanup function."""
    with signal_manager.handler():
        assert len(signal_manager._handlers) == 0
        assert signal_manager._original_handler is not None

        # Simulate SIGINT
        try:
            signal_manager._handle_signal(signal.SIGINT, None)
        except SystemExit as e:
            assert e.code == 130
            mock_logger.warning.assert_called_with(
                "Operation cancelled by user (signal 2)"
            )
