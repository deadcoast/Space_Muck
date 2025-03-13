"""
Tests for the signal handling utilities.
"""

# Standard library imports

# Third-party library imports

# Local application imports
from python_fixer.core.signals import SignalManager
from unittest.mock import patch, MagicMock
import signal
import unittest

class TestSignalManager(unittest.TestCase):
    """Test cases for the SignalManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a signal manager with a mock logger
        self.signal_manager = SignalManager()
        self.signal_manager.logger = MagicMock()

        # Save original signal handlers to restore after tests
        self.original_handlers = {}
        for sig in [signal.SIGINT, signal.SIGTERM]:
            self.original_handlers[sig] = signal.getsignal(sig)

    def tearDown(self):
        """Tear down test fixtures."""
        # Reset the signal manager
        self.signal_manager.reset()

        # Restore original signal handlers
        for sig, handler in self.original_handlers.items():
            signal.signal(sig, handler)

    def test_init_default_signals(self):
        """Test that SignalManager initializes with default signals."""
        manager = SignalManager()
        self.assertEqual(manager.supported_signals, {signal.SIGINT})
        self.assertEqual(manager._handlers, [])
        self.assertEqual(manager._active_signals, set())

    def test_init_custom_signals(self):
        """Test that SignalManager initializes with custom signals."""
        custom_signals = {signal.SIGINT, signal.SIGTERM}
        manager = SignalManager(signals=custom_signals)
        self.assertEqual(manager.supported_signals, custom_signals)

    def test_register_handler(self):
        """Test registering a handler function."""

        def handler():
            pass

        self.signal_manager.register_handler(handler)
        self.assertIn(handler, self.signal_manager._handlers)
        self.signal_manager.logger.debug.assert_called_once()

    def test_register_handler_duplicate(self):
        """Test that registering the same handler twice only adds it once."""

        def handler():
            pass

        self.signal_manager.register_handler(handler)
        self.signal_manager.register_handler(handler)
        self.assertEqual(self.signal_manager._handlers.count(handler), 1)

    def test_unregister_handler(self):
        """Test unregistering a handler function."""

        def handler():
            pass

        self.signal_manager.register_handler(handler)
        self.signal_manager.unregister_handler(handler)
        self.assertNotIn(handler, self.signal_manager._handlers)
        self.signal_manager.logger.debug.assert_called()

    def test_unregister_nonexistent_handler(self):
        """Test unregistering a handler that was never registered."""

        def handler():
            pass

        self.signal_manager.unregister_handler(handler)
        self.assertNotIn(handler, self.signal_manager._handlers)
        # Should not log anything for nonexistent handler
        self.signal_manager.logger.debug.assert_not_called()

    @patch("signal.signal")
    def test_register_signal(self, mock_signal):
        """Test registering a signal."""
        sig = signal.SIGINT
        self.signal_manager.register_signal(sig)
        mock_signal.assert_called_once()
        self.assertIn(sig, self.signal_manager._active_signals)
        self.signal_manager.logger.debug.assert_called_once()

    @patch("signal.signal")
    def test_register_signal_duplicate(self, mock_signal):
        """Test that registering the same signal twice only registers it once."""
        sig = signal.SIGINT
        self.signal_manager.register_signal(sig)
        mock_signal.reset_mock()
        self.signal_manager.register_signal(sig)
        mock_signal.assert_not_called()

    @patch("signal.signal")
    def test_unregister_signal(self, mock_signal):
        """Test unregistering a signal."""
        sig = signal.SIGINT
        self.signal_manager.register_signal(sig)
        mock_signal.reset_mock()
        self.signal_manager.unregister_signal(sig)
        mock_signal.assert_called_once()
        self.assertNotIn(sig, self.signal_manager._active_signals)
        self.assertNotIn(sig, self.signal_manager._original_handlers)

    @patch("signal.signal")
    def test_unregister_nonexistent_signal(self, mock_signal):
        """Test unregistering a signal that was never registered."""
        sig = signal.SIGINT
        self.signal_manager.unregister_signal(sig)
        mock_signal.assert_not_called()

    @patch("sys.exit")
    def test_handle_signal(self, mock_exit):
        """Test the signal handler function."""
        # Register handlers
        handler1 = MagicMock()
        handler2 = MagicMock()
        self.signal_manager.register_handler(handler1)
        self.signal_manager.register_handler(handler2)

        # Call the signal handler
        self.signal_manager._handle_signal(signal.SIGINT, None)

        # Verify handlers were called in LIFO order
        handler2.assert_called_once()
        handler1.assert_called_once()

        # Verify exit was called with the right code
        mock_exit.assert_called_once_with(130)

    @patch("sys.exit")
    def test_handle_signal_with_error(self, mock_exit):
        """Test the signal handler function when a handler raises an exception."""
        # Register handlers
        handler1 = MagicMock()
        handler2 = MagicMock(side_effect=Exception("Test error"))
        self.signal_manager.register_handler(handler1)
        self.signal_manager.register_handler(handler2)

        # Call the signal handler
        self.signal_manager._handle_signal(signal.SIGINT, None)

        # Verify both handlers were called despite the error
        handler2.assert_called_once()
        handler1.assert_called_once()

        # Verify error was logged
        self.signal_manager.logger.error.assert_called_once()

        # Verify exit was called with the right code
        mock_exit.assert_called_once_with(130)

    def test_context_manager_with_handler(self):
        """Test the context manager with a cleanup handler."""
        handler = MagicMock()

        with patch("signal.signal") as mock_signal:
            with self.signal_manager.handler(handler):
                self.assertIn(handler, self.signal_manager._handlers)
                mock_signal.assert_called()

            # After context, handler should be removed
            self.assertNotIn(handler, self.signal_manager._handlers)

    def test_context_manager_with_custom_signals(self):
        """Test the context manager with custom signals."""
        handler = MagicMock()
        signals = [signal.SIGINT, signal.SIGTERM]

        with patch("signal.signal") as mock_signal:
            with self.signal_manager.handler(handler, signals=signals):
                self.assertIn(handler, self.signal_manager._handlers)
                # Should be called twice, once for each signal
                self.assertEqual(mock_signal.call_count, 2)

            # After context, handler should be removed
            self.assertNotIn(handler, self.signal_manager._handlers)

    def test_context_manager_without_handler(self):
        """Test the context manager without a cleanup handler."""
        with patch("signal.signal") as mock_signal:
            with self.signal_manager.handler():
                mock_signal.assert_called()

            # After context, signal handlers should still be active
            # but will be cleaned up in tearDown

    def test_reset(self):
        """Test resetting the signal manager."""
        # Register handlers and signals
        handler = MagicMock()
        self.signal_manager.register_handler(handler)
        self.signal_manager.register_signal(signal.SIGINT)

        # Reset the manager
        with patch("signal.signal") as mock_signal:
            self.signal_manager.reset()

            # Verify everything was cleared
            self.assertEqual(self.signal_manager._handlers, [])
            self.assertEqual(self.signal_manager._active_signals, set())
            mock_signal.assert_called()
            self.signal_manager.logger.debug.assert_called()

if __name__ == "__main__":
    unittest.main()
