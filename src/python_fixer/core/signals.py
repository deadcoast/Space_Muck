"""
Signal handling utilities for graceful interruption and cleanup.

This module provides a robust signal handling system that supports:
1. Stack-based handler registration for proper cleanup
2. Multiple signal types (SIGINT, SIGTERM, etc.)
3. Context manager support for easy use
4. Proper cleanup and resource management
"""

import logging
import signal
import sys
from contextlib import contextmanager, ExitStack
from typing import Callable, Dict, List, Optional, Set, NoReturn


class SignalManager:
    """Manages signal handlers with a stack-based approach for proper cleanup.

    This class provides a robust way to handle interruption signals (SIGINT, SIGTERM, etc.) with proper
    cleanup and error handling. It supports nested handlers that are executed in LIFO order.

    Attributes:
        supported_signals: Set of signal types supported by this manager

    Example:
        manager = SignalManager()
        with manager.handler(cleanup_func):
            # Do work that might be interrupted
            pass

        # Or register multiple signals
        with manager.handler(cleanup_func, signals=[signal.SIGINT, signal.SIGTERM]):
            # Do work that might be interrupted by multiple signal types
            pass
    """

    # Default supported signals
    DEFAULT_SIGNALS = {signal.SIGINT}

    # Exit codes for different signals
    SIGNAL_EXIT_CODES = {
        signal.SIGINT: 130,  # Standard for SIGINT (Ctrl+C)
        signal.SIGTERM: 143,  # Standard for SIGTERM
        signal.SIGHUP: 129,  # Standard for SIGHUP
    }

    def __init__(self, signals: Optional[Set[int]] = None):
        """Initialize the signal manager.

        Args:
            signals: Optional set of signal types to handle. Defaults to SIGINT only.
        """
        self._handlers: List[Callable[[], None]] = []
        self._original_handlers: Dict[int, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self.supported_signals = signals or self.DEFAULT_SIGNALS
        self._active_signals: Set[int] = set()

    def _handle_signal(self, signum: int, frame) -> NoReturn:
        """Handle interruption signal by executing cleanup handlers in LIFO order.

        Args:
            signum: Signal number that triggered this handler
            frame: Current stack frame (unused)

        Raises:
            SystemExit: Exits with appropriate code based on the signal received
        """
        signal_name = (
            signal.Signals(signum).name
            if hasattr(signal, "Signals")
            else f"signal {signum}"
        )
        self.logger.warning(f"Operation interrupted by {signal_name}")

        # Execute handlers in reverse order (LIFO)
        for handler in reversed(self._handlers):
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Error in signal handler: {e}")
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.exception("Signal handler traceback:")

        # Use appropriate exit code based on signal type
        exit_code = self.SIGNAL_EXIT_CODES.get(signum, 1)
        sys.exit(exit_code)

    def register_signal(self, sig: int) -> None:
        """Register a signal to be handled by this manager.

        Args:
            sig: Signal number to register

        Note:
            This method is typically called internally by the handler method.
        """
        if sig not in self._active_signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
            self._active_signals.add(sig)
            self.logger.debug(
                f"Installed handler for {signal.Signals(sig).name if hasattr(signal, 'Signals') else f'signal {sig}'}"
            )

    def unregister_signal(self, sig: int) -> None:
        """Unregister a signal handler and restore the original.

        Args:
            sig: Signal number to unregister

        Note:
            This method is typically called internally by the handler method.
        """
        if sig in self._active_signals:
            signal.signal(sig, self._original_handlers[sig])
            self._active_signals.remove(sig)
            del self._original_handlers[sig]
            self.logger.debug(
                f"Restored original handler for {signal.Signals(sig).name if hasattr(signal, 'Signals') else f'signal {sig}'}"
            )

    def register_handler(self, cleanup_func: Callable[[], None]) -> None:
        """Register a cleanup handler function.

        Args:
            cleanup_func: Function to call during cleanup. Must take no arguments.
        """
        if cleanup_func not in self._handlers:
            self.logger.debug(f"Registering signal handler: {cleanup_func.__name__}")
            self._handlers.append(cleanup_func)

    def unregister_handler(self, cleanup_func: Callable[[], None]) -> None:
        """Unregister a cleanup handler function.

        Args:
            cleanup_func: Function to remove from the handler stack
        """
        if cleanup_func in self._handlers:
            self.logger.debug(f"Removing signal handler: {cleanup_func.__name__}")
            self._handlers.remove(cleanup_func)

    @contextmanager
    def handler(
        self,
        cleanup_func: Optional[Callable[[], None]] = None,
        signals: Optional[List[int]] = None,
    ):
        """Context manager for registering signal handlers with cleanup.

        Args:
            cleanup_func: Optional function to call during cleanup. Must take no arguments.
            signals: Optional list of signals to handle. Defaults to the supported_signals set.

        Yields:
            None

        Example:
            def cleanup():
                print("Cleaning up...")

            with signal_manager.handler(cleanup):
                # Do work that might be interrupted
                pass

            # Or with multiple signals
            with signal_manager.handler(cleanup, signals=[signal.SIGINT, signal.SIGTERM]):
                # Do work that might be interrupted by multiple signal types
                pass
        """
        # Use ExitStack to ensure proper cleanup even if exceptions occur
        with ExitStack() as stack:
            # Register the cleanup function if provided
            if cleanup_func:
                self.register_handler(cleanup_func)
                # Ensure we unregister on exit
                stack.callback(self.unregister_handler, cleanup_func)

            # Register signals
            signals_to_handle = signals or list(self.supported_signals)
            for sig in signals_to_handle:
                self.register_signal(sig)
                # Ensure we unregister each signal on exit
                stack.callback(self.unregister_signal, sig)

            # Yield control back to the caller
            yield

    def reset(self) -> None:
        """Reset the signal manager to its initial state.

        This method unregisters all handlers and restores original signal handlers.
        Useful for testing or when you need to completely reset the manager.
        """
        # Clear all handlers
        self._handlers.clear()

        # Restore all original signal handlers
        for sig in list(self._active_signals):
            self.unregister_signal(sig)

        self.logger.debug("Signal manager reset to initial state")
