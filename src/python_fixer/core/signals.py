"""
Signal handling utilities for graceful interruption and cleanup.
"""

import logging
import signal
import sys
from contextlib import contextmanager
from typing import Callable, List, Optional, NoReturn

class SignalManager:
    """Manages signal handlers with a stack-based approach for proper cleanup.
    
    This class provides a robust way to handle interruption signals (SIGINT) with proper
    cleanup and error handling. It supports nested handlers that are executed in LIFO order.
    
    Example:
        manager = SignalManager()
        with manager.handler(cleanup_func):
            # Do work that might be interrupted
            pass
    """
    
    def __init__(self):
        """Initialize the signal manager."""
        self._handlers: List[Callable[[], None]] = []
        self._original_handler: Optional[Callable] = None
        self.logger = logging.getLogger(__name__)
    
    def _handle_signal(self, signum: int, frame) -> NoReturn:
        """Handle interruption signal by executing cleanup handlers in LIFO order.
        
        Args:
            signum: Signal number that triggered this handler
            frame: Current stack frame (unused)
            
        Raises:
            SystemExit: Always exits with code 130 (standard Unix signal termination)
        """
        self.logger.warning(f"Operation cancelled by user (signal {signum})")
        
        # Execute handlers in reverse order (LIFO)
        for handler in reversed(self._handlers):
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Error in signal handler: {e}")
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.exception("Signal handler traceback:")
        
        sys.exit(130)
    
    @contextmanager
    def handler(self, cleanup_func: Optional[Callable[[], None]] = None):
        """Context manager for registering signal handlers with cleanup.
        
        Args:
            cleanup_func: Optional function to call during cleanup. Must take no arguments.
            
        Yields:
            None
            
        Example:
            def cleanup():
                print("Cleaning up...")
                
            with signal_manager.handler(cleanup):
                # Do work that might be interrupted
                pass
        """
        if cleanup_func:
            self.logger.debug(f"Registering signal handler: {cleanup_func.__name__}")
            self._handlers.append(cleanup_func)
        
        try:
            if not self._original_handler:
                self._original_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handle_signal)
                self.logger.debug("Installed SIGINT handler")
            yield
        finally:
            if cleanup_func:
                self.logger.debug(f"Removing signal handler: {cleanup_func.__name__}")
                self._handlers.remove(cleanup_func)
            if not self._handlers:
                signal.signal(signal.SIGINT, self._original_handler)
                self._original_handler = None
                self.logger.debug("Restored original SIGINT handler")
