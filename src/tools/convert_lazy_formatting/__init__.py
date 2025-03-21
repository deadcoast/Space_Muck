"""Convert Lazy Formatting package.

This package provides tools to transform eager logging calls with f-strings 
and .format() into lazy-format style for better performance.
"""

from .convert_to_lazy_formatting import transform_file, LazyLoggingTransformer
from .lazy_menu import run_interactive_menu

__all__ = ['transform_file', 'LazyLoggingTransformer', 'run_interactive_menu']
