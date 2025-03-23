#!/usr/bin/env python3
"""
lazy_formatter.py
A script to transform logging calls with f-strings into lazy-format style.
E.g.logging.info(f"Colony {colony_id} went extinct at generation {self.generation}")
            -> logging.info("Colony %s went extinct at generation %s", colony_id, self.generation)
Usage:
    python lazy_formatter.py <file_or_directory> [<file_or_directory> ...]
    python lazy_formatter.py -n <file_or_directory> [<file_or_directory> ...] # Dry run
    python lazy_formatter.py -h # Show help
This script uses the LibCST library to parse and transform Python code.
It requires Python 3.7 or higher.
It is designed to be run from the command line and can process multiple files or directories.
It will recursively search for .py files in the specified directories.
"""
#!/usr/bin/env python3

import argparse
import logging
import pathlib
import re
import sys
from typing import List, Tuple

import libcst as cst

# Setup logging for error reporting
logger = logging.getLogger(__name__)


class LazyLoggingTransformer(cst.CSTTransformer):
    """
    Transforms eager logging calls like:
      logging.info(f"Hello {name} count={count}")
      logging.info("Something {}".format(value))
    into lazy-format style:
      logging.info("Hello %s count=%d", name, count)
      logging.info("Something %s", value)

    Features:
      - Handles multiple placeholders in f-strings and .format() calls.
      - Guesses placeholder types (%s, %d, %f) with naive heuristics.
      - Preserves other arguments (like `exc_info=True`) to logging calls.
      - Leaves unaffected code alone.
      - Handles complex format specifiers and nested expressions.
      - Provides special handling for multi-argument f-strings.
    """

    LOG_METHODS = {"debug", "info", "warning", "error", "critical"}
    # Common integer variable name patterns
    INTEGER_PATTERNS = {
        # Counters and indices
        "count",
        "num",
        "index",
        "idx",
        "iteration",
        "size",
        "length",
        "total",
        "i",
        "j",
        "k",
        "n",
        "offset",
        "limit",
        "skip",
        "max_",
        "min_",
        "sum_",
        # Identifiers
        "id",
        "level",
        "generation",
        "step",
        "position",
        "pos",
        "priority",
        "rank",
        # Status codes and enums
        "code",
        "status",
        "error_code",
        "state",
        "flag",
        "type",
        "mode",
        "option",
        # Time and date components
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "timestamp",
        # Boolean-like values (usually printed as 0/1)
        "is_",
        "has_",
        "can_",
        "should_",
        "enable",
        "enabled",
        "active",
        "valid",
        # Coordinates and dimensions
        "row",
        "col",
        "width",
        "height",
        "depth",
        "line",
        "channel",
        "band",
        "dim",
        # Common integer suffixes
        "_count",
        "_num",
        "_id",
        "_idx",
        "_size",
        "_len",
        "_qty",
        "_total",
    }

    # Common float variable name patterns
    FLOAT_PATTERNS = {
        # Ratios and percentages
        "ratio",
        "percent",
        "pct",
        "rate",
        "value",
        "score",
        "efficiency",
        "factor",
        # Coordinate systems
        "x",
        "y",
        "z",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "coord",
        "scale",
        # Physical measurements
        "temp",
        "temperature",
        "pressure",
        "health",
        "energy",
        "time",
        "duration",
        "weight",
        "mass",
        "density",
        "volume",
        "area",
        "distance",
        "radius",
        # Financial metrics
        "price",
        "cost",
        "fee",
        "amount",
        "balance",
        "budget",
        "revenue",
        "profit",
        # Performance metrics
        "progress",
        "speed",
        "velocity",
        "acceleration",
        "frequency",
        "avg",
        "average",
        "mean",
        "median",
        "std",
        "stdev",
        "var",
        "variance",
        "error",
        "tolerance",
        # Common float suffixes
        "_ratio",
        "_pct",
        "_percent",
        "_rate",
        "_val",
        "_value",
        "_avg",
        "_score",
    }
    # Regex pattern for complex placeholders that might need special handling
    COMPLEX_PLACEHOLDER_PATTERN = re.compile(
        r"\{([^{}]+?)(\[[^{}]+?\]|\.[^{}]+?)(:[^{}]*)?\}"
    )

    # Pattern for Python 3.8+ debug expressions like f"{var=}"
    DEBUG_EXPR_PATTERN = re.compile(r"\{([^{}:=]+)=\}")

    # Pattern for extended debug expressions with format spec like f"{var=:.2f}"
    DEBUG_EXPR_WITH_FORMAT_PATTERN = re.compile(r"\{([^{}:=]+)=([^{}]*)\}")
    # Format specifier types for more detailed precision handling
    FORMAT_TYPE_MAP = {
        # Integer formats
        "d": "d",  # decimal
        "b": "b",  # binary
        "o": "o",  # octal
        "x": "x",  # hex (lowercase)
        "X": "X",  # hex (uppercase)
        "n": "d",  # locale-aware decimal
        # Float formats
        "f": "f",  # fixed point
        "F": "F",  # fixed point uppercase
        "e": "e",  # scientific notation
        "E": "E",  # scientific notation uppercase
        "g": "g",  # general format (shorter of fixed/scientific)
        "G": "G",  # general format uppercase
        "%": "%",  # percentage
        # String formats
        "s": "s",  # string
        "c": "c",  # character
        "r": "r",  # repr
        "a": "a",  # ascii
    }

    # Mapping for format specifier parts
    FORMAT_SPECIFIER_MAP = {
        # Alignment
        "<": "-",  # left-aligned ('-' in % format)
        ">": "",  # right-aligned (default in % format)
        "^": "",  # center-aligned (not directly supported in %)
        "=": "",  # padding after sign (not directly supported)
        # Sign
        "+": "+",  # always show sign
        "-": "",  # only show for negative (default)
        " ": " ",  # space for positive, minus for negative
    }

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """
        Invoked for every function call. If it's a logging.* call with an f-string or .format(),
        transform it.
        """
        func = updated_node.func
        if not isinstance(func, cst.Attribute):
            return updated_node

        # Check if base is "logging" and attr is in LOG_METHODS
        if (
            isinstance(func.value, cst.Name)
            and func.value.value == "logging"
            and func.attr.value in self.LOG_METHODS
        ):
            # Must have at least one argument
            if not updated_node.args:
                return updated_node

            first_arg = updated_node.args[0].value

            # Handle f-string calls: logging.info(f"...")
            if isinstance(first_arg, cst.FormattedString):
                return self._transform_f_string_call(updated_node)

            # Handle .format() calls: logging.info("...".format(...))
            if isinstance(first_arg, cst.Call):
                # e.g. "Colony {} ended".format(...)
                return self._transform_dot_format_call(updated_node)

        return updated_node

    # -------------------------------------------------------------------------
    # 1) Transform f-string calls
    # -------------------------------------------------------------------------
    def _transform_f_string_call(self, call_node: cst.Call) -> cst.Call:
        """
        For logging.info(f"Colony {colony_id} ended at gen {self.generation}")
        produce:
          logging.info("Colony %s ended at gen %d", colony_id, self.generation)

        Also handle multi-argument or multi-line f-strings if they appear.
        """
        first_arg = call_node.args[0].value
        if not isinstance(first_arg, cst.FormattedString):
            return call_node  # Not actually an f-string, bail out

        # Get expressions from the f-string
        expressions = self._get_formatted_string_expressions(first_arg)

        # Check if this is a complex f-string that needs special handling
        if expressions and (
            len(expressions) > 3
            or any(self._is_complex_expression(expr) for expr in expressions)
        ):
            return self._handle_multi_argument_fstring(
                call_node, first_arg, expressions
            )

        # We'll parse all f-string parts into placeholders + expressions
        new_format_str = ""
        new_exprs = []

        for part in first_arg.parts:
            if isinstance(part, cst.FormattedStringExpression):
                # Something like {colony_id}, {self.generation}, {some_func()}
                # Check for format specifiers
                format_spec = (
                    part.format_spec
                    if hasattr(part, "format_spec") and part.format_spec
                    else ""
                )

                guessed_type = self._guess_placeholder_type(
                    part.expression, format_spec
                )
                new_format_str += f"%{guessed_type}"
                new_exprs.append(part.expression)

            elif isinstance(part, cst.SimpleString):
                # Text inside the f-string
                text_chunk = self._strip_string_quotes(part.value)
                new_format_str += text_chunk

            elif isinstance(part, cst.ConcatenatedString):
                # Multi-line or concatenated segments in an f-string
                sub_fmt, sub_exprs = self._extract_from_concatenated_string(part)
                new_format_str += sub_fmt
                new_exprs.extend(sub_exprs)

        return self._transform_call_node(new_format_str, call_node, new_exprs)

    # -------------------------------------------------------------------------
    # 2) Transform .format() calls
    # -------------------------------------------------------------------------
    def _transform_dot_format_call(self, call_node: cst.Call) -> cst.Call:
        """
        For logging.info("Colony {} ended".format(colony_id))
        produce:
          logging.info("Colony %s ended", colony_id)

        For logging.info("Hi {name} you are number {idx}".format(name=username, idx=user_index))
        produce:
          logging.info("Hi %s you are number %d", username, user_index)
        """
        # First, validate and extract information from the format call
        format_info = self._extract_format_call_info(call_node)
        if not format_info:
            return call_node  # Not a valid .format() call or too complex

        base_string, base_str_val, format_args = format_info

        # Extract placeholders from the base string
        placeholders = self._find_placeholders_in_string(base_str_val)

        # Process format arguments into positional and keyword collections
        pos_args, kw_args_map = self._separate_format_args(format_args)

        # Check if we have complex placeholders that need special handling
        if self._has_complex_format_placeholders(placeholders):
            return self._handle_complex_format_call(call_node, base_string, format_args)

        # Parse the string and transform placeholders
        final_format_str, expressions_in_order = self._process_format_string(
            base_str_val, pos_args, kw_args_map
        )

        # Create the transformed node
        return self._transform_call_node(
            final_format_str, call_node, expressions_in_order
        )

    def _extract_format_call_info(self, call_node: cst.Call):
        """Extract and validate the .format() call information.

        Args:
            call_node: The logging call node to analyze

        Returns:
            tuple or None: (base_string, base_str_val, format_args) if valid, None otherwise
        """
        first_arg = call_node.args[0].value

        # Must be a call node representing .format()
        if not isinstance(first_arg, cst.Call):
            return None

        # Check it's an attribute access for .format
        func_attr = first_arg.func
        if not isinstance(func_attr, cst.Attribute) or func_attr.attr.value != "format":
            return None

        # Get the base string being formatted
        base_string = func_attr.value

        # Handle different string types
        if isinstance(base_string, cst.SimpleString):
            base_str_val = self._strip_string_quotes(base_string.value)
        elif isinstance(base_string, cst.ConcatenatedString):
            fmt_str, exprs = self._extract_from_concatenated_string(base_string)
            if exprs:  # Too complex if there are expressions in the concatenation
                return None
            base_str_val = fmt_str
        else:
            return None  # Other string types not supported

        return base_string, base_str_val, first_arg.args

    def _separate_format_args(self, format_args):
        """Separate .format() arguments into positional and keyword collections.

        Args:
            format_args: List of CST Arg nodes from the .format() call

        Returns:
            tuple: (positional_args, keyword_args_map)
        """
        pos_args = []
        kw_args_map = {}

        for arg in format_args:
            if arg.keyword is None:
                pos_args.append(arg.value)
            else:
                kw_args_map[arg.keyword.value] = arg.value

        return pos_args, kw_args_map

    def _has_complex_format_placeholders(self, placeholders):
        """Check if format placeholders contain complex expressions that need special handling.

        Args:
            placeholders: List of placeholder strings

        Returns:
            bool: True if complex placeholders are present
        """
        return (
            any(self.COMPLEX_PLACEHOLDER_PATTERN.search(p) for p in placeholders)
            and len(placeholders) >= 2
        )

    def _handle_complex_format_call(self, call_node, base_string, format_args):
        """Create a lambda wrapper for complex format expressions.

        This method is used when we have complex formatting that can't be easily
        converted to %s style formatting, so we wrap it in a lambda to preserve
        the lazy evaluation behavior.

        Args:
            call_node: The original logging call node
            base_string: The string being formatted
            format_args: The arguments to .format()

        Returns:
            cst.Call: The transformed call node with a lambda
        """
        # Create a lambda that will call the format method at runtime
        lambda_body = cst.Call(
            func=cst.Attribute(value=base_string, attr=cst.Name(value="format")),
            args=format_args,
        )
        lambda_func = cst.Lambda(params=cst.Parameters(params=[]), body=lambda_body)

        return self._replace_arg_with_lambda(call_node, lambda_func)

    def _handle_escaped_braces(self, str_val, idx):
        """Handle escaped braces in format strings.

        Args:
            str_val: The string value being processed
            idx: Current index in the string

        Returns:
            tuple: (processed_text, new_index)
        """
        if idx + 1 < len(str_val):
            if str_val[idx] == "{" and str_val[idx + 1] == "{":
                return "{", idx + 2
            elif str_val[idx] == "}" and str_val[idx + 1] == "}":
                return "}", idx + 2
        return str_val[idx], idx + 1

    def _process_placeholder(self, placeholder_body, pos_args, kw_args_map, auto_index):
        """Process a single placeholder in a format string.

        Args:
            placeholder_body: The content inside the braces
            pos_args: List of positional arguments
            kw_args_map: Dictionary of keyword arguments
            auto_index: Current auto-index for empty placeholders

        Returns:
            tuple: (expression, placeholder_type, new_auto_index)
        """
        # Special handling for auto-indexed placeholders {}
        if not placeholder_body and pos_args:
            if auto_index < len(pos_args):
                ph_expr = pos_args[auto_index]
                placeholder_type = self._guess_placeholder_type(ph_expr, "")
                return ph_expr, placeholder_type, auto_index + 1
            # More placeholders than arguments
            return None, "s", auto_index

        # Resolve the placeholder to an expression and type
        ph_expr, placeholder_type = self._resolve_placeholder_expression(
            placeholder_body, pos_args, kw_args_map
        )
        return ph_expr, placeholder_type, auto_index

    def _process_format_string(self, base_str_val, pos_args, kw_args_map):
        """Process a format string, replacing placeholders with %s/%d/%f markers.

        Args:
            base_str_val: The string value to process
            pos_args: List of positional arguments
            kw_args_map: Dictionary of keyword arguments

        Returns:
            tuple: (final_format_str, expressions_in_order)
        """
        final_format_str = ""
        expressions_in_order = []

        auto_index = 0  # For auto-indexed placeholders {}
        idx = 0

        while idx < len(base_str_val):
            idx = self._process_next_format_segment(
                base_str_val,
                idx,
                final_format_str,
                expressions_in_order,
                pos_args,
                kw_args_map,
                auto_index,
            )

        return final_format_str, expressions_in_order

    def _process_next_format_segment(
        self,
        base_str_val,
        idx,
        final_format_str,
        expressions_in_order,
        pos_args,
        kw_args_map,
        auto_index,
    ):
        """Process the next segment of a format string starting from the given index.

        Args:
            base_str_val: The base string value to process
            idx: The current index in the string
            final_format_str: The format string being built (string)
            expressions_in_order: The list of expressions being built
            pos_args: Positional arguments for format placeholders
            kw_args_map: Keyword arguments for format placeholders
            auto_index: The current auto-index for unnamed placeholders

        Returns:
            int: The updated index after processing this segment, and updated auto_index if relevant
        """
        # Handle special character cases first
        if self._is_opening_brace(base_str_val, idx):
            return self._handle_opening_brace(
                base_str_val,
                idx,
                final_format_str,
                expressions_in_order,
                pos_args,
                kw_args_map,
                auto_index,
            )
        elif self._is_closing_brace(base_str_val, idx):
            return self._handle_closing_brace(base_str_val, idx, final_format_str)

        # Regular character
        final_format_str += base_str_val[idx]
        return idx + 1

    def _is_opening_brace(self, base_str_val, idx):
        """Check if the character at index is an opening brace."""
        return base_str_val[idx] == "{"

    def _is_closing_brace(self, base_str_val, idx):
        """Check if the character at index is a closing brace with possible escape sequence."""
        return (
            base_str_val[idx] == "}"
            and idx + 1 < len(base_str_val)
            and base_str_val[idx + 1] == "}"
        )

    def _handle_opening_brace(
        self,
        base_str_val,
        idx,
        final_format_str,
        expressions_in_order,
        pos_args,
        kw_args_map,
        auto_index,
    ):
        """Handle an opening brace in a format string."""
        # Check for escaped braces
        if idx + 1 < len(base_str_val) and base_str_val[idx + 1] == "{":
            return self._add_escaped_braces_to_format(
                base_str_val, idx, final_format_str
            )
        return self._process_format_placeholder(
            base_str_val,
            idx,
            final_format_str,
            expressions_in_order,
            pos_args,
            kw_args_map,
            auto_index,
        )

    def _handle_closing_brace(self, base_str_val, idx, final_format_str):
        """Handle a closing brace in a format string."""
        return self._add_escaped_braces_to_format(
            base_str_val, idx, final_format_str
        )

    def _add_escaped_braces_to_format(self, base_str_val, idx, final_format_str):
        # Handle escaped braces using the base implementation
        # and add the result to the format string
        processed_text, new_idx = self._handle_escaped_braces(base_str_val, idx)
        final_format_str += processed_text
        return new_idx

    def _process_format_placeholder(
        self,
        base_str_val,
        idx,
        final_format_str,
        expressions_in_order,
        pos_args,
        kw_args_map,
        auto_index,
    ):
        """Process a format placeholder starting at the given index.

        Args:
            base_str_val: The base string value to process
            idx: The current index in the string
            final_format_str: The format string being built (string)
            expressions_in_order: The list of expressions being built
            pos_args: Positional arguments for format placeholders
            kw_args_map: Keyword arguments for format placeholders
            auto_index: The current auto-index for unnamed placeholders

        Returns:
            int: The updated index after processing this placeholder
        """
        # Find the matching closing brace
        closing_idx = base_str_val.find("}", idx)
        if closing_idx == -1:
            # Malformed format string, just add the character and move on
            final_format_str += base_str_val[idx]
            return idx + 1

        # Extract the placeholder content
        placeholder_body = base_str_val[idx + 1 : closing_idx].strip()

        # Process the placeholder
        ph_expr, placeholder_type, auto_index = self._process_placeholder(
            placeholder_body, pos_args, kw_args_map, auto_index
        )

        # Add to the output format string and expressions list
        final_format_str += f"%{placeholder_type}"
        if ph_expr is not None:
            expressions_in_order.append(ph_expr)

        return closing_idx + 1

    def _transform_call_node(self, format_str, call_node, expressions):
        """Helper to transform the call node after we have the new format string and expressions.

        Args:
            format_str (str): The format string with %s/%d/%f placeholders
            call_node (cst.Call): The original logging call node
            expressions (List[cst.BaseExpression]): The expressions to format

        Returns:
            cst.Call: The transformed call node with lazy formatting
        """
        # Validate and safely process the format string
        format_str, expressions = self._safely_process_format_string(
            format_str, expressions
        )

        # Create a new string literal with the format string
        new_format_node = cst.SimpleString(f'"{format_str}"')
        # Start with the format string as the first argument
        transformed_args = [call_node.args[0].with_changes(value=new_format_node)]
        # Add all the expressions as additional arguments
        transformed_args.extend(cst.Arg(value=expr) for expr in expressions)
        # Preserve any remaining original arguments (e.g., exc_info=True)
        if len(call_node.args) > 1:
            transformed_args.extend(call_node.args[1:])
        return call_node.with_changes(args=transformed_args)

    # Helper: parse out placeholders from string manually (simple approach)
    def _find_placeholders_in_string(self, s: str):
        """
        Return a list of substring placeholders like "{}", "{0}", "{name}", "{name:0.2f}".
        This method is here mainly for illustration; we won't do advanced checks,
        but we do highlight how you might approach it.
        """
        # In this script, we don't actually need the raw placeholders separately,
        # since we parse them inline in _transform_dot_format_call. This is just for reference
        results = []
        idx = 0
        while True:
            start = s.find("{", idx)
            if start == -1:
                break
            end = s.find("}", start + 1)
            if end == -1:
                break
            results.append(s[start : end + 1])
            idx = end + 1
        return results

    def _parse_format_specifier(self, spec_part):
        """Extract type information from a format specifier.

        Args:
            spec_part: The format specifier part (after the colon)

        Returns:
            str: The inferred format type ('s', 'd', or 'f')
        """
        if not spec_part:
            return ""

        # Check for explicit type at the end of spec_part
        if spec_part[-1].isalpha():
            return self._get_type_from_format_char(spec_part[-1])

        # Look for number formatting which implies float
        float_indicators = [".", "e", "E", "f", "g", "G", "%"]
        return "f" if any(c in spec_part for c in float_indicators) else ""

    def _get_type_from_format_char(self, type_char):
        """Map a format specifier character to our simplified type system.

        Args:
            type_char: The format specifier character

        Returns:
            str: The simplified type ('s', 'd', or 'f')
        """
        return next(
            (
                our_type
                for our_type, chars in self.FORMAT_TYPE_MAP.items()
                if type_char in chars
            ),
            "",
        )

    def _get_expression_for_positional_placeholder(self, index, pos_args):
        """Get the expression for a positional placeholder.

        Args:
            index: The index of the positional placeholder
            pos_args: List of positional arguments

        Returns:
            expression: The expression at the given index or None
        """
        return pos_args[index] if index < len(pos_args) else None

    def _resolve_placeholder_expression(self, placeholder_body, pos_args, kw_args_map):
        """
        Given the text inside { ... } from a string literal, figure out:
          1) Which expression it corresponds to (positional or keyword).
          2) The best format specifier (%s, %d, %f) from the placeholder text or expression type.

        Returns: (cst.BaseExpression or None, 's' / 'd' / 'f')
        """
        # Split on ':' to check for format spec
        if ":" in placeholder_body:
            name_part, spec_part = placeholder_body.split(":", 1)
            name_part = name_part.strip()
            spec_part = spec_part.strip()
        else:
            name_part = placeholder_body.strip()
            spec_part = ""

        # Get format type from specifier
        format_type = self._parse_format_specifier(spec_part)

        # Empty placeholder: {}
        if name_part == "":
            expr = pos_args.pop(0) if pos_args else None
            guessed_type = format_type or self._guess_placeholder_type(expr, spec_part)
            return (expr, guessed_type)

        # Indexed placeholder: {0}, {1}, etc.
        if name_part.isdigit():
            expr = self._get_expression_for_positional_placeholder(
                int(name_part), pos_args
            )
            guessed_type = format_type or self._guess_placeholder_type(expr, spec_part)
            return (expr, guessed_type)

        # Handle complex expressions (with dots or brackets)
        is_complex = self.COMPLEX_PLACEHOLDER_PATTERN.search(name_part)

        # Named placeholder (may be complex)
        if name_part in kw_args_map:
            expr = kw_args_map.pop(name_part)
            guessed_type = format_type or self._guess_placeholder_type(expr, spec_part)
            return (expr, guessed_type)
        elif is_complex:
            # Complex expression that we couldn't resolve
            return (None, format_type or "s")

        # Default fallback
        return (None, format_type or "s")

    # -------------------------------------------------------------------------
    # 3) Multi-argument or multi-line f-strings
    #    (We handle them in the same approach, but here's a separate helper
    #     to flatten a cst.ConcatenatedString into placeholders + expressions.)
    # -------------------------------------------------------------------------
    def _extract_from_concatenated_string(self, concat_str: cst.ConcatenatedString):
        """
        Flatten a multi-part string (which may itself contain FormattedStrings or nested
        concatenations). Return a tuple:
          (format_string_part, [expressions_list])
        """
        left = concat_str.left
        right = concat_str.right

        left_fmt, left_exprs = self._extract_from_string_node(left)
        right_fmt, right_exprs = self._extract_from_string_node(right)

        return left_fmt + right_fmt, left_exprs + right_exprs

    def _extract_from_string_node(self, node):
        """
        Helper for flattening any string-ish node into (fmt_str, [exprs]).

        Args:
            node: A CST node that might be a string-like node

        Returns:
            tuple: (format_string, expression_list)
        """
        # Process different node types
        node_type_handlers = {
            cst.SimpleString: self._handle_simple_string_node,
            cst.FormattedString: self._transform_formatted_string,
            cst.ConcatenatedString: self._extract_from_concatenated_string,
        }

        if handler := next(
            (
                handler
                for node_type, handler in node_type_handlers.items()
                if isinstance(node, node_type)
            ),
            None,
        ):
            return handler(node)

        # Not a string at all; fallback
        return ("", [])

    def _handle_simple_string_node(
        self, node: cst.SimpleString
    ) -> Tuple[str, List[cst.BaseExpression]]:
        """Extract content from a simple string node."""
        chunk = self._strip_string_quotes(node.value)
        return (chunk, [])

    def _transform_formatted_string(
        self, fstring_node: cst.FormattedString
    ) -> Tuple[str, List[cst.BaseExpression]]:
        """
        Break down a single FormattedString node (f"some {expr}...") into
        (format_str, [expressions]) with guessed placeholders.
        Supports Python 3.8+ debug expressions like f"{var=}".

        Args:
            fstring_node: The f-string node to transform

        Returns:
            tuple: (format_string, expression_list)
        """
        try:
            fmt_str = ""
            exprs = []

            # Process each part of the f-string
            for part in fstring_node.parts:
                fmt_str, exprs = self._process_fstring_part(part, fmt_str, exprs)

            # Validate and safely process the format string
            return self._safely_process_format_string(fmt_str, exprs)
        except Exception as e:
            # Use the malformed f-string handler for recovery
            logger.warning(
                f"Error in f-string transformation: {e}, attempting recovery"
            )
            return self._handle_malformed_fstring(fstring_node)

    def _get_raw_fstring_value(self, fstring_node: cst.FormattedString) -> str:
        """Extract the raw string value from an f-string node for pattern matching.

        Args:
            fstring_node: The f-string node

        Returns:
            str: The raw string value with placeholders intact
        """
        # Try to extract the value directly if available
        if hasattr(fstring_node, "value"):
            return fstring_node.value

        # Fallback: construct a simplified version from parts
        raw_str = ""
        for part in fstring_node.parts:
            if isinstance(part, cst.FormattedStringText):
                raw_str += part.value
            elif isinstance(part, cst.FormattedStringExpression):
                # Get expression name or representation
                expr_value = (
                    part.expression.value if hasattr(part.expression, "value") else ""
                )
                # Add format spec if present
                fmt_spec = f":{part.format_spec.value}" if part.format_spec else ""
                # Add equals sign if it's a debug expression
                equals = "=" if part.equal else ""
                # Construct the placeholder
                raw_str += f"{{{expr_value}{equals}{fmt_spec}}}"

        return raw_str

    def _process_fstring_part(
        self, part, fmt_str: str, exprs: List[cst.BaseExpression]
    ) -> Tuple[str, List[cst.BaseExpression]]:
        """
        Process a single part of an f-string and update the format string and expressions list.
        Support for Python 3.8+ debug expressions like f"{var=}" is included.

        Args:
            part: A part of an f-string (expression, simple string, or concatenated string)
            fmt_str: The current format string
            exprs: The current list of expressions

        Returns:
            tuple: (updated_format_string, updated_expression_list)
        """
        # Process based on part type
        if isinstance(part, cst.FormattedStringExpression):
            # Get format spec if present, otherwise empty string
            format_spec = part.format_spec.value if part.format_spec else ""

            # Check if this is a debug expression (Python 3.8+): {var=}
            if isinstance(part.expression, cst.Name) and part.equal:
                # It's a debug expression, format as "var=var" with appropriate type
                var_name = part.expression.value
                guessed_type = self._guess_placeholder_type(
                    part.expression, format_spec
                )
                return f"{fmt_str}{var_name}=%{guessed_type}", exprs + [part.expression]

            # Regular expression in f-string
            guessed_type = self._guess_placeholder_type(part.expression, format_spec)
            return f"{fmt_str}%{guessed_type}", exprs + [part.expression]

        if isinstance(part, cst.SimpleString):
            # Handle literal text in f-string
            text_chunk = self._strip_string_quotes(part.value)
            return f"{fmt_str}{text_chunk}", exprs

        if isinstance(part, cst.ConcatenatedString):
            # Handle nested concatenated strings
            sub_fmt, sub_exprs = self._extract_from_concatenated_string(part)
            return f"{fmt_str}{sub_fmt}", exprs + sub_exprs

        # Default case for unknown part types
        return fmt_str, exprs

    # -------------------------------------------------------------------------
    # 4) Guessing placeholder types
    # -------------------------------------------------------------------------
    def _guess_placeholder_type(
        self, expr: cst.BaseExpression, format_spec: str = ""
    ) -> str:
        """
        Return "s", "d", or "f" for use in a "%s" / "%d" / "%f" placeholder
        based on a more comprehensive set of heuristics:
          - If format_spec includes format type specifiers (f, d, etc.) => use that
          - If expr is a numeric literal int => %d
          - If expr is a numeric literal float => %f
          - If the variable name matches common integer/float naming patterns => %d/%f
          - If the expression involves numeric operations => guess based on operation
          - If the expression accesses a known numeric attribute => guess based on pattern
          - Default fallback => %s
        """
        # First priority: check format specification
        if format_spec:
            if format_type := self._check_format_spec(format_spec):
                return format_type

        # Check expression type
        return self._infer_type_from_expression(expr)

    def _check_format_spec(self, format_spec: str) -> str:
        """Check format specification for type indicators and extract formatting directives.

        Args:
            format_spec: Format specification string (e.g., ':>10.2f')

        Returns:
            str: The detected format type ('d', 'f', 's', etc.)
        """
        # Use str.removeprefix for Python 3.9+, or fallback for older versions
        format_spec = format_spec[1:] if format_spec.startswith(":") else format_spec

        # If the last character is a type specifier, use it
        if format_spec and format_spec[-1].isalpha():
            type_char = format_spec[-1]
            if type_char in self.FORMAT_TYPE_MAP:
                return self.FORMAT_TYPE_MAP[type_char]

        # Check for numerical format patterns
        if re.search(r"\.\d+[feEgG]?$", format_spec):  # Has precision => likely float
            return "f"
        if re.search(r"\d+d$", format_spec):  # Has width with 'd' => integer
            return "d"

        # Return type based on indicators or fallback to empty string
        return "f" if "%" in format_spec else ""

    def _infer_type_from_expression(self, expr: cst.BaseExpression) -> str:
        """Infer type from expression structure with advanced type detection."""
        # Check for numeric literals
        if isinstance(expr, cst.Integer):
            return "d"
        if isinstance(expr, cst.Float):
            return "f"
        # Check for container types
        if isinstance(expr, (cst.List, cst.Dict, cst.Set, cst.Tuple)):
            return "s"
        # Check for string literals
        if isinstance(expr, (cst.SimpleString, cst.FormattedString)):
            return "s"
        # Check for boolean values
        if isinstance(expr, cst.Name) and expr.value in ("True", "False"):
            return "d"
        # Check for function calls
        if isinstance(expr, cst.Call):
            return self._infer_type_from_function_call(expr)

        # Check for different node types
        type_checkers = {
            cst.Name: self._check_name_type,
            cst.Attribute: self._check_attribute_type,
            cst.Subscript: self._check_subscript_type,
            cst.BinaryOperation: self._check_binary_operation_type,
        }

        # Find appropriate type checker and return any definitive result
        if (
            checker := next(
                (
                    checker
                    for node_type, checker in type_checkers.items()
                    if isinstance(expr, node_type)
                ),
                None,
            )
        ) and (result := checker(expr)):
            return result

        # Fallback: everything else => %s
        return "s"

    def _infer_type_from_function_call(self, expr: cst.Call) -> str:
        """Infer type from function call based on common function naming patterns."""
        if not isinstance(expr.func, cst.Name) and not isinstance(
            expr.func, cst.Attribute
        ):
            return "s"  # Default for complex function calls

        # Extract function name from either Name or Attribute node
        if isinstance(expr.func, cst.Name):
            func_name = expr.func.value.lower()
        else:  # Must be cst.Attribute based on earlier guard clause
            func_name = expr.func.attr.value.lower()

        # Integer-returning functions
        if any(
            name in func_name
            for name in [
                "count",
                "len",
                "sum",
                "index",
                "find",
                "ord",
                "int",
                "size",
                "num",
                "enumerate",
                "position",
            ]
        ):
            return "d"
        # Float-returning functions
        elif any(
            name in func_name
            for name in [
                "float",
                "average",
                "mean",
                "median",
                "min",
                "max",
                "calc",
                "compute",
                "div",
                "percent",
                "ratio",
            ]
        ):
            return "f"
        # String-returning functions
        elif any(
            name in func_name
            for name in [
                "str",
                "format",
                "join",
                "replace",
                "name",
                "repr",
                "to_string",
                "convert",
                "decode",
                "get_",
                "fetch",
            ]
        ):
            return "s"
        # Boolean-returning functions
        elif any(
            name in func_name
            for name in [
                "is_",
                "has_",
                "can_",
                "should_",
                "contains",
                "check",
                "validate",
                "verify",
            ]
        ):
            return "d"  # Booleans formatted as 0/1

        # Default to string for unknown function calls
        return "s"

    def _check_name_type(self, expr: cst.Name) -> str:
        """Check variable name against common patterns."""
        return self._match_name_against_patterns(expr.value.lower())

    def _match_name_against_patterns(self, name: str) -> str:
        """Match a name string against integer and float patterns with enhanced detection.

        Args:
            name: Variable or attribute name to analyze

        Returns:
            str: Inferred format type ('d', 'f', 's', or empty string)
        """
        # Check for specific container types in names
        container_indicators = {
            "list",
            "array",
            "queue",
            "stack",
            "deque",
            "buffer",
            "dict",
            "map",
            "table",
            "set",
            "collection",
            "tuple",
        }

        # If it's a known container type, return string format
        if any(indicator in name.lower() for indicator in container_indicators):
            return "s"

        # Check for common integer variable names
        if next(
            (
                p
                for p in self.INTEGER_PATTERNS
                if p in name.lower() or name.lower() == p
            ),
            None,
        ):
            return "d"

        # Check for common float variable names
        if next(
            (p for p in self.FLOAT_PATTERNS if p in name.lower() or name.lower() == p),
            None,
        ):
            return "f"

        # Check for boolean indicators
        bool_indicators = {"is_", "has_", "can_", "should_", "flag"}
        if any(name.lower().startswith(indicator) for indicator in bool_indicators):
            return "d"

        return ""  # No specific type found

    def _check_attribute_type(self, expr: cst.Attribute) -> str:
        """Check attribute name against common patterns."""
        return self._match_name_against_patterns(expr.attr.value.lower())

    def _check_subscript_type(self, expr: cst.Subscript) -> str:
        """Check subscript expressions for type hints."""
        # Only process subscripts with integer indices and name values
        if not (
            isinstance(expr.slice, cst.Index)
            and isinstance(expr.slice.value, cst.Integer)
            and isinstance(expr.value, cst.Name)
        ):
            return ""

        # First try standard name pattern matching with the variable name
        var_name = expr.value.value.lower()
        if result := self._match_name_against_patterns(var_name):
            return result

        # Check against additional array-specific patterns
        common_patterns = {
            "f": ["coord", "position", "vector", "matrix"],  # Float patterns
            "d": ["count", "total", "sum", "size"],  # Integer patterns
        }

        # Check patterns more efficiently using next with default return value
        return next(
            (
                type_code
                for type_code, patterns in common_patterns.items()
                if any(pattern in var_name for pattern in patterns)
            ),
            "",
        )

    def _check_binary_operation_type(self, expr: cst.BinaryOperation) -> str:
        """Determine result type of binary operations."""
        # Only handle arithmetic operations
        if not self._is_arithmetic_operation(expr.operator):
            return ""

        # Division typically produces floats
        if isinstance(expr.operator, cst.Divide):
            return "f"

        # For other operations, check the operand types
        return self._determine_operand_result_type(expr.left, expr.right)

    def _is_arithmetic_operation(self, operator) -> bool:
        """Check if an operator is arithmetic."""
        return isinstance(operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide))

    def _determine_operand_result_type(
        self, left: cst.BaseExpression, right: cst.BaseExpression
    ) -> str:
        """Determine the result type based on operand types."""
        has_float_operand = any(isinstance(op, cst.Float) for op in (left, right))
        has_numeric_operand = has_float_operand or any(
            isinstance(op, cst.Integer) for op in (left, right)
        )

        # If we have numeric operands, determine the result type
        if has_numeric_operand:
            return "f" if has_float_operand else "d"

        return ""  # Can't determine definitively

    # -------------------------------------------------------------------------
    # 5) Misc Helpers
    # -------------------------------------------------------------------------
    def _strip_string_quotes(self, val: str) -> str:
        """
        Given a string literal token (e.g. '"hello"' or "'world'"),
        strip surrounding quotes. This is naive but works for typical cases.
        """
        if len(val) >= 2 and val[0] in {'"', "'"} and val[-1] in {'"', "'"}:
            return val[1:-1]
        return val

    def _is_complex_f_string(self, fmt_str: cst.FormattedString) -> bool:
        """
        Determine if an f-string is complex enough to warrant special handling.
        Complex f-strings may include:
        - Nested expressions (dict lookups, method calls, etc.)
        - Multiple expressions with potential for side effects
        - Complex format specifiers

        Returns:
            bool: True if the f-string should be handled specially
        """
        # Get all expressions from the f-string
        expressions = self._get_formatted_string_expressions(fmt_str)

        # Check if any expressions are complex
        has_complex_expr = any(
            self._is_complex_expression(expr) for expr in expressions
        )

        # Count expressions and check for complexity
        expression_count, _, has_format_specs = self._analyze_fstring_complexity(
            fmt_str
        )

        # Consider complex if we have many expressions or complex expressions with format specs
        return expression_count > 2 and (has_complex_expr or has_format_specs)

    def _get_formatted_string_expressions(
        self, formatted_string: cst.FormattedString
    ) -> List[cst.BaseExpression]:
        """Extract all expressions from a formatted string.

        Args:
            formatted_string: The formatted string to extract expressions from

        Returns:
            List[cst.BaseExpression]: The expressions found in the formatted string
        """
        return [
            part.expression
            for part in formatted_string.parts
            if isinstance(part, cst.FormattedStringExpression)
        ]

    def _is_complex_expression(self, expr: cst.BaseExpression) -> bool:
        """Check if an expression is complex (contains method calls, subscripts, etc.).

        Args:
            expr: The expression to check

        Returns:
            bool: True if the expression is complex
        """
        # Direct checks for complex types
        complex_types = (cst.Call, cst.Attribute, cst.Subscript, cst.BinaryOperation)
        if isinstance(expr, complex_types):
            return True

        # Check for nested expression complexity
        if hasattr(expr, "expr") and isinstance(expr.expr, cst.BaseExpression):
            return self._is_complex_expression(expr.expr)

        # Check for function-like names that might have side effects
        return isinstance(expr, cst.Name) and any(
            expr.value.startswith(prefix)
            for prefix in ["get_", "calc_", "compute_", "fetch_"]
        )

    def _analyze_fstring_complexity(
        self, fmt_str: cst.FormattedString
    ) -> Tuple[int, bool, bool, bool]:
        """Analyze an f-string to determine its complexity metrics.

        Args:
            fmt_str: The f-string to analyze

        Returns:
            tuple: (expression_count, has_complex_expr, has_format_specs, has_debug_expr)
        """
        expression_count = 0
        has_complex_expr = False
        has_format_specs = False
        has_debug_expr = False

        for part in fmt_str.parts:
            if isinstance(part, cst.FormattedStringExpression):
                expression_count += 1

                # Check for format specifiers
                if self._has_format_spec(part):
                    has_format_specs = True

                # Check for complex expressions
                if self._is_complex_expression(part.expression):
                    has_complex_expr = True

                # Check for debug expressions (Python 3.8+ f"{var=}")
                if part.equal:
                    has_debug_expr = True

        return expression_count, has_complex_expr, has_format_specs, has_debug_expr

    def _has_format_spec(self, part: cst.FormattedStringExpression) -> bool:
        """Check if a formatted string expression has a format specifier."""
        return part.format_spec and part.format_spec.strip()

    def _is_complex_expression(self, expr: cst.BaseExpression) -> bool:
        """Check if an expression is complex (contains calls, subscripts, etc.)."""
        # Check for simple expressions that are never complex
        if isinstance(expr, (cst.Integer, cst.Float, cst.SimpleString)):
            return False

        # Simple variable names are not complex
        if isinstance(expr, cst.Name):
            return False

        # Common complex expression types
        if isinstance(
            expr, (cst.Call, cst.Subscript, cst.BinaryOperation, cst.CompoundStatement)
        ):
            return True

        # Check for non-simple attribute access (e.g., obj.method rather than self.attr)
        if isinstance(expr, cst.Attribute):
            # If the value part is itself complex, the whole expression is complex
            if self._is_complex_expression(expr.value):
                return True
            # Simple attribute access (e.g., self.attr) is not complex
            return not (isinstance(expr.value, cst.Name) and expr.attr.value)

        # Container literal with complex elements
        if isinstance(expr, (cst.List, cst.Tuple, cst.Dict, cst.Set)):
            if isinstance(expr, cst.Dict):
                # For dictionaries, check both keys and values
                elements = []
                for elem in expr.elements:
                    elements.extend([elem.key, elem.value])
            else:  # List, Tuple, Set
                elements = [elem.value for elem in expr.elements]

            # If any element is complex, the container is complex
            return any(self._is_complex_expression(elem) for elem in elements)

        # By default, assume other expressions are complex
        return True

    def _handle_multi_argument_fstring(
        self,
        call_node: cst.Call,
        fstring_node: cst.FormattedString,
        expressions: List[cst.BaseExpression],
    ) -> cst.Call:
        """
        Handle an f-string with multiple arguments or complex expressions.
        For complex f-strings, we create a lambda to evaluate the f-string at runtime,
        preserving lazy evaluation. For simpler cases, we convert to regular format strings.

        Args:
            call_node: The original logging call node
            fstring_node: The f-string node
            expressions: The expressions in the f-string

        Returns:
            cst.Call: The transformed call node
        """
        # For complex f-strings with many expressions or complex expressions, use a lambda
        complex_expr_count = sum(
            self._is_complex_expression(expr) for expr in expressions
        )

        if complex_expr_count > 0 and len(expressions) > 2:
            # Create a lambda that will evaluate the f-string at runtime
            lambda_func = self._create_lambda_for_fstring(fstring_node)
            return self._replace_arg_with_lambda(call_node, lambda_func)

        # For simpler cases, extract format string and expressions
        fmt_value, expr_values = self._extract_from_string_node(fstring_node)

        # Build transformed call with format string and expressions
        return self._transform_call_with_format(call_node, fmt_value, expr_values)

    def _transform_call_with_format(
        self, call_node: cst.Call, fmt_value: str, expr_values: List[cst.BaseExpression]
    ) -> cst.Call:
        """Transform a call node by replacing its first argument with a format string and adding expressions as arguments.

        Args:
            call_node: The original call node
            fmt_value: The format string value
            expr_values: The expressions to add as arguments

        Returns:
            cst.Call: The transformed call node
        """
        # Create a new string with format specifiers
        new_first_arg = cst.SimpleString(f'"{fmt_value}"')

        # Start with the format string as the first argument
        transformed_args = [call_node.args[0].with_changes(value=new_first_arg)]

        # Add all expressions as additional arguments
        transformed_args.extend([cst.Arg(value=expr) for expr in expr_values])

        # Preserve any remaining arguments
        if len(call_node.args) > 1:
            transformed_args.extend(call_node.args[1:])

        return call_node.with_changes(args=transformed_args)

    def _create_lambda_for_fstring(
        self, fstring_node: cst.FormattedString
    ) -> cst.Lambda:
        """
        Create a lambda function that wraps an f-string to preserve its evaluation semantics.

        Args:
            fstring_node: The f-string to wrap in a lambda

        Returns:
            cst.Lambda: A lambda function that returns the f-string when called
        """
        # Create the lambda function with empty parameters
        return cst.Lambda(
            params=cst.Parameters(params=[]),
            body=cst.FormattedString(
                parts=fstring_node.parts, start=fstring_node.start, end=fstring_node.end
            ),
        )

    def _replace_arg_with_lambda(
        self, call_node: cst.Call, lambda_func: cst.Lambda
    ) -> cst.Call:
        """
        Replace the first argument of a call with a lambda function.

        Args:
            call_node: The original call node
            lambda_func: The lambda function to use as the first argument

        Returns:
            cst.Call: The transformed call node
        """
        # Replace the first argument with the lambda function
        transformed_args = [call_node.args[0].with_changes(value=lambda_func)]
        if len(call_node.args) > 1:
            transformed_args.extend(call_node.args[1:])
        return call_node.with_changes(args=transformed_args)

    # Error handling and validation methods
    def _validate_format_string(
        self, format_string: str, expressions: List[cst.BaseExpression]
    ) -> bool:
        """
        Validate that a format string has the correct number and type of placeholders
        compared to the expressions provided.

        Args:
            format_string: The format string with placeholders
            expressions: List of expressions to substitute into the placeholders

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not format_string or not expressions:
                return True  # Empty format strings or expressions are technically valid

            # Use a regex to find all format specifiers, handling escaped %% correctly
            import re

            placeholders = re.findall(
                r"%(?:%|[#0 +-]?\d*(?:\.\d+)?[diouxXeEfFgGcrsa])", format_string
            )
            # Filter out %% (escaped percent signs) which are not actual placeholders
            placeholders = [p for p in placeholders if p != "%%"]

            # Check if we have the right number of expressions
            if len(placeholders) != len(expressions):
                logger.warning(
                    f"Format string has {len(placeholders)} placeholders but {len(expressions)} expressions"
                )
                return False

            # Try to format with dummy values to catch syntax errors
            dummy_values = []
            for i, expr in enumerate(expressions):
                # Get placeholder type from the actual placeholder in format string
                ph_type = placeholders[i][-1] if i < len(placeholders) else "s"

                # Map placeholder types to appropriate dummy values
                dummy_value = {
                    "d": 0,
                    "i": 0,
                    "o": 0,
                    "u": 0,
                    "x": 0,
                    "X": 0,
                    "e": 0.0,
                    "E": 0.0,
                    "f": 0.0,
                    "F": 0.0,
                    "g": 0.0,
                    "G": 0.0,
                    "c": "a",
                    "r": "dummy",
                    "s": "dummy",
                    "a": "dummy",
                }.get(ph_type, "dummy")

                dummy_values.append(dummy_value)

            # Test the format string with dummy values
            format_string % tuple(dummy_values)
            return True

        except Exception as e:
            logger.warning(f"Invalid format string: {format_string}. Error: {e}")
            return False

    def _safely_process_format_string(
        self, format_str: str, expressions: List[cst.BaseExpression]
    ) -> Tuple[str, List[cst.BaseExpression]]:
        """
        Process a format string with built-in error recovery for malformed format strings.

        Args:
            format_str: The format string to process
            expressions: The expressions to format with

        Returns:
            Tuple[str, List[cst.BaseExpression]]: Processed format string and expressions
        """
        # Add guard clause for empty expressions
        if not expressions:
            return format_str, expressions

        # Validate the format string
        if not self._validate_format_string(format_str, expressions):
            return self._placeholder_coversion(
                format_str, expressions
            )
        # Original format string is valid
        return format_str, expressions

    def _placeholder_coversion(self, format_str, expressions):
        """
        Attempt recovery: convert all placeholders to %s for safety
        """
        safe_format = self._convert_to_safe_format(
            format_str
        )  # Convert all non-%s, non-%% to %s

        # Ensure we have the right number of placeholders
        placeholder_count = len(
            re.findall(r"%[^%]", safe_format)
        )  # Count all % not followed by %
        if placeholder_count < len(expressions):
            # Add more %s placeholders if needed
            extra_placeholders = " " + " ".join(
                ["%s"] * (len(expressions) - placeholder_count)
            )
            safe_format += extra_placeholders
        elif placeholder_count > len(expressions):
            # Get just the first part with correct placeholders
            parts = re.split(r"%[^%]", safe_format)
            safe_format = ""
            for i, part in enumerate(parts):
                safe_format += part
                if i < len(expressions):
                    safe_format += "%s"

        # Double-check our recovery worked
        if not self._validate_format_string(safe_format, expressions):
            # Last resort fallback: just use each expression as a string
            placeholders = " ".join(["%s"] * len(expressions))
            logger.warning(
                f"Format string recovery failed. Using simplest form: {placeholders}"
            )
            return placeholders, expressions

        logger.info(f"Recovered format string: {safe_format}")
        return safe_format, expressions

    def _extract_parts_from_fstring(
        self, fstring_node: cst.FormattedString
    ) -> Tuple[str, List[cst.BaseExpression]]:
        """Extract parts from an f-string node into format string and expressions list.

        Args:
            fstring_node: The f-string node to extract parts from

        Returns:
            Tuple[str, List[cst.BaseExpression]]: Format string and expressions list
        """
        expressions = []
        parts = []

        # Process each part carefully
        for part in fstring_node.parts:
            if isinstance(part, cst.FormattedStringText):
                # Simple text - just add it
                parts.append(part.value)
            elif isinstance(part, cst.FormattedStringExpression):
                # Extract the expression and add a placeholder based on expression type
                expr = part.expression
                expressions.append(expr)
                # Try to infer type for better placeholder
                placeholder_type = self._infer_type_from_expression(expr)
                parts.append(f"%{placeholder_type or 's'}")

        # Join parts to create a format string
        format_str = "".join(parts)
        return format_str, expressions

    def _handle_malformed_fstring(
        self, fstring_node: cst.FormattedString
    ) -> Tuple[str, List[cst.BaseExpression]]:
        """
        Handle malformed f-strings by extracting as much information as possible.

        Args:
            fstring_node: The potentially malformed f-string node

        Returns:
            Tuple[str, List[cst.BaseExpression]]: A safe format string and expressions list
        """
        try:
            # Try normal processing first
            return self._transform_formatted_string(fstring_node)
        except Exception as e:
            logger.warning(f"Error processing f-string: {e}")
            # Fallback: extract what we can using the helper method
            return self._extract_parts_from_fstring(fstring_node)


def transform_file(filepath: str, in_place: bool = True) -> bool:
    """
    Parse the given Python file, apply LazyLoggingTransformer, and optionally
    write the result back to disk.

    Args:
        filepath: Path to the Python file to transform
        in_place: If True, write changes to the file, otherwise print to stdout

    Returns:
        bool: True if transformation was successful, False otherwise
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original_code = f.read()

        mod = cst.parse_module(original_code)
        transformed_mod = mod.visit(LazyLoggingTransformer())
        new_code = transformed_mod.code

        if in_place:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_code)
        else:
            # Print to stdout (dry-run)
            print(new_code)

        return True
    except Exception as e:
        print(f"Error transforming {filepath}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transform eager logging calls (f-strings, .format) into lazy-format style."
    )
    parser.add_argument(
        "paths", nargs="+", help="One or more .py files or directories to transform."
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print transformed code to stdout instead of modifying files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with transformation details.",
    )
    parser.add_argument(
        "--skip-complex",
        action="store_true",
        help="Skip complex f-strings and format calls instead of using lambdas.",
    )
    args = parser.parse_args()

    # Collect .py files
    files_to_transform = []
    for path_str in args.paths:
        p = pathlib.Path(path_str)
        if p.is_dir():
            files_to_transform.extend(iter(p.rglob("*.py")))
        elif p.is_file() and p.suffix == ".py":
            files_to_transform.append(p)
        else:
            print(f"Skipping non-Python path: {p}", file=sys.stderr)

    success_count = 0
    error_count = 0
    skipped_count = 0

    for py_file in files_to_transform:
        if args.verbose:
            print(f"Transforming {py_file} ...")
        success = transform_file(str(py_file), in_place=(not args.dry_run))
        if success:
            success_count += 1
        else:
            error_count += 1

    print("\nSummary:")
    print(f"  Files processed successfully: {success_count}")
    print(f"  Files with errors: {error_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Total files: {len(files_to_transform)}")


if __name__ == "__main__":
    main()
