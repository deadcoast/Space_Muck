# Standard library imports
import json
import os

# Third-party library imports

# Local application imports
from pathlib import Path
from questionary import prompt
from typing import Any, Dict, Optional
from variant_loggers.log_analyzer import report_param
import aiofiles
import asyncio
import patch
import tempfile
import variant_loggers

# Configure variant_loggers
variant_loggers.basicConfig(level=variant_loggers.INFO)
logger = variant_loggers.getLogger(__name__)

# Example files to analyze
files = ["example1.py", "example2.py"]

# Create example files
for file in files:
    if not os.path.isfile(file):
        try:
            with open(file, "w") as f:
                f.write("# Example Python file\nprint('Hello, World!')\n")
            logger.info(f"Created example file '{file}'.")
        except Exception as e:
            logger.error(f"Failed to create example file '{file}': {e}")


class FixManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the FixManager with optional configuration path.

        :param config_path: Path to the configuration file.
        :type config_path: Optional[str]
        """

        self.config = self._load_config(config_path) if config_path else {}
        self._setup_components(self.config)

        # Validate configuration integrity
        fix_manager = self._validate_config(config_path)
        if fix_manager and fix_manager != self.config:
            logger.warning(
                "Configuration validation failed: inconsistent state detected"
            )

    def _validate_config(self, config_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Validate configuration by loading it independently.

        :param config_path: Path to the configuration file.
        :type config_path: Optional[str]
        :return: Validated configuration dictionary or None if validation fails.
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            return self._load_config(config_path) if config_path else {}
        except Exception as e:
            logger.error(f"Failed to validate configuration: {e}")
            return None

    def _load_config(self, param: str) -> Dict[str, Any]:
        """
        Loads configuration settings from a JSON file.

        :param param: Path to the configuration JSON file.
        :type param: str
        :return: Configuration as a dictionary.
        :rtype: Dict[str, Any]
        """
        config_path = param
        if not os.path.isfile(config_path):
            logger.error(f"Configuration file '{config_path}' does not exist.")
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        try:
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            logger.info(f"Configuration loaded from '{config_path}'.")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from the configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise

    def _setup_components(self, config: Dict[str, Any]) -> None:
        """
        Sets up necessary components based on the configuration.

        :param config: Configuration settings.
        :type config: Dict[str, Any]
        """
        # Example: Setup variant_loggers level based on config
        log_level = config.get("log_level", "INFO").upper()
        numeric_level = getattr(variant_loggers, log_level, None)
        if not isinstance(numeric_level, int):
            logger.warning(f"Invalid log level '{log_level}'. Using 'INFO' level.")
            numeric_level = variant_loggers.INFO
        logger.setLevel(numeric_level)
        logger.info(f"Logging level set to {log_level}.")

        # Additional component setups can be added here
        # For example, setting up paths, initializing patch handlers, etc.

    async def _generate_report(
        self, analyzer: Any, param: Dict[str, Any], custom_report_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Asynchronously generates a report based on the analysis results.

        :param analyzer: Analyzer object that contains analysis results.
        :type analyzer: Any
        :param param: Additional parameters for report generation.
        :type param: Dict[str, Any]
        :param custom_report_path: Optional custom path for the report.
        :type custom_report_path: Optional[Path]
        :return: Path to the generated report file, or None if failed.
        :rtype: Optional[str]
        """
        report_format = param.get("format", "json")  # e.g., 'json', 'txt', 'html'
        # Use custom_report_path if provided, otherwise use the one from param
        report_path = str(custom_report_path) if custom_report_path else param.get("report_path", "report.json")

        try:
            report_data = (
                analyzer.get_report_data()
            )  # Assuming analyzer has this method

            async with aiofiles.open(report_path, "w", encoding="utf-8") as report_file:
                if report_format == "json":
                    await report_file.write(json.dumps(report_data, indent=4))
                elif report_format == "txt":
                    report_content = self._format_report_txt(report_data)
                    await report_file.write(report_content)
                elif report_format == "html":
                    report_content = self._format_report_html(report_data)
                    await report_file.write(report_content)
                else:
                    logger.error(f"Unsupported report format: {report_format}")
                    return None

            logger.info(
                f"Report generated at '{report_path}' in '{report_format}' format."
            )
            return report_path

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

    def _format_report_txt(self, report_data: Dict[str, Any]) -> str:
        """
        Formats report data into plain text.

        :param report_data: Data to include in the report.
        :type report_data: Dict[str, Any]
        :return: Formatted report string.
        :rtype: str
        """
        lines = ["Analysis Report", "================\n"]
        for issue in report_data.get("issues", []):
            lines.extend(
                (
                    f"Issue ID: {issue.get('id')}",
                    f"Description: {issue.get('description')}",
                    f"File: {issue.get('file')}",
                    f"Line: {issue.get('line')}",
                    f"Severity: {issue.get('severity')}\n",
                )
            )
        return "\n".join(lines)

    def _format_report_html(self, report_data: Dict[str, Any]) -> str:
        """
        Formats report data into HTML.

        :param report_data: Data to include in the report.
        :type report_data: Dict[str, Any]
        :return: Formatted HTML report string.
        :rtype: str
        """
        html = [
            "<html>",
            "<head><title>Analysis Report</title></head>",
            "<body>",
            "<h1>Analysis Report</h1>",
            "<table border='1'>",
            "<tr><th>Issue ID</th><th>Description</th><th>File</th><th>Line</th><th>Severity</th></tr>",
        ]
        for issue in report_data.get("issues", []):
            html.extend(
                (
                    "<tr>",
                    f"<td>{issue.get('id')}</td>",
                    f"<td>{issue.get('description')}</td>",
                    f"<td>{issue.get('file')}</td>",
                    f"<td>{issue.get('line')}</td>",
                    f"<td>{issue.get('severity')}</td>",
                    "</tr>",
                )
            )
        html.extend(("</table>", "</body></html>"))
        return "\n".join(html)

    async def _automatic_fix(self, analyzer: Any) -> bool:
        """
        Asynchronously applies fixes automatically based on analysis results.

        :param analyzer: Analyzer object that contains analysis results.
        :type analyzer: Any
        :return: True if fixes were applied successfully, False otherwise.
        :rtype: bool
        """
        try:
            issues = analyzer.get_issues()  # Assuming analyzer has this method
            if not issues:
                logger.info("No issues found to fix.")
                return True

            for issue in issues:
                fix_id = issue.get("fix_id")
                if not fix_id:
                    logger.warning(
                        f"Issue {issue.get('id')} does not have a fix_id. Skipping."
                    )
                    continue

                # Fetch the patch corresponding to fix_id
                patch_content = self._fetch_patch(fix_id)
                if not patch_content:
                    logger.warning(
                        f"No patch found for fix_id: {fix_id}. Skipping issue {issue.get('id')}."
                    )
                    continue

                file_path = issue.get("file")
                if not file_path or not os.path.isfile(file_path):
                    logger.error(
                        f"File '{file_path}' does not exist. Skipping issue {issue.get('id')}."
                    )
                    continue

                # Read the file content asynchronously
                async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                    content = await file.read()

                # Apply the patch
                new_content = self._apply_patch(content, patch_content)
                if new_content is None:
                    logger.error(f"Failed to apply patch for issue {issue.get('id')}.")
                    continue

                # Write the new content back to the file asynchronously
                async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
                    await file.write(new_content)

                logger.info(f"Automatically applied fix {fix_id} to '{file_path}'.")

            logger.info("Automatic fixes applied successfully.")
            return True

        except Exception as e:
            logger.error(f"An error occurred during automatic fix: {e}")
            return False

    async def _interactive_fix(self, analyzer: Any) -> bool:
        """
        Asynchronously applies fixes interactively based on analysis results.

        :param analyzer: Analyzer object that contains analysis results.
        :type analyzer: Any
        :return: True if fixes were applied successfully, False otherwise.
        :rtype: bool
        """
        try:
            issues = analyzer.get_issues()
            if not issues:
                logger.info("No issues found to fix.")
                return True

            # Prepare questions for questionary
            questions = [
                {
                    "type": "checkbox",
                    "name": "selected_issues",
                    "message": "Select the issues you want to fix:",
                    "choices": [
                        {
                            "name": f"Issue {issue.get('id')}: {issue.get('description')} (File: {issue.get('file')}, Line: {issue.get('line')})",
                            "value": issue,
                        }
                        for issue in issues
                        if issue.get("fix_id")
                    ],
                }
            ]

            answers = prompt(questions)
            selected_issues = answers.get("selected_issues", [])

            if not selected_issues:
                logger.info("No issues selected for fixing.")
                return True

            for issue in selected_issues:
                fix_id = issue.get("fix_id")
                if not fix_id:
                    logger.warning(
                        f"Issue {issue.get('id')} does not have a fix_id. Skipping."
                    )
                    continue

                # Fetch the patch corresponding to fix_id
                patch_content = self._fetch_patch(fix_id)
                if not patch_content:
                    logger.warning(
                        f"No patch found for fix_id: {fix_id}. Skipping issue {issue.get('id')}."
                    )
                    continue

                file_path = issue.get("file")
                if not file_path or not os.path.isfile(file_path):
                    logger.error(
                        f"File '{file_path}' does not exist. Skipping issue {issue.get('id')}."
                    )
                    continue

                # Read the file content asynchronously
                async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                    content = await file.read()

                # Apply the patch
                new_content = self._apply_patch(content, patch_content)
                if new_content is None:
                    logger.error(f"Failed to apply patch for issue {issue.get('id')}.")
                    continue

                # Write the new content back to the file asynchronously
                async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
                    await file.write(new_content)

                logger.info(f"Interactively applied fix {fix_id} to '{file_path}'.")

            logger.info("Interactive fixes applied successfully.")
            return True

        except Exception as e:
            logger.error(f"An error occurred during interactive fix: {e}")
            return False

    def _fetch_patch(self, fix_id: str) -> Optional[str]:
        """
        Fetches a patch associated with a specified identifier.

        This method retrieves the patch content from a `.patch` file located
        in the patches directory. The patch file is expected to be named
        as `<fix_id>.patch`.

        :param fix_id: The unique identifier of the patch to fetch.
        :type fix_id: str
        :return: The content of the retrieved patch as a string, or None if not found.
        :rtype: Optional[str]
        """
        patches_dir = self.config.get(
            "patches_dir", "patches"
        )  # Default to 'patches' directory
        patch_filename = f"{fix_id}.patch"
        patch_path = os.path.join(patches_dir, patch_filename)

        if not os.path.isfile(patch_path):
            logger.error(f"Patch file '{patch_filename}' not found in '{patches_dir}'.")
            return None

        try:
            with open(patch_path, "r") as file:
                patch_content = file.read()
            logger.info(f"Fetched patch '{patch_filename}'.")
            return patch_content
        except Exception as e:
            logger.error(f"Error reading patch file '{patch_filename}': {e}")
            return None

    def _apply_patch(self, content: str, patch_content: str) -> Optional[str]:
        """
        Applies a unified diff patch to the provided content and returns the modified content.

        :param content: The original content to be modified.
        :type content: str
        :param patch_content: The unified diff patch detailing the changes to apply.
        :type patch_content: str
        :return: The content updated after applying the patch, or None if failed.
        :rtype: Optional[str]
        """
        # Create a temporary directory to apply the patch
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file_path = os.path.join(temp_dir, "original.txt")
            patched_file_path = os.path.join(temp_dir, "original_patched.txt")
            patch_file_path = os.path.join(temp_dir, "temp.patch")

            # Log paths for debugging and verification
            logger.debug(
                f"Using temporary files:\n  Original: {original_file_path}\n  Patched: {patched_file_path}\n  Patch: {patch_file_path}"
            )

            # Verify paths exist and are writable
            temp_paths = [original_file_path, patched_file_path, patch_file_path]
            for path in temp_paths:
                if not os.access(os.path.dirname(path), os.W_OK):
                    logger.error(f"Directory for {path} is not writable")
                    return None

            # Write the original content to 'original.txt'
            try:
                with open(original_file_path, "w") as original_file:
                    original_file.write(content)
                logger.debug(f"Wrote original content to '{original_file_path}'.")
            except Exception as e:
                logger.error(f"Error writing original content to temporary file: {e}")
                return None

            # Write the patch content to 'temp.patch'
            try:
                with open(patch_file_path, "w") as patch_file:
                    patch_file.write(patch_content)
                logger.debug(f"Wrote patch content to '{patch_file_path}'.")
            except Exception as e:
                logger.error(f"Error writing patch content to temporary file: {e}")
                return None

            # Initialize the patch set
            try:
                pset = patch.fromfile(patch_file_path)
                if not pset:
                    logger.error("Failed to parse the patch content.")
                    return None
                logger.debug("Parsed patch content successfully.")
            except Exception as e:
                logger.error(f"Error parsing patch content: {e}")
                return None

            # Apply the patch
            try:
                if not pset.apply(root=temp_dir):
                    logger.error("Failed to apply the patch cleanly.")
                    return None
                logger.debug("Applied patch successfully.")
            except Exception as e:
                logger.error(f"Error applying patch: {e}")
                return None

            # Read the patched content
            try:
                with open(original_file_path, "r") as patched_file:
                    patched_content = patched_file.read()
                logger.debug(f"Read patched content from '{original_file_path}'.")
                return patched_content
            except Exception as e:
                logger.error(f"Error reading patched content: {e}")
                return None

    async def run(self, analyzer: Any, mode: str, report_param: Dict[str, Any], report_path: Optional[Path] = None) -> None:
        """
        Runs the FixManager based on the specified mode.

        :param analyzer: Analyzer object that contains analysis results.
        :type analyzer: Any
        :param mode: Mode of operation ('report', 'auto_fix', 'interactive_fix').
        :type mode: str
        :param report_param: Parameters for report generation.
        :type report_param: Dict[str, Any]
        :param report_path: Optional path for saving the report.
        :type report_path: Optional[Path]
        """
        if mode == "report":
            # Pass the report_path parameter to the _generate_report method
            output_path = await self._generate_report(analyzer, report_param, report_path)
            if output_path:
                logger.info(f"Report successfully generated at '{output_path}'.")
            else:
                logger.error("Failed to generate report.")

        elif mode == "auto_fix":
            success = await self._automatic_fix(analyzer)
            if success:
                logger.info("Automatic fixes applied successfully.")
            else:
                logger.error("Automatic fixes failed.")

        elif mode == "interactive_fix":
            success = await self._interactive_fix(analyzer)
            if success:
                logger.info("Interactive fixes applied successfully.")
            else:
                logger.error("Interactive fixes failed.")

        else:
            logger.error(
                f"Unknown mode '{mode}'. Supported modes are 'report', 'auto_fix', 'interactive_fix'."
            )

    # Example Usage


async def main(fix_manager=None):
    # Use the imported report_param if fix_manager is not provided
    if fix_manager is None:
        fix_manager = report_param
    # Path to the configuration file
    config_path = "config.json"

    # Create a configuration file if it doesn't exist
    if not os.path.isfile(config_path):
        # Example configuration
        example_config = {"patches_dir": "patches", "log_level": "DEBUG"}

        try:
            with open(config_path, "w") as config_file:
                json.dump(example_config, config_file, indent=4)
            logger.info(f"Created example configuration file at '{config_path}'.")
        except Exception as e:
            logger.error(f"Failed to create configuration file: {e}")
            return

def run(analyzer, mode, report_param, report_path):
    # Path to the configuration file
    config_path = "config.json"  # Using the same config path as in main()

    # Initialize FixManager with configuration
    fix_manager = FixManager(config_path=config_path)

    # Generate Report
    asyncio.run(fix_manager.run(
        analyzer, mode="report", report_param=report_param, report_path=report_path
    ))

    # Apply Automatic Fixes - use mode parameter from function arguments
    if mode in ["auto_fix", "all"]:
        asyncio.run(fix_manager.run(analyzer, mode="auto_fix", report_param=report_param, report_path=report_path))

    # Apply Interactive Fixes - use mode parameter from function arguments
    if mode in ["interactive_fix", "all"]:
        asyncio.run(fix_manager.run(
            analyzer, mode="interactive_fix", report_param=report_param, report_path=report_path
        ))
    return None

if __name__ == "__main__":
    asyncio.run(main())
