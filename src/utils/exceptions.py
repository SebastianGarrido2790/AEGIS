"""
Custom exception handling for the project.

This module defines a custom exception class that captures detailed error information,
including the file name and line number where the exception occurred.
This is critical for MLOps pipelines to quickly debug failures in automated workflows and prevent silent failures.

Implementation details:
- Captures Context: Automatically extracts the file name and line number where the error occurred.
- Formatting: Wraps the error into a standardized string format for logs.
- Strict Typing: Uses ModuleType instead of untyped sys imports to satisfy modern linters.
"""

from pathlib import Path
from types import ModuleType

from src.constants import PROJECT_ROOT


def error_message_detail(error: Exception | str, error_detail: ModuleType) -> str:
    """
    Extracts the detailed error message including file name and line number.

    Args:
        error (Exception | str): The exception or error message.
        error_detail (ModuleType): The sys module to access execution info.

    Returns:
        str: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    # Safety check to prevent crashes in edge cases where the traceback might be incomplete.
    if exc_tb is not None and exc_tb.tb_frame is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    if file_name != "unknown":
        file_path = Path(file_name)
        try:
            # Try to make it relative to project root for cleaner logs
            display_path = str(file_path.relative_to(PROJECT_ROOT))
        except (ValueError, TypeError):
            display_path = str(file_path)
    else:
        display_path = "unknown"

    error_message = (
        f"Error occurred in python script: [{display_path}] "
        f"line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception class to provide detailed traceback information within the message.
    """

    def __init__(self, error_message: Exception | str, error_detail: ModuleType):
        """
        Initialize the CustomException.

        Args:
            error_message (Exception | str): The original error message or exception object.
            error_detail (ModuleType): The sys module to capture stack trace.
        """
        # Generate the detailed message
        self.detailed_message = error_message_detail(
            error=error_message, error_detail=error_detail
        )
        # Call the base class constructor with the detailed message
        super().__init__(self.detailed_message)

    def __str__(self) -> str:
        return self.detailed_message
