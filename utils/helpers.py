"""
Helper functions, custom exceptions, logging setup, and utilities for SmartForecast.
This module provides reusable decorators, directory validation, and standardized error handling.
"""

import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import List, Union, Callable, Any


# =====================================================================
# Custom Exceptions
# =====================================================================

class SmartForecastException(Exception):
    """Base exception class for all SmartForecast related errors."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataValidationError(SmartForecastException):
    """Exception raised when data validation, schema, or integrity checks fail."""
    pass


class FeatureEngineeringError(SmartForecastException):
    """Exception raised when feature calculation or lag generation fails."""
    pass


class ModelTrainingError(SmartForecastException):
    """Exception raised during model training, hyperparameter tuning, or evaluation."""
    pass


class InferenceError(SmartForecastException):
    """Exception raised during future forecasting or batch prediction."""
    pass


# =====================================================================
# Logging Setup
# =====================================================================

def setup_logger(
    name: str = "SmartForecast",
    log_file: Union[str, Path] = "smartforecast.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a standardized logger with console and file handlers.

    Args:
        name: Name of the logger instance.
        log_file: Path to the log file.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicating handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Log format with timestamp, level, logger name, and message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        log_path = Path(log_file)
        if log_path.parent != Path('.'):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        console_handler.stream.write(f"[WARNING] Could not create file logger at {log_file}: {e}\n")

    return logger


# Global default logger instance for general use
logger = setup_logger()


# =====================================================================
# Directory Verification Utilities
# =====================================================================

def ensure_directories(directories: List[Union[str, Path]]) -> None:
    """
    Ensure that a list of directory paths exist. If not, create them safely.

    Args:
        directories: List of directory paths (strings or Path objects).
    """
    for dir_path in directories:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path_obj.resolve()}")


# =====================================================================
# Performance & Timing Decorators
# =====================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func: Function to be timed.

    Returns:
        Wrapped function that logs elapsed duration.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        logger.info(f"Starting execution: '{func.__name__}'...")
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Completed '{func.__name__}' in {elapsed_time:.3f} seconds.")
            return result
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"Function '{func.__name__}' failed after {elapsed_time:.3f} seconds with error: {e}")
            raise
    return wrapper


def format_number(val: Union[int, float]) -> str:
    """
    Format large numbers into human-readable strings (e.g., 1.2M, 45.3K).

    Args:
        val: Numeric value to format.

    Returns:
        Formatted string representation.
    """
    if val is None:
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{val / 1_000:.2f}K"
    elif isinstance(val, int) or val.is_integer():
        return f"{int(val):,}"
    else:
        return f"{val:.2f}"
