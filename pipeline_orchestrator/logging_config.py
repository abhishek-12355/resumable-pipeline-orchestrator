"""Logging configuration for pipeline orchestrator."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    stream: Optional[object] = None
) -> logging.Logger:
    """
    Set up logging configuration for the pipeline orchestrator.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to INFO if not specified.
        format_string: Custom format string for log messages.
                      Defaults to a standard format if not specified.
        stream: Stream to write logs to. Defaults to sys.stderr.
    
    Returns:
        Root logger for pipeline_orchestrator package
    """
    # Get root logger for the package
    logger = logging.getLogger("pipeline_orchestrator")
    
    # Set level
    if level is None:
        level = "INFO"
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    if stream is None:
        stream = sys.stderr
    
    handler = logging.StreamHandler(stream)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Logger instance
    """
    # Ensure the root logger is set up
    root_logger = logging.getLogger("pipeline_orchestrator")
    if not root_logger.handlers:
        setup_logging()
    
    return logging.getLogger(name)

