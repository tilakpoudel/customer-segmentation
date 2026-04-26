# src/utils.py
"""Shared helper functions for the Customer Segmentation application."""

import logging
import sys

def setup_logging() -> logging.Logger:
    """
    Configure and return a standard logger.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("customer_segmentation")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def format_currency(value: float) -> str:
    """
    Format a float as a currency string.
    
    Args:
        value (float): The numeric value to format.
        
    Returns:
        str: Formatted currency string (e.g., "$1,234.56").
    """
    return f"${value:,.2f}"

def format_number(value: float, decimals: int = 0) -> str:
    """
    Format a number with commas and specified decimals.
    
    Args:
        value (float): The numeric value to format.
        decimals (int): Number of decimal places.
        
    Returns:
        str: Formatted number string.
    """
    return f"{value:,.{decimals}f}"
