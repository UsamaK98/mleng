"""
Logging module for the Parliamentary Meeting Analyzer.

This module sets up structured logging for the application, ensuring all logs
are properly formatted and saved to the logs directory.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Create logs directory if it doesn't exist
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure default logger
def setup_logger():
    """Configure the application logger with proper formatting and file output."""
    # Remove default handler
    logger.remove()
    
    # Add stdout handler for interactive use
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler for persistent logs
    log_file = LOG_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",    # Rotate logs daily
        retention="14 days", # Keep logs for 14 days
        compression="zip"    # Compress rotated logs
    )
    
    return logger

# Create and configure logger
logger = setup_logger()

# Define specialized loggers for different components
class ModelLogger:
    """Logger specialized for model operations."""
    
    @staticmethod
    def log_inference(model_name, input_text, output_text, duration_ms, success=True):
        """Log model inference operations."""
        logger.info(
            f"Model inference | {model_name} | Duration: {duration_ms:.2f}ms | Success: {success}"
        )
        if not success:
            logger.error(f"Model inference failed for {model_name}")
            logger.debug(f"Input: {input_text[:100]}...")
        else:
            logger.debug(f"Input: {input_text[:100]}... | Output: {output_text[:100]}...")

class DataLogger:
    """Logger specialized for data operations."""
    
    @staticmethod
    def log_data_load(data_source, num_records, duration_ms, success=True):
        """Log data loading operations."""
        logger.info(
            f"Data loaded | {data_source} | Records: {num_records} | Duration: {duration_ms:.2f}ms | Success: {success}"
        )
        if not success:
            logger.error(f"Failed to load data from {data_source}")

class QueryLogger:
    """Logger specialized for query operations."""
    
    @staticmethod
    def log_query(query_text, context_items, duration_ms, success=True):
        """Log query processing operations."""
        logger.info(
            f"Query processed | Duration: {duration_ms:.2f}ms | Context items: {len(context_items)} | Success: {success}"
        )
        logger.debug(f"Query: {query_text}")
        if not success:
            logger.error(f"Query processing failed: {query_text}")

# Usage example:
# from src.utils.logging import logger, ModelLogger, DataLogger, QueryLogger
# 
# logger.info("Application started")
# ModelLogger.log_inference("ollama-qwq", "What is...", "The answer is...", 120.5)
# DataLogger.log_data_load("parliamentary_minutes.csv", 1000, 50.2)
# QueryLogger.log_query("Who spoke about climate?", ["item1", "item2"], 200.3) 