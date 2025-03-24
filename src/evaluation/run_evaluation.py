"""
Script to run the evaluation of the Parliamentary Meeting Analyzer.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import logger
from src.evaluation.evaluation import Evaluator, run_evaluation

if __name__ == "__main__":
    try:
        logger.info("Starting evaluation script")
        success = run_evaluation()
        
        if success:
            logger.info("Evaluation completed successfully")
            print("Evaluation completed successfully. Results saved to the output-new directory.")
        else:
            logger.error("Evaluation failed")
            print("Evaluation failed. Check logs for details.")
    
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        print(f"Error: {str(e)}") 