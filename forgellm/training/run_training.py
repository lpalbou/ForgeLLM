#!/usr/bin/env python3
"""
Run training process for continued pre-training or instruction fine-tuning.
This module is designed to be run as a separate process.
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run training process")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()

def setup_log_file(log_file_path: str, config: Dict[str, Any]):
    """Set up log file with initial data"""
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create initial log data
    log_data = {
        "config": config,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "metrics": []
    }
    
    # Write initial log data
    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    return log_data

def update_log_file(log_file_path: str, metrics: Dict[str, Any]):
    """Update log file with new metrics"""
    try:
        # Read existing log data
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
        
        # Add new metrics
        log_data["metrics"].append(metrics)
        
        # Write updated log data
        with open(log_file_path, "w") as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating log file: {e}")

def run_training():
    """Run training process"""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load raw config data for the log file
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Load config using dataclass methods
        from forgellm.training.config import TrainingConfig, InstructTuningConfig
        
        # Determine config type and load
        if 'model_name' in config_dict and 'input_dir' in config_dict:
            # Training config
            config = TrainingConfig.from_dict(config_dict)
        elif 'base_model_path' in config_dict and 'base_model_name' in config_dict:
            # Instruction tuning config
            config = InstructTuningConfig.from_dict(config_dict)
        else:
            logger.error("Unknown configuration type")
            sys.exit(1)
        
        # Set up log file
        log_data = setup_log_file(args.log_file, config_dict)
        
        # Import trainer
        from forgellm.training.trainer import ContinuedPretrainer
        
        # Create trainer
        trainer = ContinuedPretrainer(config)
        
        # Run training
        logger.info("Starting training")
        trainer.run_training()
        
        # Update log file with final status
        log_data["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        log_data["status"] = "completed"
        with open(args.log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        
        # Update log file with error status if it exists
        if 'log_data' in locals() and args.log_file:
            log_data["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            log_data["status"] = "failed"
            log_data["error"] = str(e)
            with open(args.log_file, "w") as f:
                json.dump(log_data, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    run_training() 