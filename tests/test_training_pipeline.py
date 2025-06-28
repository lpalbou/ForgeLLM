#!/usr/bin/env python3
"""
Simple test script to verify the training pipeline integration
"""

import os
import sys
import logging
import time
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import forgellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forgellm.training.config import TrainingConfig
from forgellm.training.trainer import ContinuedPretrainer

def test_start_training():
    """Test starting a training job with minimal configuration"""
    logger.info("Testing start_training with minimal configuration")
    
    # Create a minimal configuration for testing
    config = TrainingConfig(
        model_name="mlx-community/gemma-3-1b-it-4bit",  # Small model for testing
        input_dir="mnemosyne",
        output_dir="models/test_training",
        batch_size=1,  # Small batch size for testing
        learning_rate=5e-6,
        max_iterations=2,  # Very small number for testing
        save_every=1
    )
    
    # Initialize trainer
    trainer = ContinuedPretrainer()
    
    try:
        # Start training
        trainer.start_training(config)
        logger.info("Training started successfully")
        
        # Check if training is active
        is_active = trainer.is_training_active()
        logger.info(f"Training active: {is_active}")
        
        # Wait for a moment to let the training process start
        time.sleep(5)
        
        # Get training status
        status = trainer.get_training_status()
        logger.info(f"Training status: {json.dumps(status, indent=2)}")
        
        # Stop training
        trainer.stop_training()
        logger.info("Training stopped successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_start_training: {e}")
        # Try to stop training if it was started
        try:
            trainer.stop_training()
        except:
            pass
        return False

def test_get_dashboard_data():
    """Test getting dashboard data"""
    logger.info("Testing get_dashboard_data")
    
    # Initialize trainer
    trainer = ContinuedPretrainer()
    
    try:
        # Get dashboard data
        data = trainer.get_dashboard_data()
        logger.info(f"Dashboard data: {json.dumps(data, indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_get_dashboard_data: {e}")
        return False

def run_tests():
    """Run all tests"""
    tests = [
        ("Start Training", test_start_training),
        ("Get Dashboard Data", test_get_dashboard_data)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        success = test_func()
        results.append((name, success))
        logger.info(f"Test {name}: {'PASSED' if success else 'FAILED'}")
        
    # Print summary
    logger.info("\nTest Summary:")
    all_passed = True
    for name, success in results:
        logger.info(f"{name}: {'PASSED' if success else 'FAILED'}")
        if not success:
            all_passed = False
            
    return all_passed

if __name__ == "__main__":
    logger.info("Starting training pipeline tests")
    success = run_tests()
    sys.exit(0 if success else 1) 