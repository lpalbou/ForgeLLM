#!/usr/bin/env python3
"""
Test script for dashboard generation
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import forgellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forgellm.training.dashboard import (
    DashboardGenerator, 
    create_comprehensive_dashboard, 
    identify_best_checkpoints,
    load_training_data
)

def create_sample_training_data():
    """Create sample training data for testing"""
    data = {
        "session_id": "TEST_20250627_123456",
        "base_model": "mlx-community/gemma-3-1b-it-4bit",
        "output_path": "models/test_dashboard",
        "config": {
            "model": "mlx-community/gemma-3-1b-it-4bit",
            "batch_size": 4,
            "learning_rate": 5e-6,
            "max_iterations": 100,
            "lr_schedule": "cosine_decay",
            "warmup_steps": 10,
            "max_seq_length": 2048
        },
        "metrics": []
    }
    
    # Generate sample metrics
    import random
    import math
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    train_loss_base = 4.5
    val_loss_base = 4.7
    
    for i in range(101):  # 0 to 100
        # Every 25 iterations, add validation metrics
        is_val_step = (i % 25 == 0)
        
        # Calculate loss with some random noise and general downward trend
        train_loss = max(0.5, train_loss_base * (1 - i/200) + random.uniform(-0.1, 0.1))
        
        # Validation loss follows training loss but with a gap
        val_loss = val_loss_base * (1 - i/200) + random.uniform(-0.05, 0.15) if is_val_step else None
        
        # Update base values for next iteration
        train_loss_base = train_loss
        if val_loss:
            val_loss_base = val_loss
        
        # Create metric entry
        metric = {
            "iteration": i,
            "timestamp": (base_time + timedelta(minutes=i*0.5)).isoformat(),
            "train_loss": train_loss,
            "val_loss": val_loss if is_val_step else None,
            "learning_rate": 5e-6 * (1 - i/100) if i > 10 else 5e-6 * (i/10),
            "tokens_per_sec": 1000 + random.uniform(-50, 50),
            "peak_memory_gb": 2.5 + random.uniform(-0.1, 0.1)
        }
        
        # Add checkpoint info for validation steps
        if is_val_step and i > 0:
            metric["checkpoint"] = f"{i}_adapters.safetensors"
        
        data["metrics"].append(metric)
    
    return data

def test_dashboard_generation():
    """Test dashboard generation"""
    logger.info("Testing dashboard generation")
    
    # Create sample data
    data = create_sample_training_data()
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary JSON file
        json_path = os.path.join(temp_dir, "test_metrics.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        
        # Generate dashboard
        output_path = create_comprehensive_dashboard(
            json_path,
            output_dir=temp_dir,
            output_name="test_dashboard.png"
        )
        
        # Check if dashboard was created
        if output_path and os.path.exists(output_path):
            logger.info(f"Dashboard generated successfully: {output_path}")
            return True
        else:
            logger.error("Failed to generate dashboard")
            return False

def test_best_checkpoints():
    """Test best checkpoints identification"""
    logger.info("Testing best checkpoints identification")
    
    # Create sample data
    data = create_sample_training_data()
    
    # Identify best checkpoints
    best_checkpoints = identify_best_checkpoints(data, top_k=3)
    
    # Check if we got the expected number of checkpoints
    if len(best_checkpoints) == 3:
        logger.info("Found 3 best checkpoints:")
        for i, checkpoint in enumerate(best_checkpoints):
            logger.info(f"{i+1}. Iteration {checkpoint['iteration']}: {checkpoint['selection_reason']}")
        return True
    else:
        logger.error(f"Expected 3 checkpoints, got {len(best_checkpoints)}")
        return False

def run_tests():
    """Run all tests"""
    tests = [
        ("Dashboard Generation", test_dashboard_generation),
        ("Best Checkpoints", test_best_checkpoints)
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
    logger.info("Starting dashboard tests")
    success = run_tests()
    sys.exit(0 if success else 1) 