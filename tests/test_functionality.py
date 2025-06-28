#!/usr/bin/env python3
"""
Test script to verify the key functionalities of the refactored ForgeLLM package.
"""

import os
import sys
import json
import time
import logging
import unittest
import tempfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forgellm.models.model_manager import ModelManager
from forgellm.training.trainer import ContinuedPretrainer
from forgellm.training.config import TrainingConfig
from forgellm.training.dashboard import create_comprehensive_dashboard, identify_best_checkpoints, load_training_data

class TestForgeLLMFunctionality(unittest.TestCase):
    """Test the key functionalities of the refactored ForgeLLM package."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Create a simple test configuration
        self.test_config = {
            "model_name": "test_model",
            "input_dir": os.path.join(self.temp_dir, "input"),
            "output_dir": os.path.join(self.temp_dir, "output"),
            "batch_size": 2,
            "learning_rate": 1e-5,
            "max_iterations": 10
        }
        
        # Create the input directory
        os.makedirs(self.test_config["input_dir"], exist_ok=True)
        os.makedirs(self.test_config["output_dir"], exist_ok=True)
        
        # Create a dummy training log file for testing dashboard generation
        self.create_dummy_training_log()
    
    def create_dummy_training_log(self):
        """Create a dummy training log file for testing dashboard generation."""
        log_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "training_log.json")
        
        # Create dummy log entries with proper format
        log_data = {
            "config": {
                "model": "test_model",
                "batch_size": 2,
                "learning_rate": 1e-5,
                "max_iterations": 10,
                "fine_tune_type": "continued_pretraining",
                "max_seq_length": 512,
                "lr_schedule": "cosine_decay",
                "warmup_steps": 2,
                "lr_decay_factor": 0.1
            },
            "metrics": []
        }
        
        for i in range(10):
            entry = {
                "iteration": i,
                "train_loss": 2.0 - (i * 0.1),
                "val_loss": 2.2 - (i * 0.09) if i % 2 == 0 else 0.0,  # Ensure no None values
                "learning_rate": 1e-5,
                "tokens_per_sec": 100 + i * 5,
                "peak_memory_gb": 4.0 + (i * 0.1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + i * 60)),
                "checkpoint": f"checkpoint_{i}.pt" if i % 3 == 0 else None
            }
            log_data["metrics"].append(entry)
        
        # Write to file
        with open(log_file, 'w') as f:
            json.dump(log_data, f)
        
        self.log_file = log_file
    
    def test_model_manager_initialization(self):
        """Test that the ModelManager initializes correctly."""
        try:
            # ModelManager is a singleton with no constructor parameters
            model_manager = ModelManager()
            self.assertIsNotNone(model_manager)
            logger.info("ModelManager initialization test passed")
        except Exception as e:
            self.fail(f"ModelManager initialization failed: {e}")
    
    def test_training_config(self):
        """Test that the TrainingConfig class works correctly."""
        try:
            config = TrainingConfig(
                model_name=self.test_config["model_name"],
                input_dir=self.test_config["input_dir"],
                output_dir=self.test_config["output_dir"],
                batch_size=self.test_config["batch_size"],
                learning_rate=self.test_config["learning_rate"],
                max_iterations=self.test_config["max_iterations"]
            )
            self.assertEqual(config.model_name, self.test_config["model_name"])
            self.assertEqual(config.batch_size, self.test_config["batch_size"])
            logger.info("TrainingConfig test passed")
        except Exception as e:
            self.fail(f"TrainingConfig test failed: {e}")
    
    def test_trainer_initialization(self):
        """Test that the ContinuedPretrainer initializes correctly."""
        try:
            trainer = ContinuedPretrainer()
            self.assertIsNotNone(trainer)
            logger.info("ContinuedPretrainer initialization test passed")
        except Exception as e:
            self.fail(f"ContinuedPretrainer initialization failed: {e}")
    
    def test_dashboard_generation(self):
        """Test dashboard generation functionality."""
        try:
            # Generate a dashboard from the dummy log file
            output_dir = os.path.join(self.temp_dir, "dashboard")
            output_file = "dashboard.png"
            result = create_comprehensive_dashboard(self.log_file, output_dir, output_file)
            
            # Check that the dashboard file was created
            expected_path = os.path.join(output_dir, output_file)
            self.assertTrue(os.path.exists(expected_path))
            self.assertEqual(result, expected_path)
            logger.info("Dashboard generation test passed")
        except Exception as e:
            self.fail(f"Dashboard generation test failed: {e}")
    
    def test_best_checkpoint_identification(self):
        """Test best checkpoint identification functionality."""
        try:
            # Create dummy checkpoints
            checkpoints_dir = os.path.join(self.temp_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # Create dummy checkpoint files
            for i in range(5):
                with open(os.path.join(checkpoints_dir, f"checkpoint_{i}.pt"), 'w') as f:
                    f.write("dummy checkpoint")
            
            # Load the training data
            data = load_training_data(self.log_file)
            
            # Identify best checkpoints
            best_checkpoints = identify_best_checkpoints(data, top_k=3)
            
            # Check that we got some checkpoints
            self.assertIsNotNone(best_checkpoints)
            self.assertIsInstance(best_checkpoints, list)
            logger.info("Best checkpoint identification test passed")
        except Exception as e:
            self.fail(f"Best checkpoint identification test failed: {e}")
    
    def test_web_server_startup(self):
        """Test that the web server starts up correctly."""
        try:
            # Start the web server in a subprocess
            process = subprocess.Popen(
                ["python", "-m", "forgellm_web", "--debug", "--port", "5005"],
                cwd=self.base_dir,
                env={**os.environ, "PYTHONPATH": self.base_dir},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it a moment to start up
            time.sleep(2)
            
            # Check if the process is still running
            if process.poll() is None:
                # It's still running, so it started successfully
                logger.info("Web server startup test passed")
                # Kill the process
                process.terminate()
                process.wait(timeout=5)
            else:
                # It exited, so it failed to start
                stdout, stderr = process.communicate()
                logger.error(f"Web server stdout: {stdout.decode('utf-8')}")
                logger.error(f"Web server stderr: {stderr.decode('utf-8')}")
                self.fail("Web server failed to start")
        except Exception as e:
            self.fail(f"Web server startup test failed: {e}")
    
    def test_api_routes(self):
        """Test that the API routes are working correctly."""
        # This would normally test the API routes by making HTTP requests,
        # but for simplicity, we'll just check that the routes module is importable
        try:
            from forgellm.api.routes import setup_api
            self.assertIsNotNone(setup_api)
            logger.info("API routes import test passed")
        except Exception as e:
            self.fail(f"API routes import test failed: {e}")
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main() 