#!/usr/bin/env python3
"""
Test script for the API routes.
"""

import os
import sys
import json
import logging
import unittest
import tempfile
from pathlib import Path
from flask import Flask
from flask.testing import FlaskClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forgellm.api.routes import setup_api
from forgellm.models.model_manager import ModelManager
from forgellm.training.trainer import ContinuedPretrainer


class TestAPIRoutes(unittest.TestCase):
    """Test the API routes functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Create a Flask app for testing
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test_key'
        self.app.config['MODELS_DIR'] = os.path.join(self.temp_dir, 'models')
        
        # Create model directories
        os.makedirs(os.path.join(self.app.config['MODELS_DIR'], 'base'), exist_ok=True)
        os.makedirs(os.path.join(self.app.config['MODELS_DIR'], 'cpt'), exist_ok=True)
        os.makedirs(os.path.join(self.app.config['MODELS_DIR'], 'ift'), exist_ok=True)
        
        # Create a dummy model directory
        cpt_model_dir = os.path.join(self.app.config['MODELS_DIR'], 'cpt', 'test_model')
        os.makedirs(cpt_model_dir, exist_ok=True)
        with open(os.path.join(cpt_model_dir, 'model.safetensors'), 'w') as f:
            f.write('dummy model file')
        
        # Create a dummy training log file
        log_dir = os.path.join(self.temp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.create_dummy_training_log()
        
        # Initialize model manager and trainer
        self.app.model_manager = ModelManager()
        self.app.trainer = ContinuedPretrainer()
        
        # Register API blueprint
        self.app.register_blueprint(setup_api(self.app))
        
        # Create a test client
        self.client = self.app.test_client()
    
    def create_dummy_training_log(self):
        """Create a dummy training log file."""
        log_data = {
            "config": {
                "model_name": "test_model",
                "batch_size": 2,
                "learning_rate": 1e-5,
                "max_iterations": 10
            },
            "metrics": []
        }
        
        for i in range(5):
            metric = {
                "iteration": i,
                "train_loss": 2.0 - (i * 0.1),
                "val_loss": 2.2 - (i * 0.09) if i % 2 == 0 else 0.0,
                "learning_rate": 1e-5,
                "tokens_per_sec": 100 + i * 5,
                "peak_memory_gb": 4.0 + (i * 0.1),
                "timestamp": "2023-01-01T00:00:00"
            }
            log_data["metrics"].append(metric)
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f)
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        logger.info("Health endpoint test passed")
    
    def test_cpt_models_endpoint(self):
        """Test the CPT models endpoint."""
        response = self.client.get('/api/cpt_models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('models', data)
        # Check that at least one model is returned
        self.assertGreater(len(data['models']), 0)
        # Check that each model has a name
        for model in data['models']:
            self.assertIn('name', model)
        logger.info("CPT models endpoint test passed")
    
    def test_ift_models_endpoint(self):
        """Test the IFT models endpoint."""
        response = self.client.get('/api/ift_models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('models', data)
        logger.info("IFT models endpoint test passed")
    
    def test_base_models_endpoint(self):
        """Test the base models endpoint."""
        response = self.client.get('/api/base_models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('models', data)
        logger.info("Base models endpoint test passed")
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        response = self.client.get('/api/model/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('success', data)
        logger.info("Model info endpoint test passed")
    
    def test_training_status_endpoint(self):
        """Test the training status endpoint."""
        response = self.client.get('/api/training/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('active', data)
        logger.info("Training status endpoint test passed")
    
    def test_checkpoints_endpoint(self):
        """Test the checkpoints endpoint."""
        response = self.client.get('/api/checkpoints')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('checkpoints', data)
        logger.info("Checkpoints endpoint test passed")
    
    def test_models_endpoint(self):
        """Test the models endpoint."""
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('models', data)
        # Check that we have at least one model
        self.assertGreater(len(data['models']), 0)
        # Check that each model has a type field
        for model in data['models']:
            self.assertIn('type', model)
        logger.info("Models endpoint test passed")
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main() 