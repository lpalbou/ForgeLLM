#!/usr/bin/env python3
"""
Test script for the web interface.
"""

import os
import sys
import time
import json
import logging
import requests
import subprocess
import unittest
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestWebInterface(unittest.TestCase):
    """Test the web interface functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the web server for testing."""
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.port = 5010
        cls.base_url = f"http://localhost:{cls.port}"
        
        # Start the web server in a subprocess
        cls.process = subprocess.Popen(
            ["python", "-m", "forgellm_web", "--debug", "--port", str(cls.port)],
            cwd=cls.base_dir,
            env={**os.environ, "PYTHONPATH": cls.base_dir},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it a moment to start up
        time.sleep(2)
        
        # Check if the process is still running
        if cls.process.poll() is not None:
            stdout, stderr = cls.process.communicate()
            logger.error(f"Web server stdout: {stdout.decode('utf-8')}")
            logger.error(f"Web server stderr: {stderr.decode('utf-8')}")
            raise RuntimeError("Web server failed to start")
            
        logger.info(f"Web server started on port {cls.port}")
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get('status') == 'ok')
            logger.info("Health endpoint test passed")
        except Exception as e:
            self.fail(f"Health endpoint test failed: {e}")
    
    def test_cpt_models_endpoint(self):
        """Test the CPT models endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/cpt_models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('models', data)
            self.assertIsInstance(data['models'], list)
            logger.info("CPT models endpoint test passed")
        except Exception as e:
            self.fail(f"CPT models endpoint test failed: {e}")
    
    def test_ift_models_endpoint(self):
        """Test the IFT models endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/ift_models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('models', data)
            self.assertIsInstance(data['models'], list)
            logger.info("IFT models endpoint test passed")
        except Exception as e:
            self.fail(f"IFT models endpoint test failed: {e}")
    
    def test_base_models_endpoint(self):
        """Test the base models endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/base_models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('models', data)
            self.assertIsInstance(data['models'], list)
            logger.info("Base models endpoint test passed")
        except Exception as e:
            self.fail(f"Base models endpoint test failed: {e}")
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/model/info")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('success', data)
            logger.info("Model info endpoint test passed")
        except Exception as e:
            self.fail(f"Model info endpoint test failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Tear down the web server after testing."""
        if hasattr(cls, 'process') and cls.process is not None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
            logger.info("Web server stopped")


if __name__ == "__main__":
    unittest.main() 