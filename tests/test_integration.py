#!/usr/bin/env python
"""
Integration tests for the ForgeLLM system.
"""

import os
import sys
import time
import json
import unittest
import threading
import subprocess
import requests
from pathlib import Path

# Add parent directory to path to import from forgellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestIntegration(unittest.TestCase):
    """Integration tests for the ForgeLLM system."""
    
    @classmethod
    def setUpClass(cls):
        """Start the web server and model server for testing."""
        # Find the server scripts
        model_server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_server.py')
        web_server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'forgellm_web.py')
        
        if not os.path.exists(model_server_path):
            raise FileNotFoundError(f"Model server script not found at {model_server_path}")
        
        if not os.path.exists(web_server_path):
            raise FileNotFoundError(f"Web server script not found at {web_server_path}")
        
        # Use different ports for testing to avoid conflicts
        cls.model_port = 5002
        cls.web_port = 5003
        
        cls.model_server_url = f"http://localhost:{cls.model_port}"
        cls.web_server_url = f"http://localhost:{cls.web_port}"
        
        # Start the model server as a subprocess
        cls.model_server_process = subprocess.Popen(
            [
                "python", 
                model_server_path, 
                "--host", "localhost", 
                "--port", str(cls.model_port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the model server to start
        for _ in range(10):
            try:
                response = requests.get(f"{cls.model_server_url}/api/model/status", timeout=1)
                if response.status_code == 200:
                    print("Model server started successfully")
                    break
            except:
                time.sleep(0.5)
        else:
            cls.tearDownClass()
            raise RuntimeError("Failed to start model server")
        
        # Start the web server as a subprocess
        # Set environment variables to point to the test model server
        env = os.environ.copy()
        env['MODEL_SERVER_URL'] = cls.model_server_url
        
        cls.web_server_process = subprocess.Popen(
            [
                "python", 
                web_server_path, 
                "--host", "localhost", 
                "--port", str(cls.web_port),
                "--debug", "False"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Wait for the web server to start
        for _ in range(10):
            try:
                response = requests.get(f"{cls.web_server_url}/api/health", timeout=1)
                if response.status_code == 200:
                    print("Web server started successfully")
                    break
            except:
                time.sleep(0.5)
        else:
            cls.tearDownClass()
            raise RuntimeError("Failed to start web server")
    
    @classmethod
    def tearDownClass(cls):
        """Stop the web server and model server after testing."""
        if hasattr(cls, 'web_server_process'):
            cls.web_server_process.terminate()
            try:
                cls.web_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.web_server_process.kill()
        
        if hasattr(cls, 'model_server_process'):
            cls.model_server_process.terminate()
            try:
                cls.model_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.model_server_process.kill()
    
    def test_health_endpoint(self):
        """Test that the health endpoint works."""
        response = requests.get(f"{self.web_server_url}/api/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')
    
    def test_model_status_endpoint(self):
        """Test that the model status endpoint works."""
        response = requests.get(f"{self.web_server_url}/api/model/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('loaded', data)
        self.assertIn('is_loading', data)
    
    def test_model_loading_and_generation(self):
        """Test the complete flow of loading a model and generating text."""
        # This test assumes you have a small test model available
        # You might want to skip this if no test model is available
        test_model = os.environ.get('TEST_MODEL', 'mlx-community/gemma-3-1b-it-bf16')
        
        # Check if we should skip this test
        if not test_model or test_model == 'skip':
            self.skipTest("No test model specified")
        
        # Send request to load the model through the web server
        response = requests.post(
            f"{self.web_server_url}/api/model/load",
            json={'model_name': test_model},
            timeout=5
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        
        # Wait for the model to load (up to 60 seconds)
        start_time = time.time()
        while time.time() - start_time < 60:
            response = requests.get(f"{self.web_server_url}/api/model/status")
            data = response.json()
            
            if data.get('loaded'):
                break
            
            if not data.get('is_loading'):
                # If it's not loading and not loaded, there was an error
                self.fail(f"Model loading failed: {data.get('error')}")
            
            time.sleep(1)
        else:
            self.fail("Model loading timed out")
        
        # Now generate text
        response = requests.post(
            f"{self.web_server_url}/api/model/generate",
            json={
                'prompt': 'Hello, world!',
                'max_tokens': 20
            },
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('text', data)
        self.assertIsInstance(data['text'], str)
        self.assertTrue(len(data['text']) > 0)
    
    def test_model_list_endpoint(self):
        """Test that the model list endpoint works."""
        response = requests.get(f"{self.web_server_url}/api/models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('models', data)
        self.assertIn('base', data['models'])
        self.assertIn('cpt', data['models'])
        self.assertIn('ift', data['models'])
    
    def test_training_status_endpoint(self):
        """Test that the training status endpoint works."""
        response = requests.get(f"{self.web_server_url}/api/training/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('active', data)
    
    def test_error_handling(self):
        """Test error handling in the web server."""
        # Test invalid JSON
        response = requests.post(
            f"{self.web_server_url}/api/model/load",
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        self.assertEqual(response.status_code, 400)
        
        # Test missing model_name
        response = requests.post(
            f"{self.web_server_url}/api/model/load",
            json={},
            timeout=5
        )
        self.assertEqual(response.status_code, 400)
        
        # Test invalid endpoint
        response = requests.get(f"{self.web_server_url}/api/invalid_endpoint")
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main() 