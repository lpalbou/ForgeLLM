#!/usr/bin/env python
"""
Tests for the model server component.
"""

import os
import sys
import time
import json
import unittest
import threading
import requests
import subprocess
from pathlib import Path

# Add parent directory to path to import from forgellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestModelServer(unittest.TestCase):
    """Test cases for the model server."""
    
    @classmethod
    def setUpClass(cls):
        """Start the model server for testing."""
        # Find the model server script
        model_server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_server.py')
        
        if not os.path.exists(model_server_path):
            raise FileNotFoundError(f"Model server script not found at {model_server_path}")
        
        # Use a different port for testing to avoid conflicts
        cls.port = 5002
        cls.server_url = f"http://localhost:{cls.port}"
        
        # Start the server as a subprocess
        cls.server_process = subprocess.Popen(
            [
                "python", 
                model_server_path, 
                "--host", "localhost", 
                "--port", str(cls.port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        for _ in range(10):
            try:
                response = requests.get(f"{cls.server_url}/api/model/status", timeout=1)
                if response.status_code == 200:
                    print("Model server started successfully")
                    break
            except:
                time.sleep(0.5)
        else:
            cls.tearDownClass()
            raise RuntimeError("Failed to start model server")
    
    @classmethod
    def tearDownClass(cls):
        """Stop the model server after testing."""
        if hasattr(cls, 'server_process'):
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
    
    def test_server_status(self):
        """Test that the server status endpoint works."""
        response = requests.get(f"{self.server_url}/api/model/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('loaded', data)
        self.assertIn('is_loading', data)
    
    def test_model_loading(self):
        """Test model loading functionality."""
        # This test assumes you have a small test model available
        # You might want to skip this if no test model is available
        test_model = os.environ.get('TEST_MODEL', 'mlx-community/gemma-3-1b-it-bf16')
        
        # Check if we should skip this test
        if not test_model or test_model == 'skip':
            self.skipTest("No test model specified")
        
        # Send request to load the model
        response = requests.post(
            f"{self.server_url}/api/model/load",
            json={'model_name': test_model},
            timeout=5
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['model_name'], test_model)
        
        # Wait for the model to load (up to 60 seconds)
        start_time = time.time()
        while time.time() - start_time < 60:
            response = requests.get(f"{self.server_url}/api/model/status")
            data = response.json()
            
            if data.get('loaded'):
                break
            
            if not data.get('is_loading'):
                # If it's not loading and not loaded, there was an error
                self.fail(f"Model loading failed: {data.get('error')}")
            
            time.sleep(1)
        else:
            self.fail("Model loading timed out")
    
    def test_text_generation(self):
        """Test text generation functionality."""
        # This test depends on test_model_loading, so skip if that's skipped
        test_model = os.environ.get('TEST_MODEL', 'mlx-community/gemma-3-1b-it-bf16')
        if not test_model or test_model == 'skip':
            self.skipTest("No test model specified")
        
        # First load the model if not already loaded
        response = requests.get(f"{self.server_url}/api/model/status")
        data = response.json()
        
        if not data.get('loaded'):
            # Load the model
            response = requests.post(
                f"{self.server_url}/api/model/load",
                json={'model_name': test_model},
                timeout=5
            )
            
            # Wait for the model to load
            start_time = time.time()
            while time.time() - start_time < 60:
                response = requests.get(f"{self.server_url}/api/model/status")
                data = response.json()
                
                if data.get('loaded'):
                    break
                
                if not data.get('is_loading'):
                    self.fail(f"Model loading failed: {data.get('error')}")
                
                time.sleep(1)
            else:
                self.fail("Model loading timed out")
        
        # Now generate text
        response = requests.post(
            f"{self.server_url}/api/model/generate",
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
        self.assertIn('generation_time', data)
        self.assertIsInstance(data['text'], str)
        self.assertTrue(len(data['text']) > 0)
    
    def test_error_handling(self):
        """Test error handling in the model server."""
        # Test invalid JSON
        response = requests.post(
            f"{self.server_url}/api/model/load",
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        self.assertEqual(response.status_code, 400)
        
        # Test missing model_name
        response = requests.post(
            f"{self.server_url}/api/model/load",
            json={},
            timeout=5
        )
        self.assertEqual(response.status_code, 400)
        
        # Test invalid endpoint
        response = requests.get(f"{self.server_url}/api/invalid_endpoint")
        self.assertEqual(response.status_code, 404)
        
        # Test generation without a model loaded
        # First unload any loaded model
        requests.post(f"{self.server_url}/api/model/unload", json={})
        
        # Then try to generate text
        response = requests.post(
            f"{self.server_url}/api/model/generate",
            json={
                'prompt': 'Hello, world!',
                'max_tokens': 20
            },
            timeout=5
        )
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main() 