#!/usr/bin/env python
"""
Tests for the ModelManager class.
"""

import os
import sys
import time
import unittest
import threading
import subprocess
from unittest.mock import patch, MagicMock

# Add parent directory to path to import from forgellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forgellm.forgellm.models.model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    """Test cases for the ModelManager class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Start the model server if it's not already running
        # This is handled by ModelManager._ensure_server_running, so we don't need to do it here
        pass
    
    def setUp(self):
        """Set up each test."""
        # Create a fresh ModelManager for each test
        # Patch the requests module to avoid actual HTTP calls
        self.requests_patcher = patch('forgellm.forgellm.models.model_manager.requests')
        self.mock_requests = self.requests_patcher.start()
        
        # Set up mock responses
        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            'success': True,
            'loaded': False,
            'is_loading': False,
            'model_name': None,
            'adapter_path': None
        }
        self.mock_requests.get.return_value = self.mock_response
        self.mock_requests.post.return_value = self.mock_response
        
        # Initialize ModelManager with mocked requests
        self.model_manager = ModelManager()
        
        # Reset the singleton for each test
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def tearDown(self):
        """Clean up after each test."""
        self.requests_patcher.stop()
    
    def test_singleton_pattern(self):
        """Test that ModelManager follows the singleton pattern."""
        manager1 = ModelManager()
        manager2 = ModelManager()
        self.assertIs(manager1, manager2)
    
    def test_initialization(self):
        """Test that ModelManager initializes correctly."""
        self.assertIsNotNone(self.model_manager.models_dir)
        self.assertIsNotNone(self.model_manager.base_models_dir)
        self.assertIsNotNone(self.model_manager.cpt_models_dir)
        self.assertIsNotNone(self.model_manager.ift_models_dir)
        self.assertIsNotNone(self.model_manager.server_url)
    
    def test_ensure_server_running(self):
        """Test that ModelManager ensures the server is running."""
        # First call should try to connect to the server
        self.mock_requests.get.side_effect = [Exception("Connection refused")]
        
        # Then it should start the server and check again
        self.mock_requests.get.side_effect = None
        
        with patch('forgellm.forgellm.models.model_manager.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            self.model_manager._ensure_server_running()
            
            # Check that Popen was called to start the server
            mock_popen.assert_called_once()
    
    def test_load_model(self):
        """Test loading a model."""
        # Set up mock response for load
        load_response = MagicMock()
        load_response.status_code = 200
        load_response.json.return_value = {
            'success': True,
            'message': 'Model test_model loading started',
            'model_name': 'test_model',
            'adapter_path': None
        }
        self.mock_requests.post.return_value = load_response
        
        # Call load
        result = self.model_manager.load('test_model')
        
        # Check that the request was made correctly
        self.mock_requests.post.assert_called_with(
            f"{self.model_manager.server_url}/api/model/load",
            json={'model_name': 'test_model', 'adapter_path': None},
            timeout=5
        )
        
        # Check that the result is correct
        self.assertTrue(result)
        self.assertEqual(self.model_manager.model_name, 'test_model')
        self.assertTrue(self.model_manager.loading)
        self.assertFalse(self.model_manager.loaded)
    
    def test_load_model_with_adapter(self):
        """Test loading a model with an adapter."""
        # Set up mock response for load
        load_response = MagicMock()
        load_response.status_code = 200
        load_response.json.return_value = {
            'success': True,
            'message': 'Model test_model loading started',
            'model_name': 'test_model',
            'adapter_path': 'test_adapter'
        }
        self.mock_requests.post.return_value = load_response
        
        # Call load
        result = self.model_manager.load('test_model', 'test_adapter')
        
        # Check that the request was made correctly
        self.mock_requests.post.assert_called_with(
            f"{self.model_manager.server_url}/api/model/load",
            json={'model_name': 'test_model', 'adapter_path': 'test_adapter'},
            timeout=5
        )
        
        # Check that the result is correct
        self.assertTrue(result)
        self.assertEqual(self.model_manager.model_name, 'test_model')
        self.assertEqual(self.model_manager.adapter_path, 'test_adapter')
        self.assertTrue(self.model_manager.loading)
        self.assertFalse(self.model_manager.loaded)
    
    def test_load_model_failure(self):
        """Test handling a failed model load."""
        # Set up mock response for load
        load_response = MagicMock()
        load_response.status_code = 500
        load_response.json.return_value = {
            'success': False,
            'error': 'Failed to load model'
        }
        self.mock_requests.post.return_value = load_response
        
        # Call load
        result = self.model_manager.load('test_model')
        
        # Check that the result is correct
        self.assertFalse(result)
        self.assertEqual(self.model_manager.error, 'HTTP error: 500')
    
    def test_check_loading_status(self):
        """Test checking the loading status."""
        # Set up mock responses for status checks
        loading_response = MagicMock()
        loading_response.status_code = 200
        loading_response.json.return_value = {
            'success': True,
            'loaded': False,
            'is_loading': True,
            'model_name': 'test_model',
            'adapter_path': None
        }
        
        loaded_response = MagicMock()
        loaded_response.status_code = 200
        loaded_response.json.return_value = {
            'success': True,
            'loaded': True,
            'is_loading': False,
            'model_name': 'test_model',
            'adapter_path': None
        }
        
        # Set up the side effect to return loading_response first, then loaded_response
        self.mock_requests.get.side_effect = [loading_response, loaded_response]
        
        # Set up the model manager state
        self.model_manager.model_name = 'test_model'
        self.model_manager.loading = True
        
        # Call _check_loading_status
        self.model_manager._check_loading_status()
        
        # Check that the requests were made correctly
        self.mock_requests.get.assert_called_with(
            f"{self.model_manager.server_url}/api/model/status",
            timeout=5
        )
        
        # Check that the state was updated correctly
        self.assertFalse(self.model_manager.loading)
        self.assertTrue(self.model_manager.loaded)
        self.assertIsNone(self.model_manager.error)
    
    def test_check_loading_status_failure(self):
        """Test handling a failed loading status check."""
        # Set up mock response for status check
        error_response = MagicMock()
        error_response.status_code = 200
        error_response.json.return_value = {
            'success': True,
            'loaded': False,
            'is_loading': False,
            'model_name': 'test_model',
            'adapter_path': None,
            'error': 'Failed to load model'
        }
        self.mock_requests.get.return_value = error_response
        
        # Set up the model manager state
        self.model_manager.model_name = 'test_model'
        self.model_manager.loading = True
        
        # Call _check_loading_status
        self.model_manager._check_loading_status()
        
        # Check that the state was updated correctly
        self.assertFalse(self.model_manager.loading)
        self.assertFalse(self.model_manager.loaded)
        self.assertEqual(self.model_manager.error, 'Failed to load model')
    
    def test_unload_model(self):
        """Test unloading a model."""
        # Set up the model manager state
        self.model_manager.model_name = 'test_model'
        self.model_manager.adapter_path = 'test_adapter'
        self.model_manager.loaded = True
        
        # Call unload
        result = self.model_manager.unload()
        
        # Check that the result is correct
        self.assertTrue(result)
        self.assertIsNone(self.model_manager.model_name)
        self.assertIsNone(self.model_manager.adapter_path)
        self.assertFalse(self.model_manager.loaded)
        self.assertFalse(self.model_manager.loading)
    
    def test_generate_text(self):
        """Test generating text."""
        # Set up mock response for generate
        generate_response = MagicMock()
        generate_response.status_code = 200
        generate_response.json.return_value = {
            'success': True,
            'text': 'Generated text',
            'generation_time': 0.5
        }
        self.mock_requests.post.return_value = generate_response
        
        # Set up the model manager state
        self.model_manager.loaded = True
        
        # Call generate
        result = self.model_manager.generate('Test prompt', max_tokens=50, temperature=0.7)
        
        # Check that the request was made correctly
        self.mock_requests.post.assert_called_with(
            f"{self.model_manager.server_url}/api/model/generate",
            json={'prompt': 'Test prompt', 'max_tokens': 50},
            timeout=30
        )
        
        # Check that the result is correct
        self.assertEqual(result, 'Generated text')
    
    def test_generate_text_no_model(self):
        """Test generating text with no model loaded."""
        # Set up the model manager state
        self.model_manager.loaded = False
        
        # Call generate
        result = self.model_manager.generate('Test prompt')
        
        # Check that the result is correct
        self.assertEqual(result, 'Error: No model loaded')
        
        # Check that no request was made
        self.mock_requests.post.assert_not_called()
    
    def test_generate_text_loading(self):
        """Test generating text while model is loading."""
        # Set up the model manager state
        self.model_manager.loaded = False
        self.model_manager.loading = True
        
        # Call generate
        result = self.model_manager.generate('Test prompt')
        
        # Check that the result is correct
        self.assertEqual(result, 'Error: Model is still loading')
        
        # Check that no request was made
        self.mock_requests.post.assert_not_called()
    
    def test_get_status(self):
        """Test getting model status."""
        # Set up mock response for status
        status_response = MagicMock()
        status_response.status_code = 200
        status_response.json.return_value = {
            'success': True,
            'loaded': True,
            'is_loading': False,
            'model_name': 'test_model',
            'adapter_path': None
        }
        self.mock_requests.get.return_value = status_response
        
        # Call get_status
        result = self.model_manager.get_status()
        
        # Check that the request was made correctly
        self.mock_requests.get.assert_called_with(
            f"{self.model_manager.server_url}/api/model/status",
            timeout=5
        )
        
        # Check that the result is correct
        self.assertEqual(result, {
            'success': True,
            'loaded': True,
            'is_loading': False,
            'model_name': 'test_model',
            'adapter_path': None
        })
    
    def test_list_models(self):
        """Test listing available models."""
        # Mock os.path.exists and os.listdir
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir') as mock_listdir, \
             patch('os.path.isdir', return_value=True):
            
            # Set up mock return values for os.listdir
            mock_listdir.side_effect = [
                ['model1', 'model2'],  # base models
                ['model3', 'model4'],  # cpt models
                ['model5', 'model6']   # ift models
            ]
            
            # Call list_models
            result = self.model_manager.list_models()
            
            # Check that the result is correct
            self.assertEqual(result, {
                'base': ['model1', 'model2'],
                'cpt': ['model3', 'model4'],
                'ift': ['model5', 'model6']
            })

if __name__ == '__main__':
    unittest.main() 