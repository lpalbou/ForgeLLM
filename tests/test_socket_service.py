#!/usr/bin/env python3
"""
Test script for the socket service.
"""

import os
import sys
import time
import json
import logging
import unittest
import tempfile
import threading
import socketio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forgellm.web.services.socket_service import TrainingMonitor, setup_socketio
from flask import Flask
from flask_socketio import SocketIO


class TestSocketService(unittest.TestCase):
    """Test the socket service functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Create a Flask app for testing
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'test_key'
        self.socketio = SocketIO(self.app, async_mode='threading')
        
        # Set up socket service
        setup_socketio(self.socketio, self.app)
        
        # Create a client for testing
        self.client = socketio.Client()
        
        # Create a dummy training log file
        self.log_file = os.path.join(self.temp_dir, "training_log.json")
        self.create_dummy_training_log()
    
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
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + i * 60))
            }
            log_data["metrics"].append(metric)
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f)
    
    def start_server(self):
        """Start the socketio server."""
        self.server_thread = threading.Thread(
            target=self.socketio.run,
            kwargs={
                'app': self.app,
                'host': '127.0.0.1',
                'port': 5050,
                'debug': False,
                'use_reloader': False
            }
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(1)  # Give the server time to start
    
    def test_training_monitor(self):
        """Test the TrainingMonitor class."""
        # Create a training monitor
        monitor = TrainingMonitor()
        
        # Test set_socketio
        monitor.set_socketio(self.socketio)
        self.assertEqual(monitor.socketio, self.socketio)
        
        # Test set_current_training
        training_data = {"test": "data"}
        monitor.set_current_training(training_data)
        self.assertEqual(monitor.current_training, training_data)
        
        # Test update_training_data
        updated_data = {"test": "updated"}
        monitor.update_training_data(updated_data)
        self.assertEqual(monitor.current_training, updated_data)
        
        # Test _add_eta
        metrics_data = {
            "config": {"max_iterations": 10},
            "metrics": [
                {"iteration": 1, "timestamp": "2023-01-01T00:00:00"},
                {"iteration": 2, "timestamp": "2023-01-01T00:01:00"}
            ]
        }
        monitor._add_eta(metrics_data)
        self.assertIn("eta", metrics_data)
        self.assertIn("eta_seconds", metrics_data)
        self.assertIn("progress_percentage", metrics_data)
        
        logger.info("TrainingMonitor test passed")
    
    def test_socket_connection(self):
        """Test socket connection and events."""
        # Start the server
        self.start_server()
        
        # Create event trackers
        connected = threading.Event()
        training_update_received = threading.Event()
        training_data = [None]
        
        # Define event handlers
        @self.client.event
        def connect():
            logger.info("Client connected")
            connected.set()
        
        @self.client.event
        def disconnect():
            logger.info("Client disconnected")
        
        @self.client.on('training_update')
        def on_training_update(data):
            logger.info("Received training update")
            training_data[0] = data
            training_update_received.set()
        
        try:
            # Connect to the server
            self.client.connect('http://127.0.0.1:5050')
            
            # Wait for connection
            self.assertTrue(connected.wait(5), "Failed to connect to server")
            
            # Request update
            self.client.emit('request_update')
            
            # Wait for update
            self.assertTrue(training_update_received.wait(5), "Failed to receive training update")
            
            # Check data
            self.assertIsNotNone(training_data[0])
            
            logger.info("Socket connection test passed")
        finally:
            # Disconnect
            if self.client.connected:
                self.client.disconnect()
    
    def test_load_training_log(self):
        """Test loading training log via socket."""
        # Start the server
        self.start_server()
        
        # Create event trackers
        connected = threading.Event()
        training_update_received = threading.Event()
        training_data = [None]
        
        # Define event handlers
        @self.client.event
        def connect():
            logger.info("Client connected")
            connected.set()
        
        @self.client.on('training_update')
        def on_training_update(data):
            logger.info("Received training update")
            training_data[0] = data
            training_update_received.set()
        
        try:
            # Connect to the server
            self.client.connect('http://127.0.0.1:5050')
            
            # Wait for connection
            self.assertTrue(connected.wait(5), "Failed to connect to server")
            
            # Load training log
            self.client.emit('load_training_log', {'log_file': self.log_file})
            
            # Wait for update
            self.assertTrue(training_update_received.wait(5), "Failed to receive training update")
            
            # Check data
            self.assertIsNotNone(training_data[0])
            self.assertIn('metrics', training_data[0])
            self.assertEqual(len(training_data[0]['metrics']), 5)
            self.assertIn('best_checkpoints', training_data[0])
            
            logger.info("Load training log test passed")
        finally:
            # Disconnect
            if self.client.connected:
                self.client.disconnect()
    
    def test_system_info(self):
        """Test getting system info via socket."""
        # Start the server
        self.start_server()
        
        # Create event trackers
        connected = threading.Event()
        system_info_received = threading.Event()
        system_info = [None]
        
        # Define event handlers
        @self.client.event
        def connect():
            logger.info("Client connected")
            connected.set()
        
        @self.client.on('system_info')
        def on_system_info(data):
            logger.info("Received system info")
            system_info[0] = data
            system_info_received.set()
        
        try:
            # Connect to the server
            self.client.connect('http://127.0.0.1:5050')
            
            # Wait for connection
            self.assertTrue(connected.wait(5), "Failed to connect to server")
            
            # Request system info
            self.client.emit('get_system_info')
            
            # Wait for response
            self.assertTrue(system_info_received.wait(5), "Failed to receive system info")
            
            # Check data
            self.assertIsNotNone(system_info[0])
            self.assertIn('cpu', system_info[0])
            self.assertIn('memory', system_info[0])
            self.assertIn('disk', system_info[0])
            
            logger.info("System info test passed")
        finally:
            # Disconnect
            if self.client.connected:
                self.client.disconnect()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the server if running
        if hasattr(self, 'socketio') and self.socketio:
            self.socketio.stop()
        
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main() 