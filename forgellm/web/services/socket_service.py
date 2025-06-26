"""
Socket.IO service for real-time updates
"""

import logging
import json
from typing import Dict, Any, Optional
from flask_socketio import SocketIO, emit
from pathlib import Path

from ...training.dashboard import load_training_data

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and emit updates"""
    
    def __init__(self):
        """Initialize training monitor"""
        self.current_training = None
        self.socketio = None
    
    def set_socketio(self, socketio: SocketIO):
        """Set the Socket.IO instance"""
        self.socketio = socketio
    
    def set_current_training(self, training_data: Dict[str, Any]):
        """Set the current training data"""
        self.current_training = training_data
    
    def emit_update(self):
        """Emit training update"""
        if self.socketio and self.current_training:
            self.socketio.emit('training_update', self.current_training)
    
    def emit_finished(self, data: Dict[str, Any]):
        """Emit training finished"""
        if self.socketio:
            self.socketio.emit('training_finished', data)

# Create singleton instance
training_monitor = TrainingMonitor()

def setup_socketio(socketio: SocketIO, app=None):
    """Set up Socket.IO events
    
    Args:
        socketio: Socket.IO instance
        app: Flask application
    """
    # Set socketio instance in training monitor
    training_monitor.set_socketio(socketio)
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info('Client connected')
        emit('connected', {'status': 'connected'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info('Client disconnected')
    
    @socketio.on('request_update')
    def handle_request_update():
        """Handle request for update"""
        logger.info('Client requested update')
        if training_monitor.current_training:
            emit('training_update', training_monitor.current_training)
        else:
            # Get training status from app
            if app and hasattr(app, 'trainer'):
                trainer = app.trainer
                if trainer.is_training_active():
                    status = trainer.get_training_status()
                    emit('training_update', status)
                else:
                    emit('training_update', {'active': False})
            else:
                emit('training_update', {'active': False})
    
    @socketio.on('check_training_status')
    def handle_check_training_status():
        """Handle check training status"""
        logger.info('Client checked training status')
        if app and hasattr(app, 'trainer'):
            trainer = app.trainer
            if trainer.is_training_active():
                status = trainer.get_training_status()
                emit('training_update', status)
            else:
                emit('training_update', {'active': False})
        else:
            emit('training_update', {'active': False})
    
    @socketio.on('load_training_log')
    def handle_load_training_log(data):
        """Handle load training log"""
        logger.info('Client loaded training log')
        log_file = data.get('log_file')
        if log_file:
            try:
                training_data = load_training_data(log_file)
                emit('training_update', training_data)
            except Exception as e:
                logger.error(f"Error loading training log: {e}")
                emit('error', {'message': f"Error loading training log: {e}"})
    
    @socketio.on('start_generation')
    def handle_start_generation(data):
        """Handle start generation"""
        logger.info('Client started generation')
        if app and hasattr(app, 'model_manager'):
            model_manager = app.model_manager
            try:
                text = model_manager.generate_text(data)
                emit('generation_result', {'text': text})
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                emit('error', {'message': f"Error generating text: {e}"})
        else:
            emit('error', {'message': "Model manager not available"})
    
    @socketio.on('stop_generation')
    def handle_stop_generation():
        """Handle stop generation"""
        logger.info('Client stopped generation')
        if app and hasattr(app, 'model_manager'):
            model_manager = app.model_manager
            try:
                model_manager.stop_generation()
                emit('generation_stopped')
            except Exception as e:
                logger.error(f"Error stopping generation: {e}")
                emit('error', {'message': f"Error stopping generation: {e}"})
        else:
            emit('error', {'message': "Model manager not available"})
    
    logger.info("Socket.IO events set up successfully")
    return socketio 