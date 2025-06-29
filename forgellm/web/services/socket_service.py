"""
Socket.IO service for real-time updates
"""

import logging
import json
import os
import time
from typing import Dict, Any, Optional
from flask_socketio import SocketIO, emit
from pathlib import Path

from ...training.dashboard import load_training_data, identify_best_checkpoints

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and emit updates"""
    
    def __init__(self):
        """Initialize training monitor"""
        self.current_training = None
        self.socketio = None
        self.last_update_time = 0
        self.update_interval = 3.0  # Update interval in seconds (increased from 1.0)
    
    def set_socketio(self, socketio: SocketIO):
        """Set the Socket.IO instance"""
        self.socketio = socketio
    
    def set_current_training(self, training_data: Dict[str, Any]):
        """Set the current training data"""
        self.current_training = training_data
        self.emit_update()
    
    def update_training_data(self, training_data: Dict[str, Any]):
        """Update training data and emit if enough time has passed"""
        current_time = time.time()
        time_since_last = current_time - self.last_update_time
        self.current_training = training_data
        
        # Only emit updates at most once per update_interval seconds
        if time_since_last >= self.update_interval:
            logger.info(f"🔄 Emitting training update (last update {time_since_last:.1f}s ago)")
            self.emit_update()
            self.last_update_time = current_time
        else:
            logger.debug(f"⏳ Throttling update (only {time_since_last:.1f}s since last, need {self.update_interval}s)")
    
    def emit_update(self):
        """Emit training update"""
        if self.socketio and self.current_training:
            try:
                # Add best checkpoints if available
                if 'metrics' in self.current_training and len(self.current_training['metrics']) > 5:
                    try:
                        best_checkpoints = identify_best_checkpoints(self.current_training)
                        self.current_training['best_checkpoints'] = best_checkpoints
                    except Exception as e:
                        logger.warning(f"Failed to identify best checkpoints: {e}")
                
                # Add estimated time remaining if possible
                if 'metrics' in self.current_training and len(self.current_training['metrics']) > 2:
                    try:
                        self._add_eta(self.current_training)
                    except Exception as e:
                        logger.warning(f"Failed to calculate ETA: {e}")
                
                self.socketio.emit('training_update', self.current_training)
            except Exception as e:
                logger.error(f"Error emitting training update: {e}")
    
    def _add_eta(self, training_data: Dict[str, Any]):
        """Add estimated time remaining to training data"""
        metrics = training_data.get('metrics', [])
        config = training_data.get('config', {})
        
        if not metrics or len(metrics) < 2:
            return
        
        # Get current iteration and max iterations
        current_iteration = metrics[-1].get('iteration', 0)
        max_iterations = config.get('max_iterations', 0)
        
        if max_iterations <= 0 or current_iteration >= max_iterations:
            return
        
        # Calculate average time per iteration from recent iterations
        recent_metrics = metrics[-min(10, len(metrics)):]
        if len(recent_metrics) < 2:
            return
        
        try:
            # Calculate time differences
            times = []
            for i in range(1, len(recent_metrics)):
                prev_time = recent_metrics[i-1].get('timestamp')
                curr_time = recent_metrics[i].get('timestamp')
                
                if isinstance(prev_time, str) and isinstance(curr_time, str):
                    from datetime import datetime
                    prev_dt = datetime.fromisoformat(prev_time)
                    curr_dt = datetime.fromisoformat(curr_time)
                    diff_seconds = (curr_dt - prev_dt).total_seconds()
                    times.append(diff_seconds)
                elif isinstance(prev_time, (int, float)) and isinstance(curr_time, (int, float)):
                    diff_seconds = curr_time - prev_time
                    times.append(diff_seconds)
            
            if times:
                avg_time_per_iter = sum(times) / len(times)
                remaining_iterations = max_iterations - current_iteration
                eta_seconds = avg_time_per_iter * remaining_iterations
                
                # Format ETA
                if eta_seconds < 60:
                    eta = f"{int(eta_seconds)} seconds"
                elif eta_seconds < 3600:
                    eta = f"{int(eta_seconds / 60)} minutes"
                else:
                    eta = f"{eta_seconds / 3600:.1f} hours"
                
                # Add ETA to training data
                training_data['eta'] = eta
                training_data['eta_seconds'] = eta_seconds
                training_data['progress_percentage'] = (current_iteration / max_iterations) * 100
        except Exception as e:
            logger.warning(f"Error calculating ETA: {e}")
    
    def emit_finished(self, data: Dict[str, Any]):
        """Emit training finished"""
        if self.socketio:
            try:
                # Add best checkpoints if available
                if 'metrics' in data and len(data['metrics']) > 5:
                    try:
                        best_checkpoints = identify_best_checkpoints(data)
                        data['best_checkpoints'] = best_checkpoints
                    except Exception as e:
                        logger.warning(f"Failed to identify best checkpoints: {e}")
                
                self.socketio.emit('training_finished', data)
            except Exception as e:
                logger.error(f"Error emitting training finished: {e}")

    def start_background_updates(self):
        """Start background updates - DISABLED to prevent API call spam"""
        logger.info('🚫 Socket background updates DISABLED - using main app single update approach')
        # Background updates disabled to prevent excessive API calls
        # All updates now handled by main app.js performSingleUpdate() method
        pass

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
        
        # Send initial state if training is active
        if app and hasattr(app, 'trainer'):
            trainer = app.trainer
            if trainer.is_training_active():
                status = trainer.get_training_status()
                emit('training_update', status)
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info('Client disconnected')
    
    @socketio.on('request_update')
    def handle_request_update():
        """Handle request for update"""
        logger.info('🔌 Client requested update')
        if training_monitor.current_training:
            logger.info('📤 Sending cached training data')
            emit('training_update', training_monitor.current_training)
        else:
            # Get training status from app
            if app and hasattr(app, 'trainer'):
                trainer = app.trainer
                if trainer.is_training_active():
                    logger.info('📊 Getting fresh training status')
                    status = trainer.get_training_status()
                    emit('training_update', status)
                else:
                    logger.info('⏹️ Training not active')
                    emit('training_update', {'active': False})
            else:
                logger.info('❌ No trainer available')
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
                
                # Add best checkpoints if available
                if 'metrics' in training_data and len(training_data['metrics']) > 5:
                    try:
                        best_checkpoints = identify_best_checkpoints(training_data)
                        training_data['best_checkpoints'] = best_checkpoints
                    except Exception as e:
                        logger.warning(f"Failed to identify best checkpoints: {e}")
                
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
    
    @socketio.on('get_system_info')
    def handle_get_system_info():
        """Handle get system info"""
        logger.info('Client requested system info')
        try:
            import psutil
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)
            
            # Get GPU info if available
            gpu_info = {}
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                       capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split(',')
                    if len(gpu_data) >= 3:
                        gpu_info = {
                            'utilization': float(gpu_data[0]),
                            'memory_used_gb': float(gpu_data[1]) / 1024,
                            'memory_total_gb': float(gpu_data[2]) / 1024,
                            'memory_percent': (float(gpu_data[1]) / float(gpu_data[2])) * 100 if float(gpu_data[2]) > 0 else 0
                        }
            except Exception as e:
                logger.warning(f"Error getting GPU info: {e}")
            
            # Get model memory usage if available
            model_memory = 0
            if app and hasattr(app, 'model_manager'):
                model_manager = app.model_manager
                if hasattr(model_manager, 'memory_usage_gb'):
                    try:
                        model_memory = model_manager.memory_usage_gb()
                    except Exception as e:
                        logger.warning(f"Error getting model memory usage: {e}")
            
            system_info = {
                'cpu': {
                    'percent': cpu_percent
                },
                'memory': {
                    'percent': memory_percent,
                    'used_gb': memory_used_gb,
                    'total_gb': memory_total_gb
                },
                'disk': {
                    'percent': disk_percent,
                    'used_gb': disk_used_gb,
                    'total_gb': disk_total_gb
                },
                'model': {
                    'memory_gb': model_memory
                }
            }
            
            if gpu_info:
                system_info['gpu'] = gpu_info
            
            emit('system_info', system_info)
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            emit('error', {'message': f"Error getting system info: {e}"})
    
    logger.info("Socket.IO events set up successfully")
    return socketio 