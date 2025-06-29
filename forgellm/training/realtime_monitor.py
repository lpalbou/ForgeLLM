"""
Real-time Training Monitor
Watches training files and provides live metrics without interfering with existing trainer code
"""

import json
import logging
import threading
import time
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RealtimeTrainingMonitor:
    """Real-time monitor that watches training files and provides live updates"""
    
    def __init__(self):
        self._current_training_file = None
        self._last_metrics = []
        self._is_monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        self._last_update = None
        
    def start_monitoring(self):
        """Start monitoring training files"""
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Real-time training monitor started")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Real-time training monitor stopped")
        
    def _check_mlx_processes_running(self):
        """Check if any MLX training processes are actually running
        
        Returns:
            bool: True if MLX training processes are found, False otherwise
        """
        import psutil
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        # Look for actual MLX training commands
                        if any(pattern in cmdline for pattern in [
                            'mlx_lm.lora',      # MLX LoRA training
                            'mlx_lm.fuse',      # MLX model fusion
                            'mlx-lm',           # MLX-LM package calls
                            'mlx_lm',           # MLX-LM package calls
                        ]):
                            logger.info(f"RealtimeMonitor: Found active MLX process: PID {proc.info['pid']}")
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.warning(f"Error checking MLX processes: {e}")
        
        return False

    def _find_active_training_file(self) -> Optional[str]:
        """Find the most recent active training file"""
        # FIRST: Check if any MLX processes are actually running
        if not self._check_mlx_processes_running():
            logger.info("RealtimeMonitor: No MLX processes running - no active training")
            return None
        
        possible_dirs = [
            Path("models/cpt")
        ]
        
        all_log_files = []
        
        # Look for recent training files
        for models_dir in possible_dirs:
            if models_dir.exists():
                pattern = str(models_dir / "*" / "CPT_*.json")
                log_files = glob.glob(pattern)
                all_log_files.extend(log_files)
        
        if not all_log_files:
            logger.info("RealtimeMonitor: No training log files found")
            return None
        
        # Find the most recent file that's actively being updated
        most_recent = None
        most_recent_time = 0
        
        for log_file in all_log_files:
            try:
                # Check modification time - only consider very recent files (last 10 minutes)
                mtime = os.path.getmtime(log_file)
                
                if time.time() - mtime < 600:  # 10 minutes
                    # Also check if the file doesn't have an end_time
                    try:
                        with open(log_file, 'r') as f:
                            data = json.load(f)
                            if data.get('end_time') is None:
                                if mtime > most_recent_time:
                                    most_recent_time = mtime
                                    most_recent = log_file
                                    logger.info(f"RealtimeMonitor: Found potential active training: {log_file}")
                    except:
                        continue
            except Exception:
                continue
        
        if most_recent:
            logger.info(f"RealtimeMonitor: Selected active training file: {most_recent}")
        else:
            logger.info("RealtimeMonitor: No active training files found despite MLX processes")
        
        return most_recent
    
    def _read_training_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read training file safely"""
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return None
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError) as e:
            # File might be being written to, ignore errors
            return None
        except Exception as e:
            logger.warning(f"Error reading training file {file_path}: {e}")
            return None
    
    def _monitor_loop(self):
        """Main monitoring loop - optimized to reduce unnecessary work"""
        while self._is_monitoring:
            try:
                # SMART MONITORING: Check MLX processes first (lightweight)
                mlx_running = self._check_mlx_processes_running()
                
                if not mlx_running:
                    # No MLX processes - sleep longer and skip file operations
                    with self._lock:
                        self._last_metrics = None
                        self._current_training_file = None
                        self._last_update = None
                    
                    # Sleep much longer when no training is active
                    time.sleep(10)  # 10 seconds when inactive
                    continue
                
                # MLX is running - do full monitoring
                logger.info("RealtimeMonitor: MLX processes detected - full monitoring active")
                
                # Find current active training file
                active_file = self._find_active_training_file()
                
                if active_file != self._current_training_file:
                    self._current_training_file = active_file
                    if active_file:
                        logger.info(f"Monitoring new training file: {active_file}")
                
                if self._current_training_file:
                    # Read the training data
                    data = self._read_training_file(self._current_training_file)
                    
                    if data and 'metrics' in data:
                        with self._lock:
                            self._last_metrics = data['metrics']
                            self._last_update = datetime.now()
                
                # Sleep shorter when training is active
                time.sleep(3)  # 3 seconds when training is active
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics - optimized to avoid redundant checks"""
        with self._lock:
            # Use cached state from monitor loop instead of redundant MLX check
            has_metrics = bool(self._last_metrics)
            has_recent_update = (
                self._last_update and 
                datetime.now() - self._last_update < timedelta(minutes=2)
            )
            
            # Training is active if we have recent metrics (monitor loop handles MLX checking)
            is_active = has_metrics and has_recent_update
            
            if not is_active:
                logger.info(f"RealtimeMonitor: Training inactive - metrics={has_metrics}, recent_update={has_recent_update}")
                return {
                    'active': False,
                    'metrics': [],
                    'last_update': None,
                    'message': 'No active training detected'
                }
            
            logger.info(f"RealtimeMonitor: Training active - using cached metrics")
            
            return {
                'active': True,
                'metrics': self._last_metrics.copy(),
                'last_update': self._last_update.isoformat(),
                'training_file': self._current_training_file
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        current_data = self.get_current_metrics()
        
        # ALWAYS try to get config from most recent training session
        # even when no training is currently active
        most_recent_config = self._get_most_recent_config()
        if most_recent_config:
            current_data['config'] = most_recent_config
        
        if not current_data['metrics']:
            return current_data
        
        # Read full training data to get config and other info
        if self._current_training_file:
            try:
                full_data = self._read_training_file(self._current_training_file)
                if full_data:
                    # Include config and other metadata
                    current_data['config'] = full_data.get('config', {})
                    current_data['start_time'] = full_data.get('start_time')
                    current_data['status'] = full_data.get('status', 'running')
                    
                    # Add any other fields from the training file
                    for key in ['model_name', 'output_dir', 'dataset_info']:
                        if key in full_data:
                            current_data[key] = full_data[key]
            except Exception as e:
                logger.warning(f"Error reading full training data: {e}")
        
        # Generate charts using the existing chart generation function
        try:
            from .dashboard import generate_web_chart_data
            
            # Format data for chart generation
            chart_data = {
                'metrics': current_data['metrics']
            }
            
            charts = generate_web_chart_data(chart_data)
            if charts:
                current_data['charts'] = charts
                
        except Exception as e:
            logger.warning(f"Error generating charts: {e}")
        
        # Add current values for display - INCLUDE ALL AVAILABLE FIELDS
        if current_data['metrics']:
            latest = current_data['metrics'][-1]
            
            # Extract ALL fields from the latest metrics, not just a subset
            current_values = {}
            
            # Core training metrics (always include)
            core_fields = [
                'iteration', 'epoch', 'train_loss', 'val_loss', 
                'train_perplexity', 'val_perplexity', 'learning_rate',
                'tokens_per_sec', 'trained_tokens', 'peak_memory_gb',
                'iterations_per_sec', 'warmup_steps', 'lr_decay', 'weight_decay'
            ]
            
            for field in core_fields:
                if field in latest:
                    current_values[field] = latest[field]
                else:
                    # Set appropriate default values for missing fields
                    if field in ['val_loss', 'val_perplexity']:
                        current_values[field] = None  # These are only available every N iterations
                    elif field in ['epoch', 'warmup_steps', 'lr_decay', 'weight_decay']:
                        current_values[field] = '-'  # These might not always be present
                    else:
                        current_values[field] = latest.get(field, 0)
            
            # Add any additional fields that might be present
            for key, value in latest.items():
                if key not in current_values:
                    current_values[key] = value
            
            current_data['current_values'] = current_values
        
        return current_data


# Global monitor instance
_global_monitor = None

def get_realtime_monitor() -> RealtimeTrainingMonitor:
    """Get the global real-time monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealtimeTrainingMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor

def stop_realtime_monitor():
    """Stop the global real-time monitor"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor = None 