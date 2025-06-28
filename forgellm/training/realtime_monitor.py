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
        
    def _find_active_training_file(self) -> Optional[str]:
        """Find the most recent active training file"""
        possible_dirs = [
            Path("forgellm/forgellm/models/cpt"),
            Path("forgellm/models/cpt"),
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
            return None
        
        # Find the most recent file that's actively being updated
        most_recent = None
        most_recent_time = 0
        
        for log_file in all_log_files:
            try:
                # Check modification time
                mtime = os.path.getmtime(log_file)
                
                # Only consider files modified in the last hour as potentially active
                if time.time() - mtime < 3600:
                    if mtime > most_recent_time:
                        most_recent_time = mtime
                        most_recent = log_file
            except Exception:
                continue
        
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
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
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
                
                # Sleep for a short interval
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        with self._lock:
            if not self._last_metrics:
                return {
                    'active': False,
                    'metrics': [],
                    'last_update': None
                }
            
            # Check if data is recent (updated within last 30 seconds)
            is_active = (
                self._last_update and 
                datetime.now() - self._last_update < timedelta(seconds=30)
            )
            
            return {
                'active': is_active,
                'metrics': self._last_metrics.copy(),
                'last_update': self._last_update.isoformat() if self._last_update else None,
                'training_file': self._current_training_file
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        current_data = self.get_current_metrics()
        
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
        
        # Add current values for display
        if current_data['metrics']:
            latest = current_data['metrics'][-1]
            current_data['current_values'] = {
                'iteration': latest.get('iteration', 0),
                'train_loss': latest.get('train_loss'),
                'val_loss': latest.get('val_loss'),
                'train_perplexity': latest.get('train_perplexity'),
                'val_perplexity': latest.get('val_perplexity'),
                'learning_rate': latest.get('learning_rate'),
                'tokens_per_sec': latest.get('tokens_per_sec'),
                'trained_tokens': latest.get('trained_tokens'),
                'peak_memory_gb': latest.get('peak_memory_gb'),
                'iterations_per_sec': latest.get('iterations_per_sec')
            }
        
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