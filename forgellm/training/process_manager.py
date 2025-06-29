"""
Training Process Manager - Handles training process lifecycle and monitoring
"""

import json
import logging
import math
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil

from .config import TrainingConfig
from .trainer import ContinuedPretrainer
from .dashboard import load_training_data, identify_best_checkpoints

logger = logging.getLogger(__name__)


class TrainingProcessManager:
    """
    Manages training processes and monitoring - replaces legacy TrainingManager
    
    This class provides the same interface as the legacy TrainingManager but
    uses the new ContinuedPretrainer internally.
    """
    
    def __init__(self):
        """Initialize training process manager"""
        self.current_training = None
        self.trainer = ContinuedPretrainer()
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.disable_auto_detect = False
        
        # Detect active training on startup (unless disabled)
        self._detect_active_training()
    
    def start_training(self, config: Dict) -> Dict:
        """Start training process with the given configuration"""
        try:
            # Check if training is already running
            if self.current_training and self.current_training.get("status") == "running":
                return {"success": False, "error": "Training already in progress"}
            
            # Stop any existing monitoring
            self.stop_monitoring.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
            self.stop_monitoring.clear()
            
            # Convert dict config to TrainingConfig object
            training_config = self._dict_to_training_config(config)
            
            # Start training using the new trainer
            result = self.trainer.start_training(training_config)
            
            if result.get("success"):
                # Store training configuration
                self.current_training = {
                    "config": config,
                    "start_time": datetime.now().isoformat(),
                    "status": "running",
                    "output_dir": training_config.output_dir,
                    "pid": result.get("pid"),
                    "log_file": result.get("log_file")
                }
                
                # Start monitoring thread
                self.start_monitoring()
                
                return {
                    "success": True,
                    "output_dir": training_config.output_dir,
                    "pid": result.get("pid")
                }
            else:
                return {"success": False, "error": result.get("message", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_training(self) -> Dict:
        """Stop current training"""
        if not self.current_training:
            return {"success": False, "error": "No training in progress"}
        
        try:
            # Stop the trainer
            result = self.trainer.stop_training()
            
            # Stop monitoring
            self.stop_monitoring.set()
            if self.current_training:
                self.current_training["status"] = "stopped"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return {"success": False, "error": str(e)}
    
    def start_monitoring(self):
        """Start monitoring thread"""
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_training)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_training(self):
        """Monitor training progress and emit updates"""
        while not self.stop_monitoring.is_set() and self.current_training:
            try:
                # Check if training is still active
                if not self.trainer.is_training_active():
                    self.current_training["status"] = "completed"
                    break
                
                # Find latest log file
                log_file = self._find_latest_log_file()
                if log_file and log_file.exists():
                    self.current_training["log_file"] = log_file
                    
                    # Parse training data
                    training_data = self._parse_training_data(log_file)
                    
                    # Socket updates disabled to prevent API call spam
                    # All updates now handled by main app.js performSingleUpdate() method
                    # try:
                    #     from ..web.services.socket_service import training_monitor
                    #     if training_monitor:
                    #         training_monitor.update_training_data(training_data)
                    # except ImportError:
                    #     pass  # Socket.IO not available
                    
                    # Check if training has completed
                    if training_data.get("status") == "completed":
                        self.current_training["status"] = "completed"
                        break
                
                # Sleep before next update
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error monitoring training: {e}")
                time.sleep(5)  # Sleep longer on error
        
        # Training completed or stopped
        if self.current_training and self.current_training.get("status") == "completed":
            logger.info("Training completed")
            
            # Get final training data
            try:
                log_file = self.current_training.get("log_file")
                if log_file and Path(log_file).exists():
                    final_data = self._parse_training_data(log_file)
                    
                    # Socket updates disabled to prevent API call spam
                    # try:
                    #     from ..web.services.socket_service import training_monitor
                    #     if training_monitor:
                    #         training_monitor.emit_finished(final_data)
                    # except ImportError:
                    #     pass  # Socket.IO not available
                        
            except Exception as e:
                logger.error(f"Error getting final training data: {e}")
        
        # Clear current training
        self.current_training = None
    
    def _find_latest_log_file(self) -> Optional[Path]:
        """Find the latest training log file"""
        try:
            models_dir = Path("models/cpt")
            if not models_dir.exists():
                return None
            
            # Find most recent directory
            latest_dir = None
            latest_time = 0
            
            for dir_path in models_dir.iterdir():
                if dir_path.is_dir():
                    dir_time = dir_path.stat().st_mtime
                    if dir_time > latest_time:
                        latest_time = dir_time
                        latest_dir = dir_path
            
            if latest_dir:
                # Look for log files
                log_files = list(latest_dir.glob("CPT_*.json"))
                
                # Check for nested directory structure
                if not log_files:
                    nested_cpt_dir = latest_dir / "cpt"
                    if nested_cpt_dir.exists():
                        for nested_dir in nested_cpt_dir.iterdir():
                            if nested_dir.is_dir():
                                nested_log_files = list(nested_dir.glob("CPT_*.json"))
                                if nested_log_files:
                                    log_files.extend(nested_log_files)
                
                if log_files:
                    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return log_files[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding latest log file: {e}")
            return None
    
    def _parse_training_data(self, log_file: Path) -> Dict:
        """Parse training data from JSON log"""
        try:
            if isinstance(log_file, str):
                log_file = Path(log_file)
                
            if not log_file.exists() or log_file.stat().st_size == 0:
                return {"error": f"Log file does not exist or is empty: {log_file}"}
                
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('metrics', [])
            if not metrics:
                return {"error": "No metrics available"}
            
            # Extract latest metrics
            latest = metrics[-1]
            latest_val = next(
                (m for m in reversed(metrics) if m.get("val_loss") is not None),
                latest,
            )
            latest_tokens = next(
                (m for m in reversed(metrics) if m.get("trained_tokens") is not None),
                latest,
            )
            
            # Calculate progress
            config = data.get('config', {})
            max_iterations = config.get('max_iterations', 1000)
            current_iteration = latest.get('iteration', 0)
            progress = (current_iteration / max_iterations) * 100
            
            # Calculate time estimates
            try:
                t0 = datetime.fromisoformat(metrics[0]["timestamp"])
                t_now = datetime.fromisoformat(latest["timestamp"])
                elapsed_min = max(0, (t_now - t0).total_seconds() / 60)
                if current_iteration > 0:
                    avg_sec_per_iter = (t_now - t0).total_seconds() / current_iteration
                    remaining_min = (max_iterations - current_iteration) * avg_sec_per_iter / 60
                else:
                    remaining_min = None
            except Exception:
                elapsed_min = remaining_min = None
            
            # Get best checkpoints
            best_checkpoints = identify_best_checkpoints(data, top_k=3)
            
            # Calculate epoch progress
            trained_tokens = latest_tokens.get("trained_tokens")
            dataset_token_budget = config.get('dataset_total_tokens') or config.get('dataset_token_budget')
            epoch_done = None
            if trained_tokens and dataset_token_budget:
                epoch_done = trained_tokens / dataset_token_budget
            
            # Build result
            result = {
                "current_iteration": current_iteration,
                "max_iterations": max_iterations,
                "progress": progress,
                "train_loss": latest.get('train_loss'),
                "val_loss": latest_val.get('val_loss'),
                "train_perplexity": latest.get('train_perplexity'),
                "val_perplexity": latest_val.get('val_perplexity'),
                "learning_rate": latest.get('learning_rate'),
                "tokens_per_sec": latest.get('tokens_per_sec'),
                "peak_memory_gb": latest.get('peak_memory_gb'),
                "trained_tokens": trained_tokens,
                "epoch_done": epoch_done,
                "elapsed_minutes": elapsed_min,
                "eta_minutes": remaining_min,
                "best_checkpoints": best_checkpoints[:3],
                "total_metrics": len(metrics),
                "session_id": data.get('session_id'),
                "model_name": data.get('model_name'),
                "output_path": data.get('output_path'),
                "warmup_steps": config.get('warmup_steps'),
                "lr_decay_factor": config.get('lr_decay_factor', 0.1),
                "config": config
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing training data: {e}")
            return {"error": str(e)}
    
    def _detect_active_training(self):
        """Detect if there's any active training running"""
        if self.disable_auto_detect:
            return
            
        try:
            # Check if trainer reports active training
            if self.trainer.is_training_active():
                logger.info("Detected active training from trainer")
                
                # Find the latest log file
                log_file = self._find_latest_log_file()
                if log_file and log_file.exists():
                    try:
                        with open(log_file, 'r') as f:
                            data = json.load(f)
                        
                        config = data.get('config', {})
                        if 'model' not in config and 'base_model' in data:
                            config['model'] = data['base_model']
                        start_time = data.get('session_start_time', datetime.now().isoformat())
                        
                        self.current_training = {
                            "config": config,
                            "start_time": start_time,
                            "status": "running",
                            "log_file": log_file,
                            "output_dir": log_file.parent,
                            "detected": True,
                            "monitor_only": True
                        }
                        
                        # Start monitoring
                        self.start_monitoring()
                        logger.info(f"Started monitoring detected training: {log_file}")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse detected training log {log_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error detecting active training: {e}")
    
    def _dict_to_training_config(self, config_dict: Dict) -> TrainingConfig:
        """Convert dictionary config to TrainingConfig object"""
        # Map legacy parameter names to new ones
        param_mapping = {
            "model": "model_name",
            "eval_every": "steps_per_eval"
        }
        
        # Create a clean config dict with only valid TrainingConfig parameters
        clean_config = {}
        
        # Get valid TrainingConfig field names
        from .config import TrainingConfig
        valid_fields = set(TrainingConfig.__annotations__.keys())
        
        for key, value in config_dict.items():
            # Map legacy names
            mapped_key = param_mapping.get(key, key)
            
            # Only include if it's a valid TrainingConfig field
            if mapped_key in valid_fields:
                clean_config[mapped_key] = value
        
        # Set defaults for required fields if not provided
        if "model_name" not in clean_config:
            clean_config["model_name"] = config_dict.get("model", "microsoft/DialoGPT-medium")
        if "input_dir" not in clean_config:
            clean_config["input_dir"] = config_dict.get("input_dir", "dataset")
        if "output_dir" not in clean_config:
            clean_config["output_dir"] = config_dict.get("output_dir", "models")
        
        return TrainingConfig(**clean_config) 