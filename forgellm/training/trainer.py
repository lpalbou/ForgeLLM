"""
Continued Pre-trainer with SOTA best practices
"""

import json
import logging
import math
import os
import subprocess
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import psutil

from .config import TrainingConfig
from .data_processor import PretrainingDataProcessor
from .monitor import AdvancedTrainingMonitor
from .metrics_logger import TrainingMetricsLogger
from ..utils.process_tracker import process_tracker

logger = logging.getLogger(__name__)


class ContinuedPretrainer:
    """Main class for continued pre-training with SOTA best practices"""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config
        self.training_folder_name = None
        self.actual_output_dir = None
        self.data_processor = None
        self.monitor = None
        self._training_process = None
        self._is_training_active = False
        self._output_monitor_thread = None
        self._should_stop_monitor = False
        self._log_file_path = None
        
        if config is not None:
            self._initialize_with_config(config)
    
    def _initialize_with_config(self, config: TrainingConfig):
        """Initialize the trainer with a configuration"""
        self.config = config
        
        # Generate descriptive training folder name
        self.training_folder_name = self._generate_training_folder_name()
        
        # Fix folder hierarchy: if output_dir already contains 'cpt', don't add another level
        # This handles cases where output_dir is "models" vs "models/cpt/some_custom_name"
        output_path = Path(config.output_dir)
        if "cpt" in output_path.parts:
            # If output_dir already contains 'cpt', use it directly with the training folder name
            # e.g., "models/cpt/test_fixed_lr_schedule" -> "models/cpt/{training_folder_name}"
            # Find the cpt part and rebuild the path correctly
            parts = list(output_path.parts)
            if "cpt" in parts:
                cpt_index = parts.index("cpt")
                # Take everything up to and including 'cpt', then add our training folder
                base_parts = parts[:cpt_index + 1]
                self.actual_output_dir = Path(*base_parts) / self.training_folder_name
            else:
                # Fallback: treat as if no cpt in path
                self.actual_output_dir = output_path / "cpt" / self.training_folder_name
        else:
            # Normal case: output_dir is "models", so create "models/cpt/{training_folder_name}"
            self.actual_output_dir = output_path / "cpt" / self.training_folder_name
        
        # Update config to use the new path
        self.config.output_dir = str(self.actual_output_dir)
        
        self.data_processor = PretrainingDataProcessor(self.config)
        self.monitor = AdvancedTrainingMonitor(self.config)
        
        # Ensure output directories exist
        self.actual_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log the training directory creation
        logger.info(f"📁 Created CPT training directory: {self.actual_output_dir}")
        
    def _generate_training_folder_name(self) -> str:
        """Generate a descriptive folder name based on model and training parameters"""
        from datetime import datetime
        
        # Extract model name (remove path and simplify)
        model_short = self.config.model_name.split('/')[-1].replace('-', '_')
        
        # Key training parameters
        lr_str = f"lr{self.config.learning_rate:.0e}".replace('-', '_')
        bs_str = f"bs{self.config.batch_size}"
        iter_str = f"iter{self.config.max_iterations}"
        
        # Optional parameters (only if non-default)
        params = []
        if self.config.lr_schedule != "cosine_decay":
            params.append(f"sched_{self.config.lr_schedule}")
        if self.config.data_mixture_ratio != 0.95:
            params.append(f"mix{int(self.config.data_mixture_ratio*100)}")
        if self.config.max_seq_length != 2048:
            params.append(f"seq{self.config.max_seq_length}")
            
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        # Combine all parts
        parts = [model_short, lr_str, bs_str, iter_str] + params + [timestamp]
        folder_name = "_".join(parts)
        
        # Ensure folder name is not too long (limit to 150 chars)
        if len(folder_name) > 150:
            # Keep timestamp and key params, truncate model name
            essential_parts = [lr_str, bs_str, iter_str, timestamp]
            essential_length = len("_".join(essential_parts)) + len("_".join(params))
            max_model_length = 150 - essential_length - 10  # 10 for safety
            model_short = model_short[:max_model_length]
            parts = [model_short, lr_str, bs_str, iter_str] + params + [timestamp]
            folder_name = "_".join(parts)
        
        return folder_name
    
    def prepare_data(self):
        """Prepare training data from documents"""
        if self.config is None or self.data_processor is None:
            raise ValueError("Trainer not initialized with a configuration")
            
        logger.info("=== Preparing Training Data with Data Mixture ===")
        num_train, num_valid, total_tokens_dataset = self.data_processor.create_training_data()
        
        if num_train == 0:
            raise ValueError("No training data created. Check your documents.")
            
        return num_train, num_valid, total_tokens_dataset
    
    def run_training(self):
        """Execute the continued pre-training process with SOTA best practices"""
        if self.config is None:
            raise ValueError("Trainer not initialized with a configuration")
            
        try:
            # Log organized training setup
            logger.info("=== ORGANIZED CONTINUED PRE-TRAINING SETUP ===")
            logger.info(f"📂 Parent Output Dir: {Path(self.config.output_dir).parent.parent}")
            logger.info(f"🗂️  Training Type: CPT (Continued Pre-Training)")
            logger.info(f"📁 Training Folder: cpt/{self.training_folder_name}")
            logger.info(f"🎯 Full Training Path: {self.actual_output_dir}")
            logger.info("=" * 60)
            
            # Prepare data
            num_train, num_valid, total_tokens_dataset = self.prepare_data()
            
            # Use the configured max_iterations directly
            total_steps = self.config.max_iterations
            
            logger.info("=== Starting SOTA Continued Pre-training ===")
            logger.info(f"🤖 Model: {self.config.model_name}")
            logger.info(f"📚 Training examples: {num_train:,}")
            logger.info(f"🔍 Validation examples: {num_valid:,}")
            logger.info(f"🎯 Max iterations: {total_steps:,}")
            logger.info(f"📦 Batch size: {self.config.batch_size}")
            logger.info(f"📈 Learning rate: {self.config.learning_rate}")
            logger.info(f"🔄 LR schedule: {self.config.lr_schedule}")
            logger.info(f"🎭 Data mixture: {self.config.data_mixture_ratio:.1%} domain + {1-self.config.data_mixture_ratio:.1%} general")
            logger.info(f"🚨 Overfitting threshold: {self.config.overfitting_threshold:.1%}")
            logger.info(f"💾 Checkpoints saved every {self.config.save_every} iterations")
            
            # Run MLX-LM training, passing dataset sizes for dynamic validation batching
            self._run_mlx_training(total_steps, num_train, num_valid, total_tokens_dataset)
            
            # Log final summary
            self.monitor.log_final_summary()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _run_mlx_training(self, total_steps: int, num_train: int, num_valid: int, dataset_tokens: int):
        """Run the actual MLX-LM training process with SOTA learning rate scheduling"""
        import subprocess
        import sys
        import yaml
        import tempfile
        
        # Create MLX-LM YAML configuration with SOTA learning rate scheduling
        mlx_config = {
            # Model and data
            "model": self.config.model_name,
            "data": self.config.data_dir,
            "train": True,
            
            # Fine-tuning configuration (configurable)
            "fine_tune_type": self.config.fine_tune_type,  # Can be: full, lora, dora
            "num_layers": self.config.num_layers,  # -1 = all layers, or specific number
            
            # Training parameters
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "iters": total_steps,
            "save_every": self.config.save_every,
            "adapter_path": self.config.output_dir,
            "max_seq_length": self.config.max_seq_length,
            "steps_per_report": self.config.steps_per_report,  
            # steps_per_eval: we keep every 25 steps for quick checks.
            "steps_per_eval": self.config.steps_per_eval,
            # Dynamically compute quick-validation batch count from percentage.
            "val_batches": self.config.val_batches or max(1, math.ceil(self.config.validation_fast_pct * num_valid / self.config.batch_size)),
            "seed": self.config.seed,
            
            # SOTA Learning Rate Schedule Configuration - RESTORED!
            "lr_schedule": {
                "name": self.config.lr_schedule,
                "arguments": [
                    self.config.learning_rate,  # init: initial learning rate
                    total_steps,  # decay_steps: total training steps (FIXED: was 0!)
                    float(f"{self.config.learning_rate * self.config.lr_decay_factor:.1e}")  # end: final LR = init * decay_factor, properly formatted
                ],
                "warmup": self.config.warmup_steps  # warmup steps
            },
            
            # Optimizer configuration
            "optimizer": "adamw",  # AdamW is SOTA for continued pre-training
            "optimizer_config": {
                "adamw": {
                    "weight_decay": self.config.weight_decay  # Configurable weight decay
                }
            },
            
            # Advanced training settings
            "grad_checkpoint": True,  # Enable gradient checkpointing for memory efficiency
            "mask_prompt": False,     # Don't mask prompts for continued pre-training
            "dataset_total_tokens": dataset_tokens,
        }
        
        # Create MLX-LM YAML configuration file in output directory
        config_filename = f"mlx_config_{int(time.time())}.yaml"
        config_file = Path(self.config.output_dir) / config_filename
        
        # Ensure output directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(mlx_config, f, default_flow_style=False)
            
        logger.info(f"📄 MLX-LM config saved: {config_file}")
        
        try:
            # Build MLX-LM training command using config file
            cmd = [
                sys.executable, "-m", "mlx_lm", "lora",
                "--config", str(config_file)
            ]
            
            # Initialize comprehensive training metrics logger with complete config
            config_dict = {
                "training_type": "CPT",
                "fine_tune_type": self.config.fine_tune_type,
                "num_layers": self.config.num_layers,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_iterations": total_steps,
                "save_every": self.config.save_every,
                "steps_per_report": self.config.steps_per_report,
                "steps_per_eval": self.config.steps_per_eval,
                "val_batches": self.config.val_batches or max(1, math.ceil(self.config.validation_fast_pct * num_valid / self.config.batch_size)),
                "max_seq_length": self.config.max_seq_length,
                "warmup_steps": self.config.warmup_steps,
                "data_mixture_ratio": self.config.data_mixture_ratio,
                "overfitting_threshold": self.config.overfitting_threshold,
                "early_stopping_patience": self.config.early_stopping_patience,
                "min_loss_improvement": self.config.min_loss_improvement,
                "validation_split": self.config.validation_split,
                "enable_early_stopping": self.config.enable_early_stopping,
                "use_lr_rewarming": self.config.use_lr_rewarming,
                "lr_decay_factor": self.config.lr_decay_factor,
                "lr_schedule": self.config.lr_schedule,
                "seed": self.config.seed,
                "input_dir": self.config.input_dir,
                "data_dir": self.config.data_dir,
                "max_tokens_per_file": self.config.max_tokens_per_file,
                "max_checkpoints": self.config.max_checkpoints,
                # MLX-LM specific parameters
                "optimizer": "adamw",
                "weight_decay": self.config.weight_decay,
                "grad_checkpoint": True,
                "mask_prompt": False,
                "dataset_total_tokens": dataset_tokens,
            }
            
            # Build command string for logging
            training_command = ' '.join(cmd)
            
            # Create descriptive model name for logging (similar to IFT pattern)
            model_name_for_logging = f"dataset_cpt_{self.config.model_name.split('/')[-1]}"
            
            # Import our metrics logger
            from .metrics_logger import create_training_logger
            
            metrics_logger = create_training_logger(
                training_type="CPT",
                model_name=model_name_for_logging,
                output_dir=self.config.output_dir,  # Store logs in output directory
                config=config_dict,
                base_model=self.config.model_name,
                output_path=self.config.output_dir,
                training_command=training_command
            )
            
            # Initialize variables that might be used in except blocks
            raw_log_fh = None
                
            logger.info("🚀 SOTA CONTINUED PRE-TRAINING")
            logger.info("=" * 80)
            if self.config.fine_tune_type == "full":
                logger.info(f"🔥 Training Type: FULL PARAMETER (all layers unfrozen)")
            elif self.config.fine_tune_type == "lora":
                logger.info(f"🔥 Training Type: LoRA (parameter-efficient fine-tuning)")
                logger.info(f"🔄 Number of layers: {self.config.num_layers}")
            elif self.config.fine_tune_type == "dora":
                logger.info(f"🔥 Training Type: DoRA (parameter-efficient fine-tuning)")
                logger.info(f"🔄 Number of layers: {self.config.num_layers}")
            logger.info(f"📈 Learning Rate Schedule: {self.config.lr_schedule}")
            logger.info(f"🎯 Initial LR: {self.config.learning_rate}")
            logger.info(f"📉 LR Decay Factor: {self.config.lr_decay_factor}")
            logger.info(f"🔄 Warmup Steps: {self.config.warmup_steps}")
            logger.info(f"⚙️  Optimizer: AdamW with weight decay {self.config.weight_decay}")
            logger.info(f"💾 Gradient Checkpointing: Enabled")
            logger.info(f"📄 MLX Config: {config_file}")
            logger.info(f"🚀 Running command: {training_command}")
            logger.info(f"📊 Training metrics will be logged to: {metrics_logger.log_file}")
            logger.info("=" * 80)
            
            # Open raw output log file inside training directory
            raw_log_path = Path(self.config.output_dir) / "mlx_train_output.log"
            raw_log_fh = open(raw_log_path, "w", encoding="utf-8")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            
            # Initialise before first use inside loop
            parsed_metrics = None
            
            # Define a callback function to process each line
            def process_line(raw_line, stripped_line):
                nonlocal parsed_metrics
                
                # Write raw line with original formatting to file
                raw_log_fh.write(raw_line)
                raw_log_fh.flush()
                
                logger.info(f"MLX-LM: {stripped_line}")
                
                # Parse and log metrics using enhanced logger
                parsed_metrics = metrics_logger.parse_and_log_line(stripped_line)
                if parsed_metrics:
                    logger.info(f"📊 Captured metrics for iteration {parsed_metrics.iteration}")
                    
                    # Always log training metrics (per-iteration). Validation loss may be None.
                    self.monitor.log_metrics(
                        parsed_metrics.iteration,
                        parsed_metrics.train_loss or 0.0,
                        parsed_metrics.val_loss,  # can be None
                        parsed_metrics.learning_rate or self.config.learning_rate,
                        parsed_metrics.tokens_per_sec or 0.0,
                        parsed_metrics.peak_memory_gb or 0.0,
                    )
                    
                    # Only evaluate early-stopping when a validation value is available
                    if parsed_metrics.val_loss is not None and self.monitor.should_stop_early(parsed_metrics.val_loss):
                        logger.warning("🛑 Stopping training early")
                        process.terminate()
            
            # Use the safe stream parser from the metrics logger
            thread, output_queue, stop_event = metrics_logger.parse_stream_safely(
                process.stdout, 
                callback=process_line
            )
            
            # Wait for the process to complete
            while process.poll() is None:
                time.sleep(0.1)
                
            # Stop the reader thread
            stop_event.set()
            thread.join()
            
            # Process any remaining items in the queue
            while not output_queue.empty():
                item = output_queue.get()
                if item is not None and not isinstance(item, str) and item.startswith("Error"):
                    logger.error(item)
            
            # Ensure raw log is flushed & closed
            return_code = process.poll()
            raw_log_fh.flush()
            raw_log_fh.close()
            
            # Finalize metrics logging
            metrics_logger.finalize_session()
            summary = metrics_logger.get_summary()
            
            if return_code == 0 or return_code is None:  # None means terminated by early stopping
                if return_code is None:
                    logger.info("✅ MLX-LM FULL PARAMETER training stopped early (early stopping triggered)")
                else:
                    logger.info("✅ MLX-LM FULL PARAMETER training completed successfully")
                logger.info("🔥 FULL PARAMETER TRAINING WITH SOTA LR SCHEDULING COMPLETED!")
                logger.info(f"📊 Training metrics saved: {summary['log_file']}")
                logger.info(f"📈 Training points captured: {summary['training_points']}")
                logger.info(f"📊 Validation points captured: {summary['validation_points']}")
                logger.info(f"💾 Checkpoints saved: {summary['checkpoints_saved']}")
            else:
                raise subprocess.CalledProcessError(return_code, cmd)
                
        except subprocess.CalledProcessError as e:
            # Handle the case where return_code might be None
            if e.returncode is not None:
                logger.error(f"❌ MLX-LM FULL PARAMETER training failed with return code {e.returncode}")
            else:
                logger.error(f"❌ MLX-LM FULL PARAMETER training failed: Process terminated")
            # Still finalize metrics even on failure
            if 'metrics_logger' in locals():
                metrics_logger.finalize_session()
            if raw_log_fh and not raw_log_fh.closed:
                raw_log_fh.close()
            raise
        except Exception as e:
            logger.error(f"❌ MLX-LM FULL PARAMETER training failed with unexpected error: {e}")
            # Still finalize metrics even on failure
            if 'metrics_logger' in locals():
                metrics_logger.finalize_session()
            if raw_log_fh and not raw_log_fh.closed:
                raw_log_fh.close()
            raise
    
    def validate_model(self, model_path: str) -> float:
        """Validate the trained model"""
        try:
            logger.info("=== Validating Trained Model ===")
            
            # Load the model with adapter
            from mlx_lm import load, generate
            model, tokenizer = load(self.config.model_name, adapter_path=model_path)
            
            # Test generation with correct MLX-LM API
            test_prompt = "The future of artificial intelligence"
            
            # Use minimal parameters that work with MLX-LM
            response = generate(
                model, tokenizer, 
                prompt=test_prompt,
                max_tokens=50  # Reduced for faster validation
            )
            
            logger.info(f"✅ Test generation successful:")
            logger.info(f"📝 Prompt: {test_prompt}")
            logger.info(f"🤖 Response: {response}")
            
            return 0.0  # Placeholder validation score
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            logger.warning("⚠️  Validation failed - this is normal if training hasn't completed yet")
            return float('inf')

    def start_training(self, config: TrainingConfig):
        """Start training in a separate process
        
        Args:
            config: Training configuration
        """
        if self._is_training_active:
            logger.warning("Training is already active")
            return
        
        try:
            # Initialize with config
            self._initialize_with_config(config)
            
            # Create a temporary file to store the config
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                yaml.dump(config.to_dict(), temp_file)
                temp_config_path = temp_file.name
            
            # Create a log file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(config.output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file_path = str(log_dir / f"training_log_{timestamp}.json")
            
            # Start training in a separate process
            cmd = [
                sys.executable,
                "-m", "forgellm.training.run_training",
                "--config", temp_config_path,
                "--log-file", self._log_file_path
            ]
            
            # Set environment variables
            env = os.environ.copy()
            
            # Add the current directory to PYTHONPATH to ensure imports work
            python_path = env.get('PYTHONPATH', '')
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if python_path:
                env['PYTHONPATH'] = f"{current_dir}:{python_path}"
            else:
                env['PYTHONPATH'] = current_dir
            
            # Start the process
            self._training_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Track the process for cleanup
            process_tracker.track_process(self._training_process)
            
            # Set training active flag
            self._is_training_active = True
            
            # Start output monitor thread
            self._should_stop_monitor = False
            self._output_monitor_thread = threading.Thread(
                target=self._monitor_output,
                daemon=True
            )
            self._output_monitor_thread.start()
            
            logger.info(f"Training started with PID {self._training_process.pid}")
            
            # Clean up temporary file after a delay
            def cleanup_temp_file():
                time.sleep(10)  # Wait for 10 seconds to ensure the file is read
                try:
                    os.unlink(temp_config_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary config file: {e}")
            
            threading.Thread(target=cleanup_temp_file, daemon=True).start()
            
            return {
                "success": True,
                "message": "Training started successfully",
                "pid": self._training_process.pid,
                "log_file": self._log_file_path
            }
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return {
                "success": False,
                "message": f"Failed to start training: {e}"
            }
    
    def _monitor_output(self):
        """Monitor training process output and update status"""
        # Import here to avoid circular imports
        try:
            from ..web.services.socket_service import training_monitor
        except ImportError:
            # Fallback if socket service is not available
            training_monitor = None
        
        if not self._training_process:
            return
        
        try:
            for line in iter(self._training_process.stdout.readline, ''):
                if self._should_stop_monitor:
                    break
                
                # Process output line
                line = line.strip()
                if not line:
                    continue
                
                logger.info(f"Training: {line}")
                
                # Check if training log file exists and has content
                if self._log_file_path and os.path.exists(self._log_file_path):
                    try:
                        # Check if file size is non-zero and read it
                        if os.path.getsize(self._log_file_path) > 0:
                            with open(self._log_file_path, 'r') as f:
                                try:
                                    training_data = json.load(f)
                                    
                                    # Socket updates disabled to prevent API call spam
                                    # if training_monitor:
                                    #     training_monitor.update_training_data(training_data)
                                except json.JSONDecodeError:
                                    # File might be partially written, ignore
                                    pass
                    except Exception as e:
                        logger.warning(f"Error reading training log file: {e}")
                
                # Check if process is still running
                if self._training_process.poll() is not None:
                    # Process has ended
                    self._is_training_active = False
                    
                    # Read final log file
                    if self._log_file_path and os.path.exists(self._log_file_path):
                        try:
                            with open(self._log_file_path, 'r') as f:
                                training_data = json.load(f)
                                # Socket updates disabled to prevent API call spam
                                # if training_monitor:
                                #     training_monitor.emit_finished(training_data)
                        except Exception as e:
                            logger.warning(f"Error reading final training log file: {e}")
                    
                    break
        except Exception as e:
            logger.error(f"Error monitoring training output: {e}")
        finally:
            self._is_training_active = False
    
    def stop_training(self):
        """Stop the training process"""
        if not self._is_training_active or not self._training_process:
            logger.warning("No active training to stop")
            return {
                "success": False,
                "message": "No active training to stop"
            }
        
        try:
            # Signal the monitor thread to stop
            self._should_stop_monitor = True
            
            # Terminate the process
            self._training_process.terminate()
            
            # Wait for process to end
            try:
                self._training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                self._training_process.kill()
                self._training_process.wait(timeout=5)
            
            # Set training inactive
            self._is_training_active = False
            
            # Untrack the process
            if self._training_process:
                process_tracker.untrack_process(self._training_process)
            
            logger.info("Training stopped")
            return {
                "success": True,
                "message": "Training stopped successfully"
            }
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return {
                "success": False,
                "message": f"Error stopping training: {e}"
            }
    
    def is_training_active(self):
        """Check if training is active
        
        Returns:
            bool: True if training is active, False otherwise
        """
        # Check if process is still running
        if self._is_training_active and self._training_process:
            if self._training_process.poll() is None:
                return True
            else:
                # Process has ended
                self._is_training_active = False
                return False
        
        return False
    
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
                            logger.info(f"Found active MLX process: PID {proc.info['pid']}, CMD: {cmdline}")
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.warning(f"Error checking MLX processes: {e}")
        
        return False

    def _find_active_training_log(self):
        """Find the most recent training log file that corresponds to actual running MLX processes
        
        Returns:
            str: Path to active training log file, or None if not found
        """
        # FIRST: Check if any MLX processes are actually running
        if not self._check_mlx_processes_running():
            logger.info("No MLX processes running - no active training")
            return None
        
        import glob
        from pathlib import Path
        from datetime import datetime
        
        # Look for recent training directories
        possible_dirs = [
            Path("models/cpt")
        ]
        
        all_log_files = []
        
        # Find all CPT log files from today
        today = datetime.now().strftime("%Y-%m-%d")
        
        for models_dir in possible_dirs:
            if models_dir.exists():
                # Look for today's files first
                log_pattern = str(models_dir / f"*{today}*" / "CPT_*.json")
                log_files = glob.glob(log_pattern)
                all_log_files.extend(log_files)
                
                # Also include recent files (last 2 hours) as fallback
                if not log_files:
                    log_pattern = str(models_dir / "*" / "CPT_*.json")
                    recent_files = glob.glob(log_pattern)
                    # Filter to only very recent files
                    import time
                    current_time = time.time()
                    recent_files = [f for f in recent_files 
                                  if current_time - os.path.getmtime(f) < 7200]  # 2 hours
                    all_log_files.extend(recent_files)
        
        if not all_log_files:
            logger.info("No recent training log files found")
            return None
        
        # Find the most recent log file that indicates active training
        most_recent = None
        most_recent_time = 0
        
        for log_file in all_log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    
                # Only consider files without end_time AND with recent activity
                if data.get('end_time') is None:
                    mtime = os.path.getmtime(log_file)
                    if mtime > most_recent_time:
                        most_recent_time = mtime
                        most_recent = log_file
                        logger.info(f"Found potential active training log: {log_file}")
            except Exception as e:
                logger.warning(f"Error reading log file {log_file}: {e}")
                continue
        
        if most_recent:
            logger.info(f"Selected active training log: {most_recent}")
        else:
            logger.info("No active training logs found despite MLX processes running")
        
        return most_recent

    def get_training_status(self):
        """Get training status with robust MLX process checking
        
        Returns:
            dict: Training status
        """
        # STEP 1: Check if we have our own active training process
        own_training_active = self.is_training_active()
        
        # STEP 2: Check if ANY MLX processes are running (most important check)
        mlx_processes_running = self._check_mlx_processes_running()
        
        # STEP 3: Determine if training is active
        # Training is only active if:
        # - We have our own active process, OR
        # - There are MLX processes running
        training_active = own_training_active or mlx_processes_running
        
        logger.info(f"Training status check: own_active={own_training_active}, mlx_running={mlx_processes_running}, final_active={training_active}")
        
        status = {
            "active": training_active
        }
        
        log_file_to_read = None
        
        # STEP 4: Find log file to read
        if self._log_file_path and own_training_active:
            # Use our own log file if we have active training
            status["log_file"] = self._log_file_path
            log_file_to_read = self._log_file_path
            logger.info(f"Using own training log: {self._log_file_path}")
        elif mlx_processes_running:
            # Look for active training from other processes only if MLX is running
            active_log = self._find_active_training_log()
            if active_log:
                log_file_to_read = active_log
                status["log_file"] = active_log
                logger.info(f"Using external training log: {active_log}")
        
        # STEP 5: Read the log file if we found one and training is active
        if log_file_to_read and os.path.exists(log_file_to_read) and training_active:
            try:
                if os.path.getsize(log_file_to_read) > 0:
                    with open(log_file_to_read, 'r') as f:
                        training_data = json.load(f)
                        status.update(training_data)
                        logger.info(f"Loaded training data from: {log_file_to_read}")
            except Exception as e:
                logger.warning(f"Error reading training log file: {e}")
        
        # STEP 6: If no training is active, ensure we return a clean inactive status
        if not training_active:
            status = {
                "active": False,
                "message": "No active training detected"
            }
            logger.info("No active training - returning inactive status")
        
        return status
    
    def get_dashboard_data(self):
        """Get data for dashboard
        
        Returns:
            dict: Dashboard data
        """
        # Get training status
        status = self.get_training_status()
        
        # Add additional dashboard data
        if "metrics" in status and len(status["metrics"]) > 0:
            from .dashboard import identify_best_checkpoints, generate_web_chart_data
            
            try:
                # Identify best checkpoints
                best_checkpoints = identify_best_checkpoints(status)
                status["best_checkpoints"] = best_checkpoints
                
                # Generate chart data for web interface
                charts = generate_web_chart_data(status)
                if charts:
                    status["charts"] = charts
                    
            except Exception as e:
                logger.warning(f"Error generating dashboard data: {e}")
        
        return status 