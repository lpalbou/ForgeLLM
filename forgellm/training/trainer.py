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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from forgellm.training.config import TrainingConfig
from forgellm.training.data_processor import PretrainingDataProcessor
from forgellm.training.monitor import AdvancedTrainingMonitor

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
            model_name_for_logging = f"mnemosyne_cpt_{self.config.model_name.split('/')[-1]}"
            
            # Import training metrics logger dynamically to avoid dependency issues
            import importlib.util
            spec = importlib.util.spec_from_file_location("training_metrics_logger", "training_metrics_logger.py")
            if spec is None or spec.loader is None:
                raise ImportError("Could not load training_metrics_logger.py")
            metrics_logger_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metrics_logger_module)
            
            metrics_logger = metrics_logger_module.create_training_logger(
                training_type="CPT",
                model_name=model_name_for_logging,
                output_dir=self.config.output_dir,  # Store logs in output directory
                config=config_dict,
                base_model=self.config.model_name,
                output_path=self.config.output_dir,
                training_command=training_command
            )
                
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
            def process_line(line):
                nonlocal parsed_metrics
                
                # Write raw line as-is to file and stdout logger
                raw_log_fh.write(line)
                raw_log_fh.flush()
                
                line_stripped = line.strip()
                logger.info(f"MLX-LM: {line_stripped}")
                
                # Parse and log metrics using enhanced logger
                parsed_metrics = metrics_logger.parse_and_log_line(line_stripped)
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
            metrics_logger.finalize_session()
            raw_log_fh.close()
            raise
        except Exception as e:
            logger.error(f"❌ MLX-LM FULL PARAMETER training failed with unexpected error: {e}")
            # Still finalize metrics even on failure
            metrics_logger.finalize_session()
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
        """Start training with the given configuration"""
        if self._is_training_active:
            logger.warning("Training is already active")
            return
            
        self._initialize_with_config(config)
        self._is_training_active = True
        
        # Start training in a separate process
        import subprocess
        import sys
        
        cmd = [
            sys.executable, "-m", "forgellm", "train",
            "--model-name", config.model_name,
            "--batch-size", str(config.batch_size),
            "--learning-rate", str(config.learning_rate),
            "--max-iterations", str(config.max_iterations),
            "--save-every", str(config.save_every),
            "--max-seq-length", str(config.max_seq_length),
            "--fine-tune-type", config.fine_tune_type,
            "--num-layers", str(config.num_layers),
            "--lr-schedule", config.lr_schedule,
            "--warmup-steps", str(config.warmup_steps),
            "--seed", str(config.seed)
        ]
        
        # Add input and output dirs if specified
        if config.input_dir != 'mnemosyne':
            cmd.extend(["--input-dir", config.input_dir])
        if config.output_dir != 'models':
            cmd.extend(["--output-dir", config.output_dir])
        
        # Start process
        self._training_process = subprocess.Popen(cmd)
        logger.info(f"Training started with PID {self._training_process.pid}")
    
    def stop_training(self):
        """Stop the current training process"""
        if not self._is_training_active:
            logger.warning("No active training to stop")
            return
            
        if self._training_process is not None:
            import signal
            self._training_process.send_signal(signal.SIGTERM)
            self._training_process.wait()
            logger.info("Training stopped")
            
        self._is_training_active = False
        self._training_process = None
    
    def is_training_active(self):
        """Check if training is active"""
        # First, check for the specific MLX-LM process we know is running
        try:
            import subprocess
            import re
            
            # Run ps command to get all processes
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if result.returncode == 0:
                # Check for mlx_lm process with gemma-3-4b-it-bf16
                if 'mlx_lm' in result.stdout and 'gemma-3-4b-it-bf16' in result.stdout:
                    logger.info("Found running MLX-LM process with gemma-3-4b-it-bf16")
                    
                    # Extract the training directory from the process command line
                    match = re.search(r'--config\s+([^\s]+)', result.stdout)
                    if match:
                        config_path = match.group(1)
                        training_dir = os.path.dirname(config_path)
                        
                        # Set the config if not already set
                        if not self.config:
                            try:
                                import yaml
                                with open(config_path, 'r') as f:
                                    yaml_config = yaml.safe_load(f)
                                    
                                # Create a minimal config with the essential fields
                                from .config import TrainingConfig
                                self.config = TrainingConfig(
                                    model_name=yaml_config.get('model', 'mlx-community/gemma-3-4b-it-bf16'),
                                    input_dir='mnemosyne',
                                    output_dir=training_dir,
                                    batch_size=yaml_config.get('batch_size', 4),
                                    learning_rate=yaml_config.get('learning_rate', 3e-6),
                                    max_iterations=yaml_config.get('iters', 300),
                                    max_seq_length=yaml_config.get('max_seq_length', 2048)
                                )
                                logger.info(f"Loaded config from {config_path}")
                            except Exception as e:
                                logger.error(f"Error loading config from yaml: {e}")
                    
                    # Set internal state
                    self._is_training_active = True
                    return True
        except Exception as e:
            logger.error(f"Error checking for MLX-LM processes: {e}")
        
        # If we get here, check the internal state
        if self._training_process is not None:
            # Check if process is still running
            returncode = self._training_process.poll()
            if returncode is not None:
                # Process has exited
                logger.info(f"Training process has exited with code {returncode}")
                self._is_training_active = False
                self._training_process = None
        
        logger.info(f"No active training detected, returning internal state: {self._is_training_active}")
        return self._is_training_active
    
    def get_training_status(self):
        """Get the current training status"""
        status = {
            'active': self.is_training_active(),
            'config': self.config.__dict__ if self.config else {},
            'status': 'running' if self.is_training_active() else 'stopped'
        }
        
        # Add additional status information if available
        if self.monitor:
            status.update(self.monitor.get_current_status())
            
        return status
    
    def get_dashboard_data(self):
        """Get dashboard data for the web UI"""
        if not self.is_training_active():
            return {}
            
        # Basic dashboard data
        data = {
            'active': True,
            'config': self.config.__dict__ if self.config else {},
            'current_iteration': 0,
            'max_iterations': self.config.max_iterations if self.config else 0,
            'progress': 0,
            'train_loss': None,
            'val_loss': None,
            'train_perplexity': None,
            'val_perplexity': None,
            'elapsed_minutes': 0,
            'eta_minutes': None,
            'tokens_per_sec': None,
            'trained_tokens': None,
            'peak_memory_gb': None,
            'learning_rate': None,
            'charts': {
                'loss': {'data': [], 'layout': {}},
                'perplexity': {'data': [], 'layout': {}},
                'learning_rate': {'data': [], 'layout': {}},
                'speed': {'data': [], 'layout': {}}
            }
        }
        
        # Try to read metrics from log file
        if self.config and self.config.output_dir:
            try:
                import os
                import re
                import json
                import glob
                from datetime import datetime
                
                # Check for metrics JSON file
                metrics_files = glob.glob(os.path.join(self.config.output_dir, 'CPT_*.json'))
                
                if metrics_files:
                    # Sort by modification time (newest first)
                    metrics_file = sorted(metrics_files, 
                                        key=lambda f: os.path.getmtime(f),
                                        reverse=True)[0]
                    
                    logger.info(f"Reading metrics from {metrics_file}")
                    
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Extract config
                    if 'config' in metrics_data:
                        # Update config if needed
                        if not self.config:
                            from .config import TrainingConfig
                            self.config = TrainingConfig(
                                model_name=metrics_data['base_model'],
                                input_dir=metrics_data['config'].get('input_dir', 'mnemosyne'),
                                output_dir=metrics_data['output_path']
                            )
                    
                    # Extract metrics
                    if 'metrics' in metrics_data and metrics_data['metrics']:
                        metrics = metrics_data['metrics']
                        
                        # Get the latest metric
                        latest_metric = metrics[-1]
                        
                        # Update data with latest metrics
                        data['current_iteration'] = latest_metric.get('iteration', 0)
                        data['train_loss'] = latest_metric.get('train_loss')
                        data['val_loss'] = latest_metric.get('val_loss')
                        data['train_perplexity'] = latest_metric.get('train_perplexity')
                        data['val_perplexity'] = latest_metric.get('val_perplexity')
                        data['learning_rate'] = latest_metric.get('learning_rate')
                        data['tokens_per_sec'] = latest_metric.get('tokens_per_sec')
                        data['trained_tokens'] = latest_metric.get('trained_tokens')
                        data['peak_memory_gb'] = latest_metric.get('peak_memory_gb')
                        
                        # Calculate progress
                        if data['max_iterations'] > 0:
                            data['progress'] = (data['current_iteration'] / data['max_iterations']) * 100
                        
                        # Calculate elapsed time
                        if 'start_time' in metrics_data and metrics_data['start_time']:
                            start_time = datetime.fromisoformat(metrics_data['start_time'])
                            latest_time = datetime.fromisoformat(latest_metric['timestamp'])
                            elapsed_seconds = (latest_time - start_time).total_seconds()
                            data['elapsed_minutes'] = elapsed_seconds / 60
                            
                            # Calculate ETA
                            if data['progress'] > 0:
                                total_minutes = (data['elapsed_minutes'] * 100) / data['progress']
                                data['eta_minutes'] = total_minutes - data['elapsed_minutes']
                        
                        # Add best checkpoints if available
                        data['best_checkpoints'] = []
                        
                        # Filter metrics with valid val_loss values before sorting
                        valid_metrics = [m for m in metrics if m.get('val_loss') is not None]
                        if valid_metrics:
                            # Sort by validation loss (ascending)
                            sorted_metrics = sorted(valid_metrics, key=lambda m: m.get('val_loss', float('inf')))
                            
                            # Take up to 3 best checkpoints
                            for i, metric in enumerate(sorted_metrics[:3]):
                                data['best_checkpoints'].append({
                                    'iteration': metric.get('iteration'),
                                    'val_loss': metric.get('val_loss'),
                                    'val_perplexity': metric.get('val_perplexity'),
                                    'path': metric.get('checkpoint_path'),
                                    'selection_reason': 'Best validation loss'
                                })
                        
                        # Generate chart data
                        iterations = []
                        train_losses = []
                        val_losses = []
                        train_perplexities = []
                        val_perplexities = []
                        learning_rates = []
                        tokens_per_sec = []
                        
                        for metric in metrics:
                            if metric.get('iteration') is not None:
                                iterations.append(metric['iteration'])
                                
                                # Collect train loss data
                                if metric.get('train_loss') is not None:
                                    train_losses.append(metric['train_loss'])
                                else:
                                    train_losses.append(None)
                                
                                # Collect val loss data
                                if metric.get('val_loss') is not None:
                                    val_losses.append(metric['val_loss'])
                                else:
                                    val_losses.append(None)
                                
                                # Collect perplexity data
                                if metric.get('train_perplexity') is not None:
                                    train_perplexities.append(metric['train_perplexity'])
                                else:
                                    train_perplexities.append(None)
                                
                                if metric.get('val_perplexity') is not None:
                                    val_perplexities.append(metric['val_perplexity'])
                                else:
                                    val_perplexities.append(None)
                                
                                # Collect learning rate data
                                if metric.get('learning_rate') is not None:
                                    learning_rates.append(metric['learning_rate'])
                                else:
                                    learning_rates.append(None)
                                
                                # Collect speed data
                                if metric.get('tokens_per_sec') is not None:
                                    tokens_per_sec.append(metric['tokens_per_sec'])
                                else:
                                    tokens_per_sec.append(None)
                        
                        # Create chart data
                        if iterations:
                            # Loss chart
                            data['charts']['loss']['data'] = [
                                {
                                    'x': iterations,
                                    'y': train_losses,
                                    'type': 'scatter',
                                    'mode': 'lines+markers',
                                    'name': 'Train Loss',
                                    'line': {'color': 'rgb(31, 119, 180)'}
                                },
                                {
                                    'x': iterations,
                                    'y': val_losses,
                                    'type': 'scatter',
                                    'mode': 'lines+markers',
                                    'name': 'Validation Loss',
                                    'line': {'color': 'rgb(255, 127, 14)'}
                                }
                            ]
                            data['charts']['loss']['layout'] = {
                                'title': 'Training Loss',
                                'xaxis': {'title': 'Iteration'},
                                'yaxis': {'title': 'Loss'},
                                'legend': {'orientation': 'h', 'y': -0.2}
                            }
                            
                            # Perplexity chart
                            data['charts']['perplexity']['data'] = [
                                {
                                    'x': iterations,
                                    'y': train_perplexities,
                                    'type': 'scatter',
                                    'mode': 'lines+markers',
                                    'name': 'Train Perplexity',
                                    'line': {'color': 'rgb(31, 119, 180)'}
                                },
                                {
                                    'x': iterations,
                                    'y': val_perplexities,
                                    'type': 'scatter',
                                    'mode': 'lines+markers',
                                    'name': 'Validation Perplexity',
                                    'line': {'color': 'rgb(255, 127, 14)'}
                                }
                            ]
                            data['charts']['perplexity']['layout'] = {
                                'title': 'Perplexity',
                                'xaxis': {'title': 'Iteration'},
                                'yaxis': {'title': 'Perplexity'},
                                'legend': {'orientation': 'h', 'y': -0.2}
                            }
                            
                            # Learning rate chart
                            data['charts']['learning_rate']['data'] = [
                                {
                                    'x': iterations,
                                    'y': learning_rates,
                                    'type': 'scatter',
                                    'mode': 'lines',
                                    'name': 'Learning Rate',
                                    'line': {'color': 'rgb(44, 160, 44)'}
                                }
                            ]
                            data['charts']['learning_rate']['layout'] = {
                                'title': 'Learning Rate Schedule',
                                'xaxis': {'title': 'Iteration'},
                                'yaxis': {'title': 'Learning Rate', 'type': 'log'}
                            }
                            
                            # Speed chart
                            data['charts']['speed']['data'] = [
                                {
                                    'x': iterations,
                                    'y': tokens_per_sec,
                                    'type': 'scatter',
                                    'mode': 'lines',
                                    'name': 'Tokens/sec',
                                    'line': {'color': 'rgb(214, 39, 40)'}
                                }
                            ]
                            data['charts']['speed']['layout'] = {
                                'title': 'Training Speed',
                                'xaxis': {'title': 'Iteration'},
                                'yaxis': {'title': 'Tokens/sec'}
                            }
                
                # If no metrics file or couldn't extract data, try to parse the log file
                if data['current_iteration'] == 0:
                    log_file = os.path.join(self.config.output_dir, 'mlx_train_output.log')
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                        
                        # Extract iteration
                        iter_match = re.search(r'Iter (\d+): Train loss', log_content)
                        if iter_match:
                            data['current_iteration'] = int(iter_match.group(1))
                            
                        # Extract train loss
                        loss_match = re.search(r'Train loss ([\d\.]+)', log_content)
                        if loss_match:
                            data['train_loss'] = float(loss_match.group(1))
                            
                        # Extract val loss
                        val_match = re.search(r'Val loss ([\d\.]+)', log_content)
                        if val_match:
                            data['val_loss'] = float(val_match.group(1))
                            
                        # Extract tokens per second
                        tps_match = re.search(r'Tokens/sec ([\d\.]+)', log_content)
                        if tps_match:
                            data['tokens_per_sec'] = float(tps_match.group(1))
                            
                        # Extract trained tokens
                        tt_match = re.search(r'Trained Tokens (\d+)', log_content)
                        if tt_match:
                            data['trained_tokens'] = int(tt_match.group(1))
                            
                        # Extract peak memory
                        mem_match = re.search(r'Peak mem ([\d\.]+) GB', log_content)
                        if mem_match:
                            data['peak_memory_gb'] = float(mem_match.group(1))
                            
                        # Extract learning rate
                        lr_match = re.search(r'Learning Rate ([\d\.e\-]+)', log_content)
                        if lr_match:
                            data['learning_rate'] = float(lr_match.group(1))
                            
                        # Calculate progress
                        if data['max_iterations'] > 0:
                            data['progress'] = (data['current_iteration'] / data['max_iterations']) * 100
            except Exception as e:
                logger.error(f"Error reading metrics: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return data 