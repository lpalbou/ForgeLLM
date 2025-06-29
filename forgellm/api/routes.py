"""API routes for ForgeLLM."""

import logging
import os
import re
import json
import psutil
import glob
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import tempfile
import shutil
import subprocess
from datetime import datetime
from flask import Flask, Blueprint, request, jsonify, current_app, send_file
import time

from ..models import ModelManager, ModelPublisher
from ..training.config import TrainingConfig
from ..training.trainer import ContinuedPretrainer
from ..training.process_manager import TrainingProcessManager
from ..training.dashboard import create_comprehensive_dashboard, identify_best_checkpoints, load_training_data, generate_web_chart_data

logger = logging.getLogger(__name__)

def setup_api(app: Flask) -> Blueprint:
    """Set up API routes for ForgeLLM.
    
    Args:
        app: Flask application
        
    Returns:
        Blueprint: API blueprint
    """
    bp = Blueprint('api', __name__, url_prefix='/api')
    
    # Get model manager
    model_manager = getattr(app, 'model_manager', None)
    if model_manager is None:
        model_manager = ModelManager()
        app.model_manager = model_manager
    
    # Get trainer
    trainer = getattr(app, 'trainer', None)
    if trainer is None:
        trainer = ContinuedPretrainer()
        app.trainer = trainer
    
    # Get training process manager
    training_manager = getattr(app, 'training_manager', None)
    if training_manager is None:
        training_manager = TrainingProcessManager()
        app.training_manager = training_manager
    
    @bp.route('/cpt_models', methods=['GET'])
    def get_cpt_models():
        """Get CPT models."""
        try:
            # Get models directory from environment or use default
            models_dir = os.environ.get('MODELS_DIR', 'models')
            cpt_dir = os.path.join(models_dir, 'cpt')
            
            # Check if the directory exists
            if not os.path.exists(cpt_dir):
                return jsonify({"models": []})
            
            # Get list of CPT models
            cpt_models = []
            for model_path in glob.glob(os.path.join(cpt_dir, '*')):
                if os.path.isdir(model_path):
                    model_name = os.path.basename(model_path)
                    
                    # Calculate model size using du command for accurate directory size
                    try:
                        # Use subprocess to run du command for accurate directory size with human-readable format
                        result = subprocess.run(
                            ['du', '-sh', model_path],  # -sh gives human-readable size
                            capture_output=True, 
                            text=True, 
                            check=False
                        )
                        if result.returncode == 0:
                            # Parse the output to get size with unit (e.g., "9.3G")
                            size_str = result.stdout.strip().split()[0]
                            
                            # Extract numeric part and unit
                            match = re.match(r'([0-9.]+)([KMGTP])', size_str)
                            if match:
                                size_num = float(match.group(1))
                                unit = match.group(2)
                                
                                # Convert to GB based on unit
                                if unit == 'K':
                                    size_gb = size_num / (1024 * 1024)
                                elif unit == 'M':
                                    size_gb = size_num / 1024
                                elif unit == 'G':
                                    size_gb = size_num
                                elif unit == 'T':
                                    size_gb = size_num * 1024
                                elif unit == 'P':
                                    size_gb = size_num * 1024 * 1024
                                else:
                                    size_gb = 0
                            else:
                                size_gb = 0
                        else:
                            # Fallback to a simple directory walk if du fails
                            size_gb = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                    for dirpath, dirnames, filenames in os.walk(model_path) 
                                    for filename in filenames if os.path.isfile(os.path.join(dirpath, filename))) / (1024**3)
                    except Exception as e:
                        logger.warning(f"Error calculating size for {model_name}: {e}")
                        size_gb = 0
                    
                    cpt_models.append({
                        "name": model_name,
                        "path": os.path.join('cpt', model_name),
                        "size": round(size_gb, 2)  # Round to 2 decimal places
                    })
            
            return jsonify({"models": cpt_models})
        except Exception as e:
            logger.error(f"Error getting CPT models: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/ift_models', methods=['GET'])
    def get_ift_models():
        """Get IFT models."""
        try:
            # Get models directory from environment or use default
            models_dir = os.environ.get('MODELS_DIR', 'models')
            ift_dir = os.path.join(models_dir, 'ift')
            
            # Check if the directory exists
            if not os.path.exists(ift_dir):
                return jsonify({"models": []})
            
            # Get list of IFT models
            ift_models = []
            for model_path in glob.glob(os.path.join(ift_dir, '*')):
                if os.path.isdir(model_path):
                    model_name = os.path.basename(model_path)
                    
                    # Calculate model size using du command for accurate directory size
                    try:
                        # Use subprocess to run du command for accurate directory size with human-readable format
                        result = subprocess.run(
                            ['du', '-sh', model_path],  # -sh gives human-readable size
                            capture_output=True, 
                            text=True, 
                            check=False
                        )
                        if result.returncode == 0:
                            # Parse the output to get size with unit (e.g., "9.3G")
                            size_str = result.stdout.strip().split()[0]
                            
                            # Extract numeric part and unit
                            match = re.match(r'([0-9.]+)([KMGTP])', size_str)
                            if match:
                                size_num = float(match.group(1))
                                unit = match.group(2)
                                
                                # Convert to GB based on unit
                                if unit == 'K':
                                    size_gb = size_num / (1024 * 1024)
                                elif unit == 'M':
                                    size_gb = size_num / 1024
                                elif unit == 'G':
                                    size_gb = size_num
                                elif unit == 'T':
                                    size_gb = size_num * 1024
                                elif unit == 'P':
                                    size_gb = size_num * 1024 * 1024
                                else:
                                    size_gb = 0
                            else:
                                size_gb = 0
                        else:
                            # Fallback to a simple directory walk if du fails
                            size_gb = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                    for dirpath, dirnames, filenames in os.walk(model_path) 
                                    for filename in filenames if os.path.isfile(os.path.join(dirpath, filename))) / (1024**3)
                    except Exception as e:
                        logger.warning(f"Error calculating size for {model_name}: {e}")
                        size_gb = 0
                    
                    ift_models.append({
                        "name": model_name,
                        "path": os.path.join('ift', model_name),
                        "size": round(size_gb, 2)  # Round to 2 decimal places
                    })
            
            return jsonify({"models": ift_models})
        except Exception as e:
            logger.error(f"Error getting IFT models: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/base_models', methods=['GET'])
    def get_base_models():
        """Get base models."""
        try:
            # Define a list of common base models
            base_models = []
            
            # Check HuggingFace cache for available models
            cache_path = Path.home() / '.cache' / 'huggingface' / 'hub'
            if cache_path.exists():
                # Look for models in the cache
                model_dirs = list(cache_path.glob('models--*'))
                for model_dir in model_dirs:
                    try:
                        # Extract model name from directory name
                        if model_dir.name.startswith('published--'):
                            # Handle published models - extract the actual model name
                            model_name = model_dir.name.replace('published--', '')
                        else:
                            # Handle regular cached models
                            model_name = model_dir.name.replace('models--', '').replace('--', '/')
                        
                        # Calculate model size using du command for accurate directory size
                        try:
                            # Use subprocess to run du command for accurate directory size with human-readable format
                            result = subprocess.run(
                                ['du', '-sh', str(model_dir)],  # -sh gives human-readable size
                                capture_output=True, 
                                text=True, 
                                check=False
                            )
                            if result.returncode == 0:
                                # Parse the output to get size with unit (e.g., "9.3G")
                                size_str = result.stdout.strip().split()[0]
                                
                                # Extract numeric part and unit
                                match = re.match(r'([0-9.]+)([KMGTP])', size_str)
                                if match:
                                    size_num = float(match.group(1))
                                    unit = match.group(2)
                                    
                                    # Convert to GB based on unit
                                    if unit == 'K':
                                        size_gb = size_num / (1024 * 1024)
                                    elif unit == 'M':
                                        size_gb = size_num / 1024
                                    elif unit == 'G':
                                        size_gb = size_num
                                    elif unit == 'T':
                                        size_gb = size_num * 1024
                                    elif unit == 'P':
                                        size_gb = size_num * 1024 * 1024
                                    else:
                                        size_gb = 0
                                else:
                                    size_gb = 0
                            else:
                                # Fallback to a simple directory walk if du fails
                                size_gb = sum(f.stat().st_size for f in model_dir.glob('**/*') if f.is_file()) / (1024**3)
                        except Exception as e:
                            logger.warning(f"Error calculating size for {model_name}: {e}")
                            size_gb = 0
                        
                        base_models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "size": round(size_gb, 2)  # Round to 2 decimal places
                        })
                    except Exception as e:
                        logger.warning(f"Error processing model directory {model_dir}: {e}")
            
            return jsonify({"models": base_models})
        except Exception as e:
            logger.error(f"Error getting base models: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/training/status', methods=['GET'])
    def get_training_status():
        """Get training status."""
        try:
            # Check if training is active using the training manager
            current_training = training_manager.current_training
            active = current_training is not None and current_training.get("status") == "running"
            
            # Get training status
            if active:
                status = {
                    "config": current_training.get("config", {}),
                    "start_time": current_training.get("start_time"),
                    "output_dir": current_training.get("output_dir"),
                    "pid": current_training.get("pid")
                }
            else:
                status = {}
            
            return jsonify({'success': True, 'active': active, **status})
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/training/start', methods=['POST'])
    def start_training():
        """Start training."""
        try:
            # Get training configuration from request
            if not request.is_json:
                logger.error("Request content type is not application/json")
                return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

            config_data = request.get_json(force=True)
            if config_data is None:
                logger.error("Failed to parse JSON data")
                return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400

            # Log received configuration data
            logger.info(f"Received training config data: {config_data}")

            # Start training using the training process manager
            result = training_manager.start_training(config_data)

            if result.get('success', False):
                return jsonify({'success': True, 'message': 'Training started', 'output_dir': result.get('output_dir'), 'pid': result.get('pid')})
            else:
                return jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/training/stop', methods=['POST'])
    def stop_training():
        """Stop training."""
        try:
            # Stop training using the training manager
            result = training_manager.stop_training()
            
            if result.get('success', True):
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': result.get('error', 'Failed to stop training')}), 500
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/model/load', methods=['POST'])
    def load_model():
        """Load a model."""
        try:
            data = request.get_json()
            # Accept both 'model_name' and 'model' for backward compatibility
            model_name = data.get('model_name') or data.get('model')
            adapter_path = data.get('adapter_path') or data.get('adapter')
            
            if not model_name:
                return jsonify({
                    'success': False,
                    'error': 'Missing model_name'
                }), 400
            
            success = model_manager.load(model_name, adapter_path)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Model {model_name} loading started',
                    'model_name': model_name,
                    'adapter_path': adapter_path
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to load model: {model_manager.error}'
                }), 500
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @bp.route('/model/unload', methods=['POST'])
    def unload_model():
        """Unload the model."""
        try:
            success = model_manager.unload()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Model unloaded successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to unload model'
                }), 500
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @bp.route('/model/generate', methods=['POST'])
    def generate_text():
        """Generate text from the model."""
        try:
            data = request.get_json()
            prompt = data.get('prompt')
            max_tokens = data.get('max_tokens', 100)
            temperature = data.get('temperature', 0.7)
            
            # Additional parameters from web UI
            history = data.get('history', [])
            top_p = data.get('top_p', 0.9)
            repetition_penalty = data.get('repetition_penalty', 1.1)
            max_kv_size = data.get('max_kv_size', 8192)
            system_prompt = data.get('system_prompt', '')  # Legacy support
            streaming = data.get('streaming', False)
            is_base_model = data.get('is_base_model', None)  # New parameter
            
            if not prompt:
                return jsonify({
                    'success': False,
                    'error': 'Missing prompt'
                }), 400
            
            # Check if model is loaded
            status = model_manager.get_status()
            if not status.get('loaded'):
                return jsonify({
                    'success': False,
                    'error': 'No model loaded'
                }), 400
            
            if streaming:
                # For streaming, we need to handle it differently
                # This would require implementing Server-Sent Events (SSE) or WebSocket
                # For now, let's return an error for streaming requests to the web API
                return jsonify({
                    'success': False,
                    'error': 'Streaming not supported via web API. Use model server directly.'
                }), 400
            else:
                # Generate text (non-streaming)
                start_time = time.time()
                response = model_manager.generate_text({
                    'prompt': prompt,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'history': history,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'max_kv_size': max_kv_size,
                    'system_prompt': system_prompt,  # Legacy support
                    'is_base_model': is_base_model  # New parameter
                })
                end_time = time.time()
                
                return jsonify({
                    'success': True,
                    'completion': response,
                    'generation_time': end_time - start_time
                })
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @bp.route('/dashboard/data', methods=['GET'])
    def get_dashboard_data():
        """Get dashboard data."""
        try:
            # Get training status
            active = trainer.is_training_active()
            
            # Get dashboard data
            data = trainer.get_dashboard_data() if active else {}
            
            return jsonify({'success': True, 'active': active, **data})
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/dashboard/realtime', methods=['GET'])
    def get_realtime_dashboard_data():
        """Get real-time dashboard data with direct, efficient logic."""
        try:
            # Simple, direct approach - no background monitoring needed
            import psutil
            import glob
            import json
            import os
            import time
            from pathlib import Path
            from datetime import datetime
            
            # 1. Check if MLX training is actually running (lightweight check)
            mlx_running = False
            active_training_file = None
            
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline']:
                            cmdline = ' '.join(proc.info['cmdline'])
                            if any(pattern in cmdline for pattern in [
                                'mlx_lm.lora', 'mlx_lm.fuse', 'mlx-lm', 'python -m mlx_lm'
                            ]):
                                mlx_running = True
                                logger.info(f"Active MLX training detected: PID {proc.info['pid']}")
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                logger.warning(f"Error checking MLX processes: {e}")
            
            # 2. If no MLX running, return inactive status immediately
            if not mlx_running:
                return jsonify({
                    'success': True,
                    'active': False,
                    'current_values': None,
                    'message': 'No active training detected'
                })
            
            # 3. MLX is running - find the active training file
            possible_dirs = [Path("models/cpt")]
            all_log_files = []
            
            for models_dir in possible_dirs:
                if models_dir.exists():
                    pattern = str(models_dir / "*" / "CPT_*.json")
                    log_files = glob.glob(pattern)
                    all_log_files.extend(log_files)
            
            # Find the most recent active training file
            most_recent = None
            most_recent_time = 0
            
            for log_file in all_log_files:
                try:
                    mtime = os.path.getmtime(log_file)
                    # Only consider files modified in the last 10 minutes
                    if time.time() - mtime < 600:
                        try:
                            with open(log_file, 'r') as f:
                                data = json.load(f)
                                # File should not have end_time to be considered active
                                if data.get('end_time') is None and mtime > most_recent_time:
                                    most_recent_time = mtime
                                    most_recent = log_file
                        except:
                            continue
                except Exception:
                    continue
            
            if not most_recent:
                return jsonify({
                    'success': True,
                    'active': False,
                    'current_values': None,
                    'message': 'MLX running but no active training file found'
                })
            
            # 4. Read the active training data
            try:
                with open(most_recent, 'r') as f:
                    training_data = json.load(f)
                
                if not training_data.get('metrics'):
                    return jsonify({
                        'success': True,
                        'active': False,
                        'current_values': None,
                        'message': 'No metrics found in training file'
                    })
                
                # 5. Extract current values from latest metrics
                latest_metrics = training_data['metrics'][-1]
                config = training_data.get('config', {})
                
                def format_numeric_value(value, max_decimals=3):
                    """Format numeric values with max precision and handle special cases"""
                    if value is None:
                        return None
                    if isinstance(value, str):
                        return value
                    if isinstance(value, (int, float)):
                        if isinstance(value, int) or value == int(value):
                            return int(value)  # Keep integers as integers
                        else:
                            # For very small numbers (like learning rates), preserve more precision
                            if abs(value) < 0.001 and value != 0:
                                # Use scientific notation or more decimal places for small values
                                return float(f"{value:.6g}")  # 6 significant digits
                            else:
                                # Round to max_decimals and remove trailing zeros
                                rounded = round(value, max_decimals)
                                return float(f"{rounded:.{max_decimals}f}".rstrip('0').rstrip('.'))
                    return value
                
                # Calculate epoch correctly using trained_tokens vs dataset_total_tokens
                epoch_value = latest_metrics.get('epoch')
                if epoch_value is None or epoch_value == '-':
                    # CORRECT epoch calculation: trained_tokens / dataset_total_tokens
                    trained_tokens = latest_metrics.get('trained_tokens', 0)
                    dataset_total_tokens = config.get('dataset_total_tokens', 0)
                    
                    if dataset_total_tokens > 0 and trained_tokens > 0:
                        epoch_fraction = trained_tokens / dataset_total_tokens
                        epoch_value = format_numeric_value(epoch_fraction, 3)
                    else:
                        epoch_value = 0.0
                else:
                    epoch_value = format_numeric_value(epoch_value, 3)
                
                # Calculate elapsed time and ETA
                elapsed_minutes = None
                eta_minutes = None
                
                try:
                    # Get start time from training data
                    start_time_str = training_data.get('start_time')
                    if start_time_str:
                        from datetime import datetime
                        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                        current_time = datetime.now(start_time.tzinfo) if start_time.tzinfo else datetime.now()
                        elapsed_seconds = (current_time - start_time).total_seconds()
                        elapsed_minutes = elapsed_seconds / 60
                        
                        # Calculate ETA based on progress
                        current_iteration = latest_metrics.get('iteration', 0)
                        max_iterations = config.get('max_iterations', 300)
                        
                        if current_iteration > 0 and max_iterations > current_iteration:
                            progress_fraction = current_iteration / max_iterations
                            if progress_fraction > 0:
                                total_estimated_minutes = elapsed_minutes / progress_fraction
                                eta_minutes = total_estimated_minutes - elapsed_minutes
                except Exception as e:
                    logger.debug(f"Error calculating time estimates: {e}")
                
                # Include ALL available fields with proper formatting
                current_values = {}
                core_fields = [
                    'iteration', 'epoch', 'train_loss', 'val_loss', 
                    'train_perplexity', 'val_perplexity', 'learning_rate',
                    'tokens_per_sec', 'trained_tokens', 'peak_memory_gb',
                    'iterations_per_sec', 'warmup_steps', 'lr_decay', 'weight_decay'
                ]
                
                for field in core_fields:
                    if field == 'epoch':
                        current_values[field] = epoch_value
                    elif field in latest_metrics:
                        value = latest_metrics[field]
                        # Apply formatting to numeric fields
                        if field in ['train_loss', 'val_loss', 'train_perplexity', 'val_perplexity', 
                                   'learning_rate', 'tokens_per_sec', 'peak_memory_gb', 'iterations_per_sec']:
                            current_values[field] = format_numeric_value(value, 3)
                        else:
                            current_values[field] = value
                    else:
                        # Handle missing fields - use config values where appropriate
                        if field == 'learning_rate':
                            # Use learning_rate from latest metrics first, then config
                            lr_value = latest_metrics.get('learning_rate')
                            if lr_value is None:
                                lr_value = config.get('learning_rate')
                            current_values[field] = format_numeric_value(lr_value, 3)
                        elif field == 'warmup_steps':
                            ws_value = latest_metrics.get('warmup_steps')
                            if ws_value is None:
                                ws_value = config.get('warmup_steps', '-')
                            current_values[field] = ws_value
                        elif field == 'lr_decay':
                            ld_value = latest_metrics.get('lr_decay')
                            if ld_value is None:
                                ld_value = config.get('lr_decay_factor', '-')
                            current_values[field] = ld_value
                        elif field == 'weight_decay':
                            wd_value = latest_metrics.get('weight_decay')
                            if wd_value is None:
                                wd_value = config.get('weight_decay', '-')
                            current_values[field] = wd_value
                        elif field in ['val_loss', 'val_perplexity']:
                            current_values[field] = None
                        else:
                            current_values[field] = latest_metrics.get(field, 0)
                
                # Add time estimates
                if elapsed_minutes is not None:
                    current_values['elapsed_minutes'] = format_numeric_value(elapsed_minutes, 1)
                if eta_minutes is not None:
                    current_values['eta_minutes'] = format_numeric_value(eta_minutes, 1)
                
                # 6. Generate charts if needed
                charts = None
                try:
                    from ..training.dashboard import generate_web_chart_data
                    chart_data = {'metrics': training_data['metrics']}
                    charts = generate_web_chart_data(chart_data)
                except Exception as e:
                    logger.warning(f"Error generating charts: {e}")
                
                # Return data with both current_values and individual fields for UI compatibility
                response_data = {
                    'success': True,
                    'active': True,
                    'current_values': current_values,
                    'config': config,
                    'charts': charts,
                    'training_file': most_recent,
                    'start_time': training_data.get('start_time'),
                    'last_update': datetime.now().isoformat()
                }
                
                # Add individual fields at top level for UI compatibility
                response_data.update(current_values)
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Error reading training file {most_recent}: {e}")
                return jsonify({
                    'success': True,
                    'active': False,
                    'current_values': None,
                    'message': f'Error reading training data: {e}'
                })
            
        except Exception as e:
            logger.error(f"Error getting real-time dashboard data: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/check_dashboard', methods=['GET'])
    def check_dashboard():
        """Check if a dashboard exists for a model."""
        try:
            # Get model path from request
            path = request.args.get('path')
            
            if not path:
                return jsonify({'success': False, 'error': 'No path provided'}), 400
            
            logger.info(f"Checking dashboard for path: {path}")
            
            # For published models, the path will be like: published/gemma_3_4b_it_bf16_lr3e_05_bs2_iter200_seq3072_2025-06-29_15-17_0000200_20250629_163142
            # We need to convert this to the actual cache path
            if path.startswith('published/'):
                # Extract the published model name
                published_name = path.replace('published/', '')
                # Construct the full cache path
                cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--published--{published_name}'
                actual_path = str(cache_path)
                logger.info(f"Converted published model path to: {actual_path}")
            else:
                actual_path = path
            
            # Check for published model dashboard first (assets/training_dashboard.png)
            published_dashboard_path = os.path.join(actual_path, 'assets', 'training_dashboard.png')
            if os.path.exists(published_dashboard_path):
                # Return a web-accessible URL for the published dashboard
                # Use the original path for URL generation to avoid issues with special characters
                safe_path = path.replace('/', '_').replace(' ', '_').replace(':', '_').replace('-', '_')
                dashboard_url = f"/api/dashboard/{safe_path}/training_dashboard.png"
                logger.info(f"Found published dashboard at: {published_dashboard_path}, URL: {dashboard_url}")
                return jsonify({
                    'success': True, 
                    'exists': True, 
                    'dashboard_url': dashboard_url,
                    'path': published_dashboard_path
                })
            
            # Fallback: Check for legacy dashboard directory
            legacy_dashboard_path = os.path.join(actual_path, 'dashboard')
            if os.path.exists(legacy_dashboard_path):
                safe_path = path.replace('/', '_').replace(' ', '_').replace(':', '_').replace('-', '_')
                return jsonify({
                    'success': True, 
                    'exists': True, 
                    'dashboard_url': f"/api/dashboard/{safe_path}/dashboard",
                    'path': legacy_dashboard_path
                })
            
            logger.info(f"No dashboard found for path: {actual_path}")
            return jsonify({'success': True, 'exists': False, 'dashboard_url': None, 'path': None})
            
        except Exception as e:
            logger.error(f"Error checking dashboard: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/dashboard/<path:model_path>/<filename>')
    def serve_dashboard_image(model_path, filename):
        """Serve dashboard images for published models."""
        try:
            logger.info(f"Serving dashboard for model_path: {model_path}, filename: {filename}")
            
            # Convert back from URL-safe format - need to handle the complex conversions
            # First, convert underscores back to various characters
            actual_path = model_path.replace('_', '/')
            
            # For published models, convert to the actual cache path
            if actual_path.startswith('published/'):
                published_name = actual_path.replace('published/', '')
                # Need to restore the original format with dashes and colons
                # This is a bit tricky since we converted multiple characters to underscores
                # Let's check if this is a published model pattern and construct the cache path
                cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--published--{published_name}'
                
                # Try to find the actual directory since the name might have been mangled
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
                if cache_dir.exists():
                    # Look for directories that start with the expected prefix
                    pattern = f'models--published--*{published_name.split("/")[-1]}*'
                    matching_dirs = list(cache_dir.glob(pattern))
                    if matching_dirs:
                        actual_path = str(matching_dirs[0])
                        logger.info(f"Found matching published model directory: {actual_path}")
                    else:
                        actual_path = str(cache_path)
                        logger.info(f"Using constructed cache path: {actual_path}")
                else:
                    actual_path = str(cache_path)
            
            # Construct the full path to the dashboard image
            if filename == 'training_dashboard.png':
                image_path = os.path.join(actual_path, 'assets', filename)
            else:
                # For other dashboard types (legacy)
                image_path = os.path.join(actual_path, filename)
            
            logger.info(f"Looking for dashboard image at: {image_path}")
            
            # Check if file exists and serve it
            if os.path.exists(image_path):
                from flask import send_file
                logger.info(f"Serving dashboard image: {image_path}")
                return send_file(image_path, mimetype='image/png')
            else:
                logger.warning(f"Dashboard image not found: {image_path}")
                return jsonify({'error': 'Dashboard image not found'}), 404
                
        except Exception as e:
            logger.error(f"Error serving dashboard image: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/open_folder', methods=['POST'])
    def open_folder():
        """Open a folder in the system file explorer."""
        try:
            data = request.get_json()
            folder_path = data.get('path')
            
            if not folder_path:
                return jsonify({'success': False, 'error': 'No path provided'}), 400
            
            # Convert to absolute path if relative
            if not os.path.isabs(folder_path):
                folder_path = os.path.abspath(folder_path)
            
            # Check if path exists
            if not os.path.exists(folder_path):
                return jsonify({'success': False, 'error': f'Path does not exist: {folder_path}'}), 404
            
            # If it's a file, get the parent directory
            if os.path.isfile(folder_path):
                folder_path = os.path.dirname(folder_path)
            
            # Open folder based on operating system
            import platform
            import subprocess
            
            system = platform.system()
            try:
                if system == 'Darwin':  # macOS
                    subprocess.run(['open', folder_path], check=True)
                elif system == 'Windows':
                    subprocess.run(['explorer', folder_path], check=True)
                elif system == 'Linux':
                    subprocess.run(['xdg-open', folder_path], check=True)
                else:
                    return jsonify({'success': False, 'error': f'Unsupported operating system: {system}'}), 400
                
                return jsonify({'success': True, 'message': f'Opened folder: {folder_path}'})
                
            except subprocess.CalledProcessError as e:
                return jsonify({'success': False, 'error': f'Failed to open folder: {e}'}), 500
            
        except Exception as e:
            logger.error(f"Error opening folder: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/training/publish_checkpoint', methods=['POST'])
    def publish_checkpoint():
        """Publish a checkpoint."""
        try:
            # Get checkpoint path from request
            path = request.json.get('path')
            
            # Publish checkpoint
            publisher = ModelPublisher()
            result = publisher.publish_checkpoint(path)
            
            return jsonify({'success': True, **result})
        except Exception as e:
            logger.error(f"Error publishing checkpoint: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/logs/raw', methods=['POST'])
    def get_raw_logs():
        """Get raw logs."""
        try:
            # Get log file path from request
            log_file = request.json.get('log_file')
            
            # Read log file
            with open(log_file, 'r') as f:
                logs = f.read()
            
            return jsonify({'success': True, 'logs': logs})
        except Exception as e:
            logger.error(f"Error getting raw logs: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/dashboard/historical', methods=['POST'])
    def get_historical_dashboard():
        """Get historical dashboard data."""
        try:
            # Get log file path from request
            log_file = request.json.get('log_file')
            
            if not log_file:
                return jsonify({'success': False, 'error': 'No log file specified'}), 400
            
            # Load training data
            data = load_training_data(log_file)
            
            if 'error' in data:
                return jsonify({'success': False, 'error': data['error']}), 500
            
            # Generate chart data for web display
            charts = generate_web_chart_data(data)
            
            # Identify best checkpoints
            best_checkpoints = identify_best_checkpoints(data, top_k=3)
            
            # Create summary with metrics and best checkpoints
            metrics = data.get('metrics', [])
            config = data.get('config', {})
            
            # Extract all checkpoints from metrics with proper path parsing
            all_checkpoints = []
            for metric in metrics:
                if metric.get('checkpoint_saved') and metric.get('checkpoint_path'):
                    # Parse the compound checkpoint path to extract the numbered checkpoint
                    checkpoint_path = metric.get('checkpoint_path')
                    parsed_path = None
                    
                    if checkpoint_path:
                        # The checkpoint_path contains both paths, extract the numbered one
                        # e.g., "models/cpt/.../adapters.safetensors and models/cpt/.../0000200_adapters.safetensors."
                        parts = checkpoint_path.split(' and ')
                        iteration = metric.get('iteration')
                        for part in parts:
                            part = part.rstrip('.')  # Remove trailing period
                            if iteration and f"{iteration:07d}_adapters.safetensors" in part:
                                parsed_path = part
                                break
                        else:
                            # Fallback: use the last part if no numbered match found
                            if parts:
                                parsed_path = parts[-1].rstrip('.')
                            else:
                                parsed_path = checkpoint_path.rstrip('.')
                    
                    checkpoint_info = {
                        'iteration': metric.get('iteration'),
                        'path': parsed_path,  # Use parsed path instead of raw path
                        'train_loss': metric.get('train_loss'),
                        'val_loss': metric.get('val_loss'),
                        'train_perplexity': metric.get('train_perplexity'),
                        'val_perplexity': metric.get('val_perplexity'),
                        'learning_rate': metric.get('learning_rate'),
                        'timestamp': metric.get('timestamp')
                    }
                    all_checkpoints.append(checkpoint_info)
            
            summary = {
                'total_iterations': len(metrics),
                'best_checkpoints': best_checkpoints,
                'all_checkpoints': all_checkpoints,
                'config': config
            }
            
            # Add latest metrics if available
            if metrics:
                latest = metrics[-1]
                summary.update({
                    'iteration': latest.get('iteration', 0),
                    'train_loss': latest.get('train_loss'),
                    'val_loss': latest.get('val_loss'),
                    'train_perplexity': latest.get('train_perplexity'),
                    'val_perplexity': latest.get('val_perplexity'),
                    'learning_rate': latest.get('learning_rate'),
                    'tokens_per_sec': latest.get('tokens_per_sec'),
                    'peak_memory_gb': latest.get('peak_memory_gb'),
                    'trained_tokens': latest.get('trained_tokens', 0)
                })
            
            # Add config-based metrics
            if config:
                summary.update({
                    'warmup_steps': config.get('warmup_steps'),
                    'lr_decay_factor': config.get('lr_decay_factor'),
                    'weight_decay': config.get('weight_decay'),
                    'max_iterations': config.get('max_iterations'),
                    'lr_schedule': config.get('lr_schedule')
                })
            
            # Return data in the format expected by frontend
            return jsonify({
                'success': True,
                'charts': charts,
                'summary': summary
            })
            
        except Exception as e:
            logger.error(f"Error getting historical dashboard: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/training/sessions', methods=['GET'])
    def get_training_sessions():
        """Get available training sessions."""
        try:
            import glob
            from pathlib import Path
            
            # Look for training sessions
            possible_dirs = [
                Path("models/cpt")
            ]
            
            all_sessions = []
            
            for models_dir in possible_dirs:
                if models_dir.exists():
                    # Find all CPT log files
                    log_pattern = str(models_dir / "*" / "CPT_*.json")
                    log_files = glob.glob(log_pattern)
                    
                    for log_file in log_files:
                        try:
                            log_path = Path(log_file)
                            session_dir = log_path.parent
                            
                            # Read training data
                            with open(log_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract session info
                            session_info = {
                                "session_id": session_dir.name,
                                "session_name": session_dir.name,
                                "log_file": str(log_path),
                                "start_time": data.get('start_time'),
                                "status": data.get('status', 'unknown'),
                                "model_name": data.get('config', {}).get('model_name', 'Unknown'),
                                "metrics_count": len(data.get('metrics', [])),
                                "modified": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat()
                            }
                            
                            # Add latest metrics if available
                            metrics = data.get('metrics', [])
                            if metrics:
                                latest = metrics[-1]
                                session_info.update({
                                    "latest_iteration": latest.get('iteration'),
                                    "latest_loss": latest.get('train_loss'),
                                    "latest_val_loss": latest.get('val_loss')
                                })
                            
                            all_sessions.append(session_info)
                            
                        except Exception as e:
                            logger.warning(f"Error reading training session {log_file}: {e}")
                            continue
            
            # Sort sessions by modification time (newest first)
            all_sessions.sort(key=lambda x: x['modified'], reverse=True)
            
            return jsonify({"success": True, "training_sessions": all_sessions})
            
        except Exception as e:
            logger.error(f"Error getting training sessions: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @bp.route('/checkpoints', methods=['GET'])
    def get_checkpoints():
        """Get available checkpoints."""
        try:
            # Get model directory from query parameters
            model_dir = request.args.get('model_dir', '')
            
            # Get models directory from environment or use default
            models_dir = os.environ.get('MODELS_DIR', 'models')
            
            # If model_dir is provided, look for checkpoints in that directory
            if model_dir:
                # Check if the model directory exists
                full_model_path = os.path.join(models_dir, model_dir)
                if not os.path.exists(full_model_path):
                    return jsonify({'success': False, 'error': f"Model directory {model_dir} not found"}), 404
                
                # Get list of checkpoints
                checkpoints = []
                for checkpoint_path in glob.glob(os.path.join(full_model_path, '*_adapters.safetensors')):
                    checkpoint_name = os.path.basename(checkpoint_path)
                    # Extract iteration number from checkpoint name
                    iteration = int(checkpoint_name.split('_')[0])
                    checkpoints.append({
                        'name': checkpoint_name,
                        'path': checkpoint_path,
                        'iteration': iteration,
                        'created': datetime.fromtimestamp(os.path.getctime(checkpoint_path)).isoformat(),
                        'size': os.path.getsize(checkpoint_path) / (1024 * 1024),  # Size in MB
                    })
                
                # Sort checkpoints by iteration (highest first)
                checkpoints.sort(key=lambda x: x['iteration'], reverse=True)
                
                return jsonify({'success': True, 'checkpoints': checkpoints})
            
            # If no model_dir is provided, return all checkpoints from all models
            all_checkpoints = []
            
            # Check CPT models
            cpt_dir = os.path.join(models_dir, 'cpt')
            if os.path.exists(cpt_dir):
                for model_path in glob.glob(os.path.join(cpt_dir, '*')):
                    if os.path.isdir(model_path):
                        model_name = os.path.basename(model_path)
                        for checkpoint_path in glob.glob(os.path.join(model_path, '*_adapters.safetensors')):
                            checkpoint_name = os.path.basename(checkpoint_path)
                            # Extract iteration number from checkpoint name
                            iteration = int(checkpoint_name.split('_')[0])
                            all_checkpoints.append({
                                'name': checkpoint_name,
                                'path': checkpoint_path,
                                'model': model_name,
                                'model_path': model_path,
                                'type': 'cpt',
                                'iteration': iteration,
                                'created': datetime.fromtimestamp(os.path.getctime(checkpoint_path)).isoformat(),
                                'size': os.path.getsize(checkpoint_path) / (1024 * 1024),  # Size in MB
                            })
            
            # Check IFT models
            ift_dir = os.path.join(models_dir, 'ift')
            if os.path.exists(ift_dir):
                for model_path in glob.glob(os.path.join(ift_dir, '*')):
                    if os.path.isdir(model_path):
                        model_name = os.path.basename(model_path)
                        for checkpoint_path in glob.glob(os.path.join(model_path, '*_adapters.safetensors')):
                            checkpoint_name = os.path.basename(checkpoint_path)
                            # Extract iteration number from checkpoint name
                            try:
                                iteration = int(checkpoint_name.split('_')[0])
                            except:
                                iteration = 0
                            all_checkpoints.append({
                                'name': checkpoint_name,
                                'path': checkpoint_path,
                                'model': model_name,
                                'model_path': model_path,
                                'type': 'ift',
                                'iteration': iteration,
                                'created': datetime.fromtimestamp(os.path.getctime(checkpoint_path)).isoformat(),
                                'size': os.path.getsize(checkpoint_path) / (1024 * 1024),  # Size in MB
                            })
            
            # Sort checkpoints by creation time (newest first)
            all_checkpoints.sort(key=lambda x: x['created'], reverse=True)
            
            return jsonify({'success': True, 'checkpoints': all_checkpoints})
        except Exception as e:
            logger.error(f"Error getting checkpoints: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/models', methods=['GET'])
    def get_models():
        """Get all available models (base, CPT, IFT)."""
        try:
            # Get base models - call the logic directly to avoid response parsing issues
            base_models = []
            cache_path = Path.home() / '.cache' / 'huggingface' / 'hub'
            if cache_path.exists():
                model_dirs = list(cache_path.glob('models--*'))
                for model_dir in model_dirs:
                    try:
                        if model_dir.name.startswith('published--'):
                            model_name = model_dir.name.replace('published--', '')
                        else:
                            model_name = model_dir.name.replace('models--', '').replace('--', '/')
                        
                        # Calculate model size
                        try:
                            result = subprocess.run(
                                ['du', '-sh', str(model_dir)],
                                capture_output=True, 
                                text=True, 
                                check=False
                            )
                            if result.returncode == 0:
                                size_str = result.stdout.strip().split()[0]
                                match = re.match(r'([0-9.]+)([KMGTP])', size_str)
                                if match:
                                    size_num = float(match.group(1))
                                    unit = match.group(2)
                                    
                                    if unit == 'K':
                                        size_gb = size_num / (1024 * 1024)
                                    elif unit == 'M':
                                        size_gb = size_num / 1024
                                    elif unit == 'G':
                                        size_gb = size_num
                                    elif unit == 'T':
                                        size_gb = size_num * 1024
                                    elif unit == 'P':
                                        size_gb = size_num * 1024 * 1024
                                    else:
                                        size_gb = 0
                                else:
                                    size_gb = 0
                            else:
                                size_gb = sum(f.stat().st_size for f in model_dir.glob('**/*') if f.is_file()) / (1024**3)
                        except Exception as e:
                            logger.warning(f"Error calculating size for {model_name}: {e}")
                            size_gb = 0
                        
                        base_models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "size": round(size_gb, 2)
                        })
                    except Exception as e:
                        logger.warning(f"Error processing model directory {model_dir}: {e}")
            
            # Get CPT models
            cpt_response = get_cpt_models()
            cpt_data = json.loads(cpt_response.data) if not isinstance(cpt_response, tuple) else {"models": []}
            cpt_models = cpt_data.get("models", [])
            
            # Get IFT models
            ift_response = get_ift_models()
            ift_data = json.loads(ift_response.data) if not isinstance(ift_response, tuple) else {"models": []}
            ift_models = ift_data.get("models", [])
            
            # Combine all models
            all_models = []
            
            # Add base models
            for model in base_models:
                all_models.append({
                    "id": model.get("name", ""),
                    "name": model.get("name", ""),
                    "path": model.get("path", model.get("name", "")),  # Include actual path for folder opening
                    "type": "base",
                    "size": model.get("size", 0)
                })
            
            # Get IFT models
            ift_response = get_ift_models()
            ift_data = json.loads(ift_response.data) if not isinstance(ift_response, tuple) else {"models": []}
            ift_models = ift_data.get("models", [])
            
            # Add CPT models
            for model in cpt_models:
                all_models.append({
                    "id": model.get("path", ""),
                    "name": model.get("name", ""),
                    "path": model.get("path", ""),
                    "type": "cpt",
                    "size": model.get("size", 0)
                })
            
            # Add IFT models
            for model in ift_models:
                all_models.append({
                    "id": model.get("path", ""),
                    "name": model.get("name", ""),
                    "path": model.get("path", ""),
                    "type": "ift",
                    "size": model.get("size", 0)
                })
            
            # Sort models alphabetically by name
            all_models.sort(key=lambda x: x.get("name", "").lower())
            
            return jsonify({"models": all_models})
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/dataset/info', methods=['GET'])
    def get_dataset_info():
        """Get dataset information."""
        try:
            from pathlib import Path
            import re
            
            # Get directory from query parameters (default to 'dataset')
            dir_param = request.args.get('dir', 'dataset')
            dataset_dir = Path(dir_param)
            
            # Check if directory exists
            if not dataset_dir.exists():
                return jsonify({
                    "success": False, 
                    "error": f"Directory {dir_param} not found",
                    "total_tokens": 1000000,  # Default value if directory not found
                    "total_files": 0,
                    "directory": dir_param
                })
            
            # Count tokens and files
            total_tokens = 0
            total_files = 0
            supported_extensions = {'.txt', '.md', '.rst', '.py', '.json'}
            
            # Count tokens in all supported files
            for file_path in dataset_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # Simple token estimation: split by whitespace and punctuation
                            tokens = len(re.findall(r'\b\w+\b', content))
                            total_tokens += tokens
                            total_files += 1
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
                        continue
            
            return jsonify({
                "success": True,
                "total_tokens": total_tokens or 1000000,  # Use default if no tokens found
                "total_files": total_files,
                "directory": str(dataset_dir),
                "supported_extensions": list(supported_extensions)
            })
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "total_tokens": 1000000,  # Default value on error
                "total_files": 0
            }), 500
    
    @bp.route('/model/info', methods=['GET'])
    def get_model_info():
        """Get information about the currently loaded model."""
        try:
            # Get model info
            info = model_manager.get_model_info()
            
            return jsonify({'success': True, **info})
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'success': True,
            'status': 'ok',
            'version': '0.1.0'
        })
    
    @bp.route('/model/status', methods=['GET'])
    def get_model_status():
        """Get the status of the currently loaded model."""
        try:
            status = model_manager.get_status()
            
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @bp.route('/filesystem/browse', methods=['GET'])
    def browse_filesystem():
        """Browse filesystem directories for file selection."""
        try:
            # Get path parameter (default to current working directory)
            path = request.args.get('path', os.getcwd())
            
            # Security check: only allow browsing within the project directory and common directories
            # Determine the actual project root - prefer the forgellm directory if we're in it
            current_dir = os.path.abspath(os.getcwd())
            
            # Check if we're already in the forgellm directory (contains forgellm package)
            if os.path.exists(os.path.join(current_dir, 'forgellm', '__init__.py')):
                project_root = current_dir
            else:
                # Look for the project root (directory containing forgellm folder)
                project_root = current_dir
                while project_root != os.path.dirname(project_root):  # Not at filesystem root
                    if os.path.exists(os.path.join(project_root, 'forgellm')) and os.path.isdir(os.path.join(project_root, 'forgellm')):
                        # If we found a directory containing forgellm, use the forgellm subdirectory as project root
                        project_root = os.path.join(project_root, 'forgellm')
                        break
                    project_root = os.path.dirname(project_root)
            
            # Convert relative paths to absolute paths relative to project root
            if not os.path.isabs(path):
                path = os.path.join(project_root, path)
            
            abs_path = os.path.abspath(path)
            
            # Allow browsing within project directory or common directories like /Users, /home, etc.
            allowed_roots = [project_root, '/Users', '/home', '/data', '/opt', '/tmp']
            if not any(abs_path.startswith(root) for root in allowed_roots):
                return jsonify({
                    'success': False,
                    'error': 'Access denied to this directory'
                }), 403
            
            # Check if path exists and is a directory
            if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
                return jsonify({
                    'success': False,
                    'error': 'Directory does not exist'
                }), 404
            
            # Get directory contents
            items = []
            try:
                for item_name in sorted(os.listdir(abs_path)):
                    item_path = os.path.join(abs_path, item_name)
                    
                    # Skip hidden files/directories (starting with .)
                    if item_name.startswith('.'):
                        continue
                    
                    # Get item info
                    is_dir = os.path.isdir(item_path)
                    
                    # For directories, count subdirectories and files
                    if is_dir:
                        try:
                            sub_items = os.listdir(item_path)
                            sub_dirs = sum(1 for item in sub_items if os.path.isdir(os.path.join(item_path, item)))
                            sub_files = sum(1 for item in sub_items if os.path.isfile(os.path.join(item_path, item)))
                            description = f"{sub_dirs} dirs, {sub_files} files"
                        except PermissionError:
                            description = "Access denied"
                        except Exception:
                            description = "Unknown"
                    else:
                        # For files, show file size
                        try:
                            size = os.path.getsize(item_path)
                            if size < 1024:
                                description = f"{size} B"
                            elif size < 1024 * 1024:
                                description = f"{size/1024:.1f} KB"
                            elif size < 1024 * 1024 * 1024:
                                description = f"{size/(1024*1024):.1f} MB"
                            else:
                                description = f"{size/(1024*1024*1024):.1f} GB"
                        except Exception:
                            description = "Unknown size"
                    
                    items.append({
                        'name': item_name,
                        'path': item_path,
                        'is_directory': is_dir,
                        'description': description
                    })
            except PermissionError:
                return jsonify({
                    'success': False,
                    'error': 'Permission denied'
                }), 403
            
            # Add parent directory option (if not at root)
            parent_path = os.path.dirname(abs_path)
            show_parent = abs_path != parent_path and any(abs_path.startswith(root) for root in allowed_roots)
            
            return jsonify({
                'success': True,
                'current_path': abs_path,
                'parent_path': parent_path if show_parent else None,
                'items': items
            })
        except Exception as e:
            logger.error(f"Error browsing filesystem: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return bp 