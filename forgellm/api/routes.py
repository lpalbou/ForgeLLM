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

from ..models import ModelManager, ModelPublisher
from ..training.config import TrainingConfig
from ..training.trainer import ContinuedPretrainer
from ..training.dashboard import create_comprehensive_dashboard, identify_best_checkpoints, load_training_data

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
                        model_name = model_dir.name.replace('models--', '').replace('--', '/')
                        
                        # Skip published models
                        if "published" in model_name.lower():
                            continue
                        
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
            # Check if training is active
            active = trainer.is_training_active()
            
            # Get training status
            status = trainer.get_training_status() if active else {}
            
            return jsonify({'success': True, 'active': active, **status})
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/training/start', methods=['POST'])
    def start_training():
        """Start training."""
        try:
            # Get training configuration from request
            config_data = request.json
            
            # Log received configuration data
            logger.info(f"Received training config data: {config_data}")
            
            # Log TrainingConfig expected parameters
            from forgellm.training.config import TrainingConfig
            expected_params = list(TrainingConfig.__annotations__.keys())
            logger.info(f"TrainingConfig expected parameters: {expected_params}")
            
            # Check for parameter mismatches
            for key in config_data:
                if key not in expected_params:
                    logger.warning(f"Unexpected parameter in request: '{key}' (not in TrainingConfig)")
            
            for key in expected_params:
                if key not in config_data and key != 'DEFAULT_CONFIG_PATH':
                    logger.warning(f"Missing expected parameter in request: '{key}'")
            
            # Create training configuration
            config = TrainingConfig(**config_data)
            
            # Start training
            trainer.start_training(config)
            
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/training/stop', methods=['POST'])
    def stop_training():
        """Stop training."""
        try:
            # Stop training
            trainer.stop_training()
            
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/model/load', methods=['POST'])
    def load_model():
        """Load a model."""
        try:
            # Get model name and adapter path from request
            model_name = request.json.get('model_name')
            adapter_path = request.json.get('adapter_path')
            
            # Load model
            model_manager.load_model(model_name, adapter_path)
            
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/model/unload', methods=['POST'])
    def unload_model():
        """Unload the current model."""
        try:
            # Unload model
            model_manager.unload_model()
            
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @bp.route('/model/generate', methods=['POST'])
    def generate_text():
        """Generate text."""
        try:
            # Get generation parameters from request
            params = request.json
            
            # Generate text
            text = model_manager.generate_text(params)
            
            return jsonify({'success': True, 'text': text})
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
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
    
    @bp.route('/check_dashboard', methods=['GET'])
    def check_dashboard():
        """Check if a dashboard exists for a model."""
        try:
            # Get model path from request
            path = request.args.get('path')
            
            # Check if dashboard exists
            dashboard_path = os.path.join(path, 'dashboard')
            exists = os.path.exists(dashboard_path)
            
            return jsonify({'success': True, 'exists': exists, 'path': dashboard_path if exists else None})
        except Exception as e:
            logger.error(f"Error checking dashboard: {e}")
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
            
            # Load training data
            data = load_training_data(log_file)
            
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            logger.error(f"Error getting historical dashboard: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
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
            # Get base models
            base_response = get_base_models()
            base_data = json.loads(base_response.data) if not isinstance(base_response, tuple) else {"models": []}
            base_models = base_data.get("models", [])
            
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
                    "type": "base",
                    "size": model.get("size", 0)
                })
            
            # Add CPT models
            for model in cpt_models:
                all_models.append({
                    "id": model.get("path", ""),
                    "name": model.get("name", ""),
                    "type": "cpt",
                    "size": model.get("size", 0)
                })
            
            # Add IFT models
            for model in ift_models:
                all_models.append({
                    "id": model.get("path", ""),
                    "name": model.get("name", ""),
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
            
            # Get directory from query parameters (default to 'mnemosyne')
            dir_param = request.args.get('dir', 'mnemosyne')
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
    
    return bp 