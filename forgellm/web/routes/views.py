"""
Web routes for ForgeLLM web interface.
"""

import os
import glob
import time
from pathlib import Path
import logging
from flask import Blueprint, render_template, send_from_directory, current_app

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('views', __name__)

@bp.route('/')
def index():
    """Render the main index page"""
    # Add cache buster to force JavaScript reload after changes
    # Force cache bust to reload updated JavaScript
    cache_buster = int(time.time())
    return render_template('index.html', cache_buster=cache_buster)

@bp.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(current_app.static_folder, path)

@bp.route('/models/<path:path>')
def serve_models(path):
    """Serve model files"""
    models_dir = os.environ.get('MODELS_DIR', 'models')
    return send_from_directory(models_dir, path)

@bp.route('/data/<path:path>')
def serve_data(path):
    """Serve data files"""
    data_dir = os.environ.get('DATA_DIR', 'data')
    return send_from_directory(data_dir, path)

@bp.route('/dashboard_images/<path:model_dir>/training_dashboard.png')
def serve_dashboard_image(model_dir):
    """Serve the training dashboard image"""
    try:
        logger.info(f"Request to serve dashboard image for: {model_dir}")
        
        # Special case for direct path
        if model_dir == 'direct':
            # Look for the most recently accessed dashboard in any path
            dashboard_files = []
            
            # Check in published directories
            for path in Path('published').glob('**/assets/training_dashboard.png'):
                if path.exists():
                    dashboard_files.append((path.stat().st_mtime, path))
            
            # Check in model directories
            for path in Path('models').glob('**/assets/training_dashboard.png'):
                if path.exists():
                    dashboard_files.append((path.stat().st_mtime, path))
                    
            # Check in HuggingFace cache
            hf_cache_root = Path.home() / '.cache' / 'huggingface' / 'hub'
            for path in hf_cache_root.glob('**/assets/training_dashboard.png'):
                if path.exists():
                    dashboard_files.append((path.stat().st_mtime, path))
            
            # Sort by modification time (newest first)
            dashboard_files.sort(reverse=True)
            
            if dashboard_files:
                img_path = dashboard_files[0][1]
                logger.info(f"Serving most recent dashboard from {img_path}")
                return send_from_directory(img_path.parent, img_path.name)
        
        # Look for the image in different possible locations
        possible_paths = [
            Path(f"models/cpt/{model_dir}/assets/training_dashboard.png"),
            Path(f"models/ift/{model_dir}/assets/training_dashboard.png"),
            Path(f"published/{model_dir}/assets/training_dashboard.png"),
            Path(f"{model_dir}/assets/training_dashboard.png"),
            Path(f"{model_dir}/training_dashboard.png")
        ]
        
        # Add HuggingFace cache paths
        hf_cache_root = Path.home() / '.cache' / 'huggingface' / 'hub'
        model_dir_hf = model_dir.replace('_', '--')
        
        # Try both with and without 'published--' prefix
        for prefix in ['', 'published--']:
            cache_path = hf_cache_root / f"models--{prefix}{model_dir_hf}" / "assets" / "training_dashboard.png"
            possible_paths.append(cache_path)
            
            # Also try snapshots directory
            snapshot_pattern = hf_cache_root / f"models--{prefix}{model_dir_hf}" / "snapshots" / "*" / "assets" / "training_dashboard.png"
            for img_path in glob.glob(str(snapshot_pattern)):
                possible_paths.append(Path(img_path))
        
        # Log the paths we're checking
        logger.info(f"Looking for dashboard image in: {[str(p) for p in possible_paths]}")
        
        # Check each path
        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.info(f"Serving dashboard from {path}")
                return send_from_directory(path.parent, path.name)
        
        # If not found, return 404
        logger.warning(f"Dashboard image not found for {model_dir}")
        return "Dashboard image not found", 404
        
    except Exception as e:
        logger.error(f"Error serving dashboard image: {e}")
        return "Error serving image", 500 