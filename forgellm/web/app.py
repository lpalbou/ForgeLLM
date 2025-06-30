"""
Flask application for ForgeLLM web interface.
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

from ..api.routes import setup_api
from .routes import bp as views_bp
from .services.socket_service import setup_socketio
from ..models import ModelManager, ModelQuantizer
from ..training.trainer import ContinuedPretrainer

logger = logging.getLogger(__name__)

def create_app(static_folder=None, template_folder=None):
    """Create Flask application.
    
    Args:
        static_folder: Path to static folder
        template_folder: Path to template folder
        
    Returns:
        Flask application
    """
    # Set default static and template folders if not provided
    if static_folder is None:
        static_folder = os.path.join(os.path.dirname(__file__), 'static')
    
    if template_folder is None:
        template_folder = os.path.join(os.path.dirname(__file__), 'templates')
    
    # Use package's own static and template folders
    logger.info(f"Using static folder: {static_folder}")
    logger.info(f"Using template folder: {template_folder}")
    
    # Create Flask application
    app = Flask(__name__, 
                static_folder=static_folder, 
                template_folder=template_folder)
    
    # Configure application
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key')
    app.config['MODELS_DIR'] = os.environ.get('MODELS_DIR', 'models')
    
    # Initialize model manager, trainer, and quantizer
    app.model_manager = ModelManager()
    app.trainer = ContinuedPretrainer()
    app.quantizer = ModelQuantizer()
    
    # Register blueprints
    app.register_blueprint(views_bp)
    app.register_blueprint(setup_api(app))
    
    # Create Socket.IO instance
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Set up Socket.IO
    setup_socketio(socketio, app)
    
    # Add Socket.IO to app
    app.socketio = socketio
    
    # Create directories if they don't exist
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)
    os.makedirs(os.path.join(app.config['MODELS_DIR'], 'cpt'), exist_ok=True)
    os.makedirs(os.path.join(app.config['MODELS_DIR'], 'ift'), exist_ok=True)
    os.makedirs(os.path.join(app.config['MODELS_DIR'], 'base'), exist_ok=True)
    os.makedirs(os.path.join(app.config['MODELS_DIR'], 'quantized'), exist_ok=True)
    
    # Add route to serve static files from original static directory
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory(static_folder, filename)
    
    # Add route to serve favicon.ico
    @app.route('/favicon.ico')
    def serve_favicon():
        return send_from_directory(static_folder, 'favicon.ico')
    
    return app 