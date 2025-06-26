"""
Run the web interface.
"""

import os
import sys
import logging
from .app import create_app

logger = logging.getLogger(__name__)

def run_web_interface(host='0.0.0.0', port=5001, debug=False, static_folder=None, template_folder=None):
    """Run the web interface.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Whether to run in debug mode
        static_folder: Path to static folder
        template_folder: Path to template folder
    """
    # Create app
    app = create_app(static_folder=static_folder, template_folder=template_folder)
    
    # Run app
    logger.info(f"Starting web interface on {host}:{port}")
    app.socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

def main():
    """Main entry point for the web interface."""
    run_web_interface()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 