#!/usr/bin/env python3
"""
ForgeLLM - Web Interface
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the web interface"""
    try:
        # Ensure we're running from the correct directory (where this script is located)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.getcwd() != script_dir:
            logger.info(f"Changing working directory from {os.getcwd()} to {script_dir}")
            os.chdir(script_dir)
        
        parser = argparse.ArgumentParser(description="ForgeLLM Web Interface")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
        parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--static-folder", type=str, help="Path to static folder")
        parser.add_argument("--template-folder", type=str, help="Path to template folder")
        
        args = parser.parse_args()
        
        # Use the static/template folders from the arguments or let the app use its defaults
        static_folder = args.static_folder
        template_folder = args.template_folder
        
        # Import the run_web_interface function
        from forgellm.web.run import run_web_interface
        
        # Run the web interface
        run_web_interface(
            host=args.host,
            port=args.port,
            debug=args.debug,
            static_folder=static_folder,
            template_folder=template_folder
        )
        
    except KeyboardInterrupt:
        logger.info("Web interface stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 