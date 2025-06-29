#!/usr/bin/env python3
"""
ForgeLLM - Main Entry Point

This module provides the main entry point for the ForgeLLM package when
invoked with `python -m forgellm`. It provides a unified command-line
interface with subcommands for different functionality.
"""

import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the ForgeLLM package."""
    parser = argparse.ArgumentParser(
        prog='forgellm',
        description='ForgeLLM - Comprehensive toolkit for language model training and deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  forgellm cli --help                  # Show CLI help
  forgellm server --port 5001          # Start model server
  forgellm web --port 5002             # Start web interface
  
  python -m forgellm cli --help        # Alternative invocation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CLI subcommand - allow unknown args to be forwarded
    cli_parser = subparsers.add_parser('cli', help='Command-line interface for model operations')
    cli_parser.add_argument('cli_args', nargs='*', help='Arguments to forward to the CLI')
    
    # Server subcommand  
    server_parser = subparsers.add_parser('server', help='Start the model server')
    server_parser.add_argument('--host', default='localhost', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    server_parser.add_argument('--model', help='Model to preload')
    server_parser.add_argument('--adapter', help='Adapter to preload')
    
    # Web subcommand
    web_parser = subparsers.add_parser('web', help='Start the web interface')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=5002, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args, unknown_args = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'cli':
            from .cli.main import main as cli_main
            # Forward the CLI arguments (both parsed and unknown)
            all_cli_args = args.cli_args + unknown_args
            sys.argv = ['forgellm-cli'] + all_cli_args
            return cli_main()
                
        elif args.command == 'server':
            from .server.main import main as server_main
            # Reconstruct argv for the server
            server_args = ['forgellm-server']
            if args.host != 'localhost':
                server_args.extend(['--host', args.host])
            if args.port != 5001:
                server_args.extend(['--port', str(args.port)])
            if args.model:
                server_args.extend(['--model', args.model])
            if args.adapter:
                server_args.extend(['--adapter', args.adapter])
            
            sys.argv = server_args
            return server_main()
            
        elif args.command == 'web':
            from .web.main import main as web_main
            # Reconstruct argv for the web interface
            web_args = ['forgellm-web']
            if args.host != '0.0.0.0':
                web_args.extend(['--host', args.host])
            if args.port != 5002:
                web_args.extend(['--port', str(args.port)])
            if args.debug:
                web_args.append('--debug')
            
            sys.argv = web_args
            return web_main()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 