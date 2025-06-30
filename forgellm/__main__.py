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
import subprocess
import time
import signal
import os
from pathlib import Path

# Import the process tracker for graceful shutdown
from .utils.process_tracker import process_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        process_tracker.cleanup_all_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for the ForgeLLM package."""
    parser = argparse.ArgumentParser(
        prog='forgellm',
        description='ForgeLLM - Comprehensive toolkit for language model training and deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  forgellm start                       # Start both model server and web interface
  forgellm cli generate --model <model> # Interactive chat with a model
  forgellm server --port 5001          # Start model server only
  forgellm web --port 5002             # Start web interface only
  
  python -m forgellm start             # Alternative invocation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command - launch both servers
    start_parser = subparsers.add_parser('start', help='Start both model server and web interface')
    start_parser.add_argument('--server-port', type=int, default=5001, help='Model server port')
    start_parser.add_argument('--web-port', type=int, default=5002, help='Web interface port')
    start_parser.add_argument('--host', default='localhost', help='Host to bind to')
    
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
    
    # Special handling for CLI command to preserve all arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        from .cli.main import main as cli_main
        # Forward everything after 'cli' to the CLI
        cli_args = sys.argv[2:] if len(sys.argv) > 2 else ['--help']
        sys.argv = ['forgellm-cli'] + cli_args
        return cli_main()
    
    args, unknown_args = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'start':
            return start_both_servers(args)
                
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


def start_both_servers(args):
    """Start both model server and web interface."""
    processes = []
    
    setup_signal_handlers()
    
    try:
        logger.info(f"Starting ForgeLLM servers...")
        logger.info(f"Model Server: http://{args.host}:{args.server_port}")
        logger.info(f"Web Interface: http://{args.host}:{args.web_port}")
        
        # Start model server
        server_cmd = [
            sys.executable, '-m', 'forgellm', 'server',
            '--host', args.host,
            '--port', str(args.server_port)
        ]
        
        logger.info("Starting model server...")
        server_proc = subprocess.Popen(server_cmd)
        processes.append(server_proc)
        process_tracker.track_process(server_proc)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Start web interface
        web_cmd = [
            sys.executable, '-m', 'forgellm', 'web',
            '--host', args.host,
            '--port', str(args.web_port)
        ]
        
        logger.info("Starting web interface...")
        web_proc = subprocess.Popen(web_cmd)
        processes.append(web_proc)
        process_tracker.track_process(web_proc)
        
        # Wait a moment for web to start
        time.sleep(2)
        
        logger.info("✅ Both servers started successfully!")
        logger.info(f"🌐 Open your browser to: http://{args.host}:{args.web_port}")
        logger.info("📋 Press Ctrl+C to stop both servers")
        
        # Wait for processes
        try:
            while True:
                # Check if any process has died
                for proc in processes[:]:
                    if proc.poll() is not None:
                        logger.error(f"Process {proc.pid} has died")
                        # Kill all processes and exit
                        for p in processes:
                            try:
                                p.terminate()
                            except:
                                pass
                        return 1
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        logger.error(f"Error starting servers: {e}")
        return 1
    
    finally:
        # Use process tracker for cleanup
        process_tracker.cleanup_all_processes()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 