#!/usr/bin/env python3
"""
Process Tracker - Manages all spawned processes for graceful shutdown
"""

import os
import signal
import subprocess
import threading
import time
import logging
import psutil
from typing import List, Set, Optional
import atexit

logger = logging.getLogger(__name__)


class ProcessTracker:
    """
    Tracks all spawned processes and provides graceful shutdown functionality.
    Singleton pattern to ensure all processes are tracked globally.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProcessTracker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.tracked_processes: Set[int] = set()
        self.subprocess_objects: List[subprocess.Popen] = []
        self.shutdown_handlers: List[callable] = []
        self._shutdown_in_progress = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register atexit handler
        atexit.register(self.cleanup_all_processes)
        
        logger.info("ProcessTracker initialized")
    
    def track_process(self, process: subprocess.Popen) -> None:
        """Track a subprocess for cleanup on exit."""
        if process and process.pid:
            self.tracked_processes.add(process.pid)
            self.subprocess_objects.append(process)
            logger.debug(f"Tracking process PID {process.pid}")
    
    def track_pid(self, pid: int) -> None:
        """Track a process by PID for cleanup on exit."""
        self.tracked_processes.add(pid)
        logger.debug(f"Tracking PID {pid}")
    
    def untrack_process(self, process: subprocess.Popen) -> None:
        """Stop tracking a process (when it exits normally)."""
        if process and process.pid:
            self.tracked_processes.discard(process.pid)
            if process in self.subprocess_objects:
                self.subprocess_objects.remove(process)
            logger.debug(f"Stopped tracking process PID {process.pid}")
    
    def untrack_pid(self, pid: int) -> None:
        """Stop tracking a PID (when it exits normally)."""
        self.tracked_processes.discard(pid)
        logger.debug(f"Stopped tracking PID {pid}")
    
    def add_shutdown_handler(self, handler: callable) -> None:
        """Add a custom shutdown handler to be called during cleanup."""
        self.shutdown_handlers.append(handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals (SIGINT, SIGTERM)."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.cleanup_all_processes()
        exit(0)
    
    def cleanup_all_processes(self) -> None:
        """Clean up all tracked processes gracefully."""
        if self._shutdown_in_progress:
            return
            
        self._shutdown_in_progress = True
        logger.info("🧹 Starting graceful cleanup of all processes...")
        
        # Call custom shutdown handlers first
        for handler in self.shutdown_handlers:
            try:
                logger.debug(f"Calling shutdown handler: {handler.__name__}")
                handler()
            except Exception as e:
                logger.warning(f"Error in shutdown handler {handler.__name__}: {e}")
        
        # Clean up subprocess objects
        for process in self.subprocess_objects[:]:
            try:
                if process.poll() is None:  # Process is still running
                    logger.info(f"Terminating subprocess PID {process.pid}")
                    process.terminate()
                    
                    # Wait for graceful termination
                    try:
                        process.wait(timeout=5)
                        logger.debug(f"Process PID {process.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process PID {process.pid} did not terminate, killing...")
                        process.kill()
                        process.wait(timeout=2)
                        logger.debug(f"Process PID {process.pid} killed")
            except Exception as e:
                logger.warning(f"Error cleaning up subprocess PID {process.pid}: {e}")
        
        # Clean up tracked PIDs using psutil
        for pid in self.tracked_processes.copy():
            try:
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    logger.info(f"Terminating tracked process PID {pid} ({proc.name()})")
                    
                    # Try graceful termination first
                    proc.terminate()
                    
                    # Wait for termination
                    try:
                        proc.wait(timeout=5)
                        logger.debug(f"Process PID {pid} terminated gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process PID {pid} did not terminate, killing...")
                        proc.kill()
                        proc.wait(timeout=2)
                        logger.debug(f"Process PID {pid} killed")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process already dead or inaccessible
                pass
            except Exception as e:
                logger.warning(f"Error cleaning up PID {pid}: {e}")
        
        # Clean up any remaining MLX processes
        self._cleanup_mlx_processes()
        
        logger.info("✅ Process cleanup completed")
    
    def _cleanup_mlx_processes(self) -> None:
        """Clean up any remaining MLX training processes."""
        try:
            mlx_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        # Look for MLX training processes
                        if any(pattern in cmdline for pattern in [
                            'mlx_lm.lora',      # MLX LoRA training
                            'mlx_lm.fuse',      # MLX model fusion
                            'mlx-lm',           # MLX-LM package calls
                            'mlx_lm',           # MLX-LM package calls
                        ]):
                            mlx_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if mlx_processes:
                logger.info(f"Found {len(mlx_processes)} MLX processes to clean up")
                for proc in mlx_processes:
                    try:
                        logger.info(f"Terminating MLX process PID {proc.pid}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        logger.debug(f"MLX process PID {proc.pid} terminated")
                    except psutil.TimeoutExpired:
                        logger.warning(f"MLX process PID {proc.pid} did not terminate, killing...")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                    except Exception as e:
                        logger.warning(f"Error cleaning up MLX process PID {proc.pid}: {e}")
        except Exception as e:
            logger.warning(f"Error during MLX process cleanup: {e}")
    
    def get_tracked_processes(self) -> List[dict]:
        """Get information about all tracked processes."""
        processes = []
        for pid in self.tracked_processes:
            try:
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    processes.append({
                        'pid': pid,
                        'name': proc.name(),
                        'status': proc.status(),
                        'cmdline': ' '.join(proc.cmdline()) if proc.cmdline() else ''
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes


# Global instance
process_tracker = ProcessTracker() 