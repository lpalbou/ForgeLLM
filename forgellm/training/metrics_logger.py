#!/usr/bin/env python3
"""
Training Metrics Logger Utility
===============================

Comprehensive utility for logging training metrics during MLX-LM training sessions.
Captures training loss, validation loss, learning rate, timing, and other metrics
in structured JSON format for analysis and visualization.

Features:
- Real-time parsing of MLX-LM training output
- Structured JSON logging with timestamps
- Support for both CPT and IFT training types
- Automatic checkpoint detection and logging
- Training session metadata capture
- Easy integration with existing training scripts
"""

import json
import re
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue
from dataclasses import dataclass, asdict
import threading


@dataclass
class TrainingMetrics:
    """Single training iteration metrics"""
    iteration: int
    timestamp: str
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_perplexity: Optional[float] = None
    val_perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    iterations_per_sec: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    trained_tokens: Optional[int] = None
    peak_memory_gb: Optional[float] = None
    val_time_sec: Optional[float] = None
    checkpoint_saved: bool = False
    checkpoint_path: Optional[str] = None


@dataclass
class TrainingSession:
    """Complete training session metadata and metrics"""
    session_id: str
    training_type: str  # "CPT" or "IFT"
    model_name: str
    start_time: str
    end_time: Optional[str] = None
    base_model: Optional[str] = None  # Base model used for training
    output_path: Optional[str] = None  # Where training outputs are saved
    training_command: Optional[str] = None  # Full command used for training
    config: Optional[Dict[str, Any]] = None
    metrics: List[TrainingMetrics] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []


class TrainingMetricsLogger:
    """Real-time training metrics logger for MLX-LM training"""
    
    def __init__(self, 
                 training_type: str,
                 model_name: str,
                 output_dir: str = "training_logs",
                 config: Optional[Dict[str, Any]] = None,
                 base_model: Optional[str] = None,
                 output_path: Optional[str] = None,
                 training_command: Optional[str] = None):
        """
        Initialize training metrics logger
        
        Args:
            training_type: "CPT" or "IFT"
            model_name: Name of the model being trained
            output_dir: Directory to save training logs
            config: Training configuration dictionary
            base_model: Base model used for training
            output_path: Where training outputs are saved
            training_command: Full command used for training
        """
        self.training_type = training_type
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{training_type}_{timestamp}"
        
        self.session = TrainingSession(
            session_id=self.session_id,
            training_type=training_type,
            model_name=model_name,
            start_time=datetime.now().isoformat(),
            base_model=base_model,
            output_path=output_path,
            training_command=training_command,
            config=config
        )
        
        # Output file
        self.log_file = self.output_dir / f"{self.session_id}.json"
        
        # Open log file once and keep it open for efficient writing
        self.log_file_handle = open(self.log_file, 'w', encoding='utf-8')
        
        # Regex patterns for parsing MLX-LM output
        self.patterns = {
            # Training loss line: "Iter 100: Train loss 1.234, Learning Rate 5.000e-06, It/sec 2.5, Tokens/sec 250.0, Trained Tokens 25000, Peak mem 10.5 GB"
            'train': re.compile(
                r'Iter (\d+): Train loss ([\d.]+)(?:, Learning Rate ([\d.e-]+))?(?:, It/sec ([\d.]+))?(?:, Tokens/sec ([\d.]+))?(?:, Trained Tokens ([\d.]+))?(?:, Peak mem ([\d.]+) GB)?'
            ),
            
            # Validation loss line: "Iter 100: Val loss 1.234, Val took 5.67s"
            'val': re.compile(
                r'Iter (\d+): Val loss ([\d.]+)(?:, Val took ([\d.]+)s)?'
            ),
            
            # Checkpoint save patterns - enhanced to catch more variations
            'checkpoint': re.compile(
                r'Iter (\d+): Saved adapter weights to (.+)'
            ),
            'checkpoint_alt': re.compile(
                r'Iter (\d+): saved weights to (.+)'
            ),
            'checkpoint_alt2': re.compile(
                r'Iter (\d+): Saved to (.+)'
            ),
            'checkpoint_alt3': re.compile(
                r'Iter (\d+): Saving adapter weights to (.+)'
            ),
            'checkpoint_alt4': re.compile(
                r'Iter (\d+): Checkpoint saved to (.+)'
            ),
            # Generic checkpoint pattern
            'checkpoint_generic': re.compile(
                r'Iter (\d+):.+(?:saved|Saved|SAVED).+(?:adapter|checkpoint|weights).+(?:to|at)\s+(.+)'
            ),
            
            # Alternative training format: "Iter 100: training loss 1.234, iterations/sec 2.5, Tokens/sec 250.0"
            'train_alt': re.compile(
                r'Iter (\d+): training loss ([\d.]+)(?:, iterations/sec ([\d.]+))?(?:, Tokens/sec ([\d.]+))?'
            ),
            
            # Alternative validation format: "Iter 100: validation loss 1.234, validation time 5.67s"
            'val_alt': re.compile(
                r'Iter (\d+): validation loss ([\d.]+)(?:, validation time ([\d.]+)s)?'
            ),
            
            # MLX-Swift format: "Iteration 100: training loss 1.234, iterations/sec 2.5, Tokens/sec 250.0"
            'train_swift': re.compile(
                r'Iteration (\d+): training loss ([\d.]+)(?:, iterations/sec ([\d.]+))?(?:, Tokens/sec ([\d.]+))?'
            ),
            
            # MLX-Swift validation: "Iteration 100: validation loss 1.234, validation time 5.67s"
            'val_swift': re.compile(
                r'Iteration (\d+): validation loss ([\d.]+)(?:, validation time ([\d.]+)s)?'
            )
        }
        
        self.logger = logging.getLogger(f"TrainingMetrics_{self.session_id}")
        self.logger.info(f"🔄 Training metrics logger initialized: {self.session_id}")
        
        # Save initial session
        self._save_session()
    
    @staticmethod
    def calculate_perplexity(loss: float) -> float:
        """Calculate perplexity from loss: perplexity = exp(loss)"""
        import math
        try:
            return math.exp(loss)
        except OverflowError:
            # Handle very large losses
            return float('inf')
    
    def parse_and_log_line(self, line: str) -> Optional[TrainingMetrics]:
        """
        Parse a single line of MLX-LM output and extract metrics
        
        Args:
            line: Raw output line from MLX-LM training
            
        Returns:
            TrainingMetrics object if metrics were found, None otherwise
        """
        # Add diagnostics for processing time
        start_time = time.time()
        
        line = line.strip()
        if not line:
            return None
        
        # Try to match training loss patterns
        for pattern_name in ['train', 'train_alt', 'train_swift']:
            match = self.patterns[pattern_name].match(line)
            if match:
                iteration = int(match.group(1))
                train_loss = float(match.group(2))
                
                # Find or create metrics for this iteration
                metrics = self._get_or_create_metrics(iteration)
                metrics.train_loss = train_loss
                metrics.train_perplexity = self.calculate_perplexity(train_loss)
                
                # Extract additional training metrics if available
                if pattern_name == 'train' and len(match.groups()) >= 7:
                    if match.group(3):  # Learning rate
                        metrics.learning_rate = float(match.group(3))
                    if match.group(4):  # It/sec
                        metrics.iterations_per_sec = float(match.group(4))
                    if match.group(5):  # Tokens/sec
                        metrics.tokens_per_sec = float(match.group(5))
                    if match.group(6):  # Trained tokens
                        metrics.trained_tokens = int(float(match.group(6)))
                    if match.group(7):  # Peak memory
                        metrics.peak_memory_gb = float(match.group(7))
                elif pattern_name in ['train_alt', 'train_swift'] and len(match.groups()) >= 4:
                    if match.group(3):  # iterations/sec
                        metrics.iterations_per_sec = float(match.group(3))
                    if match.group(4):  # Tokens/sec
                        metrics.tokens_per_sec = float(match.group(4))
                
                self._save_session()
                
                # Log processing time if it's unusually long
                processing_time = time.time() - start_time
                if processing_time > 0.1:  # Log if processing takes >100ms
                    self.logger.warning(f"Line processing took {processing_time:.3f}s: {line[:50]}...")
                
                return metrics
        
        # Try to match validation loss patterns
        for pattern_name in ['val', 'val_alt', 'val_swift']:
            match = self.patterns[pattern_name].match(line)
            if match:
                iteration = int(match.group(1))
                val_loss = float(match.group(2))
                
                # Find or create metrics for this iteration
                metrics = self._get_or_create_metrics(iteration)
                metrics.val_loss = val_loss
                metrics.val_perplexity = self.calculate_perplexity(val_loss)
                
                # Extract validation time if available
                if len(match.groups()) >= 3 and match.group(3):
                    metrics.val_time_sec = float(match.group(3))
                
                self._save_session()
                return metrics
        
        # Try to match checkpoint save patterns
        for pattern_name in ['checkpoint', 'checkpoint_alt', 'checkpoint_alt2', 'checkpoint_alt3', 'checkpoint_alt4', 'checkpoint_generic']:
            match = self.patterns[pattern_name].match(line)
            if match:
                iteration = int(match.group(1))
                checkpoint_path = match.group(2).strip()
                
                # Find or create metrics for this iteration
                metrics = self._get_or_create_metrics(iteration)
                metrics.checkpoint_saved = True
                metrics.checkpoint_path = checkpoint_path
                
                self._save_session()
                return metrics
        
        # No match found
        return None
    
    def _get_or_create_metrics(self, iteration: int) -> TrainingMetrics:
        """
        Get existing metrics for an iteration or create new ones
        
        Args:
            iteration: Training iteration
            
        Returns:
            TrainingMetrics object
        """
        # Check if metrics for this iteration already exist
        for metrics in self.session.metrics:
            if metrics.iteration == iteration:
                return metrics
        
        # Create new metrics
        metrics = TrainingMetrics(
            iteration=iteration,
            timestamp=datetime.now().isoformat()
        )
        self.session.metrics.append(metrics)
        
        return metrics
    
    def _save_session(self):
        """Save the current session to the log file"""
        try:
            # Convert session to dict
            session_dict = asdict(self.session)
            
            # Seek to beginning of file
            self.log_file_handle.seek(0)
            self.log_file_handle.truncate()
            
            # Write JSON
            json.dump(session_dict, self.log_file_handle, indent=2)
            self.log_file_handle.flush()
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
    
    def parse_stream_safely(self, stream, callback=None) -> Tuple[threading.Thread, Queue, threading.Event]:
        """
        Parse a stream of MLX-LM output safely in a separate thread
        
        Args:
            stream: Stream to parse
            callback: Optional callback function to call with each line
            
        Returns:
            Tuple of (thread, queue, stop_event)
        """
        import queue
        
        # Create a queue to hold output
        output_queue = queue.Queue()
        
        # Create an event to signal the thread to stop
        stop_event = threading.Event()
        
        # Create a watchdog event to detect if the reader thread is stuck
        watchdog_event = threading.Event()
        
        # Create a thread to read from the stream
        def watchdog_thread():
            """Watchdog thread to detect if the reader thread is stuck"""
            while not stop_event.is_set():
                # Wait for the watchdog event with timeout
                if not watchdog_event.wait(timeout=30):
                    # Watchdog event not set within timeout, reader might be stuck
                    self.logger.warning("Watchdog: Reader thread might be stuck, no output for 30 seconds")
                    
                    # Put a warning message in the queue
                    output_queue.put("Warning: No output for 30 seconds, reader might be stuck")
                
                # Reset the watchdog event
                watchdog_event.clear()
                
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)
        
        # Create a thread to read from the stream
        def reader_thread():
            """Reader thread to read from the stream"""
            try:
                # Set the watchdog event initially
                watchdog_event.set()
                
                # Read from the stream
                for raw_line in iter(stream.readline, ''):
                    if stop_event.is_set():
                        break
                    
                    # Set the watchdog event to indicate we're still reading
                    watchdog_event.set()
                    
                    # Process the line
                    stripped_line = raw_line.strip()
                    if not stripped_line:
                        continue
                    
                    # Parse the line
                    metrics = self.parse_and_log_line(stripped_line)
                    
                    # Call the callback if provided with both raw and stripped line
                    if callback:
                        callback(raw_line, stripped_line)
                    
                    # Put the stripped line in the queue
                    output_queue.put(stripped_line)
                    
                    # Sleep for a short time to avoid busy waiting
                    time.sleep(0.001)
                
                # Signal that we're done reading
                output_queue.put(None)
                
            except Exception as e:
                # Put the error in the queue
                output_queue.put(f"Error in reader thread: {e}")
                self.logger.error(f"Error in reader thread: {e}")
            
            finally:
                # Set the stop event to signal other threads to stop
                stop_event.set()
        
        # Create a thread to monitor the queue
        def monitor_thread():
            """Monitor thread to process the queue"""
            try:
                while not stop_event.is_set():
                    try:
                        # Get an item from the queue with timeout
                        item = output_queue.get(timeout=1)
                        
                        # Process the item
                        if item is None:
                            # End of stream
                            break
                        
                        # Log the item
                        self.logger.debug(f"Queue item: {item}")
                        
                    except queue.Empty:
                        # Queue is empty, continue waiting
                        continue
                    
                    except Exception as e:
                        # Log the error
                        self.logger.error(f"Error processing queue item: {e}")
            
            except Exception as e:
                # Log the error
                self.logger.error(f"Error in monitor thread: {e}")
            
            finally:
                # Set the stop event to signal other threads to stop
                stop_event.set()
        
        # Start the threads
        reader = threading.Thread(target=reader_thread, daemon=True)
        reader.start()
        
        # Start the watchdog thread
        watchdog = threading.Thread(target=watchdog_thread, daemon=True)
        watchdog.start()
        
        # Start the monitor thread
        monitor = threading.Thread(target=monitor_thread, daemon=True)
        monitor.start()
        
        # Return the reader thread, output queue, and stop event
        return reader, output_queue, stop_event
    
    def finalize_session(self):
        """Finalize the training session"""
        # Set end time
        self.session.end_time = datetime.now().isoformat()
        
        # Backfill missing training metrics
        self._backfill_missing_training_metrics()
        
        # Save the session
        self._save_session()
        
        # Close the log file
        try:
            self.log_file_handle.close()
        except Exception as e:
            self.logger.error(f"Error closing log file: {e}")
    
    def _backfill_missing_training_metrics(self):
        """Backfill missing training metrics by interpolating between known values"""
        if not self.session.metrics:
            return
        
        # Sort metrics by iteration
        self.session.metrics.sort(key=lambda x: x.iteration)
        
        # Identify metrics with missing values
        metrics_with_missing_values = []
        for i, metrics in enumerate(self.session.metrics):
            if metrics.train_loss is None or metrics.val_loss is None:
                metrics_with_missing_values.append((i, metrics))
        
        # Backfill missing values
        for i, metrics in metrics_with_missing_values:
            # Find previous and next metrics with values
            prev_metrics = None
            next_metrics = None
            
            # Find previous metrics with values
            for j in range(i-1, -1, -1):
                if self.session.metrics[j].train_loss is not None:
                    prev_metrics = self.session.metrics[j]
                    break
            
            # Find next metrics with values
            for j in range(i+1, len(self.session.metrics)):
                if self.session.metrics[j].train_loss is not None:
                    next_metrics = self.session.metrics[j]
                    break
            
            # Backfill train_loss if missing
            if metrics.train_loss is None:
                if prev_metrics is not None and next_metrics is not None:
                    # Interpolate between previous and next
                    prev_iter = prev_metrics.iteration
                    next_iter = next_metrics.iteration
                    curr_iter = metrics.iteration
                    
                    # Linear interpolation
                    alpha = (curr_iter - prev_iter) / (next_iter - prev_iter)
                    metrics.train_loss = prev_metrics.train_loss + alpha * (next_metrics.train_loss - prev_metrics.train_loss)
                    metrics.train_perplexity = self.calculate_perplexity(metrics.train_loss)
                elif prev_metrics is not None:
                    # Use previous value
                    metrics.train_loss = prev_metrics.train_loss
                    metrics.train_perplexity = prev_metrics.train_perplexity
                elif next_metrics is not None:
                    # Use next value
                    metrics.train_loss = next_metrics.train_loss
                    metrics.train_perplexity = next_metrics.train_perplexity
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training session
        
        Returns:
            dict: Summary of the training session
        """
        # Count metrics
        training_points = sum(1 for m in self.session.metrics if m.train_loss is not None)
        validation_points = sum(1 for m in self.session.metrics if m.val_loss is not None)
        checkpoints_saved = sum(1 for m in self.session.metrics if m.checkpoint_saved)
        
        # Get best validation loss
        best_val_loss = float('inf')
        best_val_iter = None
        for m in self.session.metrics:
            if m.val_loss is not None and m.val_loss < best_val_loss:
                best_val_loss = m.val_loss
                best_val_iter = m.iteration
        
        return {
            "session_id": self.session_id,
            "training_type": self.training_type,
            "model_name": self.model_name,
            "start_time": self.session.start_time,
            "end_time": self.session.end_time,
            "training_points": training_points,
            "validation_points": validation_points,
            "checkpoints_saved": checkpoints_saved,
            "best_val_loss": best_val_loss,
            "best_val_iter": best_val_iter,
            "log_file": str(self.log_file)
        }


def create_training_logger(training_type: str, 
                         model_name: str, 
                         config: Optional[Dict[str, Any]] = None,
                         output_dir: str = "training_logs",
                         base_model: Optional[str] = None,
                         output_path: Optional[str] = None,
                         training_command: Optional[str] = None) -> TrainingMetricsLogger:
    """
    Create a training metrics logger
    
    Args:
        training_type: "CPT" or "IFT"
        model_name: Name of the model being trained
        config: Training configuration dictionary
        output_dir: Directory to save training logs
        base_model: Base model used for training
        output_path: Where training outputs are saved
        training_command: Full command used for training
        
    Returns:
        TrainingMetricsLogger: Training metrics logger
    """
    return TrainingMetricsLogger(
        training_type=training_type,
        model_name=model_name,
        output_dir=output_dir,
        config=config,
        base_model=base_model,
        output_path=output_path,
        training_command=training_command
    )