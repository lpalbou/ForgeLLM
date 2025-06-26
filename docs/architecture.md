# Web Interface Architecture

## Overview

The web interface provides a user-friendly way to manage continued pre-training and testing of MLX models. It consists of:

1. **Flask Backend**: Handles API requests and serves the web interface
2. **Socket.IO**: Provides real-time updates for training progress
3. **Training Manager**: Manages training processes and monitors their progress

## Components

### TrainingManager

The `TrainingManager` class is the core component responsible for:

- Starting and stopping training processes
- Monitoring training progress
- Detecting active training processes
- Parsing training logs

### Key Files

- `web_training_interface.py`: Main web interface implementation
- `start_web_interface.py`: Entry point for starting the web server
- `continued_pretraining.py`: Handles the actual training process
- `training_metrics_logger.py`: Logs training metrics to JSON files

## Critical Issue: Automatic Training Restart

When the web interface is started with `python start_web_interface.py`, it can automatically start a new training process without user interaction. This happens through the following sequence:

1. **Initialization**: `TrainingManager.__init__()` calls `_detect_active_training()`
2. **Process Detection**: `_detect_active_training()` checks for running "mlx_lm" processes
3. **Alternative Detection**: If no process is found, it falls back to `_detect_training_alternative()`
4. **Log File Check**: `_detect_training_alternative()` looks for recent log files in `models/cpt`
5. **Directory Recreation**: When evidence of a previous session is found, it recreates the directory structure
6. **Config File Recreation**: The web interface recreates the previous training configuration file
7. **Training Restart**: A new training process is started using the recreated configuration

### Root Cause

The issue appears to be in the interaction between the `_detect_active_training()`, `_detect_training_alternative()`, and `_find_latest_log_file()` methods. When the web interface starts, it's actively looking for evidence of previous training sessions and trying to resume them, even if the directories were deleted and processes were killed.

The source of the configuration file recreation is not immediately obvious in the codebase, but it's likely that the web interface is storing information about previous training sessions in memory or in a cache file that persists between restarts.

### Impact

This behavior is problematic because:

1. It happens silently without user confirmation
2. It can consume significant system resources unexpectedly
3. It persists even after deleting training directories and killing all Python processes
4. It can lead to confusion about the source of training processes

## Recommended Fixes

1. **Add User Confirmation**: Modify `_detect_active_training()` to ask for user confirmation before resuming training
2. **Disable Auto-Detection**: Add a command-line flag to disable automatic training detection
3. **Clear Cache**: Ensure any cached training information is cleared when directories are deleted
4. **Improve Logging**: Add more detailed logging about training detection and resumption
5. **Fix Directory Recreation**: Prevent automatic recreation of deleted directories

## Data Flow

1. User starts web interface (`start_web_interface.py`)
2. Flask app initializes `TrainingManager` (`web_training_interface.py`)
3. `TrainingManager` checks for active training (`_detect_active_training()`)
4. If found, training is monitored (`start_monitoring()`)
5. User interacts with web interface to start/stop training
6. Training process runs (`continued_pretraining.py`)
7. Training metrics are logged (`training_metrics_logger.py`)
8. Socket.IO sends real-time updates to the web interface 