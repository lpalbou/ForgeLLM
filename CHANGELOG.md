# Changelog

All notable changes to ForgetLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-06-30

### 🎨 Enhanced User Interface
- **Beautiful Training Log Viewer**: Complete redesign with syntax-highlighted JSON display
  - Color-coded JSON formatting (keys, strings, numbers, booleans)
  - Dark theme with professional styling
  - Metadata header showing training configuration at a glance
  - Enhanced copy-to-clipboard functionality preserving original formatting
  - Modal-based viewing without disruptive tab switching

- **Improved Training Sessions Layout**: Professional card-based design
  - Fixed model name extraction showing actual model names instead of "Unknown"
  - Clean model name display (removed vendor prefixes like `mlx-community/`)
  - Better spacing and responsive Bootstrap layout
  - Enhanced visual hierarchy with icons and proper headers

### ⚡ Performance & Monitoring Optimizations
- **Consolidated Update System**: Reduced API calls from 4-5 per cycle to single call
  - Single `/api/dashboard/realtime` endpoint every 10 seconds
  - Eliminated redundant polling mechanisms
  - Smart throttling preventing excessive updates
  - Cache busting for JavaScript files

- **Enhanced Real-time Monitoring**: 
  - Robust MLX process detection for accurate training status
  - Optimized idle monitoring (10-second sleep when no training active)
  - Improved data consistency across all API endpoints
  - Better handling of optional fields in training metrics

### 🛡️ System Reliability
- **Graceful Shutdown System**: Complete process lifecycle management
  - New `ProcessTracker` utility tracking all spawned processes
  - SIGINT/SIGTERM signal handlers for clean termination
  - 5-second graceful shutdown with force-kill fallback
  - Automatic MLX process cleanup on exit
  - Integration across all entry points (web server, training, CLI)

- **Enhanced Watchdog System**: Extended timeout from 30 to 90 seconds for heavy LLM training
- **Improved Error Handling**: Better fallbacks and error messages throughout the system

### 🎯 Smart Model Management
- **Unified Model Filtering**: Consistent model lists across training and testing tabs
  - Proper separation of base, CPT, and IFT models
  - Published CPT model detection from HuggingFace cache
  - Excluded unpublished training checkpoints from UI dropdowns
  - 81 models (base + IFT) available in both training and testing tabs

### 🔧 API & Backend Improvements
- **Enhanced Training Sessions API**: Better data extraction and formatting
  - Fixed model name resolution from multiple data sources
  - Improved checkpoint path parsing for numbered checkpoints
  - Better handling of training metadata and configuration
  - Comprehensive error handling and logging

- **Optimized Dashboard Data**: 
  - Precise epoch calculation based on trained tokens vs dataset size
  - Enhanced numeric formatting with proper precision control
  - Better handling of small numbers (learning rates, etc.)
  - Improved time calculations for elapsed and remaining time

### 🐛 Bug Fixes
- Fixed training log formatting losing line breaks and JSON structure
- Resolved excessive API polling causing performance issues
- Fixed model name display showing "N/A" instead of actual model names
- Corrected training session card layout overlapping issues
- Fixed config value fallbacks showing "-" instead of actual values
- Resolved import errors in training metrics logging

### 📋 Technical Improvements
- Updated all version references consistently across the codebase
- Enhanced logging with emoji indicators for better debugging
- Improved code organization and separation of concerns
- Better documentation and inline comments
- Comprehensive test coverage for new features

---

## [0.1.1] - Previous Release
- Initial stable release with core training and monitoring functionality
- Basic web interface for model training and testing
- MLX-LM integration for Apple Silicon optimization
- Continued pre-training (CPT) and instruction fine-tuning (IFT) support 