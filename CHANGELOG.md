# Changelog

All notable changes to ForgetLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-03

### 🎨 Complete Dark Theme Implementation
- **Professional Dark Theme Toggle**: Beautiful moon/sun icon button in navigation bar
  - Persistent theme selection using localStorage
  - Smooth 0.3s transitions for all color changes
  - Comprehensive CSS variable system for theming
  - WCAG-compliant contrast ratios for accessibility

- **Comprehensive UI Theming**: Every interface element properly themed
  - Navigation bar with gradient backgrounds adapted for both themes
  - Card components with proper surface colors and shadows
  - Form controls (inputs, selects, textareas) with themed backgrounds
  - Chat interface with distinct user/assistant message styling
  - Code blocks and markdown content with syntax highlighting
  - Progress bars, badges, and status indicators

### ⚡ Enhanced User Experience
- **Quantization Functionality Fixes**: Complete overhaul of model quantization
  - Fixed page refresh issues caused by form submission
  - Harmonized model selection dropdown with emoji icons (⚡ base, 🤖 instruct)
  - Proper event handling with Bootstrap tab integration
  - Fixed API parameter format issues
  - Enhanced local model path detection and HuggingFace cache handling
  - Graceful error handling for model card creation failures

- **Generation Interface Improvements**: 
  - **Tokens per second display**: Real-time performance metrics
  - **Prompt/completion breakdown**: Token usage statistics for all modes
  - **Fullscreen mode**: Dedicated fullscreen view for Generation Output panel
  - **Context window stats**: Immediate updates when changing context size
  - **Improved chat scrolling**: Fixed fullscreen scrolling functionality

### 🛠️ Technical Improvements
- **Model Path Generation**: Robust handling of HuggingFace cache models
  - Fixed double "models--" prefix issues in quantized model paths
  - Proper namespace separation for published models
  - Enhanced folder opening API integration
  - Improved model detection for Gemma models

- **API Response Format**: Enhanced model manager to return full response objects
  - Token calculation and speed metrics
  - Consistent response format across streaming and non-streaming modes
  - Better error handling and status reporting

- **UI Layout Enhancements**:
  - Card-based design for quantized models with proper text wrapping
  - Responsive Bootstrap grid layouts
  - Badge system for metadata display
  - Visual hierarchy improvements with icons and spacing

### 🎯 Model Management
- **Quantization Workflow**: End-to-end quantization process improvements
  - Model filtering to exclude already-quantized models
  - Consistent emoji icons across all model dropdowns
  - Real-time progress tracking and status updates
  - Custom metadata file generation for quantized models

- **Model Detection**: Enhanced model type detection
  - Proper Gemma model identification for system prompts
  - Improved base vs instruct model categorization
  - Better handling of published model variants

### 🐛 Critical Bug Fixes
- **Import Error Resolution**: Fixed startup-breaking import issues
  - Removed non-existent function imports
  - Proper logging configuration
- **Event Handler Conflicts**: Resolved duplicate event listeners
- **API Parameter Mismatches**: Fixed quantization API call format
- **Fullscreen Target Issues**: Corrected fullscreen panel targeting
- **Sorting Errors**: Fixed NoneType comparison in model sorting
- **Theme Text Visibility**: Comprehensive text color fixes for dark mode

### 🎨 Visual Polish
- **Form Element Consistency**: All input elements properly themed
  - Training tab: Fixed "Layers to Fine-tune" input styling
  - Testing tab: Fixed white rectangle issues in parameter controls
  - Quantization tab: Proper background and text colors
- **Training Estimates**: Maintained readability with theme-appropriate colors
- **Chat Input**: Lighter placeholder text for better UX in dark mode
- **Fullscreen Adjustments**: Improved spacing and layout in fullscreen mode

---

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