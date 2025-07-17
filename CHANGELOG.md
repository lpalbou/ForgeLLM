# Changelog

All notable changes to ForgetLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ✨ Enhanced Features

#### **Specific Checkpoint File Support**
- **Feature**: Enhanced adapter loading to support specific checkpoint files (e.g., `0000175_adapters.safetensors`)
  - **Problem**: MLX-LM requires `adapters.safetensors` in the adapter directory, but checkpoint files are named with iteration numbers
  - **Root Cause**: The exact MLX-LM command used is `from mlx_lm import load` with `load(model_name, adapter_path=directory)` - it expects the directory to contain `adapters.safetensors`
  - **Solution**: 
    - **Backend**: Automatically detects specific checkpoint files (pattern: `^\d+_adapters\.safetensors$`)
    - **Symlink Management**: Always removes existing `adapters.safetensors` and recreates it as a symlink to the selected checkpoint
    - **Fallback**: If symlink creation fails, falls back to copying the file
    - **Critical**: Ensures each checkpoint selection updates the symlink to point to the correct iteration
  - **User Experience**: Users can now directly select specific checkpoint iterations and each selection will load the correct checkpoint
  - **Error Handling**: Provides clear error messages if checkpoint file preparation fails

#### **🚨 Critical Bug: Adapter-Only Loading Issue**
- **Issue**: When selecting only adapters (without base model), the system may fall back to the last recorded checkpoint instead of the selected checkpoint
  - **Symptoms**: 
    - Model server fails to start when loading adapter-only configurations
    - Error: "Failed to start model server" during adapter loading
    - Potential fallback to previously used checkpoint instead of user selection
  - **Impact**: Users cannot reliably test specific checkpoints in adapter-only mode
  - **Root Cause**: Potential race condition or state management issue in adapter-only loading path
  - **Status**: **UNDER INVESTIGATION** - Fix in progress
  - **Workaround**: Always select both base model and adapter explicitly until fix is deployed
  - **Priority**: HIGH - Affects core checkpoint testing functionality

#### **Random Seed Support in Testing Tab**
- **Feature**: Added support for random seed generation in the testing tab
  - **Implementation**: Setting seed to `-1` now enables random seed generation (non-reproducible)
  - **UI Updates**: 
    - Updated seed input field to accept values from `-1` to `2147483647`
    - Enhanced tooltip to explain `-1` functionality: "Random seed for reproducible generation. Set to -1 for random seed (non-reproducible)"
    - Updated help text to mention random seed option
  - **Backend Integration**: When seed is set to `-1`, the seed parameter is completely omitted from the API request, enabling true random generation
  - **Chat History**: Random seed sessions are saved with `'random'` in metadata and properly restored as `-1` when loading history
  - **Technical Details**: The request body conditionally includes the seed parameter only when seed ≠ -1, ensuring no seed interference with random generation

## [0.4.5] - 2025-07-16

### 🔧 Critical Data Processing Fix

#### **JSONL Dataset Corruption Resolution**
- **Issue**: JSONL dataset generation was adding extra newlines that weren't in the original text, corrupting document structure
  - **Examples**: 
    - Original: `"### 4. The Importance of Embodied Experience"`
    - Corrupted: `"### 4\n\nThe Importance of Embodied Experience"`
  - **Impact**: Headers, numbered lists, and paragraph structure were being damaged during chunking
- **Root Cause**: The `chunk_text` method in `DocumentProcessor` was:
  1. Splitting text by `\n\n` to get paragraphs
  2. Splitting paragraphs by `. ` (sentences) 
  3. Joining chunks back with `\n\n`
  - This corrupted formatting by adding newlines that weren't in the original text
- **Solution**: Complete rewrite of chunking logic to preserve original text structure
  - **Paragraph-Level Processing**: Work at paragraph level only (split by `\n\n`)
  - **Structure Preservation**: Preserve original paragraph structure when rejoining
  - **Eliminated Sentence Splitting**: Removed sentence-level splitting that was adding extra newlines
  - **Filter Adjustment**: Reduced minimum chunk length from 50 to 10 characters to avoid filtering valid content
- **Verification**: 
  - **Perfect Reconstruction**: Chunked text, when rejoined, is identical to original content
  - **Header Preservation**: All headers preserved correctly without extra newlines
  - **Real File Testing**: Verified on actual dataset files with complex markdown structure
  - **No Corruption**: Zero instances of triple newlines or corrupted headers

### 🔧 Technical Implementation
- **Robust General-Purpose Logic**: Designed to work for all text inputs, not just test cases
- **Original Text Integrity**: Maintains exact original formatting and structure
- **Comprehensive Testing**: Verified against real dataset files with headers, lists, and complex formatting
- **Backward Compatibility**: Maintains existing chunking API while fixing corruption issues

### 🐛 Bug Fixes
- Fixed JSONL dataset generation adding extra newlines not present in original text
- Resolved header corruption during text chunking (e.g., "### 4\n\nTitle" → "### 4. Title")
- Eliminated paragraph structure damage in training data
- Fixed chunk filtering removing valid short content
- Prevented sentence-level splitting from corrupting document formatting

---

## [0.4.4] - 2025-07-16

### 🔧 Modal System Fixes

#### **Directory Item Count Display**
- **Issue**: File browser modal was showing "Empty" for all directories instead of actual item counts
- **Root Cause**: Frontend JavaScript was looking for `item.item_count` field, but backend API was returning `item.description` with count information
- **Solution**: Updated all modal components to use the correct `description` field
  - **Format**: Backend returns "X dirs, Y files" or file sizes like "1.6 KB"
  - **Empty Detection**: Properly detects "0 dirs, 0 files" to show "Empty" annotation
  - **Consistent Display**: All tabs now show directory contents correctly

#### **Modal Scroll Lock Fix**
- **Issue**: After closing any modal (Cancel, Select Directory, etc.), main page scroll was locked and clicking was blocked
- **Root Cause**: Bootstrap modal instances weren't being properly disposed, leaving invisible backdrops that blocked interactions
- **Solution**: Comprehensive modal cleanup system
  - **Immediate Disposal**: All modal instances now properly disposed with `modal.dispose()`
  - **Complete Cleanup**: Removes all modal backdrops and resets body/HTML styles
  - **Emergency Tools**: Added Escape key handler and console debug functions
  - **Pointer Events**: Explicitly enables `pointer-events: auto` to restore clicking
  - **Applied Everywhere**: Fixed all modal types across Training, Quantization, Compare, and Test tabs

#### **Modal Title Overflow Fix**
- **Issue**: Long model names in modal titles overflowed the header, breaking layout
- **Root Cause**: Extremely long quantized model names like "models--published--Qwen3-32B-MLX-bf16_fused..." exceeded modal width
- **Solution**: Smart text truncation with tooltips
  - **Truncation Logic**: Names longer than 40 characters are truncated with ellipsis
  - **Tooltip Support**: Full names shown on hover using `title` attribute
  - **CSS Styling**: Added proper modal title styling with text overflow handling
  - **Consistent Application**: Applied to all modal types (Model, Adapter, Quantized, Session)

### 🔧 Technical Implementation
- **Enhanced Modal Cleanup**: Added `forceModalCleanup()` function with comprehensive DOM cleanup
- **Debug Tools**: Global `debugModalState()` and `cleanupAllModals()` functions for troubleshooting
- **CSS Improvements**: Added modal title styling with proper text truncation and responsive layout
- **Cross-Component Consistency**: Unified modal handling across all JavaScript components

### 🐛 Bug Fixes
- Fixed directory item counts showing "Empty" instead of actual file/folder counts
- Resolved modal scroll lock preventing page interaction after modal closure
- Fixed modal title overflow breaking header layout with long model names
- Eliminated modal backdrop remnants blocking clicks and scrolling
- Improved modal disposal preventing memory leaks and interaction issues

## [0.4.3] - 2025-07-16

### 🔧 Data Processing Improvements

#### **Bullet Point Formatting Fix**
- **Issue**: Numbered bullet points in training data were being incorrectly formatted, breaking the structure of lists
  - Original format: `1. The importance of dynamic temporal anchoring`
  - Incorrect processed format: `1\nThe importance of dynamic temporal anchoring`
- **Root Cause**: The `chunk_text` method in `DocumentProcessor` was splitting paragraphs by `\n\n` and joining with `\n`, breaking list formatting
- **Solution**: Enhanced text chunking with special handling for numbered lists
  - Added regex pattern detection for numbered list items (both `1.` and `1)` formats)
  - Preserved list items as complete units rather than splitting them
  - Improved paragraph structure preservation with double newlines
  - Standardized list formatting to dot notation (`1.` format)
- **Result**: Properly formatted numbered lists in training data, preserving document structure and readability

### 🔧 Technical Implementation
- **Pattern Recognition**: Added regex pattern matching to detect and preserve numbered list items
- **Format Standardization**: Converted different list formats (1., 1)) to consistent dot notation
- **Paragraph Structure**: Improved paragraph separation with proper newline handling
- **Comprehensive Testing**: Added test script to verify list formatting preservation

## [0.4.2] - 2025-07-15

### 🔄 Fullscreen Mode Fix

#### **Generation Output Fullscreen Toggle**
- **Issue**: Fullscreen button in Generation Output panel wasn't working correctly when toggling back to normal mode
- **Root Cause**: The chat panel wasn't being returned to its original container when exiting fullscreen mode
- **Solution**: Implemented robust container tracking and restoration system
  - Stores original container reference before entering fullscreen
  - Uses multiple fallback mechanisms to find the correct container when exiting
  - Properly resets styles when returning to normal mode

#### **Enhanced Fullscreen Experience**
- **Improved Button**: More visible primary button with "Fullscreen" and "Exit Fullscreen" text
- **Smooth Transitions**: Added CSS transitions for smoother visual changes
- **Style Preservation**: Proper style reset when exiting fullscreen mode
- **Error Handling**: Comprehensive error handling for edge cases
- **Visual Feedback**: Clear button state changes when toggling modes

#### **Technical Implementation**
- **Original Container Tracking**: Stores DOM reference to original container before entering fullscreen
- **Multi-Level Fallbacks**: Uses stored reference, selector, or tab container as fallbacks
- **Style Management**: Properly manages height, max-height, and other styles during transitions
- **Button Text Updates**: Dynamic button text changes between modes
- **Comprehensive Logging**: Detailed console logs for debugging

### 🔧 Technical Improvements
- **Enhanced Error Handling**: Better error messages and fallbacks throughout the system
- **Improved DOM Management**: More reliable element selection and manipulation
- **CSS Enhancements**: Better styling for fullscreen mode with proper transitions
- **Code Organization**: Cleaner separation of concerns in fullscreen toggle functionality

## [0.4.1] - 2025-01-16

### 🐛 Compare Tab Table Fixes

#### **Training Type Detection Fix**
- **Issue**: Compare Tab table TYPE column was defaulting to "Full" instead of showing correct LoRA/DoRA/Full values
- **Root Cause**: Table generation was using fallback logic instead of reading actual training type from session cards
- **Solution**: Modified `detectTrainingType()` to extract training type directly from session card badges (source of truth)
- **Result**: Table now correctly displays LoRA/DoRA/Full types matching the session cards

#### **Table Structure Cleanup**
- **Removed**: Iterations column from Compare Tab table as it was not informative
- **Improved**: Table is now more focused and easier to read
- **Maintained**: All essential columns (Training Session, Type, Method, Seq Len, Best Checkpoint, Best Loss, etc.)

#### **Technical Implementation**
- **Clean Solution**: Uses session cards as single source of truth for training type detection
- **Multi-layered Fallback**: Session card badges → batch API data → config data → intelligent default
- **No Breaking Changes**: All existing functionality preserved
- **Verified**: Comprehensive Playwright testing confirms fixes work correctly

### 🔧 Technical Details
- **Files Modified**: 
  - `forgellm/web/templates/index.html` - Removed iterations column from table header
  - `forgellm/web/static/js/components/compare.js` - Enhanced training type detection and table generation
- **Approach**: Simple, maintainable solution that reuses existing UI data
- **Testing**: Automated verification ensures table shows correct types and column structure

---

## [0.4.0] - 2025-01-10

### 🐛 Critical Memory Bug Fix

#### **Training Memory Overflow Resolution**
- **Issue**: Training sessions would fail with GPU memory errors (`[METAL] Command buffer execution failed: Insufficient Memory`) after the first iteration, despite working perfectly with direct MLX-LM command line calls
- **Root Cause**: Compare Tab was making 50+ simultaneous API requests when switching tabs during training, creating memory spikes that interfered with MLX training
- **Timeline**: Issue appeared between July 6-10, 2025 when Compare Tab functionality was introduced

#### **Technical Details**
- **Problem Pattern**: Training would complete first iteration successfully, then fail during second iteration with OOM error
- **Memory Impact**: 50+ sessions × 2 API calls each = 100+ concurrent requests consuming 750MB-3.5GB during critical training moments
- **Trigger Mechanism**: Compare Tab session restoration from localStorage during training would call `handleSessionChange()` for each previously selected session

#### **Fix Implementation**
- **Multi-Layer Protection**: Added training detection at 4 different levels to prevent any Compare Tab API calls during active training
  1. **SessionDataManager**: Blocks individual session data loading with training status check
  2. **Tab Activation**: Shows warning message without making API calls when training detected
  3. **Session Restoration**: Skips loading sessions that were previously selected
  4. **Badge Population**: Checks training status before fetching session data for UI updates

- **Duplicate Training Prevention**: Added `isTraining` flag to prevent race conditions that could launch multiple training processes simultaneously

- **Rate Limiting**: Added 10 requests/5 seconds limit as backup protection against API request floods

#### **Results**
- Training now works reliably without memory conflicts
- Compare Tab gracefully disabled during training with clear user messaging
- No more duplicate training process launches
- Eliminated 100+ concurrent API request spikes during training

### 🔧 Technical Infrastructure
- **Error Handling**: Added error messages when session loading blocked during training
- **User Experience**: Clear warning messages explain why Compare Tab features are temporarily unavailable
- **Backward Compatibility**: All changes maintain existing functionality when training is not active

---

## [0.3.8] - 2025-01-09

### 🔄 Compare Tab Updates

#### **Session Card System**
- **Cross-Tab Compatibility**: Unified session card design working across Compare and Training tabs
  - **Unique ID System**: Proper tab separation with `compare-session-card-` and `training-session-card-` prefixes
  - **Tab-Specific Population**: Session cards only populate when entering respective tabs, preventing conflicts
  - **Badge Function**: Single `populateLossBadges()` function works across both tabs
  - **Tab-Aware Selectors**: Container detection for accurate badge updates
  - **ID Conflicts**: Elimination of duplicate session card IDs between tabs

#### **Loss Badge Population**
- **Accurate Loss Values**: Replacement of placeholder badges with real training/validation loss data
  - **Training Tab**: 100% success rate (60/60 badges) showing actual loss values like "T: 1.375", "V: 1.398"
  - **Compare Tab**: 97% success rate (58/60 badges) with automatic population on tab activation
  - **Function**: Single function handles both tabs with container detection
  - **API Calls**: Async fetching from `/api/dashboard/historical` for accurate loss data
  - **LR Display**: Learning rate format "LR: 3.30e-5 | LDR 0.15 | WD 0.015" with decay and weight decay

#### **Base Model Selection**
- **Training Form Population**: 4-strategy matching algorithm for base model selection
  - **Strategy 1**: Exact value match (`option.value === baseModel`) - highest priority
  - **Strategy 2**: Exact clean text match (removes icons and formatting) 
  - **Strategy 3**: Normalized format matching (handles different separators)
  - **Strategy 4**: Careful partial matching with high threshold (≥0.8 score) for safety
  - **Match Scoring**: `calculateModelMatchScore()` function prevents wrong model selection
  - **Logging**: Console output for debugging model selection process
  - **Fixed Issue**: Session with `"base_model": "Qwen/Qwen3-32B-MLX-bf16"` now correctly selects exact base model instead of published variants

#### **Curve Highlighting**
- **Chart Highlighting**: Added curve highlighting when hovering over session cards
  - **Visual Connection**: Easy identification of which curve corresponds to which training session
  - **Multi-Chart Support**: Highlighting works simultaneously across all 4 comparison charts
    - **Loss Analysis**: Validation loss curves with hover highlighting
    - **Perplexity Analysis**: Perplexity evolution with visual emphasis
    - **Loss Stability**: Coefficient of variation curves with highlighting support
    - **Generalization Gap**: Train/validation gap analysis with hover effects
  - **Visual Feedback**: 
    - **Hovered Curve**: Thicker line (4px width) with full opacity
    - **Non-Hovered Curves**: Dimmed with reduced opacity (0.4) and thinner lines (1.5px)
    - **Color Preservation**: Same color scheme maintained, only visual emphasis changes
  - **Interaction**: Chart updates on hover
  - **Selected Sessions Only**: Hover effects only apply to selected session cards

#### **Summary Table Updates**
- **Column Layout**: Reordered summary table columns for better data flow
  - **New Order**: Training Session → Type → Method → Seq Len → **Iterations → Best Checkpoint → Best Loss** → Learning Rate
  - **Performance Metrics**: Iterations, checkpoint, and loss now grouped together
- **Interface Cleanup**: Removed redundant "Selected Sessions" panel from left sidebar
  - **Focus**: Left panel focuses on session selection and search
  - **Summary Table**: Full-width summary table serves as primary overview for selected sessions

#### **Single Session Support**
- **Functionality**: Compare tab now supports viewing single training sessions (previously required 2+ sessions)
  - **Chart Generation**: Charts appear as soon as one session is selected
  - **Dynamic Titles**: Titles automatically adjust between "Analysis" and "Comparison" modes
  - **Updated Instructions**: Instructions reflect single-session capability

#### **Visual Feedback & Interactivity**
- **Session Cards**: Visual feedback when hovering over session cards
  - **Hover Animation**: Cards lift 2px with smooth transition effect
  - **Shadow**: Blue-tinted shadow indicates clickable state
  - **Border Highlight**: Left border changes to primary blue on hover
  - **Theme Support**: Different shadow opacity for light/dark modes

#### **Session Management Updates**
- **Clear All Button**: Repositioned Clear All button for better accessibility
  - **Top Placement**: Moved from bottom of panel to top, right after dataset warning
  - **Always Accessible**: No longer hidden until sessions are selected
  - **Full-Width Design**: Visual prominence with `w-100` styling

#### **Session Card Actions**
- **Session Actions**: Added action buttons to each session card
  - **📁 Folder Browser**: Opens session training folder in modal browser
    - **File Explorer**: Navigate through session directories and files
    - **Path Detection**: Extracts directory from log file path
    - **Modal Integration**: Uses existing file browser with view-only mode
  - **🗑️ Delete Session**: Red delete button with confirmation for session removal
    - **Confirmation Dialog**: Two-step confirmation process to prevent accidental deletion
    - **Backend API**: New `/api/training/sessions/delete` endpoint for safe session removal
    - **Auto-Refresh**: Session list automatically updates after successful deletion

#### **Session Information Display**
- **Badge System**: Metadata display with color-coded badges
  - **Training Type Detection**: Multi-level detection for LoRA, DoRA, and Full training
    - **Heuristics**: Analyzes session names, log paths, and adapter patterns
    - **Background Enrichment**: Async function fetches actual config data for badges
    - **Updates**: Badges update with actual data from training configs
  - **Training Method Badges**: CPT/SFT indicators with yellow color coding
  - **Sequence Length Display**: Shows max_seq_length as gray badges (2048, 3072, 4096)
  - **Iteration Count**: Session progress indicators in badge format

- **Parameter Display**: Training parameter visualization
  - **Learning Rate Information**: LR display with decay and weight decay
    - **Format**: "LR: 3e-05 | LDR 0.15 | WD 0.015" with pipe separators
    - **Background Updates**: Async fetching of actual config values for accuracy
    - **Parameter Extraction**: Parsing of `lr_decay_factor` and `weight_decay`
  - **Layout**: Session card design with information density
  - **Direct Display**: All information directly visible through badges and info lines

#### **Technical Infrastructure**
- **API Updates**: Backend support for new features
  - **Session Deletion Endpoint**: Safe session removal with error handling
  - **File Browser Integration**: Leverages existing `/api/filesystem/browse` for folder viewing
  - **Session Enrichment**: Background data fetching for accurate badge information
- **Theme Compatibility**: All new features support light/dark mode switching
- **Error Handling**: Validation and user-friendly error messages throughout

### **Final Session Card Design**
```
gemma-3-27b-it-bf16                [T: 1.375] [V: 1.398]
[LoRA] [CPT] [3072]
📅 7/9/2025 05:54 AM
📈 LR: 3.30e-5 | LDR 0.15 | WD 0.015
[📄] [🔗] [🧪] [📁] [🗑️]
```

## [0.3.7] - 2025-07-09

### 🎲 Deterministic Text Generation
- **Seed Parameter Support**: Added comprehensive seed parameter for reproducible text generation
  - **Testing Tab UI**: New seed input field with default value of 42 and helpful tooltip
  - **Deterministic Generation**: Same seed produces identical outputs for reproducible results
  - **MLX-LM Integration**: Proper MLX random seed setting for deterministic generation
  - **Complete Parameter Chain**: Seed parameter flows through entire generation pipeline:
    - Frontend JavaScript collection and validation
    - API route parameter forwarding with default value 42
    - ModelManager parameter handling and documentation
    - Model server MLX seed setting with logging
  - **Configuration Management**: Seed parameter included in:
    - Reset configuration function (resets to 42)
    - Chat history save/load functionality
    - Parameter preservation across sessions
  - **User Experience**: Clean UI integration matching existing parameter styling
    - Input validation (1 to 2,147,483,647 range)
    - Helpful description text with dice icon
    - Consistent with other generation parameters

### 📊 Training Session Comparison System
- **Compare Tab**: Revolutionary new tab for comprehensive training session analysis
  - **Multi-Session Selection**: Interactive session picker with color-coded indicators
  - **Dataset Compatibility Warning**: Clear guidance on comparing sessions with same training data
  - **Advanced Visualization**: 2x2 grid of comparison charts with professional styling
  - **Intelligent Session Management**: Automatic color assignment and consistent theming across charts
  - **Auto-Update Functionality**: Comparisons generate automatically when sessions are selected/deselected
  - **Enhanced Text Handling**: Improved text wrapping and layout for long session names and details
  - **Robust Error Handling**: Comprehensive null checks and DOM readiness validation to prevent JavaScript errors
  - **Smart Initialization**: Tab-aware initialization that ensures DOM elements are ready before execution
  - **Special Character Handling**: Robust CSS selector escaping for model IDs with dots and special characters
  - **Theme-Aware Styling**: Proper background and text colors in both light and dark modes
  - **Visual Selection Feedback**: Enhanced blue highlighting with border effects for selected models

- **Comprehensive Comparison Metrics**: Four key visualization categories
  - **Loss Comparison**: Training vs validation loss curves with deviation analysis
  - **Perplexity Evolution**: Logarithmic perplexity tracking with trend visualization
  - **Loss Stability**: Variance analysis with color-coded stability bands (excellent/good/unstable)
    - Fixed x-axis range to always start at 0 for better trend visualization
    - Added data point at iteration 0 for more complete variance tracking
  - **Generalization Gap**: Val Loss - Train Loss with overfitting/underfitting indicators
    - Dynamic y-axis range adjustment based on actual data values
    - Color-coded regions for underfitting and overfitting detection

- **Statistical Analysis Dashboard**: Advanced performance insights
  - **Performance Rankings**: Automatic ranking by final validation loss with medal system
  - **Convergence Statistics**: Average metrics and best-performing session identification
  - **Session Metadata**: Model names, iteration counts, training duration, and final metrics
  - **Export Functionality**: JSON export of complete comparison data and statistics

- **Professional User Experience**: Enterprise-grade interface design
  - **Fullscreen Mode**: Immersive comparison view with responsive chart resizing
  - **Interactive Charts**: Plotly.js integration with hover details and legend management
  - **Color Consistency**: Persistent session colors across all charts and UI elements
  - **Session Summary**: Real-time selection tracking with clear enable/disable states
  - **Session Parameter Viewer**: Modal-based parameter inspection with copy-to-clipboard functionality
  - **Quick Actions**: Contextual buttons for viewing parameters, fusing adapters, and testing in playground

- **Adapter Integration**: Seamless workflow between Compare and other tabs
  - **"Fuse" Button**: Direct adapter selection for model fusion from Compare tab
  - **"Test in Playground" Button**: One-click testing of selected models in Testing tab
  - **Best Checkpoint Selection**: Automatic selection of best checkpoint (lowest validation loss)
  - **Parameter Inspection**: Detailed parameter view with copy-to-clipboard functionality

- **API Infrastructure**: Robust backend support for comparison operations
  - **Multi-Session Endpoint**: Efficient batch loading of training session data
  - **Chart Data Generation**: Server-side processing of metrics for web visualization
  - **Error Handling**: Comprehensive validation and user-friendly error messages
  - **Performance Optimization**: Parallel data processing and caching strategies

## [0.3.6] - 2025-07-06

### 🧮 Centralized Token Counting System
- **Accurate Token Statistics**: Revolutionary centralized text statistics utility (`forgellm/utils/text_stats.py`)
  - **Intelligent Tokenizer Prioritization**: 4-tier system for maximum accuracy
    1. MLX Tokenizer (most accurate - uses actual model tokenizer)
    2. HuggingFace Tokenizer (high accuracy - model-specific fallback)
    3. TikToken (good approximation - GPT-4 style tokenization)
    4. Word Estimation (last resort - consistent 1.4x multiplier)
  - **Comprehensive Statistics**: Tokens, words, lines, pages, characters, and tokenizer metadata
  - **Validation Tools**: Compare old vs new methods with accuracy analysis and recommendations

- **Consistent Accuracy Across All Components**: Eliminated token counting discrepancies
  - **Training Data Processor**: Now uses `count_tokens_accurate()` instead of word splitting
  - **API Dataset Analysis**: Replaced regex word matching with proper tokenization
  - **CLI Commands**: Updated dataset analysis to use centralized token counting
  - **CLI REPL Mode**: Enhanced to use actual tokenizer when available for chat statistics
  - **Server Generation**: Already accurate (maintained existing proper tokenization)

- **Significant Accuracy Improvements**: Test results show major error reduction
  - Technical Code: 34.5% error reduction
  - Mixed Content: 32.4% error reduction  
  - Chat Conversations: 16.2% error reduction
  - Dataset Analysis: 22.0% average improvement

### 🔄 UI State Management Fixes
- **Dropdown Selection Preservation**: Fixed model and adapter dropdowns resetting during periodic updates
  - **Root Cause**: `updateAdapterSelect()` and `updateModelDropdown()` were completely rebuilding dropdowns every 60 seconds
  - **Solution**: Implemented intelligent selection preservation that remembers and restores user choices
  - **Enhanced User Experience**: No more frustrating dropdown resets while using the interface
  - **Comprehensive Logging**: Added debug messages to track selection restoration process

- **Reset Configuration Button**: Added professional reset functionality to Testing tab
  - **Orange "Reset" Button**: Positioned to the left of Load/Unload buttons with undo icon
  - **Complete Parameter Reset**: Resets all generation parameters to defaults
    - Base Model: Empty (no selection)
    - Adapter Path: Empty (no selection)
    - System Prompt: Empty
    - Max Tokens: 2000, Context Window: 16384, Temperature: 0.7, Top P: 0.9
    - Repetition Penalty: 1.1, Streaming: Enabled
  - **Model Unloading**: Automatically unloads currently loaded model during reset
  - **User Confirmation**: Descriptive tooltip explaining the functionality

### 🔧 Technical Infrastructure Enhancements
- **TextStatsCalculator Class**: Robust text analysis with fallback mechanisms
  - Automatic tokenizer detection and loading for specific models
  - Error handling with graceful degradation to less accurate methods
  - Transparent reporting of which tokenization method was used
  - Support for both MLX and HuggingFace tokenizer ecosystems

- **Legacy Compatibility Functions**: Smooth migration from old token counting methods
  - `estimate_tokens_from_words()`: For backward compatibility during transition
  - `validate_token_count()`: Accuracy analysis tool for existing implementations
  - Clear deprecation notices encouraging migration to accurate methods

### 📚 Enhanced Documentation
- **Data Flow Documentation**: Comprehensive update to `docs/data_flow.md`
  - Detailed flowchart of centralized token counting system
  - Priority system explanation with accuracy levels
  - Usage examples and best practices
  - Integration patterns for different components

- **API Transparency**: Enhanced logging and metadata
  - Clear indication of which tokenizer was used for each count
  - Percentage error analysis for validation
  - Recommendations for improving token counting accuracy

### 💡 Developer Experience Improvements
- **Consistent API**: Unified token counting interface across all ForgeLLM components
- **Debugging Tools**: Built-in validation and comparison utilities
- **Performance Optimization**: Intelligent caching and fallback mechanisms
- **Error Resilience**: Graceful handling of tokenizer loading failures

### 🐛 Critical Bug Fixes
- **Periodic Update Race Conditions**: Fixed dropdown selections being lost every 60 seconds
- **Token Count Inconsistencies**: Resolved major discrepancies between different components
- **Model Loading State Management**: Enhanced reset functionality with proper cleanup
- **Memory Management**: Improved tokenizer loading and caching efficiency

## [0.3.5] - 2025-07-05

### 🧠 Enhanced Markdown Rendering
- **Thinking Blocks Feature**: Revolutionary `<think></think>` tag support in chat interface
  - Interactive collapsible thinking blocks with brain icon and "Thoughts" label
  - Proper CSS styling with gray italic text and hover effects
  - Smooth fold/unfold animations with visual feedback
  - Theme-aware styling for both light and dark modes
  - Post-markdown processing using placeholder token system to avoid conflicts
  - Comprehensive CSS specificity handling for `.chat-assistant` containers

### 🚀 Critical Model Loading Fixes
- **Local-Only Model Resolution**: Eliminated HuggingFace download attempts
  - Fixed ModelManager to NEVER try downloading from HuggingFace Hub
  - Proper HuggingFace cache directory structure handling (`snapshots/{hash}/`)
  - Distinguished between published models (flat structure) and regular models (snapshots structure)
  - Enhanced model path resolution for both regular and published models
  - Clear error messages when models aren't found locally

- **API Model Listing Improvements**: Fixed incorrect model paths in `/api/models` endpoint
  - Corrected snapshot directory navigation for regular HuggingFace models
  - Proper published model detection with direct file access
  - Fixed model dropdown population issues in test interface
  - Enhanced model filtering to skip invalid model directories

- **Synchronous Model Loading**: Complete overhaul of model loading in Testing tab
  - Fixed missing loading indicators by implementing synchronous API responses
  - Added proper progress monitoring with status polling until model is fully loaded
  - Enhanced user feedback with loading overlays and button state management
  - Eliminated premature success messages while models were still loading on server

### ⚡ Performance Metrics Restoration
- **Token Speed Display**: Fixed missing tokens/second in generation stats
  - Restored `tokens_per_sec` calculation in both streaming and non-streaming responses
  - Fixed server-side token speed calculation and transmission
  - Enhanced client-side stats display timing for streaming responses
  - Consistent speed metrics across all generation modes

### 🎨 UI/UX Enhancements
- **CSS Integration**: Fixed missing `styles.css` loading in web interface
  - Added proper stylesheet link to `index.html` template
  - Implemented cache busting for CSS files
  - Resolved styling conflicts with high CSS specificity
  - Enhanced paragraph styling inheritance in thinking blocks

- **Dark Theme Improvements**: Enhanced contrast and visibility
  - Fixed loading text contrast issues in dark theme mode
  - Improved thinking block styling for dark mode
  - Better color consistency across all UI elements

- **Smart Scrolling for Streaming**: Intelligent chat auto-scroll behavior
  - Only auto-scrolls during streaming when user is already at bottom of chat
  - Preserves user's scroll position when reading earlier messages
  - Maintains smooth UX without interrupting user interaction
  - Applied to both streaming and non-streaming responses

### 🔄 Model Fusion System
- **Complete Fusion Implementation**: Full MLX-LM model fusion capability
  - **Fusion Tab**: New dedicated interface for fusing foundation models with LoRA/DoRA adapters
  - **Auto-Detection**: Automatic base model detection from adapter configuration files
  - **Published Model Pattern**: Proper `models--published--` output path structure
  - **Configuration Modal**: User-friendly interface for fusion customization
    - Model suffix input (e.g., `_math`, `_reasoning`, `_mnemosyne`)
    - Description field for README.md generation
    - Real-time preview of final model name
    - Enhanced text wrapping and layout for long model names

- **Fusion Progress Monitoring**: Real-time fusion tracking
  - Live progress bar with percentage completion
  - Status messages and elapsed/remaining time estimates
  - Background thread processing with MLX-LM fuse command
  - Comprehensive error handling and user feedback

- **Enhanced Success Display**: Professional completion interface
  - Display of published model name in `published/model_name` format
  - "Open Folder" button with tooltip for local model access
  - Responsive layout handling long model names with proper text wrapping
  - Cross-platform folder opening (macOS, Windows, Linux support)

### 🔧 Technical Infrastructure
- **Model Directory Structure Analysis**: Comprehensive investigation of HuggingFace vs local model storage
  - Documented differences between HF cache symlink structure and local flat directories
  - Enhanced understanding of MLX-LM model saving patterns
  - Improved model path resolution logic for different storage types

- **ModelFuser Architecture**: Robust fusion processing system
  - Thread-safe background processing with progress tracking
  - Intelligent adapter path resolution for both files and directories
  - Comprehensive validation of base models and adapter files
  - Automatic README.md generation with usage examples and metadata

### 🐛 Critical Bug Fixes
- **Streaming Response Handling**: Fixed token count and speed display timing issues
- **Model Path Resolution**: Eliminated infinite loops and download attempts
- **CSS Specificity**: Resolved styling conflicts in chat interface
- **Markdown Processing**: Fixed thinking block rendering interference with marked.js
- **Cache Directory Navigation**: Proper handling of HuggingFace cache structure
- **Modal Theme Support**: Fixed fusion configuration modal theme inheritance
- **Text Overflow Issues**: Resolved layout problems with long model names in modals and success displays
- **Duplicate Element IDs**: Fixed HTML validation issues with duplicate loading message elements

### 💡 Developer Experience
- **Enhanced Error Messages**: Clear feedback when models aren't found locally
- **Improved Logging**: Better debugging information for model loading and training processes
- **Code Organization**: Cleaner separation between model types and storage patterns
- **API Unification**: Consistent endpoint patterns across quantization and fusion systems
- **Robust General-Purpose Logic**: Designed solutions to work for all inputs, not just test cases

## [0.3.0] - 2025-06-30

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