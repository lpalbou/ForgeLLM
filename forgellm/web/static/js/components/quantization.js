/**
 * Quantization component
 */
class QuantizationComponent {
    constructor() {
        this.initialized = false;
        this.updateInterval = null;
        this.startTime = null;
    }

    /**
     * Initialize the quantization component
     */
    init() {
        console.log('🔧 QuantizationComponent.init() called');
        
        if (this.initialized) {
            console.log('⚠️ QuantizationComponent already initialized, skipping');
            return;
        }

        console.log('🚀 Initializing QuantizationComponent...');
        
        // Load initial data
        this.loadAvailableModels();
        this.loadQuantizedModels();
        
        // Check status
        this.updateProgress();
        
        this.initialized = true;
        console.log('✅ QuantizationComponent initialization complete');
        
        // Initialize event listeners when the tab is shown
        this.initTabEventListener();
    }

    /**
     * Initialize tab event listener to reinitialize when tab is activated
     */
    initTabEventListener() {
        console.log('🔧 Setting up tab event listener...');
        
        const quantizationTab = document.getElementById('quantization-tab');
        if (quantizationTab) {
            console.log('✅ Quantization tab found, adding event listener');
            quantizationTab.addEventListener('shown.bs.tab', () => {
                console.log('📱 Quantization tab activated, reinitializing event listeners');
                this.initEventListeners();
            });
        } else {
            console.error('❌ Quantization tab not found!');
        }
        
        // Also try to initialize event listeners immediately
        this.initEventListeners();
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        console.log('🔧 Initializing quantization event listeners...');
        
        // Start quantization button click
        const startBtn = document.getElementById('start-quantization-btn');
        console.log('🔍 Start button element:', startBtn);
        
        if (startBtn) {
            console.log('✅ Start button found, adding event listener');
            startBtn.addEventListener('click', (e) => {
                console.log('🚀 START QUANTIZATION BUTTON CLICKED!');
                e.preventDefault();
                e.stopPropagation();
                console.log('🔧 Start quantization button clicked');
                this.startQuantization();
                return false;
            });
        } else {
            console.error('❌ Start quantization button not found!');
        }

        // Quantization form submission (backup)
        const quantizationForm = document.getElementById('quantization-form');
        console.log('🔍 Quantization form element:', quantizationForm);
        
        if (quantizationForm) {
            console.log('✅ Quantization form found, adding event listener');
            quantizationForm.addEventListener('submit', (e) => {
                console.log('📋 QUANTIZATION FORM SUBMITTED!');
                e.preventDefault();
                e.stopPropagation();
                console.log('Quantization form submitted, starting quantization...');
                this.startQuantization();
                return false;
            });
        } else {
            console.error('❌ Quantization form not found!');
        }

        // Stop quantization button
        const stopBtn = document.getElementById('stop-quantization-btn');
        console.log('🔍 Stop button element:', stopBtn);
        
        if (stopBtn) {
            console.log('✅ Stop button found, adding event listener');
            stopBtn.addEventListener('click', (e) => {
                console.log('🛑 STOP QUANTIZATION BUTTON CLICKED!');
                e.preventDefault();
                e.stopPropagation();
                this.stopQuantization();
                return false;
            });
        } else {
            console.error('❌ Stop quantization button not found!');
        }

        // Model selection change - update output name preview
        const modelSelect = document.getElementById('quantization-model-select');
        const bitsSelect = document.getElementById('quantization-bits');
        const groupSizeSelect = document.getElementById('quantization-group-size');
        
        console.log('🔍 Model select elements:', { modelSelect, bitsSelect, groupSizeSelect });
        
        if (modelSelect && bitsSelect && groupSizeSelect) {
            console.log('✅ All select elements found, adding change listeners');
            [modelSelect, bitsSelect, groupSizeSelect].forEach(select => {
                select.addEventListener('change', () => {
                    console.log('📝 Select changed:', select.id);
                    this.updateOutputPreview();
                });
            });
        } else {
            console.error('❌ Some select elements not found!');
        }
        
        console.log('🔧 Quantization event listeners initialization complete');
    }

    /**
     * Load available models for quantization
     */
    async loadAvailableModels() {
        try {
            // Use the same endpoint as Training and Testing tabs
            const response = await fetch('/api/models');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            if (data.models && Array.isArray(data.models)) {
                this.updateModelDropdown(data.models);
            } else {
                console.error('Failed to load models: No models found');
                this.showError('Failed to load models: No models found');
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showError('Error loading models: ' + error.message);
        }
    }

    /**
     * Update models dropdown using the same style as other tabs
     */
    updateModelDropdown(models) {
        const select = document.getElementById('quantization-model-select');
        if (!select) return;
        
        // Filter out already-quantized models (those with quantization patterns in name)
        const quantizableModels = models.filter(model => {
            const modelName = model.name.toLowerCase();
            // Exclude models that are already quantized
            const isQuantized = modelName.includes('4bit') || 
                               modelName.includes('8bit') || 
                               modelName.includes('-4bit') || 
                               modelName.includes('-8bit') || 
                               modelName.includes('_4bit') || 
                               modelName.includes('_8bit') ||
                               modelName.includes('q4') ||
                               modelName.includes('q8') ||
                               modelName.includes('gptq') ||
                               modelName.includes('awq');
            return !isQuantized;
        });
        
        // Clear existing options except the first one (placeholder)
        const firstOption = select.firstElementChild;
        select.innerHTML = '';
        if (firstOption) {
            select.appendChild(firstOption);
        } else {
            // Add placeholder option if it doesn't exist
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'Select a model to quantize...';
            select.appendChild(placeholderOption);
        }
        
        // Add quantizable models with simplified icons (same logic as other tabs)
        quantizableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.full_path || model.path || model.id; // Use full_path for quantization API
            
            // Store the actual path for folder opening (if available)
            if (model.full_path) {
                option.setAttribute('data-path', model.full_path);
            } else if (model.path) {
                option.setAttribute('data-path', model.path);
            }
            
            // Store model name for output preview
            option.setAttribute('data-model-name', model.name);
            
            // Determine if this is a base model using same logic as other tabs
            const modelName = model.name.toLowerCase();
            
            // Special handling for Qwen models: they are instruct by default EXCEPT if "base" is in the name
            let isBaseModel;
            if (modelName.includes('qwen')) {
                // For Qwen models, check if it's explicitly marked as base
                isBaseModel = modelName.includes('base');
            } else {
                // Regular detection logic for non-Qwen models
                const instructPatterns = [
                    'instruct', 'chat', 'sft', 'dpo', 'rlhf', 
                    'assistant', 'alpaca', 'vicuna', 'wizard', 'orca',
                    'dolphin', 'openhermes', 'airoboros', 'nous',
                    'claude', 'gpt', 'turbo', 'dialogue', 'conversation',
                    '_it_', '-it-'  // Add explicit patterns for _it_ and -it-
                ];
                const specialPatterns = ['it']; // 'it' needs word boundary checking
                const basePatterns = [
                    'base', 'pt', 'pretrain', 'foundation'
                ];
                
                const hasBasePattern = basePatterns.some(pattern => 
                    modelName.includes(`-${pattern}`) || 
                    modelName.includes(`_${pattern}`) ||
                    modelName.includes(`-${pattern}-`) ||
                    modelName.includes(`_${pattern}_`) ||
                    modelName.endsWith(`-${pattern}`) ||
                    modelName.endsWith(`_${pattern}`) ||
                    modelName.endsWith(pattern)
                );
                
                let hasInstructPattern = instructPatterns.some(pattern => 
                    modelName.includes(pattern)
                );
                
                if (!hasInstructPattern) {
                    hasInstructPattern = specialPatterns.some(pattern => {
                        const regex = new RegExp(`\\b${pattern}\\b`, 'i');
                        return regex.test(modelName);
                    });
                }
                
                isBaseModel = hasBasePattern || !hasInstructPattern;
            }
            
            // Simplified icons: only 2 types (same as other tabs)
            let icon = '';
            if (isBaseModel) {
                icon = '⚡ '; // Lightning for base models
            } else {
                icon = '🤖 '; // Robot for instruct models
            }
            
            // Format size
            const sizeStr = model.size > 0 ? ` (${model.size}GB)` : '';
            
            option.textContent = `${icon}${model.name}${sizeStr}`;
            select.appendChild(option);
        });
        
        console.log(`Loaded ${quantizableModels.length} quantizable models (filtered from ${models.length} total models)`);
    }

    /**
     * Update output name preview
     */
    updateOutputPreview() {
        const modelSelect = document.getElementById('quantization-model-select');
        const bitsSelect = document.getElementById('quantization-bits');
        const groupSizeSelect = document.getElementById('quantization-group-size');
        const outputInfo = document.getElementById('quantization-output-info');
        const outputName = document.getElementById('output-model-name');
        
        if (!modelSelect || !bitsSelect || !groupSizeSelect || !outputInfo || !outputName) return;
        
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        if (!selectedOption || !selectedOption.value) {
            outputInfo.style.display = 'none';
            return;
        }
        
        const modelName = selectedOption.getAttribute('data-model-name') || selectedOption.textContent.split(' (')[0].replace(/^[⚡🤖] /, '');
        const bits = bitsSelect.value;
        const groupSize = groupSizeSelect.value;
        
        // Generate output name
        let suffix = `_Q${bits}`;
        if (groupSize !== '64') {
            suffix += `_G${groupSize}`;
        }
        
        const outputModelName = `${modelName}${suffix}`;
        
        outputName.textContent = outputModelName;
        outputInfo.style.display = 'block';
    }

    /**
     * Start quantization process
     */
    async startQuantization() {
        console.log('🚀 startQuantization() method called!');
        
        try {
            // Hide any previous messages
            this.hideMessages();
            
            // Get form values
            const modelSelect = document.getElementById('quantization-model-select');
            const bitsSelect = document.getElementById('quantization-bits');
            const groupSizeSelect = document.getElementById('quantization-group-size');
            
            console.log('🔍 Form elements:', { modelSelect, bitsSelect, groupSizeSelect });
            
            if (!modelSelect || !bitsSelect || !groupSizeSelect) {
                console.error('❌ Missing form elements');
                this.showError('Form elements not found');
                return;
            }
            
            const modelPath = modelSelect.value;
            const bits = parseInt(bitsSelect.value);
            const groupSize = parseInt(groupSizeSelect.value);
            
            console.log('📋 Form values:', { modelPath, bits, groupSize });
            
            if (!modelPath) {
                console.warn('⚠️ No model selected');
                this.showError('Please select a model to quantize');
                return;
            }
            
            console.log('📤 Making API call to start quantization...');
            
            // Start quantization via API - use individual parameters, not object
            const response = await apiService.startQuantization(modelPath, bits, groupSize);
            
            console.log('📥 API response:', response);
            
            if (response.success) {
                console.log('✅ Quantization started successfully');
                this.showSuccess('Quantization started successfully!');
                
                // Update UI to show active state
                this.showQuantizationActive();
                
                // Start monitoring progress
                this.startProgressMonitoring();
                
                // Update job info
                const outputModel = `${modelPath.split('/').pop()}_Q${bits}`;
                this.updateCurrentJobInfo(modelPath, outputModel, bits, groupSize);
                
            } else {
                console.error('❌ Failed to start quantization:', response.error);
                this.showError(response.error || 'Failed to start quantization');
            }
            
        } catch (error) {
            console.error('💥 Error starting quantization:', error);
            this.showError('Error starting quantization: ' + error.message);
        }
    }

    /**
     * Stop quantization
     */
    async stopQuantization() {
        try {
            const response = await apiService.stopQuantization();
            
            if (response.success) {
                console.log('Quantization stop requested');
                this.stopProgressMonitoring();
                this.showQuantizationIdle();
            } else {
                this.showError(response.error || 'Failed to stop quantization');
            }
        } catch (error) {
            console.error('Error stopping quantization:', error);
            this.showError('Error stopping quantization: ' + error.message);
        }
    }

    /**
     * Start monitoring quantization progress
     */
    startProgressMonitoring() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            this.updateProgress();
        }, 1000); // Update every second
        
        // Initial update
        this.updateProgress();
    }

    /**
     * Stop monitoring quantization progress
     */
    stopProgressMonitoring() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Update quantization progress
     */
    async updateProgress() {
        try {
            const response = await apiService.getQuantizationStatus();
            
            if (response.success) {
                this.updateProgressUI(response);
                
                // If quantization is complete or failed, stop monitoring
                if (!response.is_quantizing) {
                    this.stopProgressMonitoring();
                    
                    if (response.error) {
                        this.showError(response.error);
                        this.showQuantizationIdle();
                    } else if (response.progress === 100) {
                        this.showSuccess(response.status_message);
                        this.showQuantizationIdle();
                        // Reload quantized models list
                        this.loadQuantizedModels();
                    }
                }
            }
        } catch (error) {
            console.error('Error updating progress:', error);
            this.stopProgressMonitoring();
            this.showError('Error monitoring progress: ' + error.message);
            this.showQuantizationIdle();
        }
    }

    /**
     * Update progress UI elements
     */
    updateProgressUI(status) {
        // Update progress bar
        const progressBar = document.getElementById('quantization-progress-bar');
        const progressPercentage = document.getElementById('progress-percentage');
        
        if (progressBar && progressPercentage) {
            const progress = status.progress || 0;
            progressBar.style.width = `${progress}%`;
            progressPercentage.textContent = progress;
        }
        
        // Update status message
        const statusMessage = document.getElementById('quantization-status-message');
        if (statusMessage) {
            statusMessage.textContent = status.status_message || 'Processing...';
        }
        
        // Update time information
        this.updateTimeInfo(status.estimated_remaining);
    }

    /**
     * Update time information
     */
    updateTimeInfo(estimatedRemaining) {
        const elapsedTimeEl = document.getElementById('elapsed-time');
        const estimatedRemainingEl = document.getElementById('estimated-remaining');
        
        if (elapsedTimeEl && this.startTime) {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            elapsedTimeEl.textContent = this.formatTime(elapsed);
        }
        
        if (estimatedRemainingEl) {
            if (estimatedRemaining && estimatedRemaining > 0) {
                estimatedRemainingEl.textContent = this.formatTime(Math.floor(estimatedRemaining));
            } else {
                estimatedRemainingEl.textContent = '--:--';
            }
        }
    }

    /**
     * Format time in MM:SS format
     */
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    /**
     * Update current job information
     */
    updateCurrentJobInfo(inputModel, outputModel, bits, groupSize) {
        const inputEl = document.getElementById('current-input-model');
        const outputEl = document.getElementById('current-output-model');
        const settingsEl = document.getElementById('current-settings');
        
        if (inputEl) inputEl.textContent = inputModel;
        if (outputEl) outputEl.textContent = outputModel;
        if (settingsEl) settingsEl.textContent = `${bits}-bit, Group ${groupSize}`;
    }

    /**
     * Show quantization active state
     */
    showQuantizationActive() {
        const idle = document.getElementById('quantization-idle');
        const active = document.getElementById('quantization-active');
        const startBtn = document.getElementById('start-quantization-btn');
        const stopBtn = document.getElementById('stop-quantization-btn');
        
        if (idle) idle.style.display = 'none';
        if (active) active.style.display = 'block';
        if (startBtn) startBtn.style.display = 'none';
        if (stopBtn) stopBtn.style.display = 'block';
    }

    /**
     * Show quantization idle state
     */
    showQuantizationIdle() {
        const idle = document.getElementById('quantization-idle');
        const active = document.getElementById('quantization-active');
        const startBtn = document.getElementById('start-quantization-btn');
        const stopBtn = document.getElementById('stop-quantization-btn');
        
        if (idle) idle.style.display = 'block';
        if (active) active.style.display = 'none';
        if (startBtn) startBtn.style.display = 'block';
        if (stopBtn) stopBtn.style.display = 'none';
    }

    /**
     * Load quantized models
     */
    async loadQuantizedModels() {
        try {
            const response = await apiService.getQuantizedModels();
            
            if (response.success) {
                this.updateQuantizedModelsList(response.models);
            } else {
                console.error('Failed to load quantized models:', response.error);
            }
        } catch (error) {
            console.error('Error loading quantized models:', error);
        }
    }

    /**
     * Update quantized models list
     */
    updateQuantizedModelsList(models) {
        const container = document.getElementById('quantized-models-list');
        if (!container) return;
        
        if (models.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-3">
                    <i class="fas fa-box-open fa-2x mb-2"></i>
                    <p>No quantized models yet</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = models.map(model => {
            // Extract meaningful parts of the model name for better display
            const modelName = model.name;
            const parts = modelName.split('/');
            const baseName = parts[parts.length - 1]; // Get the last part (actual model name)
            const namespace = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
            
            return `
            <div class="card mb-3">
                <div class="card-body p-3">
                    <div class="row align-items-start">
                        <div class="col-10">
                            <div class="d-flex flex-column">
                                ${namespace ? `<small class="text-muted mb-1"><i class="fas fa-folder me-1"></i>${namespace}/</small>` : ''}
                                <h6 class="mb-2 text-break" style="line-height: 1.3; word-break: break-word;">
                                    ${baseName}
                                </h6>
                                <div class="d-flex flex-wrap gap-2 mb-2">
                                    <span class="badge bg-primary">${model.size} GB</span>
                                    <span class="badge bg-info">${model.bits}-bit</span>
                                    <span class="badge bg-secondary">Group ${model.group_size}</span>
                                </div>
                                ${model.quantized_at ? `
                                    <small class="text-muted">
                                        <i class="fas fa-clock me-1"></i>
                                        Created: ${new Date(model.quantized_at).toLocaleString()}
                                    </small>
                                ` : ''}
                            </div>
                        </div>
                        <div class="col-2 text-end">
                            <button class="btn btn-sm btn-outline-primary" 
                                    onclick="quantizationComponent.showQuantizedModelFolder('${model.full_path}')"
                                    title="Show model folder contents">
                                <i class="fas fa-folder-open"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            `;
        }).join('');
    }

    /**
     * Show quantized model folder contents in modal
     */
    async showQuantizedModelFolder(modelPath) {
        try {
            // Use the existing file browser modal in view mode
            window.currentBrowserCallback = null; // No callback needed for viewing
            window.currentBrowserType = 'view';
            
            // Set modal title with proper truncation
            const modal = document.getElementById('file-browser-modal');
            const modalTitle = modal.querySelector('#file-browser-title');
            const modelName = modelPath.split('/').pop();
            
            // Truncate model name if too long (keep first 40 chars + ellipsis)
            const maxLength = 40;
            const displayName = modelName.length > maxLength ? 
                modelName.substring(0, maxLength) + '...' : modelName;
            
            modalTitle.innerHTML = `<i class="fas fa-folder-open me-2 text-primary"></i>Quantized: <span class="text-truncate" title="${modelName}">${displayName}</span>`;
            modalTitle.style.maxWidth = '100%';
            modalTitle.style.overflow = 'hidden';
            
            // Hide the select button and update help text since we're just viewing
            const selectBtn = modal.querySelector('#browser-select-btn');
            const helpSelect = modal.querySelector('#browser-help-select');
            selectBtn.style.display = 'none';
            helpSelect.innerHTML = '<small class="text-muted"><i class="fas fa-info-circle me-1"></i>Click on folders to navigate</small>';
            
            // Add up button event handler
            const upBtn = document.getElementById('browser-up-btn');
            const upHandler = () => {
                const currentPath = document.getElementById('browser-current-path').value;
                const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
                this.loadDirectoryContents(parentPath);
            };
            upBtn.addEventListener('click', upHandler);
            
            // Load the directory contents using the existing function
            await this.loadDirectoryContents(modelPath);
            
            // Show the modal
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
            
            // Reset modal when it's hidden to restore normal functionality for other uses
            modal.addEventListener('hidden.bs.modal', () => {
                const selectBtn = modal.querySelector('#browser-select-btn');
                const modalTitle = modal.querySelector('#file-browser-title');
                const helpSelect = modal.querySelector('#browser-help-select');
                selectBtn.style.display = 'block';
                modalTitle.innerHTML = '<i class="fas fa-folder-open me-2 text-primary"></i>Select Directory';
                helpSelect.innerHTML = '<small class="text-muted"><i class="fas fa-info-circle me-1"></i>Click on folders to navigate • Double-click to select and close</small>';
                
                // Remove up button event handler
                upBtn.removeEventListener('click', upHandler);
                
                // Dispose of the modal instance to prevent memory leaks and scroll issues
                bsModal.dispose();
                
                // Additional cleanup to ensure scrolling works
                document.body.classList.remove('modal-open');
                document.body.style.overflow = '';
                document.body.style.paddingRight = '';
                document.documentElement.style.overflow = '';
                document.body.style.overflowY = 'auto';
                
                // Remove any remaining backdrops
                const backdrops = document.querySelectorAll('.modal-backdrop');
                backdrops.forEach(backdrop => backdrop.remove());
            }, { once: true });
            
        } catch (error) {
            console.error('Error showing quantized model folder:', error);
            this.showError('Error showing quantized model folder: ' + error.message);
        }
    }

    /**
     * Load directory contents for the file browser modal
     */
    async loadDirectoryContents(path) {
        const loadingEl = document.getElementById('browser-loading');
        const contentEl = document.getElementById('browser-content');
        const currentPathEl = document.getElementById('browser-current-path');
        const upBtn = document.getElementById('browser-up-btn');
        
        // Show loading
        loadingEl.classList.remove('d-none');
        contentEl.innerHTML = '';
        
        try {
            const response = await fetch(`/api/filesystem/browse?path=${encodeURIComponent(path)}`);
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load directory');
            }
            
            // Update current path
            currentPathEl.value = path;
            
            // Enable/disable up button
            upBtn.disabled = path === '/' || path === '';
            
            // Generate directory listing
            const items = data.items || [];
            contentEl.innerHTML = items.map(item => {
                const icon = item.is_directory ? 'fas fa-folder' : 'fas fa-file';
                return `
                    <div class="list-group-item list-group-item-action d-flex align-items-center" ${item.is_directory ? 'data-path="' + item.path + '" style="cursor: pointer;"' : ''}>
                        <i class="${icon} me-2"></i>
                        <span class="flex-grow-1">${item.name}</span>
                        <small class="text-muted">${item.description}</small>
                    </div>
                `;
            }).join('');
            
            // Add click handlers for directory navigation
            contentEl.querySelectorAll('.list-group-item[data-path]').forEach(item => {
                item.addEventListener('click', () => {
                    const path = item.getAttribute('data-path');
                    this.loadDirectoryContents(path);
                });
            });
            
        } catch (error) {
            console.error('Error loading directory:', error);
            contentEl.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Error loading directory: ${error.message}
                </div>
            `;
        } finally {
            // Hide loading
            loadingEl.classList.add('d-none');
        }
    }

    /**
     * Format file size for display
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorEl = document.getElementById('quantization-error');
        const errorMessage = document.getElementById('quantization-error-message');
        
        if (errorEl && errorMessage) {
            errorMessage.textContent = message;
            errorEl.style.display = 'block';
        }
        
        // Hide after 5 seconds
        setTimeout(() => {
            if (errorEl) errorEl.style.display = 'none';
        }, 5000);
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        const successEl = document.getElementById('quantization-success');
        const successMessage = document.getElementById('quantization-success-message');
        
        if (successEl && successMessage) {
            successMessage.textContent = message;
            successEl.style.display = 'block';
        }
        
        // Hide after 5 seconds
        setTimeout(() => {
            if (successEl) successEl.style.display = 'none';
        }, 5000);
    }

    /**
     * Hide all messages
     */
    hideMessages() {
        const errorEl = document.getElementById('quantization-error');
        const successEl = document.getElementById('quantization-success');
        
        if (errorEl) errorEl.style.display = 'none';
        if (successEl) successEl.style.display = 'none';
    }

    /**
     * Called when the tab is activated
     */
    onActivate() {
        console.log('📱 Quantization tab onActivate() called');
        
        // Reinitialize event listeners to ensure they work
        console.log('🔄 Reinitializing event listeners on tab activation');
        this.initEventListeners();
        
        // Refresh models list
        this.loadAvailableModels();
        this.loadQuantizedModels();
        
        // Check current status
        this.updateProgress();
        
        console.log('✅ Quantization tab activation complete');
    }

    /**
     * Cleanup when component is destroyed
     */
    destroy() {
        this.stopProgressMonitoring();
        this.initialized = false;
    }
}

// Create global instance
const quantizationComponent = new QuantizationComponent(); 