// MLX Training Interface JavaScript
// Import function dependencies from compare.js for unified session cards

class TrainingInterface {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.updateInterval = null;
        this.isTraining = false;
        this.modelLoaded = false;
        this.chatInitialized = false;
        this.checkpointsResetDone = false;
        this.includeHistory = true;  // Default to including history
        this.currentTokenCount = 0;  // Track total token count for the current conversation
        this.promptTokens = 0;       // Track prompt tokens
        this.completionTokens = 0;   // Track completion tokens
        this.conversationTokens = 0; // Track actual conversation content tokens (not cumulative prompt)
        this.lastTokensPerSec = null; // Track latest tokens per second
        this.trainingStartTime = null; // Store training start time for timing calculations
        this.markdownEnabled = true; // Enable markdown rendering by default
        this.lastRefreshTime = 0;    // Throttle dashboard refreshes
        this.refreshThrottleMs = 2000; // Minimum 2 seconds between refreshes
        this.lastCheckpointsUpdate = 0; // Track checkpoint updates
        this.detectedBaseModel = null; // Store auto-detected base model for fusion
        this.isMonitoringPollingActive = false; // Flag to prevent duplicate polling
        
        this.init();
    }
    
    init() {
        // Initialize Socket.IO connection
        this.initSocket();
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Load initial data
        this.loadInitialData();
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        // Initialize tooltips
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
        
        // Initialize learning rate chart
        this.updateLearningRateChart();
        
        // Initialize chat toolbar
        this.initChatToolbar();
        
        // Ensure loading overlay is properly initialized
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            // Make sure it's hidden initially
            loadingOverlay.classList.add('d-none');
        }
        
        // Direct fullscreen button initialization
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        if (fullscreenBtn) {
            console.log('🔍 Initializing fullscreen button with direct handler');
            // Remove any existing click handlers
            fullscreenBtn.removeEventListener('click', () => this.toggleFullscreen());
            // Add new click handler
            fullscreenBtn.addEventListener('click', (e) => {
                console.log('👆 Fullscreen button clicked directly');
                e.preventDefault();
                e.stopPropagation();
                this.toggleFullscreen();
            });
        } else {
            console.error('❌ Fullscreen button not found during init');
        }
    }
    
    initSocket() {
        // DISABLED - Socket connections causing 404 errors and unnecessary with single update approach
        console.log('🚫 Socket.IO connections DISABLED - using single API update approach');
        
        // Simulate connection status as connected since we're using HTTP API
        this.updateConnectionStatus(true);
        
        // No socket initialization to prevent 404 errors
        this.socket = null;
    }
    
    initEventListeners() {
        // Training form
        document.getElementById('training-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startTraining();
        });
        
        // Stop training button in training form
        document.getElementById('stop-training-form-btn').addEventListener('click', () => {
            this.stopTraining();
        });
        
        // Stop training button in monitoring section
        document.getElementById('stop-training-monitor-btn').addEventListener('click', () => {
            this.stopTraining();
        });
        
        // Fuse form submission
        document.getElementById('fuse-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startFusion();
        });
        
        // Start fusion button - show modal first
        document.getElementById('start-fuse-btn').addEventListener('click', () => {
            this.showFusionConfigModal();
        });
        
        // Fusion config modal confirm button
        document.getElementById('fusion-config-confirm').addEventListener('click', () => {
            this.confirmFusion();
        });
        
        // Fusion suffix input - update preview in real-time
        document.getElementById('fusion-suffix').addEventListener('input', () => {
            this.updateFusionPreview();
        });
        
        // Update output preview when adapter selection changes
        document.getElementById('fuse-adapter-select').addEventListener('change', () => {
            this.detectBaseModelForFusion();
            this.updateFuseOutputPreview();
        });
        
        // Stop fusion button
        document.getElementById('stop-fuse-btn').addEventListener('click', () => {
            this.stopFusion();
        });
        
        // Open fusion folder button
        document.getElementById('open-fusion-folder-btn').addEventListener('click', () => {
            this.openFusionFolder();
        });
        
        // Fuse adapter selection change - update output name preview
        const fuseAdapterSelect = document.getElementById('fuse-adapter-select');
        
        if (fuseAdapterSelect) {
            fuseAdapterSelect.addEventListener('change', () => {
                this.detectBaseModelForFusion();
                this.updateFuseOutputPreview();
            });
        }
        
        // Add event listeners for training parameter changes to update estimates
        const parameterInputs = ['batch-size', 'max-iterations', 'max-seq-length', 'save-every', 'eval-every', 'val-fast-pct', 'input-dir'];
        parameterInputs.forEach(inputId => {
            document.getElementById(inputId).addEventListener('input', () => {
                this.updateTrainingEstimates();
                this.updateLearningRateChart();
            });
        });
        
        // Add event listeners for LR-specific parameters
        const lrInputs = ['learning-rate', 'warmup-steps', 'lr-schedule', 'lr-decay-factor'];
        lrInputs.forEach(inputId => {
            document.getElementById(inputId).addEventListener('change', () => {
                this.updateLearningRateChart();
            });
        });
        
        // Fine-tuning type change handler
        document.getElementById('fine-tune-type').addEventListener('change', (e) => {
            const fineTuneType = e.target.value;
            const numLayersInput = document.getElementById('num-layers');
            const numLayersContainer = numLayersInput.closest('.col-md-4');
            const loraParamsSection = document.getElementById('lora-params-section');
            
            // For full fine-tuning, set to -1 and disable
            if (fineTuneType === 'full') {
                numLayersInput.value = '-1';
                numLayersInput.disabled = true;
                loraParamsSection.style.display = 'none';
            } else {
                // For LoRA/DoRA, enable and set to 16 if currently -1
                numLayersInput.disabled = false;
                if (numLayersInput.value === '-1') {
                    numLayersInput.value = '16';
                }
                loraParamsSection.style.display = 'flex';
            }
        });
        
        // Initialize the num-layers state based on current fine-tune-type
        const initFineTuneType = document.getElementById('fine-tune-type').value;
        const loraParamsSection = document.getElementById('lora-params-section');
        if (initFineTuneType === 'full') {
            document.getElementById('num-layers').value = '-1';
            document.getElementById('num-layers').disabled = true;
            loraParamsSection.style.display = 'none';
        } else {
            loraParamsSection.style.display = 'flex';
        }
        
        // Model loading form
        document.getElementById('model-load-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.loadModel();
        });
        
        document.getElementById('unload-model-btn').addEventListener('click', () => {
            this.unloadModel();
        });
        
        // Dashboard button
        document.getElementById('view-dashboard-btn').addEventListener('click', () => {
            this.viewTrainingDashboard();
        });
        
        // When model is selected, check for dashboard
        document.getElementById('test-model-select').addEventListener('change', (e) => {
            this.checkForTrainingDashboard(e.target.value);
        });
        
        // System prompt listener
        document.getElementById('system-prompt').addEventListener('input', () => {
            if (this.modelLoaded) {
                // Update model status
                this.updateModelStatus(
                    document.getElementById('test-model-select').value,
                    document.getElementById('adapter-path').value
                );
                
                // Clear any system prompt content from the chat input field
                const systemPrompt = document.getElementById('system-prompt').value.trim();
                const chatInput = document.getElementById('chat-input');
                if (systemPrompt && chatInput && chatInput.value.includes(systemPrompt)) {
                    chatInput.value = chatInput.value.replace(systemPrompt, '').trim();
                }
            }
        });
        
        // Advanced parameter listeners
        const advancedParams = ['temperature', 'top-p', 'repetition-penalty'];
        advancedParams.forEach(paramId => {
            document.getElementById(paramId).addEventListener('input', () => {
                if (this.modelLoaded) {
                    this.updateModelStatus(
                        document.getElementById('test-model-select').value,
                        document.getElementById('adapter-path').value
                    );
                }
            });
        });
        
        // Context window change listener - update stats immediately
        document.getElementById('max-kv-size').addEventListener('change', () => {
            if (this.modelLoaded && this.conversationTokens > 0) {
                // Update chat stats immediately when context window changes
                this.updateChatStats();
            }
            if (this.modelLoaded) {
                this.updateModelStatus(
                    document.getElementById('test-model-select').value,
                    document.getElementById('adapter-path').value
                );
            }
        });
        
        // Chat input handling
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-message-btn');
        
        if (chatInput && sendBtn) {
            // Auto-resize textarea
            chatInput.addEventListener('input', () => {
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
                
                // Enable/disable send button based on content
                const hasContent = chatInput.value.trim().length > 0;
                sendBtn.disabled = !hasContent;
            });
            
            // Enter key handling
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (chatInput.value.trim() && this.modelLoaded) {
                        this.generateTextFromInput();
                    }
                }
            });
            
            // Send button click
            sendBtn.addEventListener('click', () => {
                if (chatInput.value.trim() && this.modelLoaded) {
                    this.generateTextFromInput();
                }
            });
        }
        
        // Fullscreen toggle functionality
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        if (fullscreenBtn) {
            console.log('🔍 Setting up fullscreen button click handler');
            fullscreenBtn.addEventListener('click', (e) => {
                console.log('👆 Fullscreen button clicked');
                e.preventDefault();
                e.stopPropagation();
                this.toggleFullscreen();
                return false;
            });
        } else {
            console.error('❌ Fullscreen button not found during event setup');
        }
        
        // Escape key to exit fullscreen
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const fullscreenOverlay = document.getElementById('fullscreen-overlay');
                if (fullscreenOverlay && (fullscreenOverlay.style.display === 'flex' || 
                    window.getComputedStyle(fullscreenOverlay).display === 'flex')) {
                    console.log('⌨️ Escape key pressed, exiting fullscreen');
                    this.toggleFullscreen();
                }
            }
        });


        
        // File browser buttons
        document.getElementById('browse-input-dir').addEventListener('click', () => {
            this.openFileBrowser('input');
        });
        
        document.getElementById('browse-output-dir').addEventListener('click', () => {
            this.openFileBrowser('output');
        });
        
        // Tab switching - ONLY attach to monitoring tab to prevent duplicates
        const monitoringTab = document.querySelector('#monitoring-tab');
        if (monitoringTab && !monitoringTab.hasAttribute('data-listener-attached')) {
            monitoringTab.addEventListener('shown.bs.tab', (e) => {
                if (e.target.id === 'monitoring-tab') {
                    console.log('🔄 Switched to monitoring tab - checking training status');
                    // Always do one check to update the display and start polling if needed
                    this.checkTrainingStatusOnce();
                }
            });
            monitoringTab.setAttribute('data-listener-attached', 'true');
        }
        
        // Training tab switching - initialize search when switching to training tab
        const trainingTab = document.querySelector('#training-tab');
        if (trainingTab && !trainingTab.hasAttribute('data-search-listener-attached')) {
            trainingTab.addEventListener('shown.bs.tab', (e) => {
                if (e.target.id === 'training-tab') {
                    console.log('🔄 Switched to training tab - initializing search');
                    setTimeout(() => {
                        if (typeof window.initializeUnifiedSearch === 'function') {
                            window.initializeUnifiedSearch();
                        }
                        if (typeof window.applySearchToAllContainers === 'function') {
                            window.applySearchToAllContainers();
                        }
                    }, 100);
                }
            });
            trainingTab.setAttribute('data-search-listener-attached', 'true');
        }
        
        // Quantization tab activation - separate listener
        const quantizationTab = document.querySelector('#quantization-tab');
        if (quantizationTab && !quantizationTab.hasAttribute('data-listener-attached')) {
            quantizationTab.addEventListener('shown.bs.tab', (e) => {
                if (e.target.id === 'quantization-tab') {
                    // Activate quantization component when tab is shown
                    if (typeof quantizationComponent !== 'undefined' && quantizationComponent.onActivate) {
                        quantizationComponent.onActivate();
                    }
                }
            });
            quantizationTab.setAttribute('data-listener-attached', 'true');
        }
        
        // Stop polling when switching away from monitoring tab
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            if (tab.id !== 'monitoring-tab' && !tab.hasAttribute('data-stop-listener-attached')) {
                tab.addEventListener('shown.bs.tab', (e) => {
                    if (e.target.id !== 'monitoring-tab') {
                        this.stopMonitoringPolling();
                    }
                });
                tab.setAttribute('data-stop-listener-attached', 'true');
            }
        });
        
        // Training session selection
        document.getElementById('load-session-btn').addEventListener('click', () => {
            this.loadSelectedTrainingSession();
        });
        
        // View session logs button
        document.getElementById('view-logs-btn').addEventListener('click', () => {
            this.viewSelectedSessionLogs();
        });
        

        
        // Publish selected checkpoint from monitoring tab
        document.getElementById('publish-selected-checkpoint-btn').addEventListener('click', () => {
            const checkpointPath = document.getElementById('checkpoint-select').value;
            if (!checkpointPath) {
                this.showAlert('Select a checkpoint first.', 'warning');
                return;
            }
            this.publishCheckpoint(encodeURIComponent(checkpointPath));
        });
        
        // Copy model and adapter paths
        document.getElementById('copy-model-path-btn').addEventListener('click', () => {
            const modelPath = document.getElementById('test-model-select').value;
            if (modelPath) {
                navigator.clipboard.writeText(modelPath)
                    .then(() => this.showTooltip('copy-model-path-btn', 'Copied!'))
                    .catch(err => this.showAlert('Failed to copy: ' + err, 'danger'));
            } else {
                this.showAlert('No model selected', 'warning');
            }
        });
        
        document.getElementById('open-model-folder-btn').addEventListener('click', () => {
            const modelSelect = document.getElementById('test-model-select');
            const selectedOption = modelSelect.selectedOptions[0];
            
            if (selectedOption) {
                // Use the stored path if available, otherwise fall back to the value
                const modelPath = selectedOption.getAttribute('data-path') || selectedOption.value;
                this.openModelFolder(modelPath);
            } else {
                this.showAlert('No model selected', 'warning');
            }
        });
        
        document.getElementById('copy-adapter-path-btn').addEventListener('click', () => {
            const adapterPath = document.getElementById('adapter-path').value;
            if (adapterPath) {
                navigator.clipboard.writeText(adapterPath)
                    .then(() => this.showTooltip('copy-adapter-path-btn', 'Copied!'))
                    .catch(err => this.showAlert('Failed to copy: ' + err, 'danger'));
            } else {
                this.showAlert('No adapter selected', 'warning');
            }
        });
        
        // When adapter is selected, check for dashboard and optionally clear base model
        document.getElementById('adapter-path').addEventListener('change', (e) => {
            const modelSelect = document.getElementById('test-model-select');
            const modelPath = modelSelect.value;
            const adapterPath = e.target.value;
            
            console.log(`🔄 Adapter selection changed:`);
            console.log(`   📂 Selected adapter: ${adapterPath}`);
            console.log(`   🤖 Current base model: ${modelPath}`);
            
            // If an adapter is selected, check its path for a dashboard
            if (adapterPath) {
                this.checkForTrainingDashboard(adapterPath);
                
                // Optionally clear base model selection to indicate adapter-only loading
                // (User can still select both if they want to override)
                // modelSelect.value = '';  // Uncomment to auto-clear base model
            } else if (modelPath) {
                // If no adapter but model is selected, check model path
                this.checkForTrainingDashboard(modelPath);
            }
        });
        
        // Handle form submission to prevent default behavior and preserve selections
        document.getElementById('model-load-form').addEventListener('submit', (e) => {
            e.preventDefault();
            console.log('🚫 Form submission prevented - using button click handler instead');
            return false;
        });
        
        // Handle load model button click
        document.getElementById('load-model-btn').addEventListener('click', (e) => {
            e.preventDefault();
            console.log('🔘 Load model button clicked');
            this.loadModel();
        });
        
        // Handle reset config button click
        document.getElementById('reset-config-btn').addEventListener('click', (e) => {
            e.preventDefault();
            console.log('🔄 Reset config button clicked');
            this.resetModelConfiguration();
        });
    }
    
    async loadInitialData() {
        await this.loadModels();
        await this.loadCheckpoints();
        // Training status will be checked when monitoring tab is visited
        this.updateTrainingEstimates();
        this.updateLearningRateChart();
        await this.loadFuseModels();
    }
    
    startPeriodicUpdates() {
        // DISABLED: No automatic polling - only manual tab-based polling
        console.log('🔄 Periodic updates disabled - polling controlled by tab switching');
    }
    
    startMonitoringPolling() {
        // Prevent duplicate polling
        if (this.isMonitoringPollingActive || this.updateInterval) {
            console.log('🔄 Monitoring polling already active');
            return;
        }
        
        console.log('🔄 Starting monitoring polling (10s interval)');
        this.isMonitoringPollingActive = true;
        this.updateInterval = setInterval(() => {
            this.performMonitoringUpdate();
        }, 10000);
        
        // Immediate update when starting
        this.performMonitoringUpdate();
    }
    
    stopMonitoringPolling() {
        if (this.updateInterval) {
            console.log('⏹️ Stopping monitoring polling');
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        this.isMonitoringPollingActive = false;
    }
    
    async performMonitoringUpdate() {
        try {
            // Only make API call if monitoring tab is active AND training is running
            const monitoringTabButton = document.querySelector('#monitoring-tab');
            const isMonitoringTabActive = monitoringTabButton && monitoringTabButton.classList.contains('active');
            
            if (!isMonitoringTabActive) {
                console.log('📋 Monitoring tab not active - stopping polling');
                this.stopMonitoringPolling();
                return;
            }
            
            console.log('📊 Updating monitoring dashboard');
            
            // Make the API call
            const response = await fetch('/api/dashboard/realtime');
            const data = await response.json();
            
            this.isTraining = data.active || false;
            this.updateTrainingButtons(this.isTraining);
            
            // Expose training status globally for other components
            window.isTrainingActive = this.isTraining;
            
            // Keep polling while on monitoring tab (regardless of training status)
            // Polling will be stopped when switching away from monitoring tab
            
            // Update dashboard data
            if (data.current_values) {
                this.updateAllFields(data.current_values, data.config);
                this.updateTrainingStatus(data);
                if (data.charts) {
                    this.renderCharts(data.charts);
                }
            } else {
                this.updateTrainingStatus({active: this.isTraining});
            }
            
            // Load checkpoints every 60 seconds (only when monitoring)
            if (Date.now() - (this.lastCheckpointsUpdate || 0) > 60000) {
                this.loadCheckpoints();
                this.lastCheckpointsUpdate = Date.now();
            }
            
        } catch (error) {
            console.error('Error in monitoring update:', error);
        }
    }
    
    // NEW: Single status check for startup and training tab
    async checkTrainingStatusOnce() {
        try {
            // If we're on monitoring tab, just start polling (which will make the API call)
            const monitoringTabButton = document.querySelector('#monitoring-tab');
            const isMonitoringTabActive = monitoringTabButton && monitoringTabButton.classList.contains('active');
            
            if (isMonitoringTabActive && !this.isMonitoringPollingActive) {
                console.log('🔄 On monitoring tab - starting polling (no duplicate API call)');
                this.startMonitoringPolling();
                return; // Exit early to avoid duplicate API call
            }
            
            // Only make API call if NOT on monitoring tab
            console.log('🔍 Checking training status once');
            const response = await fetch('/api/dashboard/realtime');
            const data = await response.json();
            
            this.isTraining = data.active || false;
            this.updateTrainingButtons(this.isTraining);
            this.updateTrainingStatus({active: this.isTraining});
            
            // Expose training status globally for other components
            window.isTrainingActive = this.isTraining;
            
            if (!isMonitoringTabActive) {
                console.log('📋 Not on monitoring tab - no polling needed');
            } else if (this.isMonitoringPollingActive) {
                console.log('🔄 Polling already active');
            }
        } catch (error) {
            console.error('Error checking training status:', error);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        const indicator = statusElement.querySelector('.status-indicator');
        
        if (connected) {
            indicator.className = 'status-indicator status-running';
            statusElement.innerHTML = '<span class="status-indicator status-running"></span>Connected';
        } else {
            indicator.className = 'status-indicator status-stopped';
            statusElement.innerHTML = '<span class="status-indicator status-stopped"></span>Disconnected';
        }
    }
    
    async loadModels() {
        try {
            // Load all models (including CPT) for training dropdown
            const allModelsResponse = await fetch('/api/models');
            const allModelsData = await allModelsResponse.json();
            
            // Both training and testing tabs now use the same model list (no CPT models)
            if (allModelsData.models) {
                // Update both training and testing dropdowns with the same models
                this.updateModelDropdown('model-select', allModelsData.models);
                this.updateModelDropdown('test-model-select', allModelsData.models);
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showAlert('Failed to load models', 'danger');
        }
    }
    
    updateModelDropdown(selectId, models) {
        const select = document.getElementById(selectId);
        if (!select) return;
        
        // Remember currently selected value so we can restore it after refresh
        const previousValue = select.value;
        
        // Clear existing options except the first one
        const firstOption = select.firstElementChild;
        select.innerHTML = '';
        if (firstOption) {
            select.appendChild(firstOption);
        }
        
        // Add models with simplified icons
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            
            // Store the actual path for folder opening (if available)
            if (model.path) {
                option.setAttribute('data-path', model.path);
            }
            
            // Determine if this is a base model using same logic as isCurrentModelBase
            const modelName = model.name.toLowerCase();
            
            // Special handling for Qwen models: they are instruct by default EXCEPT if "base" is in the name
            let isBaseModel;
            if (modelName.includes('qwen')) {
                // For Qwen models, check if it's explicitly marked as base
                isBaseModel = modelName.includes('base');
                console.log(`🔍 Qwen model '${model.name}' detected as ${isBaseModel ? 'BASE' : 'INSTRUCT'}`);
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
            
            // Simplified icons: only 2 types
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
        
        // Restore previous selection if it still exists
        if (previousValue && [...select.options].some(o => o.value === previousValue)) {
            select.value = previousValue;
            console.log(`🔄 Restored model selection for ${selectId}: ${previousValue}`);
        }
    }
    
    async loadCheckpoints() {
        try {
            // Load training sessions
            const sessionsResponse = await fetch('/api/training/sessions');
            const sessionsData = await sessionsResponse.json();
            
            if (sessionsData.success && sessionsData.training_sessions) {
                this.updateTrainingSessionsList(sessionsData.training_sessions);
                this.updateAdapterSelect(sessionsData.training_sessions);
            } else {
                console.log('No training sessions found');
                this.updateTrainingSessionsList([]);
                this.updateAdapterSelect([]);
            }
        } catch (error) {
            console.error('Error loading training sessions:', error);
            this.updateTrainingSessionsList([]);
            this.updateAdapterSelect([]);
        }
    }
    
    updateTrainingSessionsList(trainingSessions) {
        // Update the dropdown in monitoring tab
        this.updateTrainingSessionsDropdown(trainingSessions);
        
        const container = document.getElementById('checkpoints-list');
        
        if (trainingSessions.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-graduation-cap fa-2x mb-3"></i>
                    <p>No training sessions found</p>
                    <small>Start a training session to see checkpoints here</small>
                </div>
            `;
            return;
        }

        // Use the unified session card generator from the Compare tab
        const sessionsHTML = trainingSessions.map(session => {
            return window.generateSessionCard(session, {
                isCompareTab: false,
                showSelectionFeatures: false,
                containerId: 'training-session-card'
            });
        }).join('');
        
        // Use the same compact layout as Compare tab
        container.innerHTML = sessionsHTML;
        
        // Populate loss badges using the new optimized batch function
        console.log('Populating loss badges for Training tab session cards using batch API');
        setTimeout(() => {
            if (typeof window.populateLossBadges === 'function') {
                window.populateLossBadges(trainingSessions);
            } else {
                console.warn('populateLossBadges function not found on window');
            }
        }, 100); // Small delay to ensure DOM rendering
        
        // Initialize unified search functionality for training tab
        setTimeout(() => {
            if (typeof window.initializeUnifiedSearch === 'function') {
                window.initializeUnifiedSearch();
            } else if (typeof initializeUnifiedSearch === 'function') {
                initializeUnifiedSearch();
            }
            
            // Apply current search term to the training tab
            if (typeof window.applySearchToAllContainers === 'function') {
                window.applySearchToAllContainers();
            }
        }, 150); // Delay to ensure DOM is ready
    }
    
    updateTrainingSessionsDropdown(trainingSessions) {
        const select = document.getElementById('training-session-select');
        
        // Remember currently selected value so we can restore it after refresh
        const previousValue = select.value;
        
        select.innerHTML = '<option value="">Current training session</option>';
        
        if (trainingSessions && trainingSessions.length > 0) {
            trainingSessions.forEach(session => {
                const option = document.createElement('option');
                option.value = session.log_file || session.session_id;
                
                // Format date with both date and time to avoid conflicts
                // Use start_time from training data if available, otherwise fall back to created timestamp
                let sessionDate;
                if (session.start_time) {
                    sessionDate = new Date(session.start_time);
                } else {
                    sessionDate = new Date(session.created * 1000);
                }
                const dateStr = sessionDate.toLocaleDateString();
                const timeStr = sessionDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                option.textContent = `${session.session_name} (${dateStr} ${timeStr})`;
                select.appendChild(option);
            });
        }
        
        // Restore previous selection if still present
        if (previousValue && [...select.options].some(o => o.value === previousValue)) {
            select.value = previousValue;
        }
    }
    
    async loadSelectedTrainingSession() {
        const select = document.getElementById('training-session-select');
        const logFile = select.value;
        
        console.log('🔄 Loading training session:', logFile);
        
        if (!logFile) {
            // No session selected, show current training
            console.log('📊 No session selected, no need to check training status again');
            // Status already checked when switching to monitoring tab
            return;
        }
        
        this.showLoading('Loading training session data...');
        
        try {
            console.log('📡 Sending request to /api/dashboard/historical with log_file:', logFile);
            
            const response = await fetch('/api/dashboard/historical', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: logFile })
            });
            
            console.log('📥 Response status:', response.status);
            const data = await response.json();
            console.log('📊 Response data:', data);
            
            if (!data.success) {
                this.showAlert(data.error || 'Failed to load training session', 'danger');
            } else {
                console.log('✅ Successfully loaded training session data');
                
                // Reset the checkpointsResetDone flag when loading historical data
                this.checkpointsResetDone = false;
                
                // Display historical charts and metrics
                console.log('📈 Displaying historical charts with data:', data);
                this.displayHistoricalCharts(data);
                
                // Update metrics
                if (data.summary) {
                    console.log('📊 Updating training metrics with summary:', data.summary);
                    this.updateAllFields(data.summary, data.summary.config || {});
                    
                    // Populate the checkpoint-select dropdown with all available checkpoints
                    console.log('🎯 Populating checkpoint select with summary:', data.summary);
                    this.populateCheckpointSelect(data.summary);
                }
                
                this.showAlert('Training session loaded successfully', 'success');
            }
        } catch (error) {
            console.error('❌ Error loading training session:', error);
            this.showAlert('Error loading training session: ' + error.message, 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    async viewSelectedSessionLogs() {
        const select = document.getElementById('training-session-select');
        const logFile = select.value;
        
        if (!logFile) {
            this.showAlert('Please select a training session', 'warning');
            return;
        }
        
        // The log file path is already stored in the select value
        this.viewTrainingLogs(logFile);
    }
    
    updateAdapterSelect(trainingSessions) {
        const select = document.getElementById('adapter-path');
        
        // Remember currently selected value so we can restore it after refresh
        const previousValue = select.value;
        
        select.innerHTML = '<option value="">No adapter (base model only)</option>';
        
        // Load checkpoints separately from the checkpoints API
        this.loadAdapterCheckpoints();
        
        // Restore previous selection if still present after loading checkpoints
        // We'll do this in loadAdapterCheckpoints() since it's async
        this.previousAdapterSelection = previousValue;
    }
    
    async loadAdapterCheckpoints() {
        try {
            const response = await fetch('/api/checkpoints');
            const data = await response.json();
            
            if (data.success && data.checkpoints) {
                const select = document.getElementById('adapter-path');
                
                data.checkpoints.forEach(checkpoint => {
                    const option = document.createElement('option');
                    option.value = checkpoint.path;
                    
                    // Format the display name with adapter type info
                    const modelName = checkpoint.model || 'Unknown';
                    const iteration = checkpoint.iteration || 0;
                    const size = checkpoint.size ? `${checkpoint.size.toFixed(1)}MB` : '';
                    
                    // Extract training type and fine-tune type if available
                    let typeInfo = '';
                    if (checkpoint.path) {
                        const pathParts = checkpoint.path.split('/');
                        // Look for CPT or IFT in path
                        if (pathParts.some(part => part.includes('cpt'))) {
                            typeInfo = ' [CPT]';
                        } else if (pathParts.some(part => part.includes('ift'))) {
                            typeInfo = ' [IFT]';
                        }
                        
                        // Look for fine-tune type hints in path
                        if (pathParts.some(part => part.includes('lora'))) {
                            typeInfo += ' LoRA';
                        } else if (pathParts.some(part => part.includes('dora'))) {
                            typeInfo += ' DoRA';
                        }
                    }
                    
                    option.textContent = `${modelName}${typeInfo} - iter ${iteration} ${size}`;
                    select.appendChild(option);
                });
                
                // Restore previous selection if it was saved and still exists
                if (this.previousAdapterSelection && [...select.options].some(o => o.value === this.previousAdapterSelection)) {
                    select.value = this.previousAdapterSelection;
                    console.log(`🔄 Restored adapter selection: ${this.previousAdapterSelection}`);
                }
                
                // Clear the stored selection
                this.previousAdapterSelection = null;
            }
        } catch (error) {
            console.error('Error loading adapter checkpoints:', error);
        }
    }
    
    async checkTrainingStatus() {
        // Redirect to the new single-check method
        return this.checkTrainingStatusOnce();
    }
    
    /**
     * Start training with current configuration
     */
    async startTraining() {
        // CRITICAL FIX: Prevent duplicate training submissions
        if (this.isTraining) {
            this.showAlert('Training is already in progress', 'warning');
            return;
        }
        
        // Set flag immediately to prevent race conditions
        this.isTraining = true;
        
        // Get form values
        const config = {
            model_name: document.getElementById('model-select').value, // Changed from 'model' to 'model_name'
            output_dir: document.getElementById('output-dir').value,
            input_dir: document.getElementById('input-dir').value,
            batch_size: parseInt(document.getElementById('batch-size').value),
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            max_iterations: parseInt(document.getElementById('max-iterations').value),
            warmup_steps: parseInt(document.getElementById('warmup-steps').value),
            max_seq_length: parseInt(document.getElementById('max-seq-length').value),
            weight_decay: parseFloat(document.getElementById('weight-decay').value),
            lr_decay_factor: parseFloat(document.getElementById('lr-decay-factor').value),
            fine_tune_type: document.getElementById('fine-tune-type').value,
            num_layers: parseInt(document.getElementById('num-layers').value),
            lr_schedule: document.getElementById('lr-schedule').value,
            save_every: parseInt(document.getElementById('save-every').value),
            steps_per_eval: parseInt(document.getElementById('eval-every').value), // Changed from 'eval_every' to 'steps_per_eval'
            validation_fast_pct: parseFloat(document.getElementById('val-fast-pct').value),
            validation_split: parseFloat(document.getElementById('validation-split').value),
            enable_early_stopping: document.getElementById('enable-early-stopping').checked,
            
            // Add LoRA parameters (only used if fine_tune_type is 'lora' or 'dora')
            lora_rank: parseInt(document.getElementById('lora-rank').value),
            lora_scale: parseFloat(document.getElementById('lora-scale').value),
            lora_dropout: parseFloat(document.getElementById('lora-dropout').value),
            lora_modules: document.getElementById('lora-modules').value,
            
            // Add missing required parameters with default values
            data_dir: 'data/pretraining',
            max_tokens_per_file: 1000000,
            max_checkpoints: 5,
            data_mixture_ratio: 0.95,
            overfitting_threshold: 0.30,
            early_stopping_patience: 3,
            min_loss_improvement: 0.001,
            steps_per_report: 5,
            use_lr_rewarming: true,
            seed: 42,
            val_batches: null
        };
        
        if (!config.model_name) { // Changed from 'model' to 'model_name'
            this.showAlert('Please select a model', 'warning');
            return;
        }
        
        console.log('Training config being sent to API:', config); // Add logging to debug
        
        this.showLoading('Starting training...');
        
        try {
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            const data = await response.json();
            
            // Always hide loading first
            this.hideLoading();
            
            if (data.success) {
                // Training started successfully - isTraining already set to true above
                this.updateTrainingButtons(true);
                this.showAlert('Training started successfully!', 'success');
                
                // Reset best checkpoints display when starting a new training session
                const bestCheckpointsContainer = document.getElementById('best-checkpoints');
                bestCheckpointsContainer.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-chart-line fa-2x mb-3"></i>
                        <p>Training in progress. Best checkpoints will appear here.</p>
                    </div>
                `;
                
                // Reset the checkpoints flag to ensure we don't keep old data
                this.checkpointsResetDone = false;
                
                // Switch to monitoring tab and start polling
                setTimeout(() => {
                    const monitoringTab = new bootstrap.Tab(document.getElementById('monitoring-tab'));
                    monitoringTab.show();
                    // Start polling since training just started
                    this.startMonitoringPolling();
                }, 1000);
            } else {
                // Training failed to start - reset flag
                this.isTraining = false;
                this.showAlert(data.error || 'Failed to start training', 'danger');
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.hideLoading();
            // Training failed to start - reset flag  
            this.isTraining = false;
            this.showAlert('Error starting training', 'danger');
        }
    }
    
    async stopTraining() {
        this.showLoading('Stopping training...');
        
        try {
            const response = await fetch('/api/training/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.isTraining = false;
                this.updateTrainingButtons(false);
                this.showAlert(data.message || 'Training stopped successfully', 'info');
                
                // Force a status update to refresh the UI
                await this.checkTrainingStatusOnce();
            } else {
                this.showAlert(data.error || 'Failed to stop training', 'danger');
                console.error('Failed to stop training:', data.error);
            }
        } catch (error) {
            console.error('Error stopping training:', error);
            this.showAlert('Error stopping training: ' + (error.message || 'Unknown error'), 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    updateTrainingButtons(isTraining) {
        // Update training form buttons
        document.getElementById('start-training-btn').disabled = isTraining;
        
        const formStopBtn = document.getElementById('stop-training-form-btn');
        if (formStopBtn) {
            formStopBtn.disabled = !isTraining;
        }
        
        // Update monitoring section stop button visibility
        const monitoringStopBtn = document.getElementById('stop-training-monitor-btn');
        if (monitoringStopBtn) {
            if (isTraining) {
                monitoringStopBtn.style.display = 'inline-block';
            } else {
                monitoringStopBtn.style.display = 'none';
            }
        }
    }
    
    updateTrainingStatus(data) {
        const container = document.getElementById('training-status');
        
        if (data.active) {
            // Store training start time for timing calculations
            this.trainingStartTime = data.start_time;
            
            // Default status to 'running' if not provided
            const status = data.status || 'running';
            const statusClass = status === 'running' ? 'status-running' : 
                              status === 'completed' ? 'status-completed' : 'status-stopped';
            
            container.innerHTML = `
                <div class="mb-3">
                    <span class="status-indicator ${statusClass}"></span>
                    <strong>${status.toUpperCase()}</strong>
                </div>
                <div class="mb-2">
                    <small class="text-muted">Model:</small><br>
                    <strong>${data.config.model_name || data.config.model}</strong>
                </div>
                <div class="mb-2">
                    <small class="text-muted">Started:</small><br>
                    <strong>${data.start_time ? new Date(data.start_time).toLocaleString() : 'N/A'}</strong>
                </div>
                <div class="mb-2">
                    <small class="text-muted">Fine-tune Type:</small> <strong>${data.config.fine_tune_type || 'full'}</strong><br>
                    <small class="text-muted">Batch Size:</small> <strong>${data.config.batch_size}</strong><br>
                    <small class="text-muted">Learning Rate:</small> <strong>${data.config.learning_rate}</strong><br>
                    <small class="text-muted">Max Iterations:</small> <strong>${data.config.max_iterations}</strong><br>
                    <small class="text-muted">Save Every:</small> <strong>${data.config.save_every || 50} steps</strong>
                </div>
            `;
        } else {
            // Clear training start time when no training is active
            this.trainingStartTime = null;
            
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-clock fa-2x mb-3"></i>
                    <p>No training in progress</p>
                </div>
            `;
        }
    }
    


    updateAllFields(values, config) {
        console.log('=== UPDATE ALL FIELDS ===');
        console.log('values:', values);
        console.log('config:', config);
        
        // Update basic metrics
        this.setElement('current-iteration', values.iteration || '-');
        this.setElement('epoch-done', values.epoch || '-');
        this.setElement('train-loss', this.formatNumber(values.train_loss));
        this.setElement('val-loss', this.formatNumber(values.val_loss));
        this.setElement('perplexity', this.formatNumber(values.val_perplexity || values.train_perplexity));
        this.setElement('tokens-per-sec', values.tokens_per_sec ? Math.round(values.tokens_per_sec) : '-');
        this.setElement('trained-tokens', values.trained_tokens ? values.trained_tokens.toLocaleString() : '-');
        this.setElement('memory-usage', this.formatNumber(values.peak_memory_gb, 1));
        
        // Update time and progress
        if (values.elapsed_minutes !== undefined) {
            this.setElement('elapsed-time', `E.Time: ${Math.round(values.elapsed_minutes)}m`);
        }
        if (values.eta_minutes !== undefined) {
            this.setElement('eta-time', `R.Time: ${Math.round(values.eta_minutes)}m`);
        }
        
        // Calculate and update progress percentage
        if (values.elapsed_minutes !== undefined && values.eta_minutes !== undefined) {
            const totalTime = values.elapsed_minutes + values.eta_minutes;
            const progressPercent = totalTime > 0 ? (values.elapsed_minutes / totalTime * 100) : 0;
            this.setElement('progress-text', `${progressPercent.toFixed(1)}%`);
            
            // Update progress bar
            const progressBar = document.getElementById('progress-bar');
            if (progressBar) {
                progressBar.style.width = `${progressPercent}%`;
                progressBar.setAttribute('aria-valuenow', progressPercent);
            }
        }
        
        // Update config values directly from config object
        if (config) {
            console.log('CONFIG EXISTS - updating config fields');
            console.log('config.learning_rate:', config.learning_rate);
            console.log('config.warmup_steps:', config.warmup_steps);
            console.log('config.lr_decay_factor:', config.lr_decay_factor);
            console.log('config.weight_decay:', config.weight_decay);
            
            this.setElement('display-learning-rate', config.learning_rate ? config.learning_rate.toExponential(2) : '-');
            this.setElement('display-warmup-steps', config.warmup_steps || '-');
            this.setElement('display-lr-decay-factor', config.lr_decay_factor || '-');
            this.setElement('display-weight-decay', config.weight_decay || '-');
        } else {
            console.log('NO CONFIG PROVIDED!');
        }
    }
    
    setElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            console.log(`Setting ${id} = ${value}`);
            element.textContent = value;
        } else {
            console.log(`ELEMENT NOT FOUND: ${id}`);
        }
    }
    
    formatNumber(value, decimals = 3) {
        if (value === null || value === undefined) return '-';
        if (typeof value === 'number') {
            return value.toFixed(decimals);
        }
        return value;
    }


    
    updateBestCheckpoints(checkpoints) {
        const container = document.getElementById('best-checkpoints');
        
        console.log('Updating best checkpoints:', checkpoints);
        
        if (!checkpoints || checkpoints.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-chart-line fa-2x mb-3"></i>
                    <p>No best checkpoints available</p>
                </div>
            `;
            return;
        }
        
        // Filter out checkpoints without iterations (invalid data)
        const validCheckpoints = checkpoints.filter(cp => cp && cp.iteration !== undefined);
        
        if (validCheckpoints.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-chart-line fa-2x mb-3"></i>
                    <p>No valid checkpoint data available</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = validCheckpoints.map((checkpoint, index) => {
            // Handle missing values gracefully
            const valLoss = checkpoint.val_loss !== undefined ? 
                `Val Loss: <strong>${checkpoint.val_loss.toFixed(4)}</strong>` : 
                `Val Loss: <strong>N/A</strong>`;
                
            const valPpl = checkpoint.val_perplexity !== undefined ? 
                `Perplexity: <strong>${checkpoint.val_perplexity.toFixed(2)}</strong>` : 
                `Perplexity: <strong>N/A</strong>`;
                
            const selectionReason = checkpoint.selection_reason || 'Best checkpoint';
            const iteration = checkpoint.iteration || 'Unknown';
            
            return `
            <div class="checkpoint-item">
                <div class="d-flex align-items-center">
                    <div class="checkpoint-rank">${index + 1}</div>
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between">
                            <div>
                                <strong>Iteration ${iteration}</strong>
                                <br>
                                <small class="text-muted">${selectionReason}</small>
                            </div>
                            <div class="text-end">
                                <small>${valLoss}</small><br>
                                <small>${valPpl}</small><br>
                                <button type="button" 
                                   class="btn btn-outline-primary btn-sm mt-1 publish-checkpoint-btn" 
                                   onclick="event.preventDefault(); trainingInterface.publishCheckpoint('${checkpoint.path || ''}');"
                                   style="cursor: pointer;" 
                                   ${checkpoint.path ? '' : 'disabled'}>
                                    <i class="fas fa-cloud-upload-alt"></i> Publish
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            `;
        }).join('');
    }
    
    async refreshDashboard() {
        // DISABLED: All dashboard updates now go through performMonitoringUpdate
        console.log('🚫 refreshDashboard() disabled - use performMonitoringUpdate() instead');
        return;
    }
    
    renderCharts(charts) {
        // Define chart IDs and their corresponding data in the charts object
        const chartConfigs = [
            { id: 'loss-chart', key: 'loss' },
            { id: 'perplexity-chart', key: 'perplexity' },
            { id: 'lr-chart', key: 'learning_rate' },
            { id: 'speed-chart', key: 'speed' }
        ];
        
        // Clear all charts first
        chartConfigs.forEach(config => {
            const chartElement = document.getElementById(config.id);
            if (chartElement) {
                // Clear the chart completely
                chartElement.innerHTML = '';
            }
        });
        
        // If no charts data, show empty state
        if (!charts) {
            console.log('No chart data provided - charts cleared');
            return;
        }
        
        // Render each chart if the element exists and data is available
        chartConfigs.forEach(config => {
            const chartElement = document.getElementById(config.id);
            const chartData = charts[config.key];
            
            if (chartElement) {
                if (chartData && chartData.data && chartData.data.length > 0) {
                    // Check if the chart data actually has points
                    const hasData = chartData.data.some(trace => 
                        trace.x && trace.y && trace.x.length > 0 && trace.y.length > 0
                    );
                    
                    if (hasData) {
                        console.log(`Rendering chart ${config.id} with data:`, chartData);
                        
                        try {
                            Plotly.newPlot(config.id, chartData.data, chartData.layout || {}, {
                                responsive: true,
                                displayModeBar: false
                            });
                        } catch (error) {
                            console.error(`Error rendering chart ${config.id}:`, error);
                        }
                    } else {
                        console.log(`Chart ${config.id} has no data points - keeping empty`);
                    }
                } else {
                    console.log(`Chart ${config.id} has no data - keeping empty`);
                }
            }
        });
    }
    
    updateCharts(data) {
        // Update charts with real-time data
        if (!data || data.error) return;
        
        // Don't call refreshDashboard here - this creates a loop
        // The data should already be processed in updateTrainingMetrics
        console.log('Charts updated with real-time data');
    }
    
    async loadModel() {
        // Prevent multiple simultaneous load operations
        if (this.isLoadingModel) {
            console.log('🚫 Model loading already in progress, ignoring click');
            return;
        }
        
        const modelSelect = document.getElementById('test-model-select');
        const adapterSelect = document.getElementById('adapter-path');
        const systemPrompt = document.getElementById('system-prompt');
        
        const model = modelSelect.value;
        const adapter = adapterSelect.value;
        const prompt = systemPrompt.value;
        
        console.log(`🚀 loadModel() called with:`);
        console.log(`   🤖 Base model: "${model}"`);
        console.log(`   📂 Adapter: "${adapter}"`);
        
        if (!model && !adapter) {
            this.showAlert('Please select a base model or adapter', 'warning');
            return;
        }
        
        // Set loading state immediately
        this.isLoadingModel = true;
        
        // Show loading and disable UI immediately
        this.showLoading('Preparing to load model...');
        this.setLoadingState(true);
        
        try {
            // Step 1: Force unload with visible feedback
            console.log('🔄 Step 1: Unloading any existing model...');
            this.showLoading('Unloading existing model...');
            
            const unloadResponse = await fetch('/api/model/unload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (unloadResponse.ok) {
                console.log('✅ Model unloaded successfully');
            } else {
                console.log('⚠️ Unload failed, continuing anyway');
            }
            
            // Small delay to ensure unload is processed
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Step 2: Load new model with visible feedback
            console.log('🔄 Step 2: Loading new model...');
            this.showLoading('Loading model...');
            
            const actualAdapterPath = adapter && adapter.endsWith('.safetensors') 
                ? adapter.substring(0, adapter.lastIndexOf('/'))
                : adapter;
            
            const startTime = Date.now();
            console.log(`⏱️ Starting model load at: ${startTime}`);
            
            const response = await fetch('/api/model/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: model,
                    adapter_path: actualAdapterPath,
                    system_prompt: prompt
                })
            });
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            console.log(`⏱️ Model load completed in: ${duration}ms`);
            
            const data = await response.json();
            console.log(`📡 Load response:`, data);
            
            if (data.success) {
                this.modelLoaded = true;
                this.updateModelButtons(true);
                
                const actualModel = data.model_name || model;
                const actualAdapter = data.adapter_path || adapter;
                this.updateModelStatus(actualModel, actualAdapter);
                
                const loadingTime = data.loading_time ? ` (${data.loading_time}s)` : '';
                if (!model && adapter && data.model_name) {
                    this.showAlert(
                        `✅ Adapter loaded successfully!${loadingTime}\n` +
                        `Auto-detected base model: ${data.model_name}\n` +
                        `Adapter: ${adapter.split('/').pop()}`, 
                        'success'
                    );
                } else {
                    this.showAlert(`Model loaded successfully${loadingTime}`, 'success');
                }
                
                this.checkForTrainingDashboard(model);
                
                if (!this.chatInitialized) {
                    this.initChatToolbar();
                    this.chatInitialized = true;
                }
                
                document.getElementById('chat-history').innerHTML = `
                    <div class="text-center text-muted mt-3 mb-3">
                        <i class="fas fa-robot fa-2x mb-3"></i>
                        <p>Model ready! Type a message below to start the conversation.</p>
                    </div>
                `;
                
                this.currentTokenCount = 0;
                this.promptTokens = 0;
                this.completionTokens = 0;
                this.conversationTokens = 0;
                this.updateChatStats();
                
                document.getElementById('clear-chat-btn').disabled = false;
            } else {
                console.error(`❌ Model loading failed:`, data.error);
                this.showAlert(`Error: ${data.error}`, 'danger');
            }
            
        } catch (error) {
            console.error('❌ Error in model loading process:', error);
            this.showAlert('Failed to load model', 'danger');
        } finally {
            // Always clean up state
            this.isLoadingModel = false;
            this.hideLoading();
            this.setLoadingState(false);
            console.log('🔄 Model loading process completed');
        }
    }
    
    setLoadingState(loading) {
        const loadButton = document.getElementById('load-model-btn');
        const unloadButton = document.getElementById('unload-model-btn');
        const generateButton = document.getElementById('send-message-btn');
        
        if (loadButton) {
            loadButton.disabled = loading;
            loadButton.innerHTML = loading 
                ? '<i class="fas fa-spinner fa-spin me-2"></i>Loading...' 
                : '<i class="fas fa-download me-2"></i>Load Model';
        }
        
        if (unloadButton) {
            unloadButton.disabled = loading;
        }
        
        if (generateButton) {
            generateButton.disabled = loading;
        }
        
        console.log(`🔘 Set loading state: ${loading}`);
    }

    async loadAdapterWithAutoDetectedBase(adapterPath, systemPrompt) {
        this.showLoading('Auto-detecting base model from adapter...');
        
        try {
            // Extract the directory from adapter path to find training config
            // If adapterPath is a .safetensors file, get its directory
            let adapterDir = adapterPath;
            if (adapterPath.endsWith('.safetensors')) {
                adapterDir = adapterPath.substring(0, adapterPath.lastIndexOf('/'));
            }
            
            console.log(`🔍 Original adapter file selection: ${adapterPath}`);
            console.log(`📁 Extracting adapter directory: ${adapterDir}`);
            console.log(`📁 Is .safetensors file: ${adapterPath.endsWith('.safetensors')}`);
            
            // Try to find the training config JSON file
            const response = await fetch('/api/directories/contents', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: adapterDir })
            });
            
            const data = await response.json();
            
            if (data.success && data.contents) {
                // Look for CPT_*.json file
                const configFile = data.contents.find(file => 
                    file.name.startsWith('CPT_') && file.name.endsWith('.json')
                );
                
                if (configFile) {
                    // Read the config file to get base model
                    const configResponse = await fetch('/api/directories/read-file', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            path: `${adapterDir}/${configFile.name}` 
                        })
                    });
                    
                    const configData = await configResponse.json();
                    
                    if (configData.success) {
                        const trainingConfig = JSON.parse(configData.content);
                        const baseModel = trainingConfig.base_model;
                        
                        if (baseModel) {
                            // Load the adapter with the detected base model
                            // Use the directory path for MLX-LM (it expects the directory, not the .safetensors file)
                            const actualAdapterPath = adapterPath.endsWith('.safetensors') ? adapterDir : adapterPath;
                            
                            console.log(`🚀 Auto-detected base model: ${baseModel}`);
                            console.log(`📂 Original adapter selection: ${adapterPath}`);
                            console.log(`📂 Using adapter directory for MLX-LM: ${actualAdapterPath}`);
                            
                            const loadPayload = {
                                model_name: baseModel,
                                adapter_path: actualAdapterPath,
                                system_prompt: systemPrompt
                            };
                            console.log(`📡 Sending to server:`, loadPayload);
                            
                            const loadResponse = await fetch('/api/model/load', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(loadPayload)
                            });
                            
                            const loadData = await loadResponse.json();
                            
                            if (loadData.success) {
                                this.modelLoaded = true;
                                this.updateModelButtons(true);
                                this.updateModelStatus(baseModel, adapterPath);
                                
                                // Show success message with base model info
                                const adapterName = adapterPath.split('/').pop();
                                this.showAlert(
                                    `✅ Adapter loaded successfully!\n` +
                                    `Base Model: ${baseModel}\n` +
                                    `Adapter: ${adapterName}`, 
                                    'success'
                                );
                                
                                // Check for training dashboard
                                this.checkForTrainingDashboard(adapterPath);
                                
                                // Initialize chat if not already done
                                if (!this.chatInitialized) {
                                    this.initChatToolbar();
                                    this.chatInitialized = true;
                                }
                                
                                // Clear any existing chat history and show ready message
                                document.getElementById('chat-history').innerHTML = `
                                    <div class="text-center text-muted mt-3 mb-3">
                                        <i class="fas fa-robot fa-2x mb-3"></i>
                                        <p>Model ready! Type a message below to start the conversation.</p>
                                    </div>
                                `;
                                
                                // Reset all token counters
                                this.currentTokenCount = 0;
                                this.promptTokens = 0;
                                this.completionTokens = 0;
                                this.conversationTokens = 0;
                                this.updateChatStats();
                                
                                // Enable chat buttons
                                document.getElementById('clear-chat-btn').disabled = false;
                                
                                return;
                            } else {
                                this.showAlert(`Error loading adapter: ${loadData.error}`, 'danger');
                                return;
                            }
                        }
                    }
                }
            }
            
            // Fallback: if auto-detection fails, show helpful error
            this.showAlert(
                'Could not auto-detect base model from adapter.<br>' +
                'Please manually select both base model and adapter.', 
                'warning'
            );
            
        } catch (error) {
            console.error('Error auto-detecting base model:', error);
            this.showAlert('Failed to auto-detect base model from adapter', 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    async unloadModel() {
        this.showLoading('Unloading model...');
        
        try {
            const response = await fetch('/api/model/unload', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.modelLoaded = false;
                this.updateModelButtons(false);
                this.updateModelStatus();
                this.showAlert('Model unloaded', 'info');
            } else {
                this.showAlert(data.error || 'Failed to unload model', 'danger');
            }
        } catch (error) {
            console.error('Error unloading model:', error);
            this.showAlert('Error unloading model', 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    async resetModelConfiguration() {
        console.log('🔄 Resetting model configuration to defaults...');
        
        // Show confirmation dialog
        if (!confirm('Are you sure you want to reset all model configuration to defaults? This will unload any currently loaded model and clear all selections.')) {
            return;
        }
        
        this.showLoading('Resetting configuration...');
        
        try {
            // First, unload any currently loaded model
            if (this.modelLoaded) {
                console.log('🔄 Unloading current model before reset...');
                await fetch('/api/model/unload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
            }
            
            // Reset model selections
            console.log('🔄 Resetting model and adapter selections...');
            const modelSelect = document.getElementById('test-model-select');
            const adapterSelect = document.getElementById('adapter-path');
            const systemPrompt = document.getElementById('system-prompt');
            
            if (modelSelect) {
                modelSelect.value = '';
                console.log('✅ Reset base model selection');
            }
            
            if (adapterSelect) {
                adapterSelect.value = '';
                console.log('✅ Reset adapter selection');
            }
            
            if (systemPrompt) {
                systemPrompt.value = '';
                console.log('✅ Reset system prompt');
            }
            
            // Reset generation parameters to defaults
            console.log('🔄 Resetting generation parameters to defaults...');
            const maxTokens = document.getElementById('max-tokens');
            const maxKvSize = document.getElementById('max-kv-size');
            const temperature = document.getElementById('temperature');
            const topP = document.getElementById('top-p');
            const repetitionPenalty = document.getElementById('repetition-penalty');
            const seed = document.getElementById('seed');
            const streaming = document.getElementById('streaming-toggle');
            
            if (maxTokens) {
                maxTokens.value = '2000';
                console.log('✅ Reset max tokens to 2000');
            }
            
            if (maxKvSize) {
                maxKvSize.value = '16384';
                console.log('✅ Reset context window to 16K');
            }
            
            if (temperature) {
                temperature.value = '0.7';
                console.log('✅ Reset temperature to 0.7');
            }
            
            if (topP) {
                topP.value = '0.9';
                console.log('✅ Reset top-p to 0.9');
            }
            
            if (repetitionPenalty) {
                repetitionPenalty.value = '1.1';
                console.log('✅ Reset repetition penalty to 1.1');
            }
            
            if (seed) {
                seed.value = '42';
                console.log('✅ Reset seed to 42');
            }
            
            if (streaming) {
                streaming.checked = true;
                console.log('✅ Reset streaming to enabled');
            }
            
            // Reset model state
            this.modelLoaded = false;
            this.updateModelButtons(false);
            this.updateModelStatus();
            
            // Clear chat history
            console.log('🔄 Clearing chat history...');
            document.getElementById('chat-history').innerHTML = `
                <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted" id="chat-welcome">
                    <i class="fas fa-robot fa-3x mb-3" style="color: var(--text-muted);"></i>
                    <p class="text-center">Load a model to start chatting!</p>
                </div>
            `;
            
            // Reset chat stats
            this.currentTokenCount = 0;
            this.promptTokens = 0;
            this.completionTokens = 0;
            this.conversationTokens = 0;
            this.updateChatStats();
            
            // Hide chat input container and disable buttons
            document.getElementById('chat-input-container').classList.add('d-none');
            document.getElementById('clear-chat-btn').disabled = true;
            document.getElementById('save-chat-btn').disabled = true;
            
            this.showAlert('Configuration reset to defaults successfully', 'success');
            console.log('✅ Model configuration reset completed');
            
        } catch (error) {
            console.error('❌ Error resetting configuration:', error);
            this.showAlert('Failed to reset configuration', 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    updateModelButtons(loaded) {
        document.getElementById('load-model-btn').disabled = loaded;
        document.getElementById('unload-model-btn').disabled = !loaded;
        
        // Show/hide chat input based on model state
        const chatInputContainer = document.getElementById('chat-input-container');
        const chatHistory = document.getElementById('chat-history');
        
        if (loaded) {
            chatInputContainer.classList.remove('d-none');
            // Initialize chat history if empty
            if (chatHistory.children.length === 0) {
                chatHistory.innerHTML = `
                    <div class="text-center text-muted mt-3 mb-3">
                        <i class="fas fa-robot fa-2x mb-3"></i>
                        <p>Model ready! Type a message below to start the conversation.</p>
                    </div>
                `;
            }
        } else {
            chatInputContainer.classList.add('d-none');
            // Clear chat history when no model is loaded
            chatHistory.innerHTML = `
                <div class="text-center text-muted mt-5">
                    <i class="fas fa-cloud fa-2x mb-3"></i>
                    <p>Load a model to start generating text</p>
                </div>
            `;
        }
    }
    
    updateModelStatus(model = null, adapter = null) {
        // Model status display has been removed from the UI
        // This function is kept as a no-op to maintain compatibility
        // with existing code that calls it
    }
    
    // Function to check and clean system prompt from prompt input
    cleanSystemPromptFromInput() {
        const systemPrompt = document.getElementById('system-prompt').value.trim();
        const chatInput = document.getElementById('chat-input');
        
        if (systemPrompt && chatInput && chatInput.value.includes(systemPrompt)) {
            // Remove system prompt from the chat input
            chatInput.value = chatInput.value.replace(systemPrompt, '').trim();
        }
    }
    
    /**
     * Format markdown text to HTML
     * @param {string} text - The markdown text to format
     * @returns {string} - Formatted HTML
     */
    formatMarkdown(text) {
        if (!text) return '';
        
        // First, escape any angle brackets that aren't part of HTML tags
        // This prevents them from being interpreted as HTML tags
        text = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        // Replace headers (# Header)
        text = text.replace(/^(#{1,6})\s+(.+)$/gm, (match, hashes, content) => {
            const level = hashes.length;
            return `<h${level} class="mt-2 mb-2">${content}</h${level}>`;
        });
        
        // Replace bold (**text**)
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Replace italic (*text*)
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Replace code blocks (```code```)
        text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre class="bg-dark text-light p-2 rounded"><code>${code}</code></pre>`;
        });
        
        // Replace inline code (`code`)
        text = text.replace(/`([^`]+)`/g, '<code class="bg-light px-1 rounded">$1</code>');
        
        // Process tables
        text = this.processMarkdownTables(text);
        
        // Process lists - first split text into blocks (paragraphs and lists)
        const blocks = text.split(/\n\n+/);
        const processedBlocks = blocks.map(block => {
            // Check if block is an unordered list
            if (/^(\s*)-\s+.+/m.test(block)) {
                // Process unordered list
                let listHtml = '<ul class="mb-3 ms-3">\n';
                
                block.split('\n').forEach(line => {
                    const listMatch = line.match(/^(\s*)-\s+(.+)$/);
                    if (listMatch) {
                        listHtml += `<li>${listMatch[2]}</li>\n`;
                    } else if (line.trim()) {
                        // Handle continued text in list items (indented lines)
                        listHtml = listHtml.slice(0, -5); // Remove closing </li>
                        listHtml += ` ${line.trim()}</li>\n`;
                    }
                });
                
                listHtml += '</ul>';
                return listHtml;
            } 
            // Check if block is an ordered list - improved pattern to better match numbered lists
            else if (/^\s*\d+\.\s+.+/m.test(block)) {
                // Process ordered list
                let listHtml = '<ol class="mb-3 ms-3">\n';
                let inListItem = false;
                let currentItemContent = '';
                
                block.split('\n').forEach(line => {
                    // Match numbered list items (1. Item)
                    const listMatch = line.match(/^\s*(\d+)\.\s+(.+)$/);
                    
                    if (listMatch) {
                        // If we were processing a previous item, add it to the list
                        if (inListItem) {
                            listHtml += `<li>${currentItemContent}</li>\n`;
                        }
                        
                        // Start a new list item
                        currentItemContent = listMatch[2];
                        inListItem = true;
                    } else if (line.trim() && inListItem) {
                        // This is a continuation of the current list item
                        currentItemContent += ` ${line.trim()}`;
                    }
                });
                
                // Add the last item if there is one
                if (inListItem) {
                    listHtml += `<li>${currentItemContent}</li>\n`;
                }
                
                listHtml += '</ol>';
                return listHtml;
            }
            
            return block; // Return unchanged if not a list
        });
        
        text = processedBlocks.join('\n\n');
        
        // Replace blockquotes
        text = text.replace(/^>\s+(.+)$/gm, '<blockquote class="border-start border-3 ps-3 text-muted">$1</blockquote>');
        
        // Replace horizontal rules
        text = text.replace(/^---+$/gm, '<hr>');
        
        // Replace paragraphs (double newlines)
        text = text.replace(/\n\n/g, '</p><p>');
        text = `<p>${text}</p>`;
        text = text.replace(/<p>\s*<\/p>/g, '');
        
        // Fix nested paragraphs in lists
        text = text.replace(/<\/li>\n<p>/g, '</li>\n');
        text = text.replace(/<\/p>\n<li>/g, '\n<li>');
        text = text.replace(/<p><ul>/g, '<ul>');
        text = text.replace(/<\/ul><\/p>/g, '</ul>');
        text = text.replace(/<p><ol>/g, '<ol>');
        text = text.replace(/<\/ol><\/p>/g, '</ol>');
        
        return text;
    }
    
    /**
     * Render markdown content using marked.js library
     * @param {HTMLElement} element - The element to render content into
     * @param {string} text - The raw text content to render
     * @param {boolean} forceMarkdown - Force markdown rendering even if not detected
     */
    renderMarkdownContent(element, text, forceMarkdown = false) {
        try {
            // Store the raw text for potential future use
            element.setAttribute('data-raw-text', text);
            
            console.log('🔍 renderMarkdownContent called with:', {
                textLength: text.length,
                textPreview: text.substring(0, 100),
                markdownEnabled: this.markdownEnabled,
                markedAvailable: typeof marked !== 'undefined',
                forceMarkdown
            });
            
            // Test markdown detection with a simple example
            const testMd = "## Header\n**bold** text";
            console.log('🧪 Test markdown detection:', this.isMarkdown(testMd));
            
            // Check if marked.js is available
            if (typeof marked === 'undefined') {
                console.warn('⚠️ Marked.js library not loaded, falling back to plain text');
                element.innerText = text;
                return;
            }
            
            // Check if content appears to be markdown and markdown is enabled
            const isMarkdownContent = forceMarkdown || this.isMarkdown(text);
            console.log('🔍 Markdown detection:', {
                isMarkdownContent,
                markdownEnabled: this.markdownEnabled,
                willRender: this.markdownEnabled && isMarkdownContent,
                forceMarkdown
            });
            
            if (this.markdownEnabled && isMarkdownContent) {
                console.log('📝 Rendering markdown content with marked.js');
                
                // Configure marked.js options for security and formatting
                marked.setOptions({
                    breaks: true,        // Enable line breaks
                    gfm: true,          // Enable GitHub Flavored Markdown
                    sanitize: false,    // We'll handle sanitization ourselves if needed
                    smartLists: true,   // Use smarter list behavior
                    smartypants: false  // Don't use smart quotes (can interfere with code)
                });
                
                // Process thinking blocks before markdown rendering
                const processedText = this.processThinkingBlocks(text);
                
                // Render markdown to HTML
                const htmlContent = marked.parse(processedText);
                element.innerHTML = htmlContent;
                
                // Add Bootstrap classes to rendered elements for better styling
                this.enhanceMarkdownStyling(element);
                
                // Initialize thinking block interactions
                this.initializeThinkingBlocks(element);
            } else {
                console.log('📝 Content doesn\'t appear to be markdown, using plain text');
                // Still process thinking blocks even for plain text
                const processedText = this.processThinkingBlocks(text);
                if (processedText !== text) {
                    // Thinking blocks were found, render as HTML
                    element.innerHTML = processedText;
                    this.initializeThinkingBlocks(element);
                } else {
                    // No thinking blocks, use plain text
                    element.innerText = text;
                }
            }
        } catch (error) {
            console.error('❌ Error rendering markdown:', error);
            // Fallback to plain text on error
            element.innerText = text;
        }
    }
    
    /**
     * Enhance the styling of rendered markdown by adding Bootstrap classes
     * @param {HTMLElement} container - The container with rendered markdown
     */
    enhanceMarkdownStyling(container) {
        // Add Bootstrap classes to tables
        const tables = container.querySelectorAll('table');
        tables.forEach(table => {
            table.className = 'table table-bordered table-striped';
            // Wrap table in responsive container if not already wrapped
            if (!table.parentElement.classList.contains('table-responsive')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'table-responsive';
                table.parentNode.insertBefore(wrapper, table);
                wrapper.appendChild(table);
            }
        });
        
        // Add Bootstrap classes to code blocks
        const codeBlocks = container.querySelectorAll('pre code');
        codeBlocks.forEach(code => {
            const pre = code.parentElement;
            pre.className = 'bg-dark text-light p-3 rounded';
        });
        
        // Add Bootstrap classes to inline code
        const inlineCodes = container.querySelectorAll('code:not(pre code)');
        inlineCodes.forEach(code => {
            code.className = 'bg-light px-1 rounded';
        });
        
        // Add Bootstrap classes to blockquotes
        const blockquotes = container.querySelectorAll('blockquote');
        blockquotes.forEach(blockquote => {
            blockquote.className = 'border-start border-3 ps-3 text-muted';
        });
        
        // Add Bootstrap classes to lists
        const lists = container.querySelectorAll('ul, ol');
        lists.forEach(list => {
            list.classList.add('mb-3');
            if (!list.closest('li')) { // Only add margin to top-level lists
                list.classList.add('ms-3');
            }
        });
        
        // Add Bootstrap classes to headers
        const headers = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headers.forEach(header => {
            header.classList.add('mt-3', 'mb-2');
        });
    }
    
    /**
     * Process markdown tables
     * @param {string} text - The markdown text containing tables
     * @returns {string} - Text with tables converted to HTML
     */
    processMarkdownTables(text) {
        // Split text into lines
        const lines = text.split('\n');
        const tableLines = [];
        let inTable = false;
        let tableStart = 0;
        
        // Find table sections
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Detect table header separator (| :--- | :--- |)
            if (!inTable && line.includes('|') && line.includes('---')) {
                // This is a table header separator, check if previous line has |
                if (i > 0 && lines[i - 1].includes('|')) {
                    inTable = true;
                    tableStart = i - 1;
                }
            }
            
            // End of table detection
            if (inTable && (!line.includes('|') || line.trim() === '')) {
                tableLines.push({ start: tableStart, end: i - 1 });
                inTable = false;
            }
        }
        
        // If we're still in a table at the end of the text
        if (inTable) {
            tableLines.push({ start: tableStart, end: lines.length - 1 });
        }
        
        // Process each table
        for (let i = tableLines.length - 1; i >= 0; i--) {
            const { start, end } = tableLines[i];
            const tableRows = lines.slice(start, end + 1);
            
            // Create HTML table
            let tableHtml = '<div class="table-responsive"><table class="table table-bordered table-striped">\n';
            
            // Process header row
            const headerCells = tableRows[0].split('|')
                .filter(cell => cell.trim() !== '')
                .map(cell => `<th>${cell.trim()}</th>`);
            tableHtml += `<thead><tr>${headerCells.join('')}</tr></thead>\n`;
            
            // Process body rows (skip header and separator)
            tableHtml += '<tbody>\n';
            for (let j = 2; j < tableRows.length; j++) {
                const rowCells = tableRows[j].split('|')
                    .filter(cell => cell.trim() !== '')
                    .map(cell => `<td>${cell.trim()}</td>`);
                tableHtml += `<tr>${rowCells.join('')}</tr>\n`;
            }
            tableHtml += '</tbody></table></div>';
            
            // Replace the table in the original text
            const tableText = tableRows.join('\n');
            text = text.replace(tableText, tableHtml);
        }
        
        return text;
    }
    
    /**
     * Process thinking blocks in text
     * @param {string} text - The text to process
     * @returns {string} - Text with thinking blocks converted to HTML
     */
    processThinkingBlocks(text) {
        // Regular expression to match <think>...</think> blocks
        const thinkRegex = /<think>([\s\S]*?)<\/think>/gi;
        
        return text.replace(thinkRegex, (match, content) => {
            // Generate a unique ID for this thinking block
            const blockId = 'thinking-' + Math.random().toString(36).substr(2, 9);
            
            // Create the HTML structure for the thinking block
            return `
<div class="thinking-block">
    <div class="thinking-header" data-target="${blockId}">
        <i class="fas fa-brain thinking-icon"></i>
                 <span class="thinking-label">Thoughts</span>
    </div>
    <div class="thinking-content" id="${blockId}">
${content.trim()}
    </div>
</div>`;
        });
    }

    /**
     * Initialize thinking block interactions
     * @param {HTMLElement} container - The container with thinking blocks
     */
    initializeThinkingBlocks(container) {
        const thinkingHeaders = container.querySelectorAll('.thinking-header');
        
        thinkingHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const targetId = header.getAttribute('data-target');
                const content = document.getElementById(targetId);
                const icon = header.querySelector('.thinking-icon');
                
                if (content) {
                    // Toggle collapsed state
                    const isCollapsed = content.classList.contains('collapsed');
                    
                    if (isCollapsed) {
                        // Expand
                        content.classList.remove('collapsed');
                        header.classList.remove('collapsed');
                        content.style.display = 'block';
                    } else {
                        // Collapse
                        content.classList.add('collapsed');
                        header.classList.add('collapsed');
                        content.style.display = 'none';
                    }
                }
            });
        });
        
        // Initially collapse all thinking blocks
        const thinkingContents = container.querySelectorAll('.thinking-content');
        thinkingContents.forEach(content => {
            content.classList.add('collapsed');
            content.style.display = 'none';
            
            // Also mark the header as collapsed
            const blockId = content.id;
            const header = container.querySelector(`[data-target="${blockId}"]`);
            if (header) {
                header.classList.add('collapsed');
            }
        });
    }
    
    /**
     * Check if text appears to be markdown
     * @param {string} text - The text to check
     * @returns {boolean} - True if the text appears to be markdown
     */
    isMarkdown(text) {
        if (!text) return false;
        
        // Check for common markdown patterns
        const markdownPatterns = [
            { name: 'Headers', pattern: /^#+\s+.+$/m },
            { name: 'Bold', pattern: /\*\*.+\*\*/ },
            { name: 'Italic', pattern: /\*.+\*/ },
            { name: 'Code blocks', pattern: /```[\s\S]*?```/ },
            { name: 'Inline code', pattern: /`.+`/ },
            { name: 'Unordered lists', pattern: /^(\s*)-\s+.+$/m },
            { name: 'Ordered lists', pattern: /^\s*\d+\.\s+.+$/m },
            { name: 'Blockquotes', pattern: /^>\s+.+$/m },
            { name: 'Horizontal rules', pattern: /^---+$/m },
            { name: 'Tables', pattern: /\|[\s-:]+\|/ },
            { name: 'Links', pattern: /\[.+\]\(.+\)/ },
            { name: 'Images', pattern: /!\[.*\]\(.+\)/ },
            { name: 'Strikethrough', pattern: /~~.+~~/ },
            { name: 'Line breaks', pattern: /  \n/ }
        ];
        
        // Check if any pattern matches
        for (const { name, pattern } of markdownPatterns) {
            if (pattern.test(text)) {
                console.log(`✅ Markdown detected: ${name} pattern matched`);
                return true;
            }
        }
        
        // Additional check for common list patterns that might be missed
        // Look for multiple lines starting with numbers or dashes
        const lines = text.split('\n');
        let listItemCount = 0;
        let hasMultipleLineBreaks = 0;
        
        for (const line of lines) {
            if (/^\s*\d+\.\s+.+$/.test(line) || /^\s*-\s+.+$/.test(line) || /^\s*\*\s+.+$/.test(line)) {
                listItemCount++;
                if (listItemCount >= 2) {
                    console.log(`✅ Markdown detected: Multiple list items found`);
                    return true;
                }
            }
            
            // Check for multiple blank lines (common in structured text)
            if (line.trim() === '') {
                hasMultipleLineBreaks++;
            }
        }
        
        // If text has multiple paragraphs (separated by blank lines), it might benefit from markdown rendering
        if (hasMultipleLineBreaks >= 2 && lines.length > 5) {
            console.log(`✅ Markdown detected: Multiple paragraphs with blank lines`);
            return true;
        }
        
        // Check for common markdown-like structures that models often generate
        if (text.includes('\n\n') && (
            text.includes('**') || 
            text.includes('*') || 
            text.includes('`') ||
            text.includes('#') ||
            /^\d+\./.test(text) ||
            text.includes('- ')
        )) {
            console.log(`✅ Markdown detected: Common markdown elements with paragraph breaks`);
            return true;
        }
        
        console.log(`❌ No markdown patterns detected in text`);
        return false;
    }

    // Helper method to estimate user message tokens
    estimateUserMessageTokens(text) {
        // Simple estimation: ~1.3 tokens per word for English text
        // This matches our centralized token counting fallback estimation
        const words = text.trim().split(/\s+/).length;
        return Math.ceil(words * 1.3);
    }
    
    async generateTextFromInput() {
        const chatInput = document.getElementById('chat-input');
        const prompt = chatInput.value.trim();
        
        if (!prompt) {
            this.showAlert('Please enter a message', 'warning');
            return;
        }
        
        // Clear the input and reset its height
        chatInput.value = '';
        chatInput.style.height = 'auto';
        document.getElementById('send-message-btn').disabled = true;
        
        // Call the main generation function
        await this.generateText(prompt);
    }

    async generateText(prompt = null) {
        // If no prompt provided, try to get it from the old input (for backward compatibility)
        if (!prompt) {
            const promptInput = document.getElementById('prompt-input');
            if (promptInput) {
                prompt = promptInput.value.trim();
            }
        }
        if (!prompt) {
            this.showAlert('Please enter a prompt', 'warning');
            return;
        }
        
        const maxTokens = parseInt(document.getElementById('max-tokens').value);
        const temperature = parseFloat(document.getElementById('temperature').value);
        const topP = parseFloat(document.getElementById('top-p').value);
        const repetitionPenalty = parseFloat(document.getElementById('repetition-penalty').value);
        const maxKvSize = parseInt(document.getElementById('max-kv-size').value);
        const seed = parseInt(document.getElementById('seed').value);
        const systemPrompt = document.getElementById('system-prompt').value.trim();
        const streaming = document.getElementById('streaming-toggle').checked;
        
        // Disable chat input during generation
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-message-btn');
        if (chatInput) {
            chatInput.disabled = true;
            sendBtn.disabled = true;
        }
        
        const chatHistory = document.getElementById('chat-history');
        
        // First interaction – clear placeholder
        if (!this.chatInitialized) {
            chatHistory.innerHTML = '';
            this.chatInitialized = true;
            document.getElementById('clear-chat-btn').disabled = false;
        }
        
        // Detect model type for intelligent prompt formatting
        const isBaseModel = await this.isCurrentModelBase();
        console.log('🔍 Base model detection result:', isBaseModel);
        
        // Intelligent prompt formatting based on model type
        let finalPrompt = prompt;
        let historyArray = [];
        
        if (this.includeHistory) {
            const bubbles = chatHistory.querySelectorAll('.list-group-item');
            
            if (isBaseModel) {
                // BASE MODEL: Use plain text continuation with system prompt prepended
                console.log('📚 Using BASE model formatting (plain text continuation)');
                let historyText = '';
                
                // Add system prompt at the beginning if provided
                if (systemPrompt) {
                    historyText += `${systemPrompt}\n\n`;
                }
                
                // Add conversation history as plain text
                for (let i = 0; i < bubbles.length; i += 2) {
                    const user = bubbles[i]?.innerText ?? '';
                    const assistant = bubbles[i + 1]?.innerText ?? '';
                    if (user) historyText += `${user}\n`;
                    if (assistant) historyText += `${assistant}\n`;
                }
                finalPrompt = historyText + prompt;
                
                // For base models, we still send history as array for backend processing
                // but the backend should use raw text formatting
                bubbles.forEach(b => {
                    if (b.classList.contains('chat-user')) {
                        historyArray.push({ role: 'user', content: b.innerText });
                    } else if (b.classList.contains('chat-assistant')) {
                        historyArray.push({ role: 'assistant', content: b.innerText });
                    }
                });
            } else {
                // INSTRUCT MODEL: Use proper message format for chat template
                console.log('📚 Using INSTRUCT model formatting (message structure)');
                
                // Build proper message history for instruct models
                if (systemPrompt) {
                    historyArray.push({ role: 'system', content: systemPrompt });
                }
                
                bubbles.forEach(b => {
                    if (b.classList.contains('chat-user')) {
                        historyArray.push({ role: 'user', content: b.innerText });
                    } else if (b.classList.contains('chat-assistant')) {
                        historyArray.push({ role: 'assistant', content: b.innerText });
                    }
                });
                
                // For instruct models, let the backend handle chat template application
                // Send the current prompt as-is, history will be used for template
                finalPrompt = prompt;
            }
        } else {
            // Single turn formatting
            if (isBaseModel) {
                console.log('📝 Single turn BASE model formatting');
                // For base models, prepend system prompt directly to the user prompt
                if (systemPrompt) {
                    finalPrompt = `${systemPrompt}\n\n${prompt}`;
                }
            } else {
                console.log('📝 Single turn INSTRUCT model formatting');
                // For instruct models, use message structure
                if (systemPrompt) {
                    historyArray.push({ role: 'system', content: systemPrompt });
                }
                // Current prompt will be added as user message by backend
                finalPrompt = prompt;
            }
        }
        
        console.log('📝 Original prompt:', prompt);
        console.log('📤 Final prompt being sent:', finalPrompt);
        console.log('📋 History array:', historyArray);
        console.log('🤖 Model type:', isBaseModel ? 'BASE' : 'INSTRUCT');

        // Hide welcome message when first message is sent
        const welcomeMsg = document.getElementById('chat-welcome');
        if (welcomeMsg) {
            welcomeMsg.style.display = 'none';
        }
        
        // USER bubble
        const userBubble = document.createElement('div');
        userBubble.className = 'list-group-item chat-user';
        userBubble.innerText = prompt;
        chatHistory.appendChild(userBubble);
        
        // Add user message tokens to conversation total
        const userMessageTokens = this.estimateUserMessageTokens(prompt);
        this.conversationTokens += userMessageTokens;
        console.log(`👤 User message: ${userMessageTokens} tokens (conversation total: ${this.conversationTokens})`);
        
        // Update stats immediately after user prompt is added
        this.updateChatStats();
        
        // ASSISTANT placeholder bubble
        const botBubble = document.createElement('div');
        botBubble.className = 'list-group-item chat-assistant';
        botBubble.innerHTML = '<div class="spinner-border spinner-border-sm text-primary me-2"></div> Generating...';
        chatHistory.appendChild(botBubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        try {
            const requestBody = { 
                prompt: finalPrompt, 
                history: historyArray, 
                max_tokens: maxTokens,
                // Remove system_prompt parameter - it's now handled in prompt/history
                temperature: temperature || undefined,
                top_p: topP || undefined,
                repetition_penalty: repetitionPenalty || undefined,
                max_kv_size: maxKvSize || undefined,
                seed: seed || undefined,
                streaming: streaming,
                // Add model type hint for backend
                is_base_model: isBaseModel
            };
            
            console.log('🚀 Sending request:', requestBody);
            
            if (streaming) {
                // Handle streaming response
                await this.handleStreamingGeneration(requestBody, botBubble);
                    } else {
            // Handle non-streaming response (use model server directly)
            const response = await fetch('http://localhost:5001/api/model/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                this.handleNonStreamingResponse(data, botBubble);
            }
            
            // Response handling moved to separate methods
        } catch (error) {
            console.error('Error generating text:', error);
            botBubble.classList.remove('chat-assistant');
            botBubble.classList.add('list-group-item-danger');
            botBubble.innerText = 'Failed to generate text';
        } finally {
            // Re-enable chat input after generation
            const chatInput = document.getElementById('chat-input');
            const sendBtn = document.getElementById('send-message-btn');
            if (chatInput) {
                chatInput.disabled = false;
                sendBtn.disabled = chatInput.value.trim().length === 0;
                // Focus the input for better UX
                chatInput.focus();
            }
            
            // Update stats after generation is complete
            this.updateChatStats();
        }
    }
    
    async handleStreamingGeneration(requestBody, botBubble) {
        try {
            // Connect directly to model server for streaming (temporary fix)
            const response = await fetch('http://localhost:5001/api/model/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let completion = '';
            
            // Clear the loading message
            botBubble.innerHTML = '';
            
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            
                            if (data.type === 'chunk') {
                                completion += data.text;
                                // Removed excessive chunk logging for cleaner console output
                                
                                // For streaming, use plain text during streaming to avoid processing issues
                                // Store raw text for final markdown rendering
                                botBubble.innerText = completion;
                                botBubble.setAttribute('data-raw-text', completion);
                                
                                // Smart auto-scroll: only scroll to bottom if user is already at/near the bottom
                                const chatHistory = document.getElementById('chat-history');
                                this.smartScrollToBottom(chatHistory);
                            } else if (data.type === 'complete') {
                                // Generation complete - capture token counts
                                console.log(`🎉 Streaming generation completed in ${data.generation_time?.toFixed(2)}s`);
                                
                                // Update token counts from server response
                                if (data.prompt_tokens && data.completion_tokens) {
                                    this.promptTokens = data.prompt_tokens;
                                    this.completionTokens = data.completion_tokens;
                                    this.currentTokenCount = data.total_tokens;
                                    this.lastTokensPerSec = data.tokens_per_sec || null;
                                    
                                    // Add only completion tokens to conversation total (not total_tokens which includes cumulative prompt)
                                    this.conversationTokens += this.completionTokens;
                                    
                                    console.log(`🤖 Assistant response: ${this.completionTokens} tokens (conversation total: ${this.conversationTokens})`);
                                    console.log(`📊 Server reported: ${this.promptTokens} prompt + ${this.completionTokens} completion = ${this.currentTokenCount} total`);
                                    if (this.lastTokensPerSec) {
                                        console.log(`⚡ Generation speed: ${this.lastTokensPerSec} tokens/sec`);
                                    }
                                    
                                    // Update stats display now that we have all the data
                                    this.updateChatStats();
                                }
                                break;
                            }
                        } catch (e) {
                            console.warn('Failed to parse streaming chunk:', line, e);
                        }
                    }
                }
            }
            
            // Clean up the final completion
            completion = this.cleanCompletion(completion);
            console.log(`🎯 Final completion (${completion.length} chars): "${completion.substring(0, 100)}..."`);
            
            // Apply markdown rendering after streaming is complete
            this.renderMarkdownContent(botBubble, completion, false);
            
            // Enable save button
            document.getElementById('save-chat-btn').disabled = false;
            
        } catch (error) {
            console.error('Streaming error:', error);
            botBubble.classList.remove('chat-assistant');
            botBubble.classList.add('list-group-item-danger');
            botBubble.innerText = `Streaming Error: ${error.message}`;
        }
    }
    
    handleNonStreamingResponse(data, botBubble) {
        const chatHistory = document.getElementById('chat-history');
        
        if (data.success) {
            // Clean and process the completion
            let completion = this.cleanCompletion(data.completion);
            
            console.log('🔍 Raw completion:', completion.substring(0, 100) + '...');
            
            // Apply markdown rendering for non-streaming response
            this.renderMarkdownContent(botBubble, completion, false);
            
            // Enable save button as soon as we have at least one answer
            document.getElementById('save-chat-btn').disabled = false;
            
            // Update token counts with detailed information
            this.promptTokens = data.prompt_tokens || 0;
            this.completionTokens = data.completion_tokens || 0;
            this.currentTokenCount = data.total_tokens || (this.promptTokens + this.completionTokens);
            this.lastTokensPerSec = data.tokens_per_sec || null;
            
            // Add only completion tokens to conversation total (not total_tokens which includes cumulative prompt)
            this.conversationTokens += this.completionTokens;
            
            console.log(`🤖 Assistant response: ${this.completionTokens} tokens (conversation total: ${this.conversationTokens})`);
            console.log(`📊 Server reported: ${this.promptTokens} prompt + ${this.completionTokens} completion = ${this.currentTokenCount} total`);
            if (this.lastTokensPerSec) {
                console.log(`⚡ Generation speed: ${this.lastTokensPerSec} tokens/sec`);
            }
        } else {
            botBubble.classList.remove('chat-assistant');
            botBubble.classList.add('list-group-item-danger');
            botBubble.innerText = `Error: ${data.error}`;
        }
        
        // Smart auto-scroll: only scroll to bottom if user is already at/near the bottom
        this.smartScrollToBottom(chatHistory);
    }
    
    cleanCompletion(completion) {
        // Remove any model-specific end tokens (</q>, </s>, etc.)
        return completion
            .replace(/<\/q>/g, '')
            .replace(/<\/s>/g, '')
            .replace(/<\/S>/g, '')
            .replace(/<end_of_turn>/g, '');
    }
    
    updateChatStats() {
        const chatHistory = document.getElementById('chat-history');
        const turns = chatHistory.querySelectorAll('.list-group-item').length / 2; // user+bot pairs
        
        // Format token information with conversation total
        let statsHtml = '';
        if (this.conversationTokens > 0) {
            const contextWindow = parseInt(document.getElementById('max-kv-size').value);
            
            // For context window usage, use the last prompt tokens (which includes system prompt + full conversation)
            // This gives a more accurate view of actual LLM context usage
            const actualContextUsage = this.promptTokens || this.conversationTokens;
            const tokenPercentage = contextWindow ? 
                ` [${Math.round((actualContextUsage / contextWindow) * 100)}%]` : '';
            
            statsHtml = `${turns} turn${turns !== 1 ? 's' : ''} | ${this.conversationTokens} tokens${tokenPercentage}`;
            
            // Add tokens per second if available
            if (this.lastTokensPerSec) {
                statsHtml += ` | ${this.lastTokensPerSec} tk/s`;
            }
            
            // Add breakdown if we have prompt/completion details from last exchange
            if (this.promptTokens && this.completionTokens) {
                statsHtml += `<br><i style="font-size: 0.85em;">(last: ${this.promptTokens} prompt, ${this.completionTokens} completion)</i>`;
            }
        } else {
            statsHtml = `${turns} turn${turns !== 1 ? 's' : ''}`;
        }
        
        const chatStats = document.getElementById('chat-stats');
        chatStats.innerHTML = statsHtml;
        
        // Visual indicator for context window usage (based on actual prompt tokens, not just conversation content)
        chatStats.classList.remove('text-warning', 'text-danger');
        
        if (this.conversationTokens) {
            const contextWindow = parseInt(document.getElementById('max-kv-size').value);
            if (contextWindow) {
                // Use actual context usage (prompt tokens) for warning thresholds
                const actualContextUsage = this.promptTokens || this.conversationTokens;
                const tokenRatio = actualContextUsage / contextWindow;
                if (tokenRatio > 0.9) {
                    chatStats.classList.add('text-danger');
                } else if (tokenRatio > 0.7) {
                    chatStats.classList.add('text-warning');
                }
            }
        }
    }

    // Helper method for smart scrolling - only scrolls if user is already at/near the bottom
    smartScrollToBottom(chatHistory) {
        const isNearBottom = chatHistory.scrollTop + chatHistory.clientHeight >= chatHistory.scrollHeight - 50;
        if (isNearBottom) {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }
    
    handleTrainingFinished(data) {
        this.isTraining = false;
        this.updateTrainingButtons(false);
        
        if (data.status === 'completed') {
            this.showAlert('Training completed successfully!', 'success');
        } else {
            this.showAlert('Training stopped', 'info');
        }
        
        // Refresh checkpoints
        this.loadCheckpoints();
        
        // Reset dashboard metrics for next training session
        const metricsIds = ['current-iteration', 'train-loss', 'val-loss', 'perplexity', 
                           'tokens-per-sec', 'trained-tokens', 'epoch-done', 'memory-usage'];
        metricsIds.forEach(id => {
            document.getElementById(id).textContent = '-';
        });
        
        // Reset progress bar
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('elapsed-time').textContent = 'E.Time: --m';
        document.getElementById('eta-time').textContent = 'R.Time: --m';
    }
    
    showLoading(message = 'Loading...') {
        console.log(`🔄 showLoading called with message: "${message}"`);
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        
        console.log(`🔄 Loading overlay element: ${loadingOverlay ? 'found' : 'NOT FOUND'}`);
        console.log(`🔄 Loading message element: ${loadingMessage ? 'found' : 'NOT FOUND'}`);
        
        if (loadingMessage) {
            loadingMessage.textContent = message;
            console.log(`🔄 Set loading message to: "${loadingMessage.textContent}"`);
        }
        
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
            console.log(`🔄 Removed d-none class. Current classes: "${loadingOverlay.className}"`);
        }
    }
    
    hideLoading() {
        console.log(`🔄 hideLoading called`);
        const loadingOverlay = document.getElementById('loading-overlay');
        
        console.log(`🔄 Loading overlay element: ${loadingOverlay ? 'found' : 'NOT FOUND'}`);
        
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
            console.log(`🔄 Added d-none class. Current classes: "${loadingOverlay.className}"`);
        }
    }
    
    showAlert(message, type = 'info') {
        // Create and show bootstrap alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // New methods for improved checkpoint UX
    loadCheckpointForTesting(checkpointPath, modelName) {
        // Auto-fill the testing form
        document.getElementById('test-model-select').value = modelName;
        document.getElementById('adapter-path').value = checkpointPath;
        
        // Switch to testing tab
        const testingTab = new bootstrap.Tab(document.getElementById('testing-tab'));
        testingTab.show();
        
        this.showAlert('Checkpoint loaded for testing! Click "Load Model" to start.', 'info');
    }
    
    showCheckpointDetails(sessionId) {
        // This could open a modal with detailed checkpoint information
        this.showAlert('Checkpoint details view - coming soon!', 'info');
    }
    
    async deleteTrainingSession(sessionId) {
        if (!confirm(`Are you sure you want to delete the training session "${sessionId}"? This will remove all checkpoints and cannot be undone.`)) {
            return;
        }
        
        this.showAlert('Training session deletion - coming soon!', 'warning');
    }
    
    async updateTrainingEstimates() {
        try {
            // Get current parameter values
            const inputDir = document.getElementById('input-dir').value;
            const batchSize = parseInt(document.getElementById('batch-size').value) || 4;
            const maxIterations = parseInt(document.getElementById('max-iterations').value) || 100;
            const maxSeqLength = parseInt(document.getElementById('max-seq-length').value) || 2048;
            const saveEvery = parseInt(document.getElementById('save-every').value) || 50;
            const evalEvery = parseInt(document.getElementById('eval-every').value) || 25;
            const valFastPct = parseFloat(document.getElementById('val-fast-pct').value) || 0.5;
            const validationSplit = parseFloat(document.getElementById('validation-split').value) || 0.1;
            
            // Fetch dataset info
            const response = await fetch(`/api/dataset/info?dir=${encodeURIComponent(inputDir)}`);
            const data = await response.json();
            
            if (data.error) {
                document.getElementById('epoch-estimate').textContent = 
                    `Could not estimate (using default values)`;
                return;
            }
            
            // Calculate estimates
            const totalTokens = data.total_tokens || 1000000;
            const tokensPerBatch = batchSize * maxSeqLength;
            const totalBatches = Math.ceil(totalTokens / tokensPerBatch);
            
            // Account for validation split
            const trainTokens = Math.floor(totalTokens * (1 - validationSplit));
            const trainBatches = Math.ceil(trainTokens / tokensPerBatch);
            
            // Calculate epoch progress
            const epochProgress = (maxIterations / trainBatches * 100).toFixed(1);
            const epochsCompleted = (maxIterations / trainBatches).toFixed(2);
            
            // Calculate checkpoints
            const totalCheckpoints = Math.floor(maxIterations / saveEvery);
            
            // Calculate validations
            const totalValidations = Math.floor(maxIterations / evalEvery);
            
            // Update the estimates display
            document.getElementById('epoch-estimate').innerHTML = `
                <strong>Dataset:</strong> ~${(totalTokens / 1000000).toFixed(1)}M tokens in ${data.total_files || 'unknown'} files<br>
                <strong>Training:</strong> ${trainTokens.toLocaleString()} tokens (${(100-validationSplit*100).toFixed(0)}%), ${trainBatches.toLocaleString()} batches<br>
                <strong>Validation:</strong> ${Math.floor(totalTokens * validationSplit).toLocaleString()} tokens (${(validationSplit*100).toFixed(0)}%)<br>
                <strong>Epoch Progress:</strong> ${epochProgress}% (${epochsCompleted} epochs)<br>
                <strong>Checkpoints:</strong> ${totalCheckpoints} (every ${saveEvery} iterations)<br>
                <strong>Validations:</strong> ${totalValidations} (every ${evalEvery} iterations)
            `;
            
        } catch (error) {
            console.error('Error updating training estimates:', error);
            document.getElementById('epoch-estimate').textContent = 
                'Error calculating estimates';
        }
    }
    
    viewTrainingLogs(logFile) {
        // Load the training logs and show them in a monitoring-style view
        this.showLoading('Loading training logs...');
        
        // Set a timeout to prevent infinite loading
        const timeoutId = setTimeout(() => {
            this.hideLoading();
            this.showAlert('Loading training logs is taking too long. Please try again.', 'warning');
        }, 30000); // 30 second timeout
        
        // First, fetch the raw JSON content of the log file
        fetch('/api/logs/raw', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ log_file: logFile })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(response => {
            if (!response.success) {
                throw new Error(response.error || 'Failed to load logs');
            }
            
            // Parse the raw logs content
            let rawData;
            try {
                rawData = JSON.parse(response.logs);
            } catch (e) {
                // If parsing fails, treat as plain text
                rawData = { raw_content: response.logs };
            }
            
            // Extract metadata for display
            let metadataHtml = '';
            if (rawData.config || rawData.base_model) {
                const config = rawData.config || {};
                
                // Get model name from top level or config
                const modelName = rawData.base_model || config.base_model || rawData.model_name || config.model_name || 'N/A';
                // Clean up model name (remove mlx-community/ prefix if present)
                const cleanModelName = modelName.replace(/^mlx-community\//, '');
                
                metadataHtml = `
                    <div class="alert alert-info mb-3">
                        <h6 class="mb-2"><i class="fas fa-info-circle me-2"></i>Training Metadata</h6>
                        <div class="row">
                            <div class="col-md-6">
                                <small><strong>Model:</strong> ${cleanModelName}</small><br>
                                <small><strong>Type:</strong> ${rawData.training_type || config.training_type || 'N/A'}</small><br>
                                <small><strong>Fine-tune Type:</strong> ${config.fine_tune_type || 'N/A'}</small>
                            </div>
                            <div class="col-md-6">
                                <small><strong>Batch Size:</strong> ${config.batch_size || 'N/A'}</small><br>
                                <small><strong>Learning Rate:</strong> ${config.learning_rate || 'N/A'}</small><br>
                                <small><strong>Max Iterations:</strong> ${config.max_iterations || 'N/A'}</small>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            // Show the formatted logs in the modal with metadata
            const logsContent = document.getElementById('logs-content');
            
            // Clear existing content
            logsContent.innerHTML = '';
            
            // Add metadata section if available
            if (metadataHtml) {
                const metadataDiv = document.createElement('div');
                metadataDiv.innerHTML = metadataHtml;
                logsContent.appendChild(metadataDiv);
            }
            
            // Add formatted JSON content
            const jsonPre = document.createElement('pre');
            jsonPre.className = 'json-content bg-dark text-light p-3 rounded';
            jsonPre.style.cssText = 'white-space: pre-wrap; word-wrap: break-word; font-family: "Courier New", monospace; font-size: 12px; line-height: 1.4; max-height: 60vh; overflow-y: auto; border: 1px solid #444;';
            
            if (rawData.raw_content) {
                // Plain text content
                jsonPre.textContent = rawData.raw_content;
            } else {
                // Formatted JSON content with syntax highlighting
                const formattedJson = JSON.stringify(rawData, null, 2);
                jsonPre.innerHTML = this.syntaxHighlightJson(formattedJson);
            }
            
            logsContent.appendChild(jsonPre);
            
            // Setup copy button (store rawData in closure)
            const originalData = rawData;
            document.getElementById('copy-logs-btn').onclick = () => {
                // Get the JSON content from the pre element
                const jsonElement = logsContent.querySelector('.json-content');
                let textToCopy;
                
                if (jsonElement) {
                    // For syntax highlighted JSON, we need to get the original text
                    if (originalData.raw_content) {
                        textToCopy = originalData.raw_content;
                    } else {
                        textToCopy = JSON.stringify(originalData, null, 2);
                    }
                } else {
                    textToCopy = logsContent.textContent;
                }
                
                navigator.clipboard.writeText(textToCopy)
                    .then(() => this.showTooltip('copy-logs-btn', 'Copied!'))
                    .catch(err => this.showAlert('Failed to copy logs: ' + err, 'danger'));
            };
            
            // Show the modal
            const logsModal = new bootstrap.Modal(document.getElementById('logsModal'));
            logsModal.show();
            
            // Hide loading and show success message
            clearTimeout(timeoutId);
            this.hideLoading();
            this.showAlert(`Successfully loaded training logs from ${logFile}`, 'success');
        })
        .catch(error => {
            clearTimeout(timeoutId);
            this.hideLoading();
            console.error('Error loading training logs:', error);
            this.showAlert(`Failed to load training logs: ${error.message}`, 'danger');
        });
    }
    
    switchToMonitoringTab() {
        // Switch to the monitoring tab programmatically
        const monitoringTab = document.getElementById('monitoring-tab');
        const monitoringPane = document.getElementById('monitoring');
        
        if (monitoringTab && monitoringPane) {
            // Remove active class from all tabs and panes
            document.querySelectorAll('.nav-link').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('show', 'active');
            });
            
            // Activate monitoring tab
            monitoringTab.classList.add('active');
            monitoringPane.classList.add('show', 'active');
        }
    }

    displayHistoricalCharts(data) {
        console.log('📈 displayHistoricalCharts called with data:', data);
        
        // Update charts with historical data
        if (data.charts) {
            console.log('📊 Rendering charts:', Object.keys(data.charts));
            this.renderCharts(data.charts);
        } else {
            console.log('⚠️ No charts data found in response');
        }
        
        // Update metrics with historical summary
        if (data.summary) {
            console.log('📊 Updating metrics with summary:', data.summary);
            
            // Update all metrics with the historical data
            this.updateAllFields(data.summary, data.summary.config || {});
            
            // Always set the flag to true when displaying historical data
            // This prevents checkpoints from being cleared during periodic updates
            this.checkpointsResetDone = true;
            
            // Explicitly update best checkpoints if available
            if (data.summary.best_checkpoints && data.summary.best_checkpoints.length > 0) {
                console.log("✅ Historical data contains best checkpoints:", data.summary.best_checkpoints);
                this.updateBestCheckpoints(data.summary.best_checkpoints);
            } else {
                console.log("⚠️ No best checkpoints found in historical data");
                // If no best checkpoints available, show a message
                const bestCheckpointsContainer = document.getElementById('best-checkpoints');
                bestCheckpointsContainer.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-chart-line fa-2x mb-3"></i>
                        <p>No best checkpoints available for this training session</p>
                    </div>
                `;
            }
            
            // Populate the checkpoint-select dropdown with all available checkpoints
            console.log('🎯 Populating checkpoint select with summary');
            this.populateCheckpointSelect(data.summary);
        } else {
            console.log('⚠️ No summary data found in response');
        }
    }
    
    updateLearningRateChart() {
        try {
            const learningRate = parseFloat(document.getElementById('learning-rate').value) || 5e-6;
            const maxIterations = parseInt(document.getElementById('max-iterations').value) || 1000;
            const warmupSteps = parseInt(document.getElementById('warmup-steps').value) || 150;
            const lrSchedule = document.getElementById('lr-schedule').value || 'cosine_decay';
            const lrDecayFactor = parseFloat(document.getElementById('lr-decay-factor').value) || 0.1;
            
            // Generate learning rate schedule data
            const steps = [];
            const lrValues = [];
            const numPoints = Math.min(maxIterations, 200); // Reduce points for better performance
            const stepSize = Math.max(1, Math.floor(maxIterations / numPoints));
            
            // Start from step 1 to avoid LR=0 issues with log scale
            for (let step = 1; step <= maxIterations; step += stepSize) {
                const lr = this.calculateLearningRate(step, learningRate, maxIterations, warmupSteps, lrSchedule, lrDecayFactor);
                if (lr > 0) { // Only include positive learning rates for log scale
                    steps.push(step);
                    lrValues.push(lr);
                }
            }
            
            // Ensure we include the final step
            if (steps[steps.length - 1] !== maxIterations) {
                const finalLr = this.calculateLearningRate(maxIterations, learningRate, maxIterations, warmupSteps, lrSchedule, lrDecayFactor);
                if (finalLr > 0) {
                    steps.push(maxIterations);
                    lrValues.push(finalLr);
                }
            }
            
            // Create main trace
            const trace = {
                x: steps,
                y: lrValues,
                type: 'scatter',
                mode: 'lines',
                name: `${lrSchedule.replace('_', ' ').toUpperCase()}`,
                line: {
                    color: '#2E86AB',
                    width: 3
                }
            };
            
            // Create warmup trace if warmup exists
            const traces = [trace];
            if (warmupSteps > 0) {
                const warmupStepsFiltered = steps.filter(s => s <= warmupSteps);
                const warmupValuesFiltered = lrValues.filter((_, i) => steps[i] <= warmupSteps);
                
                if (warmupStepsFiltered.length > 0) {
                    const warmupTrace = {
                        x: warmupStepsFiltered,
                        y: warmupValuesFiltered,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Warmup Phase',
                        line: {
                            color: '#F24236',
                            width: 3
                        }
                    };
                    traces.unshift(warmupTrace); // Add warmup trace first
                }
            }
            
            // Calculate final LR for annotation
            const finalLr = this.calculateLearningRate(maxIterations, learningRate, maxIterations, warmupSteps, lrSchedule, lrDecayFactor);
            
            // Calculate reasonable y-axis range
            const minLr = Math.min(...lrValues);
            const maxLr = Math.max(...lrValues);
            const logRange = Math.log10(maxLr) - Math.log10(minLr);
            const padding = Math.max(0.5, logRange * 0.1); // At least 0.5 orders of magnitude padding
            
            const layout = {
                title: {
                    text: `Learning Rate Schedule: ${lrSchedule.replace('_', ' ').toUpperCase()}`,
                    font: { size: 16, color: '#2c3e50' }
                },
                xaxis: {
                    title: 'Training Step',
                    gridcolor: '#E5E5E5',
                    showgrid: true,
                    zeroline: false
                },
                yaxis: {
                    title: 'Learning Rate',
                    type: 'log',
                    gridcolor: '#E5E5E5',
                    showgrid: true,
                    zeroline: false,
                    // Set reasonable y-axis range with padding
                    range: [Math.log10(minLr) - padding, Math.log10(maxLr) + padding]
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                margin: { l: 80, r: 30, t: 60, b: 60 },
                showlegend: true,
                legend: {
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#E5E5E5',
                    borderwidth: 1
                },
                annotations: []
            };
            
            // Add annotations only if they make sense
            if (warmupSteps > 0 && warmupSteps < maxIterations) {
                layout.annotations.push({
                    x: warmupSteps,
                    y: learningRate,
                    text: `Warmup End<br>Step ${warmupSteps}<br>LR: ${learningRate.toExponential(1)}`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: '#F24236',
                    ax: 30,
                    ay: -40,
                    font: { size: 10, color: '#2c3e50' },
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#F24236',
                    borderwidth: 1
                });
            }
            
            if (lrSchedule !== 'constant') {
                layout.annotations.push({
                    x: maxIterations,
                    y: finalLr,
                    text: `Final LR<br>${finalLr.toExponential(1)}<br>(${Math.round(finalLr/learningRate*100)}% of initial)`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: '#2E86AB',
                    ax: -40,
                    ay: -30,
                    font: { size: 10, color: '#2c3e50' },
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#2E86AB',
                    borderwidth: 1
                });
            }
            
            // Add warmup background highlighting
            if (warmupSteps > 0) {
                layout.shapes = [{
                    type: 'rect',
                    x0: 1, // Start from 1 instead of 0
                    x1: warmupSteps,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    fillcolor: 'rgba(255, 235, 59, 0.1)',
                    line: { width: 0 },
                    layer: 'below'
                }];
            }
            
            Plotly.newPlot('lr-schedule-chart', traces, layout, {
                responsive: true,
                displayModeBar: false
            });
            
        } catch (error) {
            console.error('Error updating learning rate chart:', error);
            document.getElementById('lr-schedule-chart').innerHTML = `
                <div class="text-center text-muted p-4">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error generating learning rate chart</p>
                </div>
            `;
        }
    }
    
    calculateLearningRate(step, baseLr, maxSteps, warmupSteps, schedule, lrDecayFactor = 0.1) {
        const endLr = baseLr * lrDecayFactor; // Use configurable decay factor
        
        if (step <= warmupSteps && warmupSteps > 0) {
            // Linear warmup: 0 -> baseLr over warmupSteps
            return baseLr * (step / warmupSteps);
        } else {
            // Cosine decay: adjusted to use remaining steps after warmup
            // This matches MLX-LM's join_schedules behavior more closely
            const adjustedStep = step - warmupSteps;
            const adjustedDecaySteps = maxSteps - warmupSteps;
            
            if (adjustedStep >= adjustedDecaySteps) {
                return endLr;
            }
            
            switch (schedule) {
                case 'cosine_decay':
                    const progress = adjustedStep / adjustedDecaySteps;
                    return endLr + (baseLr - endLr) * 0.5 * (1 + Math.cos(Math.PI * progress));
                    
                case 'linear_decay':
                    const linearProgress = adjustedStep / adjustedDecaySteps;
                    return baseLr - (baseLr - endLr) * linearProgress;
                    
                case 'constant':
                default:
                    return baseLr;
            }
        }
    }

    /* ---------- Chat helpers ---------- */
    initChatToolbar() {
        const clearBtn = document.getElementById('clear-chat-btn');
        const toggleBtn = document.getElementById('toggle-chat-btn');
        const markdownToggleBtn = document.getElementById('toggle-markdown-btn');
        const saveBtn = document.getElementById('save-chat-btn');

        // Initialize tooltips
        try {
            const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
            [...tooltipTriggerList].map(tooltipTriggerEl => {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        } catch (e) {
            console.warn('Failed to initialize tooltips:', e);
        }

        // State flags
        this.chatHidden = false;
        this.markdownEnabled = true; // Markdown enabled by default

        clearBtn.addEventListener('click', () => {
            const chatHistory = document.getElementById('chat-history');
            if (this.modelLoaded) {
                chatHistory.innerHTML = `
                    <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted" id="chat-welcome">
                        <i class="fas fa-robot fa-3x mb-3" style="color: #C7C7CC;"></i>
                        <p class="text-center">Model ready! Type a message below to start the conversation.</p>
                    </div>
                `;
            } else {
                chatHistory.innerHTML = `
                    <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted" id="chat-welcome">
                        <i class="fas fa-cloud fa-3x mb-3" style="color: #C7C7CC;"></i>
                        <p class="text-center">Load a model to start generating text</p>
                    </div>
                `;
            }
            this.chatInitialized = false;
            document.getElementById('clear-chat-btn').disabled = true;
            document.getElementById('save-chat-btn').disabled = true;
            
            // Reset all token counters
            this.currentTokenCount = 0;
            this.promptTokens = 0;
            this.completionTokens = 0;
            this.conversationTokens = 0;
            this.lastTokensPerSec = null;
            document.getElementById('chat-stats').innerText = '';
        });

        toggleBtn.addEventListener('click', (e) => {
            // Shift-click toggles history mode, regular click toggles visibility
            if (e.shiftKey) {
                this.includeHistory = !this.includeHistory;
                toggleBtn.innerHTML = this.includeHistory ? '<i class="fas fa-book-open"></i> History: On' : '<i class="fas fa-book"></i> History: Off';
                this.showAlert(`Chat history ${this.includeHistory ? 'enabled' : 'disabled'} for context`, 'info');
                return;
            }

            const chatHistory = document.getElementById('chat-history');
            this.chatHidden = !this.chatHidden;
            if (this.chatHidden) {
                chatHistory.style.display = 'none';
            } else {
                chatHistory.style.display = 'block';
            }
        });

        // Markdown toggle functionality
        markdownToggleBtn.addEventListener('click', () => {
            this.markdownEnabled = !this.markdownEnabled;
            markdownToggleBtn.innerHTML = this.markdownEnabled ? 
                '<i class="fab fa-markdown"></i> Markdown: On' : 
                '<i class="fab fa-markdown"></i> Markdown: Off';
            
            // Re-render all assistant messages with new setting
            this.refreshChatRendering();
            
            this.showAlert(`Markdown rendering ${this.markdownEnabled ? 'enabled' : 'disabled'}`, 'info');
        });
    }

    /**
     * Refresh the rendering of all chat messages based on current markdown setting
     */
    refreshChatRendering() {
        const chatHistory = document.getElementById('chat-history');
        const assistantBubbles = chatHistory.querySelectorAll('.chat-assistant');
        
        assistantBubbles.forEach(bubble => {
            const rawText = bubble.getAttribute('data-raw-text');
            if (rawText) {
                this.renderMarkdownContent(bubble, rawText, false);
            }
        });
    }

    async publishCheckpoint(path) {
        // Make sure we have a path - path might be empty string if decoding failed
        if (!path || path === 'undefined') {
            console.error('Empty or undefined path received:', path);
            this.showAlert('Invalid checkpoint path (empty)', 'danger');
            return;
        }
        
        const decodedPath = decodeURIComponent(path);
        if (!decodedPath) {
            console.error('Path decoding failed:', path, '→', decodedPath);
            this.showAlert('Invalid checkpoint path (decode failed)', 'danger');
            return;
        }
        
        console.log("Publishing checkpoint path:", path);
        console.log("Decoded checkpoint path:", decodedPath);
        this.showLoading('Publishing model...');
        try {
            const response = await fetch('/api/training/publish_checkpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    path: decodedPath
                })
            });
            const data = await response.json();
            if (data.success) {
                this.showAlert('Model published to cache!', 'success');
                // refresh model lists so new folder appears
                await this.loadModels();
            } else {
                this.showAlert(data.error || 'Publish failed', 'danger');
            }
        } catch (err) {
            console.error('Publish error', err);
            this.showAlert('Publish failed', 'danger');
        } finally {
            this.hideLoading();
        }
    }

    // Function to save chat history in a format suitable for training
    saveChatHistory() {
        const chatHistory = document.getElementById('chat-history');
        const bubbles = chatHistory.querySelectorAll('.list-group-item');
        const systemPrompt = document.getElementById('system-prompt').value.trim();
        
        if (bubbles.length === 0) {
            this.showAlert('No chat history to save', 'warning');
            return;
        }
        
        // Format for training: collect messages in a format suitable for continued pre-training
        const messages = [];
        
        // Add system prompt if available
        if (systemPrompt) {
            messages.push({
                role: 'system',
                content: systemPrompt
            });
        }
        
        // Add user and assistant messages
        for (let i = 0; i < bubbles.length; i++) {
            const bubble = bubbles[i];
            const role = bubble.classList.contains('chat-user') ? 'user' : 'assistant';
            // Use the raw text attribute if available (for markdown responses), otherwise use innerText
            const content = bubble.hasAttribute('data-raw-text') ? bubble.getAttribute('data-raw-text') : bubble.innerText;
            
            messages.push({
                role: role,
                content: content
            });
        }
        
        // Collect model metadata
        const modelSelect = document.getElementById('test-model-select');
        const adapterSelect = document.getElementById('adapter-path');
        const modelName = modelSelect ? modelSelect.value : 'Unknown model';
        const modelDisplayName = modelSelect ? (modelSelect.selectedOptions[0]?.text || modelName) : 'Unknown model';
        const adapterPath = adapterSelect ? adapterSelect.value : '';
        const adapterDisplayName = adapterSelect ? (adapterSelect.selectedOptions[0]?.text || '') : '';
        
        // Collect generation parameters
        const temperature = parseFloat(document.getElementById('temperature').value);
        const topP = parseFloat(document.getElementById('top-p').value);
        const repetitionPenalty = parseFloat(document.getElementById('repetition-penalty').value);
        const maxKvSize = parseInt(document.getElementById('max-kv-size').value);
        const seed = parseInt(document.getElementById('seed').value);
        
        // Create metadata object
        const metadata = {
            model: {
                name: modelName,
                display_name: modelDisplayName,
                adapter_path: adapterPath,
                adapter_display_name: adapterDisplayName
            },
            parameters: {
                temperature: temperature,
                top_p: topP,
                repetition_penalty: repetitionPenalty,
                max_kv_size: maxKvSize,
                seed: seed
                // system_prompt now handled in message structure
            },
            timestamp: new Date().toISOString(),
            token_count: this.currentTokenCount || 0
        };
        
        // Create simplified chat history format
        const historyData = {
            // Metadata about the model and parameters
            metadata: metadata,
            // The conversation messages
            messages: messages
        };
        
        // Convert to JSON and create download link
        const jsonData = JSON.stringify(historyData, null, 2);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Create descriptive filename with model info
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const modelShortName = modelDisplayName.split('/').pop().replace(/[^a-zA-Z0-9]/g, '_').substring(0, 30);
        
        // Create and trigger download
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_history_${modelShortName}_${timestamp}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showAlert('Chat history saved successfully with model metadata', 'success');
    }

    loadChatHistory() {
        // Trigger the hidden file input
        const fileInput = document.getElementById('load-history-file-input');
        if (fileInput) {
            fileInput.click();
        }
    }

    async handleHistoryFileLoad(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const historyData = JSON.parse(text);

            // Validate the file format
            if (!historyData.metadata || !historyData.messages) {
                this.showAlert('Invalid history file format. Expected metadata and messages.', 'danger');
                return;
            }

            // Clear current chat history
            const chatHistory = document.getElementById('chat-history');
            chatHistory.innerHTML = '';

            // Hide welcome message when loading history
            const welcomeMsg = document.getElementById('chat-welcome');
            if (welcomeMsg) {
                welcomeMsg.style.display = 'none';
            }

            // Load system prompt from message structure (new format)
            let systemPrompt = '';
            const systemMessage = historyData.messages.find(msg => msg.role === 'system');
            if (systemMessage) {
                systemPrompt = systemMessage.content;
            } else {
                // Fallback to old format for backwards compatibility
                systemPrompt = historyData.metadata.parameters?.system_prompt || 
                              historyData.metadata.system_prompt || '';
            }
            
            if (systemPrompt) {
                const systemPromptInput = document.getElementById('system-prompt');
                if (systemPromptInput) {
                    systemPromptInput.value = systemPrompt;
                    this.showAlert(`System prompt loaded: "${systemPrompt.substring(0, 50)}${systemPrompt.length > 50 ? '...' : ''}"`, 'info');
                }
            }

            // Load generation parameters if available
            const params = historyData.metadata.parameters;
            if (params) {
                if (params.temperature !== undefined) {
                    const tempInput = document.getElementById('temperature');
                    if (tempInput) tempInput.value = params.temperature;
                }
                if (params.top_p !== undefined) {
                    const topPInput = document.getElementById('top-p');
                    if (topPInput) topPInput.value = params.top_p;
                }
                if (params.repetition_penalty !== undefined) {
                    const repPenInput = document.getElementById('repetition-penalty');
                    if (repPenInput) repPenInput.value = params.repetition_penalty;
                }
                if (params.max_kv_size !== undefined) {
                    const maxKvInput = document.getElementById('max-kv-size');
                    if (maxKvInput) maxKvInput.value = params.max_kv_size;
                }
                if (params.seed !== undefined) {
                    const seedInput = document.getElementById('seed');
                    if (seedInput) seedInput.value = params.seed;
                }
            }

            // Load conversation messages and count tokens
            let messageCount = 0;
            this.conversationTokens = 0; // Reset conversation token count
            
            for (const message of historyData.messages) {
                if (message.role === 'system') {
                    // System messages are already handled above
                    continue;
                }

                if (message.role === 'user' || message.role === 'assistant') {
                    const bubble = document.createElement('div');
                    bubble.className = `list-group-item ${message.role === 'user' ? 'chat-user' : 'chat-assistant'}`;
                    
                    // Handle markdown for assistant messages
                    if (message.role === 'assistant') {
                        this.renderMarkdownContent(bubble, message.content, false);
                    } else {
                        bubble.innerText = message.content;
                    }
                    
                    chatHistory.appendChild(bubble);
                    messageCount++;
                    
                    // Count tokens for this message
                    const messageTokens = this.estimateUserMessageTokens(message.content);
                    this.conversationTokens += messageTokens;
                }
            }

            // Restore token count from metadata if available (more accurate than estimation)
            if (historyData.metadata && historyData.metadata.token_count) {
                console.log(`📁 Loaded chat with metadata token count: ${historyData.metadata.token_count}`);
                console.log(`📊 Estimated token count: ${this.conversationTokens}`);
                // Use the metadata token count if it seems reasonable (not too different from estimation)
                const estimatedTokens = this.conversationTokens;
                const metadataTokens = historyData.metadata.token_count;
                if (Math.abs(metadataTokens - estimatedTokens) / estimatedTokens < 0.5) {
                    this.conversationTokens = metadataTokens;
                    console.log(`✅ Using metadata token count: ${this.conversationTokens}`);
                } else {
                    console.log(`⚠️ Metadata token count seems inaccurate, using estimation: ${this.conversationTokens}`);
                }
            }

            console.log(`📁 Loaded conversation: ${messageCount} messages, ${this.conversationTokens} tokens`);

            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;

            // Enable relevant buttons
            if (messageCount > 0) {
                document.getElementById('save-chat-btn').disabled = false;
                document.getElementById('clear-chat-btn').disabled = false;
                this.chatInitialized = true;
            }
            
            // Update chat stats to reflect loaded conversation
            this.updateChatStats();

            // Show success message
            const modelInfo = historyData.metadata.model?.display_name || historyData.metadata.model?.name || 'Unknown';
            this.showAlert(`Chat history loaded successfully! ${messageCount} messages from ${modelInfo}`, 'success');

        } catch (error) {
            console.error('Error loading chat history:', error);
            this.showAlert(`Error loading chat history: ${error.message}`, 'danger');
        } finally {
            // Reset the file input
            event.target.value = '';
        }
    }

    // Helper method to show a temporary tooltip
    showTooltip(elementId, message) {
        const button = document.getElementById(elementId);
        const originalTitle = button.getAttribute('title');
        button.setAttribute('title', message);
        
        // Create Bootstrap tooltip if it doesn't exist
        if (!button._tooltip) {
            button._tooltip = new bootstrap.Tooltip(button, {
                trigger: 'manual'
            });
        }
        
        // Show the tooltip
        button._tooltip.show();
        
        // Hide after 1.5 seconds and restore original title
        setTimeout(() => {
            button._tooltip.hide();
            button.setAttribute('title', originalTitle);
        }, 1500);
    }
    
    // Method to open model folder in system file explorer
    async openModelFolder(modelPath) {
        try {
            const response = await fetch('/api/open_folder', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: modelPath })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showTooltip('open-model-folder-btn', 'Folder opened!');
            } else {
                this.showAlert(`Failed to open folder: ${data.error}`, 'danger');
            }
        } catch (error) {
            console.error('Error opening folder:', error);
            this.showAlert(`Error opening folder: ${error.message}`, 'danger');
        }
    }

    // Add this method after loadModel() method
    async checkForTrainingDashboard(modelPath) {
        // Disable dashboard button by default
        const dashboardBtn = document.getElementById('view-dashboard-btn');
        dashboardBtn.disabled = true;
        dashboardBtn.setAttribute('data-dashboard-url', '');
        
        if (!modelPath) return;
        
        console.log(`Checking for dashboard: ${modelPath}`);
        
        try {
            const response = await fetch(`/api/check_dashboard?path=${encodeURIComponent(modelPath)}`);
            const data = await response.json();
            
            console.log('Dashboard check response:', data);
            
            if (data.exists) {
                // Enable button and store dashboard URL
                console.log(`Dashboard found! URL: ${data.dashboard_url}`);
                dashboardBtn.disabled = false;
                dashboardBtn.setAttribute('data-dashboard-url', data.dashboard_url);
            } else {
                console.log('No dashboard found for this model');
            }
        } catch (error) {
            console.error('Error checking for dashboard:', error);
        }
    }
    
    viewTrainingDashboard() {
        const dashboardBtn = document.getElementById('view-dashboard-btn');
        const dashboardUrl = dashboardBtn.getAttribute('data-dashboard-url');
        
        if (!dashboardUrl) return;
        
        // Set image source
        const dashboardImage = document.getElementById('dashboard-image');
        dashboardImage.src = dashboardUrl;
        
        // Show modal
        const dashboardModal = new bootstrap.Modal(document.getElementById('dashboardModal'));
        dashboardModal.show();
    }
    
    /**
     * Populates the checkpoint-select dropdown with all available checkpoints from the training session
     * @param {Object} summary - The training summary data containing checkpoints
     */
    populateCheckpointSelect(summary) {
        const select = document.getElementById('checkpoint-select');
        select.innerHTML = '<option value="">Select a checkpoint...</option>';
        
        // First, check if we have all checkpoints in the summary
        if (!summary || !summary.all_checkpoints || summary.all_checkpoints.length === 0) {
            // If not available in summary, try to extract from best_checkpoints
            if (summary && summary.best_checkpoints && summary.best_checkpoints.length > 0) {
                // Sort checkpoints by iteration
                const sortedCheckpoints = [...summary.best_checkpoints].sort((a, b) => {
                    return (a.iteration || 0) - (b.iteration || 0);
                });
                
                // Add each checkpoint to the select
                sortedCheckpoints.forEach(checkpoint => {
                    if (checkpoint && checkpoint.path) {
                        const option = document.createElement('option');
                        option.value = checkpoint.path;
                        
                        // Format the label with iteration and validation loss
                        let label = `Iteration ${checkpoint.iteration || 'Unknown'}`;
                        if (checkpoint.val_loss !== undefined) {
                            label += ` - Val Loss: ${checkpoint.val_loss.toFixed(4)}`;
                        }
                        if (checkpoint.is_best) {
                            label = `⭐ ${label} (Best)`;
                        }
                        
                        option.textContent = label;
                        select.appendChild(option);
                    }
                });
            }
            return;
        }
        
        // If we have all_checkpoints, use them
        const sortedCheckpoints = [...summary.all_checkpoints].sort((a, b) => {
            return (a.iteration || 0) - (b.iteration || 0);
        });
        
        // Create a set of best checkpoint paths for quick lookup
        const bestCheckpointPaths = new Set();
        if (summary.best_checkpoints) {
            summary.best_checkpoints.forEach(cp => {
                if (cp && cp.path) bestCheckpointPaths.add(cp.path);
            });
        }
        
        // Add each checkpoint to the select
        sortedCheckpoints.forEach(checkpoint => {
            if (checkpoint && checkpoint.path) {
                const option = document.createElement('option');
                option.value = checkpoint.path;
                
                // Format the label with iteration and validation loss
                let label = `Iteration ${checkpoint.iteration || 'Unknown'}`;
                if (checkpoint.val_loss !== undefined) {
                    label += ` - Val Loss: ${checkpoint.val_loss.toFixed(4)}`;
                }
                
                // Mark best checkpoints with a star
                if (bestCheckpointPaths.has(checkpoint.path)) {
                    label = `⭐ ${label} (Best)`;
                }
                
                option.textContent = label;
                select.appendChild(option);
            }
        });
    }

    openFileBrowser(type) {
        // Store the type (input or output) for later use
        this.browserType = type;
        this.selectedPath = null;
        
        // Set modal title
        const title = type === 'input' ? 'Select Training Dataset Directory' : 'Select Output Directory';
        document.getElementById('file-browser-title').textContent = title;
        
        // Get current path from the input field
        const inputId = type === 'input' ? 'input-dir' : 'output-dir';
        let currentPath = document.getElementById(inputId).value || '.';
        
        // Smart path handling for better user experience
        if (!currentPath.startsWith('/')) {
            // For relative paths, try to use them directly (API will resolve them)
            // If it's a known relative path like 'dataset' or 'models', use it as-is
            // Otherwise, start from current working directory
            if (currentPath === '.' || currentPath === '') {
                currentPath = '.';
            }
            // Keep relative paths like 'dataset', 'models' as they are
            // The API will resolve them relative to the project root
        }
        
        // Load directory contents
        this.loadDirectoryContents(currentPath);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('file-browser-modal'));
        modal.show();
        
        // Add event listeners for modal buttons
        this.initFileBrowserEventListeners();
    }
    
    initFileBrowserEventListeners() {
        // Remove existing listeners to prevent duplicates
        const upBtn = document.getElementById('browser-up-btn');
        const selectBtn = document.getElementById('browser-select-btn');
        
        // Clone to remove all event listeners
        const newUpBtn = upBtn.cloneNode(true);
        const newSelectBtn = selectBtn.cloneNode(true);
        upBtn.parentNode.replaceChild(newUpBtn, upBtn);
        selectBtn.parentNode.replaceChild(newSelectBtn, selectBtn);
        
        // Add new event listeners
        document.getElementById('browser-up-btn').addEventListener('click', () => {
            const currentPath = document.getElementById('browser-current-path').value;
            const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
            this.loadDirectoryContents(parentPath);
        });
        
        document.getElementById('browser-select-btn').addEventListener('click', () => {
            if (this.selectedPath) {
                const inputId = this.browserType === 'input' ? 'input-dir' : 'output-dir';
                document.getElementById(inputId).value = this.selectedPath;
                
                // Update training estimates if input directory changed
                if (this.browserType === 'input') {
                    this.updateTrainingEstimates();
                }
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('file-browser-modal'));
                modal.hide();
            }
        });
    }
    
    async loadDirectoryContents(path) {
        const loadingEl = document.getElementById('browser-loading');
        const contentEl = document.getElementById('browser-content');
        const currentPathEl = document.getElementById('browser-current-path');
        const upBtn = document.getElementById('browser-up-btn');
        const selectBtn = document.getElementById('browser-select-btn');
        
        // Show loading
        loadingEl.classList.remove('d-none');
        contentEl.innerHTML = '';
        selectBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/filesystem/browse?path=${encodeURIComponent(path)}`);
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load directory');
            }
            
            // Update current path
            currentPathEl.value = data.current_path;
            this.selectedPath = data.current_path; // Default to current directory
            selectBtn.disabled = false;
            
            // Enable/disable up button
            upBtn.disabled = !data.parent_path;
            
            // Clear content
            contentEl.innerHTML = '';
            
            // Add directory items
            if (data.items && data.items.length > 0) {
                data.items.forEach(item => {
                    if (item.is_directory) {
                        const itemEl = this.createDirectoryItem(item);
                        contentEl.appendChild(itemEl);
                    }
                });
                
                // Add files (for reference, but not selectable)
                data.items.forEach(item => {
                    if (!item.is_directory) {
                        const itemEl = this.createFileItem(item);
                        contentEl.appendChild(itemEl);
                    }
                });
            } else {
                contentEl.innerHTML = '<div class="text-center text-muted p-3">No directories found</div>';
            }
        } catch (error) {
            console.error('Error loading directory:', error);
            contentEl.innerHTML = `<div class="text-center text-danger p-3">Error: ${error.message}</div>`;
            upBtn.disabled = true;
            selectBtn.disabled = true;
        } finally {
            loadingEl.classList.add('d-none');
        }
    }
    
    createDirectoryItem(item) {
        const div = document.createElement('div');
        div.className = 'list-group-item list-group-item-action d-flex align-items-center';
        div.style.cursor = 'pointer';
        
        div.innerHTML = `
            <i class="fas fa-folder text-primary me-3"></i>
            <div class="flex-grow-1">
                <div class="fw-bold">${item.name}</div>
                <small class="text-muted">${item.description}</small>
            </div>
        `;
        
        // Add click handler for navigation
        div.addEventListener('click', () => {
            this.loadDirectoryContents(item.path);
        });
        
        // Add double-click handler for selection
        div.addEventListener('dblclick', () => {
            this.selectedPath = item.path;
            const inputId = this.browserType === 'input' ? 'input-dir' : 'output-dir';
            document.getElementById(inputId).value = item.path;
            
            // Update training estimates if input directory changed
            if (this.browserType === 'input') {
                this.updateTrainingEstimates();
            }
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('file-browser-modal'));
            modal.hide();
        });
        
        return div;
    }
    
    createFileItem(item) {
        const div = document.createElement('div');
        div.className = 'list-group-item d-flex align-items-center';
        div.style.opacity = '0.6';
        
        div.innerHTML = `
            <i class="fas fa-file text-secondary me-3"></i>
            <div class="flex-grow-1">
                <div>${item.name}</div>
                <small class="text-muted">${item.description}</small>
            </div>
        `;
        
        return div;
    }

    async isCurrentModelBase() {
        // Check if current loaded model is a base model
        try {
            const response = await fetch('/api/model/status');
            const data = await response.json();
            
            console.log('🔍 Model status response:', data);
            
            if (data.loaded && data.model_name) {
                // Use the same logic as the backend ModelManager._is_base_model
                const modelName = data.model_name.toLowerCase();
                console.log('📋 Model name (lowercase):', modelName);
                
                // Special handling for Qwen models: they are instruct by default EXCEPT if "base" is in the name
                if (modelName.includes('qwen')) {
                    // For Qwen models, check if it's explicitly marked as base
                    const isBase = modelName.includes('base');
                    console.log(`🔍 Qwen model '${data.model_name}' detected as ${isBase ? 'BASE' : 'INSTRUCT'}`);
                    return isBase;
                }
                
                // Special handling for Gemma models: they are instruct by default EXCEPT if "base" is in the name
                if (modelName.includes('gemma') || modelName.includes('recurrentgemma')) {
                    // For Gemma models, check if it's explicitly marked as base
                    const isBase = modelName.includes('base');
                    console.log(`🔍 Gemma model '${data.model_name}' detected as ${isBase ? 'BASE' : 'INSTRUCT'}`);
                    return isBase;
                }
                
                // Regular detection logic for non-Qwen models
                // Instruction-tuned model patterns with more precise matching
                const instructPatterns = [
                    'instruct', 'chat', 'sft', 'dpo', 'rlhf', 
                    'assistant', 'alpaca', 'vicuna', 'wizard', 'orca',
                    'dolphin', 'openhermes', 'airoboros', 'nous',
                    'claude', 'gpt', 'turbo', 'dialogue', 'conversation',
                    '_it_', '-it-'  // Add explicit patterns for _it_ and -it-
                ];
                
                // Special patterns that need word boundary checking
                const specialPatterns = ['it']; // 'it' needs to be at word boundary to avoid matching '8bit'
                
                // Base model patterns (these indicate base models even if they might have other keywords)
                const basePatterns = [
                    'base', 'pt', 'pretrain', 'foundation'
                ];
                
                // Check for explicit base model indicators first
                const hasBasePattern = basePatterns.some(pattern => 
                    modelName.includes(`-${pattern}`) || 
                    modelName.includes(`_${pattern}`) ||
                    modelName.includes(`-${pattern}-`) ||
                    modelName.includes(`_${pattern}_`) ||
                    modelName.endsWith(`-${pattern}`) ||
                    modelName.endsWith(`_${pattern}`) ||
                    modelName.endsWith(pattern)
                );
                
                console.log('⚡ Has base pattern:', hasBasePattern);
                
                if (hasBasePattern) {
                    console.log('✅ Detected as base model (explicit base pattern found)');
                    return true; // Explicitly a base model
                }
                
                // Check for regular instruct patterns
                let hasInstructPattern = instructPatterns.some(pattern => 
                    modelName.includes(pattern)
                );
                
                console.log('🤖 Has instruct pattern (regular):', hasInstructPattern);
                
                // Check special patterns with word boundaries
                if (!hasInstructPattern) {
                    hasInstructPattern = specialPatterns.some(pattern => {
                        // Use regex for word boundary matching
                        const regex = new RegExp(`\\b${pattern}\\b`, 'i');
                        const matches = regex.test(modelName);
                        console.log(`🔍 Testing special pattern "${pattern}" with regex:`, matches);
                        return matches;
                    });
                }
                
                console.log('🤖 Has instruct pattern (final):', hasInstructPattern);
                
                // Return true if NO instruct patterns found (likely base model)
                const isBase = !hasInstructPattern;
                console.log('📊 Final base model decision:', isBase);
                return isBase;
            }
            
            console.log('❌ No model loaded, defaulting to false');
            return false; // Default to false if no model loaded
        } catch (error) {
            console.error('Error checking model type:', error);
            return false; // Default to false on error
        }
    }

    syntaxHighlightJson(json) {
        // Simple JSON syntax highlighting
        json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return json.replace(
            /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
            function (match) {
                let cls = 'number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'key';
                    } else {
                        cls = 'string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'boolean';
                } else if (/null/.test(match)) {
                    cls = 'null';
                }
                
                const colors = {
                    'key': '#9CDCFE',      // Light blue for keys
                    'string': '#CE9178',   // Light orange for strings
                    'number': '#B5CEA8',   // Light green for numbers
                    'boolean': '#569CD6',  // Blue for booleans
                    'null': '#569CD6'      // Blue for null
                };
                
                return `<span style="color: ${colors[cls] || '#FFFFFF'}">${match}</span>`;
            }
        );
    }

    /**
     * Test markdown rendering functionality (for debugging)
     * Call this from browser console: trainingInterface.testMarkdownRendering()
     */
    testMarkdownRendering() {
        console.log('🧪 Testing markdown rendering...');
        
        // Test marked.js availability
        console.log('📚 Marked.js available:', typeof marked !== 'undefined');
        if (typeof marked !== 'undefined') {
            console.log('📚 Marked.js version:', marked.defaults);
        }
        
        // Test markdown detection
        const testCases = [
            "## Header\n**bold** text",
            "Simple text without markdown",
            "1. First item\n2. Second item",
            "- Bullet point\n- Another point",
            "Some text with `inline code` and\n\nMultiple paragraphs.",
            "```javascript\nconsole.log('code block');\n```"
        ];
        
        testCases.forEach((test, index) => {
            const isMarkdown = this.isMarkdown(test);
            console.log(`Test ${index + 1}: "${test.substring(0, 30)}..." → ${isMarkdown ? '✅ Markdown' : '❌ Plain text'}`);
        });
        
        // Test actual rendering if marked is available
        if (typeof marked !== 'undefined') {
            const testElement = document.createElement('div');
            testElement.style.cssText = 'border: 2px solid #007bff; padding: 10px; margin: 10px; background: #f8f9fa;';
            
            const testMarkdown = `# Test Markdown
            
**Bold text** and *italic text*

- List item 1
- List item 2

\`inline code\` and:

\`\`\`javascript
console.log('code block');
\`\`\`

> Blockquote text

[Link example](https://example.com)`;
            
            this.renderMarkdownContent(testElement, testMarkdown, true);
            document.body.appendChild(testElement);
            
            console.log('🎯 Test element added to page with rendered markdown');
            
            // Remove test element after 10 seconds
            setTimeout(() => {
                if (testElement.parentNode) {
                    testElement.remove();
                    console.log('🧹 Test element removed');
                }
            }, 10000);
        }
    }
    
    toggleFullscreen() {
        console.log('🔍 toggleFullscreen called');
        const fullscreenOverlay = document.getElementById('fullscreen-overlay');
        // Target the specific Generation Output card by finding the card that contains the fullscreen button
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        const chatPanel = fullscreenBtn.closest('.card');
        const fullscreenIcon = fullscreenBtn.querySelector('i');
        
        if (!fullscreenOverlay || !chatPanel || !fullscreenBtn) {
            console.error('❌ Fullscreen elements not found:', {
                fullscreenOverlay: !!fullscreenOverlay,
                chatPanel: !!chatPanel,
                fullscreenBtn: !!fullscreenBtn
            });
            return;
        }
        
        console.log('🔍 Current fullscreenOverlay display:', fullscreenOverlay.style.display);
        console.log('🔍 Current computed style:', window.getComputedStyle(fullscreenOverlay).display);
        
        // Check both inline style and computed style to determine if it's currently fullscreen
        const isCurrentlyFullscreen = fullscreenOverlay.style.display === 'flex' || 
                                     window.getComputedStyle(fullscreenOverlay).display === 'flex';
        
        console.log('🔍 isCurrentlyFullscreen:', isCurrentlyFullscreen);
        
        // Store a reference to the original container before entering fullscreen
        // This is needed for proper restoration when exiting fullscreen
        if (!isCurrentlyFullscreen) {
            // Save the original container reference before going fullscreen
            this._originalContainer = chatPanel.parentElement;
            this._originalContainerSelector = '#testing .col-lg-8';
            console.log('💾 Saved original container reference:', this._originalContainer);
        }
        
        if (isCurrentlyFullscreen) {
            console.log('🔄 Exiting fullscreen mode');
            
            // Get the original container where the panel should be returned
            let originalContainer = null;
            
            // First try using the stored reference (most reliable)
            if (this._originalContainer && document.body.contains(this._originalContainer)) {
                originalContainer = this._originalContainer;
                console.log('🔍 Using stored original container reference');
            } 
            // Then try using the selector
            else if (this._originalContainerSelector) {
                originalContainer = document.querySelector(this._originalContainerSelector);
                console.log('🔍 Using original container selector:', this._originalContainerSelector);
            }
            // Fallback to the default location
            else {
                originalContainer = document.querySelector('#testing .col-lg-8');
                console.log('🔍 Using fallback container selector: #testing .col-lg-8');
            }
            
            if (!originalContainer) {
                console.error('❌ Original container not found, trying to find testing tab container');
                // Last resort - look for any container in the testing tab
                originalContainer = document.querySelector('#testing');
                
                if (!originalContainer) {
                    console.error('❌ Testing tab not found, using body as fallback');
                    // Absolute last resort - just add it to the body
                    originalContainer = document.body;
                }
            }
            
            // Make sure the container is visible
            if (originalContainer !== document.body) {
                originalContainer.style.display = '';
            }
            
            // Move the chat panel back to its original location
            originalContainer.appendChild(chatPanel);
            
            // Hide the overlay
            fullscreenOverlay.style.display = 'none';
            
            // Update button icon and tooltip
            fullscreenIcon.className = 'fas fa-expand';
            fullscreenBtn.textContent = '';
            fullscreenBtn.appendChild(fullscreenIcon);
            fullscreenBtn.appendChild(document.createTextNode(' Fullscreen'));
            fullscreenBtn.setAttribute('title', 'Toggle fullscreen view');
            
            // Re-enable body scrolling
            document.body.style.overflow = 'auto';
            
            // Ensure the chat panel has the correct styling for normal mode
            chatPanel.style.height = '';
            chatPanel.style.maxHeight = '';
            
            // Ensure chat history has the correct height
            const chatHistory = document.getElementById('chat-history');
            if (chatHistory) {
                chatHistory.style.height = '60vh';
                chatHistory.style.maxHeight = '';
            }
            
        } else {
            console.log('🔄 Entering fullscreen mode');
            // Enter fullscreen mode
            // Move the chat panel to the fullscreen overlay
            fullscreenOverlay.appendChild(chatPanel);
            
            // Show the overlay
            fullscreenOverlay.style.display = 'flex';
            
            // Update button icon and tooltip
            fullscreenIcon.className = 'fas fa-compress';
            fullscreenBtn.textContent = '';
            fullscreenBtn.appendChild(fullscreenIcon);
            fullscreenBtn.appendChild(document.createTextNode(' Exit Fullscreen'));
            fullscreenBtn.setAttribute('title', 'Exit fullscreen view');
            
            // Disable body scrolling
            document.body.style.overflow = 'hidden';
            
            // Ensure the chat panel takes full height in fullscreen mode
            chatPanel.style.height = '100vh';
            chatPanel.style.maxHeight = '100vh';
            
            // Ensure chat history has the correct height in fullscreen
            const chatHistory = document.getElementById('chat-history');
            if (chatHistory) {
                chatHistory.style.height = 'calc(100vh - 200px)';
                chatHistory.style.maxHeight = 'calc(100vh - 290px)';
            }
        }
        
        // Update tooltip if using Bootstrap tooltips
        try {
            const tooltip = bootstrap.Tooltip.getInstance(fullscreenBtn);
            if (tooltip) {
                tooltip.dispose();
                new bootstrap.Tooltip(fullscreenBtn);
            }
        } catch (e) {
            // Tooltip might not be initialized yet
            console.log('⚠️ Tooltip error:', e);
        }
        
        console.log('✅ toggleFullscreen completed');
    }
    
    // Fusion methods
    async loadFuseModels() {
        try {
            // Load adapters using the same checkpoints API as Testing tab
            const adaptersResponse = await fetch('/api/checkpoints');
            const adaptersData = await adaptersResponse.json();
            
            if (adaptersData.success && adaptersData.checkpoints) {
                this.updateFuseAdapterDropdown(adaptersData.checkpoints);
            }
        } catch (error) {
            console.error('Error loading fuse models:', error);
            this.showAlert('Failed to load fusion models', 'danger');
        }
    }
    

    
    updateFuseAdapterDropdown(checkpoints) {
        const select = document.getElementById('fuse-adapter-select');
        if (!select) return;
        
        // Clear existing options except the first one
        const firstOption = select.firstElementChild;
        select.innerHTML = '';
        if (firstOption) {
            select.appendChild(firstOption);
        }
        
        // Add checkpoints using the same format as Testing tab
        checkpoints.forEach(checkpoint => {
            const option = document.createElement('option');
            option.value = checkpoint.path;
            
            // Format the display name with adapter type info (same as Testing tab)
            const modelName = checkpoint.model || 'Unknown';
            const iteration = checkpoint.iteration || 0;
            const size = checkpoint.size ? `${checkpoint.size.toFixed(1)}MB` : '';
            
            // Extract training type and fine-tune type if available
            let typeInfo = '';
            if (checkpoint.path) {
                const pathParts = checkpoint.path.split('/');
                // Look for CPT or IFT in path
                if (pathParts.some(part => part.includes('cpt'))) {
                    typeInfo = ' [CPT]';
                } else if (pathParts.some(part => part.includes('ift'))) {
                    typeInfo = ' [IFT]';
                }
                
                // Look for fine-tune type hints in path
                if (pathParts.some(part => part.includes('lora'))) {
                    typeInfo += ' LoRA';
                } else if (pathParts.some(part => part.includes('dora'))) {
                    typeInfo += ' DoRA';
                }
            }
            
            option.textContent = `${modelName}${typeInfo} - iter ${iteration} ${size}`;
            select.appendChild(option);
        });
    }
    
    async detectBaseModelForFusion() {
        const adapterSelect = document.getElementById('fuse-adapter-select');
        const baseModelInfo = document.getElementById('fuse-base-model-info');
        const detectedBaseModel = document.getElementById('fuse-detected-base-model');
        
        if (!adapterSelect || !baseModelInfo || !detectedBaseModel) return;
        
        const adapterPath = adapterSelect.value;
        
        if (!adapterPath) {
            baseModelInfo.style.display = 'none';
            this.detectedBaseModel = null;
            return;
        }
        
        try {
            // Extract the directory from adapter path to find training config
            let adapterDir = adapterPath;
            if (adapterPath.endsWith('.safetensors')) {
                adapterDir = adapterPath.substring(0, adapterPath.lastIndexOf('/'));
            }
            
            console.log(`🔍 Detecting base model for adapter: ${adapterPath}`);
            console.log(`📁 Adapter directory: ${adapterDir}`);
            
            // Try to find the training config JSON file
            const response = await fetch('/api/directories/contents', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: adapterDir })
            });
            
            const data = await response.json();
            
            if (data.success && data.contents) {
                // Look for CPT_*.json file
                const configFile = data.contents.find(file => 
                    file.name.startsWith('CPT_') && file.name.endsWith('.json')
                );
                
                if (configFile) {
                    // Read the config file to get base model
                    const configResponse = await fetch('/api/directories/read-file', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            path: `${adapterDir}/${configFile.name}` 
                        })
                    });
                    
                    const configData = await configResponse.json();
                    
                    if (configData.success) {
                        const trainingConfig = JSON.parse(configData.content);
                        const baseModel = trainingConfig.base_model;
                        
                        if (baseModel) {
                            console.log(`🤖 Auto-detected base model: ${baseModel}`);
                            this.detectedBaseModel = baseModel;
                            detectedBaseModel.textContent = baseModel;
                            baseModelInfo.style.display = 'block';
                            return;
                        }
                    }
                }
            }
            
            // If we get here, detection failed
            console.log('❌ Could not detect base model from adapter');
            this.detectedBaseModel = null;
            detectedBaseModel.textContent = 'Could not detect base model';
            baseModelInfo.style.display = 'block';
            
        } catch (error) {
            console.error('Error detecting base model:', error);
            this.detectedBaseModel = null;
            detectedBaseModel.textContent = 'Error detecting base model';
            baseModelInfo.style.display = 'block';
        }
    }
    
    updateFuseOutputPreview() {
        const adapterSelect = document.getElementById('fuse-adapter-select');
        const outputInfo = document.getElementById('fuse-output-info');
        const outputName = document.getElementById('fuse-output-model-name');
        
        if (adapterSelect && outputInfo && outputName) {
            const adapterPath = adapterSelect.value;
            
            if (adapterPath && this.detectedBaseModel) {
                // Extract base model name
                const baseModelName = this.detectedBaseModel.split('/').pop();
                
                // Extract adapter name from checkpoint path
                const adapterText = adapterSelect.options[adapterSelect.selectedIndex].text;
                const adapterName = adapterText.split(' ')[0];
                
                const fusedName = `${baseModelName}_fused_${adapterName}`;
                outputName.textContent = fusedName;
                outputInfo.style.display = 'block';
            } else {
                outputInfo.style.display = 'none';
            }
        }
    }
    
    showFusionConfigModal() {
        const adapterPath = document.getElementById('fuse-adapter-select').value;
        
        if (!adapterPath) {
            this.showAlert('Please select an adapter', 'warning');
            return;
        }
        
        if (!this.detectedBaseModel) {
            this.showAlert('Base model could not be detected from adapter', 'warning');
            return;
        }
        
        // Update preview with current selection
        this.updateFusionPreview();
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('fusion-config-modal'));
        modal.show();
    }
    
    updateFusionPreview() {
        const adapterSelect = document.getElementById('fuse-adapter-select');
        const suffix = document.getElementById('fusion-suffix').value;
        
        if (adapterSelect && this.detectedBaseModel) {
            const adapterPath = adapterSelect.value;
            
            if (adapterPath) {
                // Extract base model name
                const baseModelName = this.detectedBaseModel.split('/').pop();
                
                // Extract adapter name from checkpoint path
                const adapterText = adapterSelect.options[adapterSelect.selectedIndex].text;
                const adapterName = adapterText.split(' ')[0];
                
                let fusedName = `${baseModelName}_fused_${adapterName}`;
                if (suffix) {
                    fusedName += suffix;
                }
                
                document.getElementById('fusion-preview-name').textContent = fusedName;
            }
        }
    }
    
    confirmFusion() {
        // Get values from modal
        const suffix = document.getElementById('fusion-suffix').value;
        const description = document.getElementById('fusion-description').value;
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('fusion-config-modal'));
        modal.hide();
        
        // Start fusion with config
        this.startFusion(suffix, description);
    }
    
    async startFusion(suffix = '', description = '') {
        const adapterPath = document.getElementById('fuse-adapter-select').value;
        
        if (!adapterPath) {
            this.showAlert('Please select an adapter', 'warning');
            return;
        }
        
        if (!this.detectedBaseModel) {
            this.showAlert('Base model could not be detected from adapter', 'warning');
            return;
        }
        
        this.showLoading('Starting fusion...');
        
        try {
            const response = await fetch('/api/fusion/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base_model_path: this.detectedBaseModel,
                    adapter_path: adapterPath,
                    suffix: suffix,
                    description: description
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showAlert('Fusion started successfully!', 'success');
                this.startFusionProgressMonitoring();
            } else {
                this.showAlert(data.error || 'Failed to start fusion', 'danger');
            }
        } catch (error) {
            console.error('Error starting fusion:', error);
            this.showAlert('Error starting fusion', 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    async stopFusion() {
        this.showLoading('Stopping fusion...');
        
        try {
            const response = await fetch('/api/fusion/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showAlert('Fusion stopped', 'info');
                this.stopFusionProgressMonitoring();
            } else {
                this.showAlert(data.error || 'Failed to stop fusion', 'danger');
            }
        } catch (error) {
            console.error('Error stopping fusion:', error);
            this.showAlert('Error stopping fusion', 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    startFusionProgressMonitoring() {
        this.stopFusionProgressMonitoring(); // Stop any existing monitoring
        
        this.fusionProgressInterval = setInterval(() => {
            this.updateFusionProgress();
        }, 2000); // Update every 2 seconds
        
        // Initial update
        this.updateFusionProgress();
    }
    
    stopFusionProgressMonitoring() {
        if (this.fusionProgressInterval) {
            clearInterval(this.fusionProgressInterval);
            this.fusionProgressInterval = null;
        }
    }
    
    async updateFusionProgress() {
        try {
            const response = await fetch('/api/fusion/status');
            const data = await response.json();
            
            if (data.success) {
                this.displayFusionProgress(data);
                
                // Stop monitoring if fusion is complete or stopped
                if (!data.is_fusing) {
                    this.stopFusionProgressMonitoring();
                }
            }
        } catch (error) {
            console.error('Error updating fusion progress:', error);
        }
    }
    
    displayFusionProgress(data) {
        const idleDiv = document.getElementById('fuse-idle');
        const activeDiv = document.getElementById('fuse-active');
        const errorDiv = document.getElementById('fuse-error');
        const successDiv = document.getElementById('fuse-success');
        const startBtn = document.getElementById('start-fuse-btn');
        const stopBtn = document.getElementById('stop-fuse-btn');
        
        if (data.is_fusing) {
            // Show active state
            if (idleDiv) idleDiv.style.display = 'none';
            if (activeDiv) activeDiv.style.display = 'block';
            if (errorDiv) errorDiv.style.display = 'none';
            if (successDiv) successDiv.style.display = 'none';
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'block';
            
            // Update progress
            const progressBar = document.getElementById('fuse-progress-bar');
            const progressPercentage = document.getElementById('fuse-progress-percentage');
            const statusMessage = document.getElementById('fuse-status-message');
            
            if (progressBar) {
                progressBar.style.width = `${data.progress}%`;
                progressBar.setAttribute('aria-valuenow', data.progress);
            }
            
            if (progressPercentage) {
                progressPercentage.textContent = Math.round(data.progress);
            }
            
            if (statusMessage) {
                statusMessage.textContent = data.status_message || 'Processing...';
            }
            
            // Update time information
            if (data.elapsed_time) {
                const elapsedElement = document.getElementById('fuse-elapsed-time');
                if (elapsedElement) {
                    const minutes = Math.floor(data.elapsed_time / 60);
                    const seconds = Math.floor(data.elapsed_time % 60);
                    elapsedElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
            }
            
            if (data.estimated_remaining) {
                const remainingElement = document.getElementById('fuse-estimated-remaining');
                if (remainingElement) {
                    const minutes = Math.floor(data.estimated_remaining / 60);
                    const seconds = Math.floor(data.estimated_remaining % 60);
                    remainingElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
            }
            
            // Update job information
            const baseModelElement = document.getElementById('fuse-current-base-model');
            const adapterElement = document.getElementById('fuse-current-adapter');
            const outputElement = document.getElementById('fuse-current-output-model');
            
            if (baseModelElement && data.base_model_path) {
                baseModelElement.textContent = data.base_model_path.split('/').pop();
            }
            
            if (adapterElement && data.adapter_path) {
                adapterElement.textContent = data.adapter_path.split('/').pop();
            }
            
            if (outputElement && data.output_name) {
                outputElement.textContent = data.output_name;
            }
            
        } else {
            // Show idle state
            if (idleDiv) idleDiv.style.display = 'block';
            if (activeDiv) activeDiv.style.display = 'none';
            if (startBtn) startBtn.style.display = 'block';
            if (stopBtn) stopBtn.style.display = 'none';
            
            // Show error or success if applicable
            if (data.error) {
                if (errorDiv) {
                    errorDiv.style.display = 'block';
                    const errorMessage = document.getElementById('fuse-error-message');
                    if (errorMessage) {
                        errorMessage.textContent = data.error;
                    }
                }
            } else if (data.progress === 100) {
                if (successDiv) {
                    successDiv.style.display = 'block';
                    const successMessage = document.getElementById('fuse-success-message');
                    if (successMessage) {
                        // Display the published model name
                        const modelName = data.output_name ? `published/${data.output_name}` : 'Unknown';
                        successMessage.innerHTML = `
                            <div>${data.status_message || 'Fusion completed successfully!'}</div>
                            <div class="mt-2 p-2 bg-light rounded">
                                <small class="text-muted">Model: </small>
                                <code class="text-break" style="word-break: break-all; white-space: pre-wrap; font-size: 0.85em;">${modelName}</code>
                            </div>
                        `;
                    }
                    
                    // Store the output path for the open folder button
                    if (data.output_path) {
                        this.fusionOutputPath = data.output_path;
                    }
                }
            }
        }
    }
    
    async openFusionFolder() {
        if (!this.fusionOutputPath) {
            this.showAlert('No fusion output path available', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/open_folder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: this.fusionOutputPath })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showAlert('Opened fusion folder', 'success');
            } else {
                this.showAlert(data.error || 'Failed to open folder', 'danger');
            }
        } catch (error) {
            console.error('Error opening fusion folder:', error);
            this.showAlert('Error opening folder (this feature only works for locally trained models)', 'warning');
        }
    }
}

// Global helper function for other components to check training status
// This avoids duplicate API calls
window.getTrainingStatus = function() {
    return {
        isActive: window.isTrainingActive || false,
        lastChecked: Date.now()
    };
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize global training status
    window.isTrainingActive = false;
    
    window.trainingInterface = new TrainingInterface();
    window.trainingInterface.init();
    
    // Add event listener for the save button
    const saveButton = document.getElementById('save-chat-btn');
    if (saveButton) {
        saveButton.addEventListener('click', () => {
            if (window.trainingInterface) {
                window.trainingInterface.saveChatHistory();
            }
        });
    }

    // Add event listener for the load button
    const loadButton = document.getElementById('load-chat-btn');
    if (loadButton) {
        loadButton.addEventListener('click', () => {
            if (window.trainingInterface) {
                window.trainingInterface.loadChatHistory();
            }
        });
    }

    // Add event listener for the hidden file input
    const fileInput = document.getElementById('load-history-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (window.trainingInterface) {
                window.trainingInterface.handleHistoryFileLoad(e);
            }
        });
    }
}); 