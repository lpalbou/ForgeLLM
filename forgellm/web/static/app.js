// MLX Training Interface JavaScript
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
        this.conversationTokens = 0; // Track total tokens across entire conversation
        this.trainingStartTime = null; // Store training start time for timing calculations
        this.markdownEnabled = true; // Enable markdown rendering by default
        
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
    }
    
    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('training_update', (data) => {
            this.updateTrainingMetrics(data);
            this.updateCharts(data);
            // Hide loading modal if training has actually started
            if (data.current_iteration > 0) {
                this.hideLoading();
            }
        });
        
        this.socket.on('training_finished', (data) => {
            this.handleTrainingFinished(data);
        });
    }
    
    initEventListeners() {
        // Training form
        document.getElementById('training-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startTraining();
        });
        
        document.getElementById('stop-training-btn').addEventListener('click', () => {
            this.stopTraining();
        });
        
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
            
            // For full fine-tuning, set to -1 and disable
            if (fineTuneType === 'full') {
                numLayersInput.value = '-1';
                numLayersInput.disabled = true;
            } else {
                // For LoRA/DoRA, enable and set to 16 if currently -1
                numLayersInput.disabled = false;
                if (numLayersInput.value === '-1') {
                    numLayersInput.value = '16';
                }
            }
        });
        
        // Initialize the num-layers state based on current fine-tune-type
        const initFineTuneType = document.getElementById('fine-tune-type').value;
        if (initFineTuneType === 'full') {
            document.getElementById('num-layers').value = '-1';
            document.getElementById('num-layers').disabled = true;
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


        
        // File browser buttons
        document.getElementById('browse-input-dir').addEventListener('click', () => {
            this.openFileBrowser('input');
        });
        
        document.getElementById('browse-output-dir').addEventListener('click', () => {
            this.openFileBrowser('output');
        });
        
        // Tab switching
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                if (e.target.id === 'monitoring-tab') {
                    this.refreshDashboard();
                }
            });
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
        
        // When adapter is selected, check for dashboard
        document.getElementById('adapter-path').addEventListener('change', (e) => {
            const modelPath = document.getElementById('test-model-select').value;
            const adapterPath = e.target.value;
            
            // If an adapter is selected, check its path for a dashboard
            if (adapterPath) {
                this.checkForTrainingDashboard(adapterPath);
            } else if (modelPath) {
                // If no adapter but model is selected, check model path
                this.checkForTrainingDashboard(modelPath);
            }
        });
    }
    
    async loadInitialData() {
        await this.loadModels();
        await this.loadCheckpoints();
        await this.checkTrainingStatus();
        this.updateTrainingEstimates();
        this.updateLearningRateChart();
    }
    
    startPeriodicUpdates() {
        this.updateInterval = setInterval(() => {
            this.loadCheckpoints();
            this.checkTrainingStatus(); // Check training status regularly
            if (this.isTraining) {
                this.socket.emit('request_update');
                this.refreshDashboard(); // Refresh dashboard during training
            }
        }, 5000);
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
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (data.models) {
                // Update model dropdowns with icons
                this.updateModelDropdown('model-select', data.models);
                this.updateModelDropdown('test-model-select', data.models);
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showAlert('Failed to load models', 'danger');
        }
    }
    
    updateModelDropdown(selectId, models) {
        const select = document.getElementById(selectId);
        if (!select) return;
        
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
        
        container.innerHTML = trainingSessions.map(session => {
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
            const formattedDate = `${dateStr} ${timeStr}`;
            
            return `
            <div class="training-session-card mb-3">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">
                                <i class="fas fa-brain text-primary me-2"></i>
                                ${session.session_name}
                            </h6>
                            <small class="text-muted">
                                <i class="fas fa-calendar me-1"></i>
                                ${formattedDate}
                            </small>
                        </div>
                        <div class="text-end">
                            <span class="badge bg-primary">${session.metrics_count || 0} metrics</span>
                            ${session.latest_val_loss ? `<span class="badge bg-success ms-1">Val Loss: ${session.latest_val_loss.toFixed(4)}</span>` : ''}
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row mb-2">
                            <div class="col-md-6">
                                <small class="text-muted">Model:</small><br>
                                <strong>${session.model_name || 'Unknown'}</strong>
                            </div>
                            <div class="col-md-6">
                                <small class="text-muted">Training Progress:</small><br>
                                <strong>${session.latest_iteration || 0} iterations</strong>
                            </div>
                        </div>
                        
                        ${session.latest_loss || session.latest_val_loss ? `
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <small class="text-muted">Latest Train Loss:</small><br>
                                <strong>${session.latest_loss ? session.latest_loss.toFixed(4) : 'N/A'}</strong>
                            </div>
                            <div class="col-md-6">
                                <small class="text-muted">Latest Val Loss:</small><br>
                                <strong>${session.latest_val_loss ? session.latest_val_loss.toFixed(4) : 'N/A'}</strong>
                            </div>
                        </div>
                        ` : ''}
                        
                        <div class="d-flex gap-2 flex-wrap">
                            <button class="btn btn-outline-primary btn-sm" onclick="trainingInterface.showCheckpointDetails('${session.session_id}')">
                                <i class="fas fa-list me-1"></i>View Details
                            </button>
                            ${session.log_file ? `
                            <button class="btn btn-outline-info btn-sm" onclick="trainingInterface.viewTrainingLogs('${session.log_file}')">
                                <i class="fas fa-chart-line me-1"></i>View Logs
                            </button>
                            ` : ''}
                            <button class="btn btn-outline-danger btn-sm" onclick="trainingInterface.deleteTrainingSession('${session.session_id}')">
                                <i class="fas fa-trash me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            `;
        }).join('');
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
            console.log('📊 No session selected, refreshing current dashboard');
            this.refreshDashboard();
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
                    this.updateTrainingMetrics(data.summary);
                    
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
        select.innerHTML = '<option value="">No adapter (base model only)</option>';
        
        // Load checkpoints separately from the checkpoints API
        this.loadAdapterCheckpoints();
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
                    
                    // Format the display name
                    const modelName = checkpoint.model || 'Unknown';
                    const iteration = checkpoint.iteration || 0;
                    const size = checkpoint.size ? `${checkpoint.size.toFixed(1)}MB` : '';
                    
                    option.textContent = `${modelName} - iter ${iteration} ${size}`;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading adapter checkpoints:', error);
        }
    }
    
    async checkTrainingStatus() {
        try {
            const response = await fetch('/api/training/status');
            const data = await response.json();
            
            // Check if we're viewing historical data (a session is selected)
            const isViewingHistorical = document.getElementById('training-session-select').value !== "";
            
            if (data.active) {
                // If active is true, consider training as running
                this.isTraining = true;
                this.updateTrainingStatus(data);
                this.updateTrainingButtons(this.isTraining);
                
                // If training is active, also refresh the dashboard
                if (this.isTraining) {
                    this.refreshDashboard();
                }
                // Reset the flag when training is active and we're not viewing historical data
                if (!isViewingHistorical) {
                    this.checkpointsResetDone = false;
                }
            } else {
                this.isTraining = false;
                this.updateTrainingButtons(false);
                this.updateTrainingStatus(data);
                
                // Reset best checkpoints display when no training is active
                // BUT ONLY if we're not viewing historical data
                if (!this.checkpointsResetDone && !isViewingHistorical) {
                    const bestCheckpointsContainer = document.getElementById('best-checkpoints');
                    bestCheckpointsContainer.innerHTML = `
                        <div class="text-center text-muted">
                            <i class="fas fa-chart-line fa-2x mb-3"></i>
                            <p>No training data available</p>
                        </div>
                    `;
                    this.checkpointsResetDone = true;
                }
            }
        } catch (error) {
            console.error('Error checking training status:', error);
        }
    }
    
    /**
     * Start training with current configuration
     */
    async startTraining() {
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
                this.isTraining = true;
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
                
                // Switch to monitoring tab after a short delay
                setTimeout(() => {
                    const monitoringTab = new bootstrap.Tab(document.getElementById('monitoring-tab'));
                    monitoringTab.show();
                }, 1000);
            } else {
                this.showAlert(data.error || 'Failed to start training', 'danger');
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.hideLoading();
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
                this.showAlert('Training stopped', 'info');
            } else {
                this.showAlert(data.error || 'Failed to stop training', 'danger');
            }
        } catch (error) {
            console.error('Error stopping training:', error);
            this.showAlert('Error stopping training', 'danger');
        } finally {
            this.hideLoading();
        }
    }
    
    updateTrainingButtons(isTraining) {
        document.getElementById('start-training-btn').disabled = isTraining;
        document.getElementById('stop-training-btn').disabled = !isTraining;
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
    
    updateTrainingMetrics(data) {
        if (!data) {
            console.error('No training metrics data provided');
            return;
        }
        
        console.log('Updating training metrics with:', data);
        
        // Update metric cards with proper data extraction
        const formatValue = (value, decimals = 4) => {
            if (value === null || value === undefined) return '-';
            return typeof value === 'number' ? value.toFixed(decimals) : value;
        };
        
        // Update iteration
        const currentIterationElement = document.getElementById('current-iteration');
        if (currentIterationElement) {
            currentIterationElement.textContent = data.current_iteration || '-';
        }
        
        // Update loss values
        const trainLoss = document.getElementById('train-loss');
        if (trainLoss) {
            trainLoss.textContent = formatValue(data.train_loss);
        }
        
        const valLoss = document.getElementById('val-loss');
        if (valLoss) {
            valLoss.textContent = formatValue(data.val_loss);
        }
        
        // Update perplexity
        const perplexity = document.getElementById('perplexity');
        if (perplexity) {
            perplexity.textContent = formatValue(data.val_perplexity, 2) || 
                                    formatValue(data.train_perplexity, 2) || '-';
        }
        
        // Update tokens per second
        const tokensPerSec = document.getElementById('tokens-per-sec');
        if (tokensPerSec) {
            tokensPerSec.textContent = data.tokens_per_sec ? Math.round(data.tokens_per_sec) : '-';
        }
        
        // Update trained tokens
        const trainedTokens = document.getElementById('trained-tokens');
        if (trainedTokens) {
            trainedTokens.textContent = data.trained_tokens ? data.trained_tokens.toLocaleString() : '-';
        }
        
        // Update memory usage
        const memoryUsage = document.getElementById('memory-usage');
        if (memoryUsage) {
            memoryUsage.textContent = formatValue(data.peak_memory_gb, 1);
        }
        
        // Update learning rate
        const learningRate = document.getElementById('learning-rate');
        if (learningRate) {
            learningRate.textContent = data.learning_rate ? 
                (typeof data.learning_rate === 'number' ? data.learning_rate.toExponential(2) : data.learning_rate) : 
                '-';
        }
        
        // Update warmup steps
        const warmupSteps = document.getElementById('warmup-steps');
        if (warmupSteps) {
            warmupSteps.textContent = data.warmup_steps || (data.config && data.config.warmup_steps) || '-';
        }
        
        // Update LR decay factor
        const lrDecayFactor = document.getElementById('lr-decay-factor');
        if (lrDecayFactor) {
            lrDecayFactor.textContent = data.lr_decay_factor || (data.config && data.config.lr_decay_factor) || '-';
        }
        
        // Update weight decay
        const weightDecay = document.getElementById('weight-decay');
        if (weightDecay) {
            weightDecay.textContent = data.weight_decay || (data.config && data.config.weight_decay) || '-';
        }
        
        // Calculate epoch and progress
        const iterationValue = data.current_iteration || 0;
        const maxIterations = data.max_iterations || 500;
        const progress = maxIterations > 0 ? (iterationValue / maxIterations) * 100 : 0;
        
        // Update progress bar
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
        
        if (progressText) {
            progressText.textContent = `${progress.toFixed(1)}%`;
        }
        
        // Calculate epoch done (approximate)
        const epochDone = data.training_progress ? (data.training_progress * 100).toFixed(2) : progress.toFixed(1);
        const epochElement = document.getElementById('epoch-done');
        if (epochElement) {
            epochElement.textContent = epochDone;
        }
        
        // Update elapsed & remaining time
        const elapsedTime = document.getElementById('elapsed-time');
        const etaTime = document.getElementById('eta-time');
        
        if (elapsedTime && data.elapsed_minutes !== undefined) {
            elapsedTime.textContent = `E.Time: ${Math.round(data.elapsed_minutes)}m`;
        }
        
        if (etaTime && data.eta_minutes !== undefined) {
            etaTime.textContent = `R.Time: ${Math.round(data.eta_minutes)}m`;
        }
        
        // Update best checkpoints
        if (data.best_checkpoints && data.best_checkpoints.length > 0) {
            this.updateBestCheckpoints(data.best_checkpoints);
        }
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
        try {
            const response = await fetch('/api/dashboard/realtime');
            const data = await response.json();
            
            console.log('Dashboard data received:', data);
            
            if (data.active) {
                // Update training metrics with current values and config
                if (data.current_values) {
                    this.updateTrainingMetrics({...data.current_values, config: data.config});
                }
                
                // Update training status
                this.updateTrainingStatus(data);
                
                // Render charts if available
                if (data.charts) {
                    this.renderCharts(data.charts);
                }
            }
        } catch (error) {
            console.error('Error refreshing dashboard:', error);
        }
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
        
        // Refresh the dashboard to get updated charts
        this.refreshDashboard();
    }
    
    async loadModel() {
        const modelSelect = document.getElementById('test-model-select');
        const adapterSelect = document.getElementById('adapter-path');
        const systemPrompt = document.getElementById('system-prompt');
        
        const model = modelSelect.value;
        const adapter = adapterSelect.value;
        const prompt = systemPrompt.value;
        
        if (!model) {
            this.showAlert('Please select a model', 'warning');
            return;
        }
        
        this.showLoading('Loading model...');
        
        try {
            const response = await fetch('/api/model/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: model,
                    adapter_path: adapter,
                    system_prompt: prompt
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.modelLoaded = true;
                this.updateModelButtons(true);
                this.updateModelStatus(model, adapter);
                this.showAlert('Model loaded successfully', 'success');
                
                // Check for training dashboard
                this.checkForTrainingDashboard(model);
                
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
            } else {
                this.showAlert(`Error: ${data.error}`, 'danger');
            }
        } catch (error) {
            console.error('Error loading model:', error);
            this.showAlert('Failed to load model', 'danger');
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
                
                // Render markdown to HTML
                const htmlContent = marked.parse(text);
                element.innerHTML = htmlContent;
                
                // Add Bootstrap classes to rendered elements for better styling
                this.enhanceMarkdownStyling(element);
            } else {
                console.log('📝 Content doesn\'t appear to be markdown, using plain text');
                element.innerText = text;
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
                streaming: streaming,
                // Add model type hint for backend
                is_base_model: isBaseModel
            };
            
            console.log('🚀 Sending request:', requestBody);
            
            if (streaming) {
                // Handle streaming response
                await this.handleStreamingGeneration(requestBody, botBubble);
            } else {
                // Handle non-streaming response (original behavior)
                const response = await fetch('/api/model/generate', {
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
            // Connect directly to model server for streaming
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
                                
                                // Auto-scroll to bottom
                                const chatHistory = document.getElementById('chat-history');
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            } else if (data.type === 'complete') {
                                // Generation complete - capture token counts
                                console.log(`🎉 Streaming generation completed in ${data.generation_time?.toFixed(2)}s`);
                                
                                // Update token counts from server response
                                if (data.prompt_tokens && data.completion_tokens) {
                                    this.promptTokens = data.prompt_tokens;
                                    this.completionTokens = data.completion_tokens;
                                    this.currentTokenCount = data.total_tokens;
                                    
                                    // Add to conversation total
                                    this.conversationTokens += this.currentTokenCount;
                                    
                                    console.log(`📊 Tokens: ${this.promptTokens} prompt + ${this.completionTokens} completion = ${this.currentTokenCount} total (conversation: ${this.conversationTokens})`);
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
            
            // Update stats display
            this.updateChatStats();
            
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
            
            // Add to conversation total
            this.conversationTokens += this.currentTokenCount;
            
            console.log(`📊 Tokens: ${this.promptTokens} prompt + ${this.completionTokens} completion = ${this.currentTokenCount} total (conversation: ${this.conversationTokens})`);
        } else {
            botBubble.classList.remove('chat-assistant');
            botBubble.classList.add('list-group-item-danger');
            botBubble.innerText = `Error: ${data.error}`;
        }
        chatHistory.scrollTop = chatHistory.scrollHeight;
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
            const tokenPercentage = contextWindow ? 
                ` [${Math.round((this.conversationTokens / contextWindow) * 100)}%]` : '';
            
            statsHtml = `${turns} turn${turns !== 1 ? 's' : ''} | ${this.conversationTokens} tokens${tokenPercentage}`;
            
            // Add breakdown if we have prompt/completion details
            if (this.promptTokens && this.completionTokens) {
                statsHtml += `<br><i style="font-size: 0.85em;">(prompt: ${this.promptTokens}, completion: ${this.completionTokens})</i>`;
            }
        } else {
            statsHtml = `${turns} turn${turns !== 1 ? 's' : ''}`;
        }
        
        const chatStats = document.getElementById('chat-stats');
        chatStats.innerHTML = statsHtml;
        
        // Visual indicator for token usage (based on conversation total)
        chatStats.classList.remove('text-warning', 'text-danger');
        
        if (this.conversationTokens) {
            const contextWindow = parseInt(document.getElementById('max-kv-size').value);
            if (contextWindow) {
                const tokenRatio = this.conversationTokens / contextWindow;
                if (tokenRatio > 0.9) {
                    chatStats.classList.add('text-danger');
                } else if (tokenRatio > 0.7) {
                    chatStats.classList.add('text-warning');
                }
            }
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
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        
        if (loadingMessage) {
            loadingMessage.textContent = message;
        }
        
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
        }
    }
    
    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
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
        .then(rawData => {
            // Extract metadata for display
            let metadataHtml = '';
            if (rawData.config) {
                const config = rawData.config;
                metadataHtml = `
                    <div class="alert alert-info mb-3">
                        <h6 class="mb-2"><i class="fas fa-info-circle me-2"></i>Training Metadata</h6>
                        <div class="row">
                            <div class="col-md-6">
                                <small><strong>Model:</strong> ${config.base_model || 'N/A'}</small><br>
                                <small><strong>Type:</strong> ${config.training_type || 'N/A'}</small><br>
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
            
            // Show the raw JSON in the modal with metadata
            const logsContent = document.getElementById('logs-content');
            
            // Add metadata section before the JSON content
            logsContent.innerHTML = metadataHtml;
            
            // Add a pre element for the JSON content
            const jsonPre = document.createElement('pre');
            jsonPre.className = 'json-content';
            jsonPre.textContent = JSON.stringify(rawData, null, 2);
            logsContent.appendChild(jsonPre);
            
            // Setup copy button
            document.getElementById('copy-logs-btn').onclick = () => {
                navigator.clipboard.writeText(logsContent.textContent)
                    .then(() => this.showTooltip('copy-logs-btn', 'Copied!'))
                    .catch(err => this.showAlert('Failed to copy logs: ' + err, 'danger'));
            };
            
            // Show the modal
            const logsModal = new bootstrap.Modal(document.getElementById('logsModal'));
            logsModal.show();
            
            // Now also fetch the processed data for charts
            return fetch('/api/dashboard/historical', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: logFile })
            });
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            this.hideLoading();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Switch to monitoring tab and display the data
            this.switchToMonitoringTab();
            
            // Clear existing charts
            const chartContainer = document.getElementById('training-charts');
            if (chartContainer) {
                chartContainer.innerHTML = '<div class="text-center"><h5>Historical Training Data</h5></div>';
            }
            
            // Display the historical data using the same format as live monitoring
            if (data.charts && Object.keys(data.charts).length > 0) {
                this.displayHistoricalCharts(data.charts);
                this.showAlert(`Successfully loaded training logs from ${logFile}`, 'success');
            } else {
                this.showAlert('No chart data found in the training logs', 'warning');
            }
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
            this.updateTrainingMetrics(data.summary);
            
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
                max_kv_size: maxKvSize
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
            }

            // Load conversation messages
            let messageCount = 0;
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
                }
            }

            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;

            // Enable relevant buttons
            if (messageCount > 0) {
                document.getElementById('save-chat-btn').disabled = false;
                document.getElementById('clear-chat-btn').disabled = false;
            }

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
}

// Add global variable at the top of the file
let trainingInterface;

document.addEventListener('DOMContentLoaded', () => {
    trainingInterface = new TrainingInterface();
    window.trainingInterface = trainingInterface; // Make it available globally

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