/**
 * Dashboard Component
 * 
 * Displays training overview and metrics.
 */

class DashboardComponent {
    constructor() {
        this.activeTrainingSection = document.getElementById('active-jobs-container');
        this.noJobsMessage = document.getElementById('no-jobs-message');
        this.activeJobsList = document.getElementById('active-jobs-list');
        this.recentModelsContainer = document.getElementById('recent-models-container');
        this.noModelsMessage = document.getElementById('no-models-message');
        this.recentModelsList = document.getElementById('recent-models-list');
        
        // Quick action buttons
        this.btnNewTraining = document.getElementById('btn-new-training');
        this.btnQuickGenerate = document.getElementById('btn-quick-generate');
        this.btnPublishModel = document.getElementById('btn-publish-model');
    }
    
    /**
     * Initialize the dashboard component
     */
    init() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadDashboardData();
        
        // Set up periodic updates
        this.startPeriodicUpdates();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Quick action buttons
        if (this.btnNewTraining) {
            this.btnNewTraining.addEventListener('click', () => {
                window.app.showView('training');
            });
        }
        
        if (this.btnQuickGenerate) {
            this.btnQuickGenerate.addEventListener('click', () => {
                window.app.showView('generate');
            });
        }
        
        if (this.btnPublishModel) {
            this.btnPublishModel.addEventListener('click', () => {
                window.app.showView('models');
            });
        }
        
        // Listen for training updates
        document.addEventListener('training-update', (event) => {
            this.updateActiveTraining(event.detail);
        });
        
        document.addEventListener('training-finished', (event) => {
            this.handleTrainingFinished(event.detail);
        });
    }
    
    /**
     * Load dashboard data
     */
    async loadDashboardData() {
        try {
            // Check for active training
            const hasActiveTraining = await trainingService.checkActiveTraining();
            
            if (hasActiveTraining) {
                this.showActiveTraining();
            } else {
                this.hideActiveTraining();
            }
            
            // Load recent models
            await this.loadRecentModels();
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }
    
    /**
     * Start periodic updates
     */
    startPeriodicUpdates() {
        // Update dashboard every 30 seconds
        setInterval(() => {
            this.loadDashboardData();
        }, 30000);
    }
    
    /**
     * Show active training section
     */
    showActiveTraining() {
        if (this.noJobsMessage) {
            this.noJobsMessage.classList.add('d-none');
        }
        
        if (this.activeJobsList) {
            this.activeJobsList.classList.remove('d-none');
        }
    }
    
    /**
     * Hide active training section
     */
    hideActiveTraining() {
        if (this.noJobsMessage) {
            this.noJobsMessage.classList.remove('d-none');
        }
        
        if (this.activeJobsList) {
            this.activeJobsList.classList.add('d-none');
            this.activeJobsList.innerHTML = '';
        }
    }
    
    /**
     * Update active training display
     * 
     * @param {Object} data - Training data
     */
    updateActiveTraining(data) {
        if (!data || !this.activeJobsList) {
            return;
        }
        
        this.showActiveTraining();
        
        // Clear existing content
        this.activeJobsList.innerHTML = '';
        
        // Create training card
        const card = document.createElement('div');
        card.className = 'card mb-3';
        
        // Format progress
        const progress = data.progress || 0;
        const progressClass = progress < 33 ? 'bg-info' : progress < 66 ? 'bg-primary' : 'bg-success';
        
        // Format metrics
        const trainLoss = data.train_loss !== undefined ? formatUtil.formatDecimal(data.train_loss, 4) : '—';
        const valLoss = data.val_loss !== undefined ? formatUtil.formatDecimal(data.val_loss, 4) : '—';
        const trainPpl = data.train_perplexity !== undefined ? formatUtil.formatDecimal(data.train_perplexity, 2) : '—';
        const valPpl = data.val_perplexity !== undefined ? formatUtil.formatDecimal(data.val_perplexity, 2) : '—';
        const tokensPerSec = data.tokens_per_sec !== undefined ? formatUtil.formatNumber(Math.round(data.tokens_per_sec)) : '—';
        const memoryUsage = data.peak_memory_gb !== undefined ? `${formatUtil.formatDecimal(data.peak_memory_gb, 1)} GB` : '—';
        
        // Format time estimates
        let timeInfo = '';
        if (data.elapsed_minutes !== undefined) {
            const elapsed = formatUtil.formatDuration(data.elapsed_minutes * 60);
            
            if (data.eta_minutes !== undefined) {
                const eta = formatUtil.formatDuration(data.eta_minutes * 60);
                timeInfo = `<div class="text-muted">Elapsed: ${elapsed} | ETA: ${eta}</div>`;
            } else {
                timeInfo = `<div class="text-muted">Elapsed: ${elapsed}</div>`;
            }
        }
        
        // Create card content
        card.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">Training: ${formatUtil.formatModelName(data.model_name || 'Model')}</h5>
                <span class="badge bg-primary">Active</span>
            </div>
            <div class="card-body">
                <div class="progress mb-3" style="height: 20px;">
                    <div class="progress-bar ${progressClass}" role="progressbar" style="width: ${progress}%;" 
                        aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100">
                        ${Math.round(progress)}%
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-2">
                            <strong>Iteration:</strong> ${formatUtil.formatNumber(data.current_iteration || 0)} / ${formatUtil.formatNumber(data.max_iterations || 0)}
                        </div>
                        <div class="mb-2">
                            <strong>Train Loss:</strong> ${trainLoss} (ppl: ${trainPpl})
                        </div>
                        <div class="mb-2">
                            <strong>Valid Loss:</strong> ${valLoss} (ppl: ${valPpl})
                        </div>
                        ${timeInfo}
                    </div>
                    <div class="col-md-6">
                        <div class="mb-2">
                            <strong>Speed:</strong> ${tokensPerSec} tokens/sec
                        </div>
                        <div class="mb-2">
                            <strong>Memory:</strong> ${memoryUsage}
                        </div>
                        <div class="mb-2">
                            <strong>Learning Rate:</strong> ${data.learning_rate ? data.learning_rate.toExponential(2) : '—'}
                        </div>
                        ${data.epoch_done !== undefined ? `<div class="mb-2"><strong>Epoch:</strong> ${formatUtil.formatDecimal(data.epoch_done, 2)}</div>` : ''}
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-sm btn-primary me-2" id="btn-view-training">
                        <i class="bi bi-graph-up"></i> View Details
                    </button>
                    <button class="btn btn-sm btn-danger" id="btn-stop-training">
                        <i class="bi bi-stop-fill"></i> Stop Training
                    </button>
                </div>
            </div>
        `;
        
        // Add event listeners
        const btnViewTraining = card.querySelector('#btn-view-training');
        if (btnViewTraining) {
            btnViewTraining.addEventListener('click', () => {
                window.app.showView('training');
            });
        }
        
        const btnStopTraining = card.querySelector('#btn-stop-training');
        if (btnStopTraining) {
            btnStopTraining.addEventListener('click', async () => {
                if (confirm('Are you sure you want to stop the training?')) {
                    try {
                        window.app.showLoading('Stopping training...');
                        await trainingService.stopTraining();
                        window.app.hideLoading();
                        window.app.showAlert('Training stopped successfully.', 'success');
                        this.hideActiveTraining();
                    } catch (error) {
                        window.app.hideLoading();
                        window.app.showAlert(`Failed to stop training: ${error.message}`, 'danger');
                    }
                }
            });
        }
        
        // Add card to list
        this.activeJobsList.appendChild(card);
    }
    
    /**
     * Handle training finished event
     * 
     * @param {Object} data - Training finished data
     */
    handleTrainingFinished(data) {
        // Hide active training section
        this.hideActiveTraining();
        
        // Show alert
        window.app.showAlert('Training completed successfully!', 'success');
        
        // Reload recent models
        this.loadRecentModels();
    }
    
    /**
     * Load recent models
     */
    async loadRecentModels() {
        try {
            // Get CPT models
            const response = await apiService.getCPTModels();
            
            if (response.models && response.models.length > 0) {
                this.showRecentModels(response.models);
            } else {
                this.hideRecentModels();
            }
            
        } catch (error) {
            console.error('Failed to load recent models:', error);
            this.hideRecentModels();
        }
    }
    
    /**
     * Show recent models section
     * 
     * @param {Array} models - List of models
     */
    showRecentModels(models) {
        if (this.noModelsMessage) {
            this.noModelsMessage.classList.add('d-none');
        }
        
        if (this.recentModelsList) {
            this.recentModelsList.classList.remove('d-none');
            this.recentModelsList.innerHTML = '';
            
            // Show up to 5 most recent models
            const recentModels = models.slice(0, 5);
            
            recentModels.forEach(model => {
                const card = document.createElement('div');
                card.className = 'card mb-2';
                
                card.innerHTML = `
                    <div class="card-body">
                        <h5 class="card-title">${formatUtil.formatModelName(model.name)}</h5>
                        <p class="card-text">
                            <small class="text-muted">
                                Checkpoint: ${model.checkpoint || 'N/A'} 
                                (Iteration: ${formatUtil.formatNumber(model.iteration || 0)})
                            </small>
                        </p>
                        <div class="d-flex">
                            <button class="btn btn-sm btn-primary me-2" data-model-id="${model.id}">
                                <i class="bi bi-chat-dots"></i> Generate
                            </button>
                            <button class="btn btn-sm btn-secondary" data-model-id="${model.id}">
                                <i class="bi bi-info-circle"></i> Details
                            </button>
                        </div>
                    </div>
                `;
                
                // Add event listeners
                const generateBtn = card.querySelector('.btn-primary');
                if (generateBtn) {
                    generateBtn.addEventListener('click', () => {
                        // Switch to generate view and load this model
                        window.app.showView('generate');
                        
                        // Dispatch event to load model
                        document.dispatchEvent(new CustomEvent('load-model', { 
                            detail: { model: model.id } 
                        }));
                    });
                }
                
                const detailsBtn = card.querySelector('.btn-secondary');
                if (detailsBtn) {
                    detailsBtn.addEventListener('click', () => {
                        // Switch to models view and show details
                        window.app.showView('models');
                        
                        // Dispatch event to show model details
                        document.dispatchEvent(new CustomEvent('show-model-details', { 
                            detail: { model: model.id } 
                        }));
                    });
                }
                
                this.recentModelsList.appendChild(card);
            });
        }
    }
    
    /**
     * Hide recent models section
     */
    hideRecentModels() {
        if (this.noModelsMessage) {
            this.noModelsMessage.classList.remove('d-none');
        }
        
        if (this.recentModelsList) {
            this.recentModelsList.classList.add('d-none');
            this.recentModelsList.innerHTML = '';
        }
    }
    
    /**
     * Called when the dashboard view is activated
     */
    onActivate() {
        // Refresh dashboard data
        this.loadDashboardData();
    }
} 