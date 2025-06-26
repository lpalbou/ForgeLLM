/**
 * Models Component
 * 
 * Handles model management, publishing, and details.
 */

class ModelsComponent {
    constructor() {
        // Models list container
        this.modelsContainer = document.getElementById('models-container');
        this.noModelsMessage = document.getElementById('no-models-message');
        this.modelsList = document.getElementById('models-list');
        
        // Model details container
        this.modelDetailsContainer = document.getElementById('model-details-container');
        this.modelDetailsContent = document.getElementById('model-details-content');
        this.backToListBtn = document.getElementById('back-to-models-list');
        
        // Publish model form
        this.publishModelForm = document.getElementById('publish-model-form');
        this.publishModelBtn = document.getElementById('publish-model-btn');
        
        // Current model ID
        this.currentModelId = null;
    }
    
    /**
     * Initialize the models component
     */
    init() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Load models list
        this.loadModelsList();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Back to list button
        if (this.backToListBtn) {
            this.backToListBtn.addEventListener('click', () => {
                this.showModelsList();
            });
        }
        
        // Publish model button
        if (this.publishModelBtn) {
            this.publishModelBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.publishModel();
            });
        }
        
        // Listen for model details request
        document.addEventListener('show-model-details', (event) => {
            if (event.detail && event.detail.model) {
                this.showModelDetails(event.detail.model);
            }
        });
    }
    
    /**
     * Load models list
     */
    async loadModelsList() {
        try {
            window.app.showLoading('Loading models...');
            
            // Get models from API
            const response = await apiService.getCPTModels();
            
            window.app.hideLoading();
            
            if (response.models && response.models.length > 0) {
                this.renderModelsList(response.models);
            } else {
                this.showNoModelsMessage();
            }
        } catch (error) {
            window.app.hideLoading();
            console.error('Failed to load models:', error);
            window.app.showAlert('Failed to load models', 'danger');
            this.showNoModelsMessage();
        }
    }
    
    /**
     * Render models list
     * 
     * @param {Array} models - List of models
     */
    renderModelsList(models) {
        if (!this.modelsList) {
            return;
        }
        
        // Clear existing content
        this.modelsList.innerHTML = '';
        
        // Hide no models message
        if (this.noModelsMessage) {
            this.noModelsMessage.classList.add('d-none');
        }
        
        // Show models list
        this.modelsList.classList.remove('d-none');
        
        // Create table
        const table = document.createElement('table');
        table.className = 'table table-hover';
        
        // Create table header
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Checkpoint</th>
                <th>Created</th>
                <th>Actions</th>
            </tr>
        `;
        table.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        models.forEach(model => {
            const tr = document.createElement('tr');
            
            // Format date
            const createdDate = model.created_at ? formatUtil.formatDate(model.created_at) : 'N/A';
            
            // Determine model type
            const modelType = model.is_lora ? 'LoRA' : 'Full';
            
            tr.innerHTML = `
                <td>${formatUtil.formatModelName(model.name)}</td>
                <td>${modelType}</td>
                <td>${model.checkpoint || 'N/A'}</td>
                <td>${createdDate}</td>
                <td>
                    <div class="btn-group btn-group-sm" role="group">
                        <button type="button" class="btn btn-primary btn-details" data-model-id="${model.id}">
                            <i class="bi bi-info-circle"></i> Details
                        </button>
                        <button type="button" class="btn btn-success btn-generate" data-model-id="${model.id}">
                            <i class="bi bi-chat-dots"></i> Generate
                        </button>
                        <button type="button" class="btn btn-info text-white btn-publish" data-model-id="${model.id}">
                            <i class="bi bi-cloud-upload"></i> Publish
                        </button>
                    </div>
                </td>
            `;
            
            // Add event listeners
            const detailsBtn = tr.querySelector('.btn-details');
            if (detailsBtn) {
                detailsBtn.addEventListener('click', () => {
                    this.showModelDetails(model.id);
                });
            }
            
            const generateBtn = tr.querySelector('.btn-generate');
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
            
            const publishBtn = tr.querySelector('.btn-publish');
            if (publishBtn) {
                publishBtn.addEventListener('click', () => {
                    this.prepareModelPublish(model);
                });
            }
            
            tbody.appendChild(tr);
        });
        
        table.appendChild(tbody);
        this.modelsList.appendChild(table);
    }
    
    /**
     * Show no models message
     */
    showNoModelsMessage() {
        if (this.noModelsMessage) {
            this.noModelsMessage.classList.remove('d-none');
        }
        
        if (this.modelsList) {
            this.modelsList.classList.add('d-none');
        }
    }
    
    /**
     * Show models list
     */
    showModelsList() {
        if (this.modelsContainer) {
            this.modelsContainer.classList.remove('d-none');
        }
        
        if (this.modelDetailsContainer) {
            this.modelDetailsContainer.classList.add('d-none');
        }
    }
    
    /**
     * Show model details
     * 
     * @param {string} modelId - Model ID
     */
    async showModelDetails(modelId) {
        try {
            window.app.showLoading('Loading model details...');
            
            // Get model details from API
            const response = await apiService.getModelDetails(modelId);
            
            window.app.hideLoading();
            
            if (response.model) {
                this.renderModelDetails(response.model);
                
                // Hide models list
                if (this.modelsContainer) {
                    this.modelsContainer.classList.add('d-none');
                }
                
                // Show model details
                if (this.modelDetailsContainer) {
                    this.modelDetailsContainer.classList.remove('d-none');
                }
                
                // Store current model ID
                this.currentModelId = modelId;
            } else {
                window.app.showAlert('Model not found', 'danger');
            }
        } catch (error) {
            window.app.hideLoading();
            console.error('Failed to load model details:', error);
            window.app.showAlert('Failed to load model details', 'danger');
        }
    }
    
    /**
     * Render model details
     * 
     * @param {Object} model - Model data
     */
    renderModelDetails(model) {
        if (!this.modelDetailsContent) {
            return;
        }
        
        // Clear existing content
        this.modelDetailsContent.innerHTML = '';
        
        // Format date
        const createdDate = model.created_at ? formatUtil.formatDate(model.created_at) : 'N/A';
        
        // Determine model type
        const modelType = model.is_lora ? 'LoRA' : 'Full';
        
        // Create model info card
        const infoCard = document.createElement('div');
        infoCard.className = 'card mb-4';
        infoCard.innerHTML = `
            <div class="card-header">
                <h5 class="mb-0">${formatUtil.formatModelName(model.name)}</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-2">
                            <strong>Model Type:</strong> ${modelType}
                        </div>
                        <div class="mb-2">
                            <strong>Checkpoint:</strong> ${model.checkpoint || 'N/A'}
                        </div>
                        <div class="mb-2">
                            <strong>Iteration:</strong> ${formatUtil.formatNumber(model.iteration || 0)}
                        </div>
                        <div class="mb-2">
                            <strong>Created:</strong> ${createdDate}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-2">
                            <strong>Training Loss:</strong> ${model.train_loss ? formatUtil.formatDecimal(model.train_loss, 4) : 'N/A'}
                        </div>
                        <div class="mb-2">
                            <strong>Validation Loss:</strong> ${model.val_loss ? formatUtil.formatDecimal(model.val_loss, 4) : 'N/A'}
                        </div>
                        <div class="mb-2">
                            <strong>Training Perplexity:</strong> ${model.train_perplexity ? formatUtil.formatDecimal(model.train_perplexity, 2) : 'N/A'}
                        </div>
                        <div class="mb-2">
                            <strong>Validation Perplexity:</strong> ${model.val_perplexity ? formatUtil.formatDecimal(model.val_perplexity, 2) : 'N/A'}
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-success me-2" id="btn-generate-model">
                        <i class="bi bi-chat-dots"></i> Generate
                    </button>
                    <button class="btn btn-info text-white me-2" id="btn-publish-model-details">
                        <i class="bi bi-cloud-upload"></i> Publish
                    </button>
                    <button class="btn btn-danger" id="btn-delete-model">
                        <i class="bi bi-trash"></i> Delete
                    </button>
                </div>
            </div>
        `;
        
        // Add event listeners
        const generateBtn = infoCard.querySelector('#btn-generate-model');
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
        
        const publishBtn = infoCard.querySelector('#btn-publish-model-details');
        if (publishBtn) {
            publishBtn.addEventListener('click', () => {
                this.prepareModelPublish(model);
            });
        }
        
        const deleteBtn = infoCard.querySelector('#btn-delete-model');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                this.deleteModel(model.id);
            });
        }
        
        this.modelDetailsContent.appendChild(infoCard);
        
        // Create training metrics card if available
        if (model.metrics && Object.keys(model.metrics).length > 0) {
            const metricsCard = document.createElement('div');
            metricsCard.className = 'card mb-4';
            metricsCard.innerHTML = `
                <div class="card-header">
                    <h5 class="mb-0">Training Metrics</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <div id="model-loss-chart" style="height: 300px;"></div>
                        </div>
                        <div class="col-md-6">
                            <div id="model-perplexity-chart" style="height: 300px;"></div>
                        </div>
                    </div>
                </div>
            `;
            
            this.modelDetailsContent.appendChild(metricsCard);
            
            // Create charts
            setTimeout(() => {
                this.createModelCharts(model.metrics);
            }, 100);
        }
    }
    
    /**
     * Create model charts
     * 
     * @param {Object} metrics - Model metrics
     */
    createModelCharts(metrics) {
        if (!metrics) {
            return;
        }
        
        // Prepare data
        const data = {
            iterations: metrics.iterations || [],
            trainLoss: metrics.train_loss || [],
            valLoss: metrics.val_loss || [],
            learningRate: metrics.learning_rate || [],
            tokensPerSec: metrics.tokens_per_sec || [],
            memoryUsage: metrics.memory_gb || []
        };
        
        // Create loss chart
        if (document.getElementById('model-loss-chart')) {
            chartsUtil.createLossChart('model-loss-chart', data);
        }
        
        // Create perplexity chart
        if (document.getElementById('model-perplexity-chart')) {
            chartsUtil.createPerplexityChart('model-perplexity-chart', data);
        }
    }
    
    /**
     * Prepare model publish form
     * 
     * @param {Object} model - Model data
     */
    prepareModelPublish(model) {
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('publish-model-modal'));
        modal.show();
        
        // Fill form fields
        const nameInput = document.getElementById('publish-model-name');
        if (nameInput) {
            nameInput.value = model.name;
        }
        
        const descriptionInput = document.getElementById('publish-model-description');
        if (descriptionInput) {
            descriptionInput.value = '';
        }
        
        const modelIdInput = document.getElementById('publish-model-id');
        if (modelIdInput) {
            modelIdInput.value = model.id;
        }
    }
    
    /**
     * Publish model
     */
    async publishModel() {
        // Get form data
        const modelId = document.getElementById('publish-model-id').value;
        const name = document.getElementById('publish-model-name').value;
        const description = document.getElementById('publish-model-description').value;
        const includeReadme = document.getElementById('publish-model-readme').checked;
        const includeMetrics = document.getElementById('publish-model-metrics').checked;
        const includeDashboard = document.getElementById('publish-model-dashboard').checked;
        
        if (!modelId || !name) {
            window.app.showAlert('Model ID and name are required', 'danger');
            return;
        }
        
        try {
            window.app.showLoading('Publishing model...');
            
            // Publish model
            const response = await apiService.publishModel(modelId, {
                name,
                description,
                include_readme: includeReadme,
                include_metrics: includeMetrics,
                include_dashboard: includeDashboard
            });
            
            window.app.hideLoading();
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('publish-model-modal'));
            if (modal) {
                modal.hide();
            }
            
            // Show success message
            window.app.showAlert('Model published successfully!', 'success');
            
            // Reload models list
            this.loadModelsList();
            
        } catch (error) {
            window.app.hideLoading();
            console.error('Failed to publish model:', error);
            window.app.showAlert(`Failed to publish model: ${error.message}`, 'danger');
        }
    }
    
    /**
     * Delete model
     * 
     * @param {string} modelId - Model ID
     */
    async deleteModel(modelId) {
        if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
            return;
        }
        
        try {
            window.app.showLoading('Deleting model...');
            
            // Delete model
            await apiService.deleteModel(modelId);
            
            window.app.hideLoading();
            
            // Show success message
            window.app.showAlert('Model deleted successfully', 'success');
            
            // Go back to models list
            this.showModelsList();
            
            // Reload models list
            this.loadModelsList();
            
        } catch (error) {
            window.app.hideLoading();
            console.error('Failed to delete model:', error);
            window.app.showAlert(`Failed to delete model: ${error.message}`, 'danger');
        }
    }
    
    /**
     * Called when the models view is activated
     */
    onActivate() {
        // Reload models list
        this.loadModelsList();
    }
} 