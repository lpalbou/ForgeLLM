/**
 * Models component
 */
class ModelsComponent {
    constructor() {
        this.initialized = false;
    }

    /**
     * Initialize the component
     */
    init() {
        if (this.initialized) return;
        this.initialized = true;
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Load models list
        this.loadModelsList();
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Add event listeners for model loading form
        const modelLoadForm = document.getElementById('model-load-form');
        if (modelLoadForm) {
            modelLoadForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.loadModel();
            });
        }

        // Add event listener for unload button
        const unloadModelBtn = document.getElementById('unload-model-btn');
        if (unloadModelBtn) {
            unloadModelBtn.addEventListener('click', () => {
                this.unloadModel();
            });
        }
    }

    /**
     * Load models list
     */
    async loadModelsList() {
        try {
            // Load all models
            const response = await apiService.get('models');
            const models = response.models || [];
            
            // Update models UI
            this.updateModelsList(models);
            
            // Load CPT models
            const cptResponse = await apiService.getCPTModels();
            
            // Update CPT models UI
            this.updateCPTModelsList(cptResponse.models || []);
            
            // Load IFT models
            const iftResponse = await apiService.getIFTModels();
            
            // Update IFT models UI
            this.updateIFTModelsList(iftResponse.models || []);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    /**
     * Update models list UI
     * @param {Array} models - All models
     */
    updateModelsList(models) {
        const modelSelect = document.getElementById('test-model-select');
        if (!modelSelect) return;
        
        // Clear existing options
        modelSelect.innerHTML = '<option value="">Select model...</option>';
        
        // Add models to select
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} (${model.type})`;
            modelSelect.appendChild(option);
        });
    }

    /**
     * Update CPT models list UI
     * @param {Array} models - CPT models
     */
    updateCPTModelsList(models) {
        // Update CPT models list UI if needed
        const adapterPath = document.getElementById('adapter-path');
        if (!adapterPath) return;
        
        // Add CPT models to adapter select
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = `CPT: ${model.name}`;
            adapterPath.appendChild(option);
        });
    }

    /**
     * Update IFT models list UI
     * @param {Array} models - IFT models
     */
    updateIFTModelsList(models) {
        // Update IFT models list UI if needed
        const adapterPath = document.getElementById('adapter-path');
        if (!adapterPath) return;
        
        // Add IFT models to adapter select
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = `IFT: ${model.name}`;
            adapterPath.appendChild(option);
        });
    }

    /**
     * Load a model
     */
    async loadModel() {
        const modelSelect = document.getElementById('test-model-select');
        const adapterPath = document.getElementById('adapter-path');
        const systemPrompt = document.getElementById('system-prompt');
        const loadModelBtn = document.getElementById('load-model-btn');
        const unloadModelBtn = document.getElementById('unload-model-btn');
        const generateBtn = document.getElementById('generate-btn');
        
        if (!modelSelect || !adapterPath) return;
        
        const modelName = modelSelect.value;
        const adapter = adapterPath.value;
        const system = systemPrompt ? systemPrompt.value : '';
        
        if (!modelName) {
            alert('Please select a model');
            return;
        }
        
        try {
            // Show loading overlay
            document.getElementById('loading-overlay').classList.remove('d-none');
            document.getElementById('loading-message').textContent = 'Loading model...';
            
            // Disable buttons
            if (loadModelBtn) loadModelBtn.disabled = true;
            
            // Load model
            const response = await apiService.loadModel(modelName, adapter);
            
            if (response.success) {
                // Update UI
                if (unloadModelBtn) unloadModelBtn.disabled = false;
                if (generateBtn) generateBtn.disabled = false;
                
                // Update model status
                this.updateModelStatus(modelName, adapter);
                
                // Show success message
                console.log(`Model ${modelName} loaded successfully`);
            } else {
                // Show error message
                alert(`Failed to load model: ${response.error}`);
                
                // Re-enable buttons
                if (loadModelBtn) loadModelBtn.disabled = false;
            }
        } catch (error) {
            console.error('Failed to load model:', error);
            alert(`Failed to load model: ${error.message}`);
            
            // Re-enable buttons
            if (loadModelBtn) loadModelBtn.disabled = false;
        } finally {
            // Hide loading overlay
            document.getElementById('loading-overlay').classList.add('d-none');
        }
    }

    /**
     * Unload the current model
     */
    async unloadModel() {
        const loadModelBtn = document.getElementById('load-model-btn');
        const unloadModelBtn = document.getElementById('unload-model-btn');
        const generateBtn = document.getElementById('generate-btn');
        
        try {
            // Show loading overlay
            document.getElementById('loading-overlay').classList.remove('d-none');
            document.getElementById('loading-message').textContent = 'Unloading model...';
            
            // Disable buttons
            if (unloadModelBtn) unloadModelBtn.disabled = true;
            
            // Unload model
            const response = await apiService.unloadModel();
            
            if (response.success) {
                // Update UI
                if (loadModelBtn) loadModelBtn.disabled = false;
                if (generateBtn) generateBtn.disabled = true;
                
                // Update model status
                this.updateModelStatus();
                
                // Show success message
                console.log('Model unloaded successfully');
            } else {
                // Show error message
                alert(`Failed to unload model: ${response.error}`);
                
                // Re-enable buttons
                if (unloadModelBtn) unloadModelBtn.disabled = false;
            }
        } catch (error) {
            console.error('Failed to unload model:', error);
            alert(`Failed to unload model: ${error.message}`);
            
            // Re-enable buttons
            if (unloadModelBtn) unloadModelBtn.disabled = false;
        } finally {
            // Hide loading overlay
            document.getElementById('loading-overlay').classList.add('d-none');
        }
    }

    /**
     * Update model status UI
     * @param {string} modelName - Model name
     * @param {string} adapterPath - Adapter path
     */
    updateModelStatus(modelName = null, adapterPath = null) {
        const modelStatus = document.getElementById('model-status');
        if (!modelStatus) return;
        
        if (modelName) {
            // Model is loaded
            modelStatus.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="status-indicator status-running me-2"></div>
                    <div>
                        <h5 class="mb-0">${modelName}</h5>
                        <small class="text-muted">${adapterPath ? `Adapter: ${adapterPath}` : 'No adapter'}</small>
                    </div>
                </div>
            `;
        } else {
            // No model loaded
            modelStatus.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-cloud fa-2x mb-3"></i>
                    <p>No model loaded</p>
                </div>
            `;
        }
    }

    /**
     * Called when the models tab is activated
     */
    onActivate() {
        // Refresh models list
        this.loadModelsList();
    }
}

// Create a singleton instance
const modelsComponent = new ModelsComponent();
window.modelsComponent = modelsComponent; 