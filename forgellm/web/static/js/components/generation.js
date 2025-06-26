/**
 * Generation Component
 * 
 * Handles text generation with trained models.
 */

class GenerationComponent {
    constructor() {
        // Model selection
        this.modelSelect = document.getElementById('generation-model');
        this.refreshModelsBtn = document.getElementById('refresh-models-btn-generation');
        
        // Generation form
        this.generationForm = document.getElementById('generation-form');
        this.promptInput = document.getElementById('generation-prompt');
        this.systemPromptInput = document.getElementById('generation-system-prompt');
        this.maxTokensInput = document.getElementById('generation-max-tokens');
        this.temperatureInput = document.getElementById('generation-temperature');
        this.topPInput = document.getElementById('generation-top-p');
        this.repetitionPenaltyInput = document.getElementById('generation-repetition-penalty');
        this.generateBtn = document.getElementById('generate-btn');
        this.stopGenerationBtn = document.getElementById('stop-generation-btn');
        
        // Generation output
        this.outputContainer = document.getElementById('generation-output-container');
        this.outputText = document.getElementById('generation-output');
        this.clearOutputBtn = document.getElementById('clear-output-btn');
        
        // Generation history
        this.historyContainer = document.getElementById('generation-history-container');
        this.historyList = document.getElementById('generation-history-list');
        this.clearHistoryBtn = document.getElementById('clear-history-btn');
        
        // Generation state
        this.isGenerating = false;
        this.history = [];
        this.currentModelId = null;
    }
    
    /**
     * Initialize the generation component
     */
    init() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Load models
        this.loadModels();
        
        // Load generation history
        this.loadHistory();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Refresh models button
        if (this.refreshModelsBtn) {
            this.refreshModelsBtn.addEventListener('click', () => {
                this.loadModels();
            });
        }
        
        // Generate button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.generate();
            });
        }
        
        // Stop generation button
        if (this.stopGenerationBtn) {
            this.stopGenerationBtn.addEventListener('click', () => {
                this.stopGeneration();
            });
        }
        
        // Clear output button
        if (this.clearOutputBtn) {
            this.clearOutputBtn.addEventListener('click', () => {
                this.clearOutput();
            });
        }
        
        // Clear history button
        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => {
                this.clearHistory();
            });
        }
        
        // Listen for model load request
        document.addEventListener('load-model', (event) => {
            if (event.detail && event.detail.model) {
                this.loadModel(event.detail.model);
            }
        });
    }
    
    /**
     * Load available models
     */
    async loadModels() {
        try {
            window.app.showLoading('Loading models...');
            
            // Get models from API
            const response = await apiService.getCPTModels();
            
            window.app.hideLoading();
            
            if (response.models && response.models.length > 0) {
                this.renderModelOptions(response.models);
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
     * Render model options
     * 
     * @param {Array} models - List of models
     */
    renderModelOptions(models) {
        if (!this.modelSelect) {
            return;
        }
        
        // Clear existing options
        this.modelSelect.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a model';
        defaultOption.disabled = true;
        defaultOption.selected = true;
        this.modelSelect.appendChild(defaultOption);
        
        // Add model options
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = formatUtil.formatModelName(model.name);
            option.dataset.modelType = model.is_lora ? 'lora' : 'full';
            this.modelSelect.appendChild(option);
        });
        
        // Enable the model select
        this.modelSelect.disabled = false;
    }
    
    /**
     * Show no models message
     */
    showNoModelsMessage() {
        if (!this.modelSelect) {
            return;
        }
        
        // Clear existing options
        this.modelSelect.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'No models available';
        defaultOption.disabled = true;
        defaultOption.selected = true;
        this.modelSelect.appendChild(defaultOption);
        
        // Disable the model select
        this.modelSelect.disabled = true;
    }
    
    /**
     * Load a specific model
     * 
     * @param {string} modelId - Model ID
     */
    loadModel(modelId) {
        if (!this.modelSelect) {
            return;
        }
        
        // Find the option with the matching model ID
        const option = Array.from(this.modelSelect.options).find(opt => opt.value === modelId);
        
        if (option) {
            // Select the option
            this.modelSelect.value = modelId;
            
            // Store the current model ID
            this.currentModelId = modelId;
        } else {
            // Model not found, reload models and then select
            this.loadModels().then(() => {
                const newOption = Array.from(this.modelSelect.options).find(opt => opt.value === modelId);
                if (newOption) {
                    this.modelSelect.value = modelId;
                    this.currentModelId = modelId;
                }
            });
        }
    }
    
    /**
     * Generate text
     */
    async generate() {
        if (this.isGenerating) {
            return;
        }
        
        // Get selected model
        const modelId = this.modelSelect.value;
        
        if (!modelId) {
            window.app.showAlert('Please select a model', 'warning');
            return;
        }
        
        // Get prompt
        const prompt = this.promptInput.value.trim();
        
        if (!prompt) {
            window.app.showAlert('Please enter a prompt', 'warning');
            return;
        }
        
        // Get generation parameters
        const systemPrompt = this.systemPromptInput.value.trim();
        const maxTokens = parseInt(this.maxTokensInput.value, 10);
        const temperature = parseFloat(this.temperatureInput.value);
        const topP = parseFloat(this.topPInput.value);
        const repetitionPenalty = parseFloat(this.repetitionPenaltyInput.value);
        
        try {
            // Set generating state
            this.isGenerating = true;
            this.setGeneratingUI(true);
            
            // Clear output
            this.outputText.textContent = '';
            this.outputContainer.classList.remove('d-none');
            
            // Send generation request
            await socketService.generate({
                model: modelId,
                prompt,
                system_prompt: systemPrompt,
                max_tokens: maxTokens,
                temperature,
                top_p: topP,
                repetition_penalty: repetitionPenalty,
                history: this.history
            });
            
            // Add to history
            this.addToHistory({
                prompt,
                system_prompt: systemPrompt,
                max_tokens: maxTokens,
                temperature,
                top_p: topP,
                repetition_penalty: repetitionPenalty,
                timestamp: new Date()
            });
            
        } catch (error) {
            console.error('Failed to generate text:', error);
            window.app.showAlert(`Failed to generate text: ${error.message}`, 'danger');
            
            // Reset generating state
            this.isGenerating = false;
            this.setGeneratingUI(false);
        }
    }
    
    /**
     * Handle generation response
     * 
     * @param {Object} data - Generation response data
     */
    handleGenerationResponse(data) {
        // Update output
        if (this.outputText) {
            this.outputText.textContent = data.response;
        }
        
        // Reset generating state
        this.isGenerating = false;
        this.setGeneratingUI(false);
    }
    
    /**
     * Stop generation
     */
    stopGeneration() {
        if (!this.isGenerating) {
            return;
        }
        
        // Send stop request
        socketService.stopGeneration();
        
        // Reset generating state
        this.isGenerating = false;
        this.setGeneratingUI(false);
    }
    
    /**
     * Set UI for generating state
     * 
     * @param {boolean} isGenerating - Whether text is being generated
     */
    setGeneratingUI(isGenerating) {
        if (this.generateBtn) {
            this.generateBtn.disabled = isGenerating;
        }
        
        if (this.stopGenerationBtn) {
            this.stopGenerationBtn.disabled = !isGenerating;
        }
        
        if (this.modelSelect) {
            this.modelSelect.disabled = isGenerating;
        }
    }
    
    /**
     * Clear output
     */
    clearOutput() {
        if (this.outputText) {
            this.outputText.textContent = '';
        }
        
        if (this.outputContainer) {
            this.outputContainer.classList.add('d-none');
        }
    }
    
    /**
     * Load generation history
     */
    loadHistory() {
        const savedHistory = localStorage.getItem('forgellm_generation_history');
        
        if (savedHistory) {
            try {
                this.history = JSON.parse(savedHistory);
                this.renderHistory();
            } catch (error) {
                console.error('Failed to load generation history:', error);
                this.history = [];
            }
        }
    }
    
    /**
     * Add to history
     * 
     * @param {Object} item - History item
     */
    addToHistory(item) {
        // Add to history
        this.history.unshift(item);
        
        // Limit history size
        if (this.history.length > 20) {
            this.history = this.history.slice(0, 20);
        }
        
        // Save history
        localStorage.setItem('forgellm_generation_history', JSON.stringify(this.history));
        
        // Render history
        this.renderHistory();
    }
    
    /**
     * Render history
     */
    renderHistory() {
        if (!this.historyList) {
            return;
        }
        
        // Clear existing history
        this.historyList.innerHTML = '';
        
        // Show history container
        if (this.historyContainer) {
            this.historyContainer.classList.remove('d-none');
        }
        
        // Add history items
        this.history.forEach((item, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'card mb-2';
            
            // Format timestamp
            const timestamp = formatUtil.formatRelativeTime(item.timestamp);
            
            historyItem.innerHTML = `
                <div class="card-body">
                    <h6 class="card-title">${formatUtil.truncateText(item.prompt, 50)}</h6>
                    <p class="card-text">
                        <small class="text-muted">${timestamp}</small>
                    </p>
                    <button class="btn btn-sm btn-primary btn-use-prompt" data-index="${index}">
                        <i class="bi bi-arrow-clockwise"></i> Use Again
                    </button>
                </div>
            `;
            
            // Add event listener
            const usePromptBtn = historyItem.querySelector('.btn-use-prompt');
            if (usePromptBtn) {
                usePromptBtn.addEventListener('click', () => {
                    this.useHistoryItem(index);
                });
            }
            
            this.historyList.appendChild(historyItem);
        });
    }
    
    /**
     * Use history item
     * 
     * @param {number} index - History item index
     */
    useHistoryItem(index) {
        const item = this.history[index];
        
        if (!item) {
            return;
        }
        
        // Fill form fields
        if (this.promptInput) {
            this.promptInput.value = item.prompt;
        }
        
        if (this.systemPromptInput) {
            this.systemPromptInput.value = item.system_prompt || '';
        }
        
        if (this.maxTokensInput) {
            this.maxTokensInput.value = item.max_tokens || 100;
        }
        
        if (this.temperatureInput) {
            this.temperatureInput.value = item.temperature || 0.7;
        }
        
        if (this.topPInput) {
            this.topPInput.value = item.top_p || 0.9;
        }
        
        if (this.repetitionPenaltyInput) {
            this.repetitionPenaltyInput.value = item.repetition_penalty || 1.1;
        }
    }
    
    /**
     * Clear history
     */
    clearHistory() {
        if (!confirm('Are you sure you want to clear your generation history?')) {
            return;
        }
        
        // Clear history
        this.history = [];
        
        // Save history
        localStorage.removeItem('forgellm_generation_history');
        
        // Hide history container
        if (this.historyContainer) {
            this.historyContainer.classList.add('d-none');
        }
        
        // Clear history list
        if (this.historyList) {
            this.historyList.innerHTML = '';
        }
    }
    
    /**
     * Called when the generation view is activated
     */
    onActivate() {
        // Reload models if needed
        if (!this.currentModelId) {
            this.loadModels();
        }
    }
} 