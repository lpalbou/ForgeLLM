/**
 * Generation component
 */
class GenerationComponent {
    constructor() {
        this.initialized = false;
        this.modelLoaded = false;
        this.generating = false;
    }

    /**
     * Initialize the component
     */
    init() {
        if (this.initialized) return;
        this.initialized = true;
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Load models
        this.loadModels();
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Add event listeners here
    }

    /**
     * Load models
     */
    async loadModels() {
        try {
            // Load base models
            const baseResponse = await apiService.getBaseModels();
            
            // Update base models UI
            this.updateBaseModelsList(baseResponse.models || []);
            
            // Load CPT models
            const cptResponse = await apiService.getCPTModels();
            
            // Update CPT models UI
            this.updateCPTModelsList(cptResponse.models || []);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    /**
     * Update base models list UI
     * @param {Array} models - Base models
     */
    updateBaseModelsList(models) {
        // Update base models list UI
    }

    /**
     * Update CPT models list UI
     * @param {Array} models - CPT models
     */
    updateCPTModelsList(models) {
        // Update CPT models list UI
    }

    /**
     * Load a model
     * @param {string} modelName - Model name
     * @param {string} adapterPath - Adapter path
     */
    async loadModel(modelName, adapterPath = null) {
        try {
            // Load model
            const response = await apiService.loadModel(modelName, adapterPath);
            
            // Update UI
            if (response.success) {
                // Model loaded successfully
                this.modelLoaded = true;
                this.updateModelStatus(modelName, adapterPath);
            } else {
                // Model failed to load
                console.error('Failed to load model:', response.error);
            }
        } catch (error) {
            console.error('Failed to load model:', error);
        }
    }

    /**
     * Unload the current model
     */
    async unloadModel() {
        try {
            // Unload model
            const response = await apiService.unloadModel();
            
            // Update UI
            if (response.success) {
                // Model unloaded successfully
                this.modelLoaded = false;
                this.updateModelStatus();
            } else {
                // Model failed to unload
                console.error('Failed to unload model:', response.error);
            }
        } catch (error) {
            console.error('Failed to unload model:', error);
        }
    }

    /**
     * Update model status UI
     * @param {string} modelName - Model name
     * @param {string} adapterPath - Adapter path
     */
    updateModelStatus(modelName = null, adapterPath = null) {
        // Update model status UI
    }

    /**
     * Generate text
     * @param {string} prompt - Prompt
     * @param {object} params - Generation parameters
     */
    async generateText(prompt, params = {}) {
        if (!this.modelLoaded || this.generating) {
            return;
        }
        
        this.generating = true;
        
        try {
            // Generate text
            const response = await apiService.generateText({
                prompt,
                ...params
            });
            
            // Update UI
            if (response.success) {
                // Text generated successfully
                this.updateGeneratedText(response.text);
            } else {
                // Text generation failed
                console.error('Failed to generate text:', response.error);
            }
        } catch (error) {
            console.error('Failed to generate text:', error);
        } finally {
            this.generating = false;
        }
    }

    /**
     * Update generated text UI
     * @param {string} text - Generated text
     */
    updateGeneratedText(text) {
        // Update generated text UI
    }

    /**
     * Called when the generation tab is activated
     */
    onActivate() {
        // Refresh models
        this.loadModels();
    }
}

// Create a singleton instance
const generationComponent = new GenerationComponent();

// Add this function to handle the initialization of the Testing tab
function initTestingTab() {
    // Check if there's a pending test session from the Compare tab
    const pendingTestSession = localStorage.getItem('forge-test-session');
    if (pendingTestSession) {
        try {
            const { model, adapter } = JSON.parse(pendingTestSession);
            console.log('Found pending test session:', { model, adapter });
            
            // Set a small delay to ensure the model dropdown has been populated
            setTimeout(() => {
                // Set the model in the dropdown
                const modelSelect = document.getElementById('model-select');
                if (modelSelect) {
                    for (let i = 0; i < modelSelect.options.length; i++) {
                        if (modelSelect.options[i].value === model || 
                            modelSelect.options[i].text.includes(model) || 
                            model.includes(modelSelect.options[i].value)) {
                            modelSelect.selectedIndex = i;
                            console.log(`Selected model: ${modelSelect.options[i].value}`);
                            
                            // Trigger change event to load adapters
                            modelSelect.dispatchEvent(new Event('change'));
                            
                            // Set another timeout to wait for adapters to load
                            setTimeout(() => {
                                // Look for the adapter-path dropdown instead of adapter-select
                                const adapterSelect = document.getElementById('adapter-path');
                                if (!adapterSelect) {
                                    console.log('Looking for alternative adapter select element...');
                                    // Try alternative IDs
                                    const possibleSelects = ['adapter-select', 'adapter-path', 'adapter_select', 'adapter_path'];
                                    for (const id of possibleSelects) {
                                        const select = document.getElementById(id);
                                        if (select) {
                                            console.log(`Found adapter select with ID: ${id}`);
                                            selectAdapter(select, adapter);
                                            break;
                                        }
                                    }
                                    
                                    // If still not found, try using a CSS selector
                                    if (!adapterSelect) {
                                        const selects = document.querySelectorAll('select');
                                        console.log(`Found ${selects.length} select elements`);
                                        selects.forEach((select, index) => {
                                            console.log(`Select ${index}: id=${select.id}, name=${select.name}`);
                                            if (select.id.includes('adapter') || select.name.includes('adapter') ||
                                                (select.options.length > 0 && Array.from(select.options).some(opt => opt.value.includes('models/')))) {
                                                console.log(`Found likely adapter select: ${select.id || select.name}`);
                                                selectAdapter(select, adapter);
                                            }
                                        });
                                    }
                                } else {
                                    selectAdapter(adapterSelect, adapter);
                                }
                                
                                function selectAdapter(selectElement, adapterValue) {
                                    console.log('Adapter options:', Array.from(selectElement.options).map(o => o.value));
                                    
                                    // First try exact match
                                    let found = false;
                                    for (let j = 0; j < selectElement.options.length; j++) {
                                        if (selectElement.options[j].value === adapterValue) {
                                            selectElement.selectedIndex = j;
                                            // Trigger change event
                                            selectElement.dispatchEvent(new Event('change'));
                                            found = true;
                                            console.log('Found exact adapter match:', adapterValue);
                                            break;
                                        }
                                    }
                                    
                                    // If no exact match, try to find a partial match
                                    if (!found) {
                                        console.log('No exact adapter match found, trying partial match');
                                        for (let j = 0; j < selectElement.options.length; j++) {
                                            // Check if the adapter ID is contained within the option value or text
                                            const optionValue = selectElement.options[j].value;
                                            const optionText = selectElement.options[j].text;
                                            
                                            if (optionValue.includes(adapterValue) || 
                                                adapterValue.includes(optionValue) ||
                                                optionText.includes(adapterValue) ||
                                                adapterValue.includes(optionText)) {
                                                selectElement.selectedIndex = j;
                                                // Trigger change event
                                                selectElement.dispatchEvent(new Event('change'));
                                                found = true;
                                                console.log('Found partial adapter match:', optionValue);
                                                break;
                                            }
                                        }
                                    }
                                    
                                    // If still not found, try matching just the session name part
                                    if (!found) {
                                        console.log('No partial match found, trying session name match');
                                        // Extract just the session name part (without the full path)
                                        const sessionNameMatch = adapterValue.match(/([^\/]+)$/);
                                        if (sessionNameMatch && sessionNameMatch[1]) {
                                            const sessionName = sessionNameMatch[1];
                                            console.log('Extracted session name:', sessionName);
                                            
                                            for (let j = 0; j < selectElement.options.length; j++) {
                                                const optionValue = selectElement.options[j].value;
                                                const optionText = selectElement.options[j].text;
                                                
                                                if (optionValue.includes(sessionName) || 
                                                    sessionName.includes(optionValue) ||
                                                    optionText.includes(sessionName) ||
                                                    sessionName.includes(optionText)) {
                                                    selectElement.selectedIndex = j;
                                                    // Trigger change event
                                                    selectElement.dispatchEvent(new Event('change'));
                                                    found = true;
                                                    console.log('Found session name match:', optionValue);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    
                                    if (!found) {
                                        console.error('Could not find matching adapter:', adapterValue);
                                    }
                                }
                            }, 1000); // Increase timeout to ensure adapters are loaded
                            break;
                        }
                    }
                } else {
                    console.error('Model select element not found');
                }
            }, 1000); // Increase timeout to ensure models are loaded
        } catch (error) {
            console.error('Error processing test session:', error);
        } finally {
            // Clear the pending test session
            localStorage.removeItem('forge-test-session');
        }
    }
}

// Add a listener for tab activation to initialize the Testing tab
document.addEventListener('DOMContentLoaded', () => {
    // Listen for tab activation events
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (event) => {
            if (event.target.id === 'testing-tab') {
                initTestingTab();
            }
        });
    });
    
    // Also check on initial load in case we're starting on the Testing tab
    if (document.querySelector('#testing-tab').classList.contains('active')) {
        initTestingTab();
    }
}); 