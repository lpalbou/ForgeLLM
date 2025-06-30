/**
 * Main application
 */
class TrainingInterface {
    constructor() {
        this.initialized = false;
    }

    /**
     * Initialize the application
     */
    init() {
        if (this.initialized) return;
        this.initialized = true;
        
        console.log('Initializing Training Interface...');
        
        // Initialize Socket.IO
        this.initSocket();
        
        // Initialize components
        this.initComponents();
        
        // Initialize navigation
        this.initNavigation();
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Load initial data
        this.loadInitialData();
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        console.log('Training Interface initialized');
    }

    /**
     * Initialize Socket.IO
     */
    initSocket() {
        // Initialize Socket.IO
        socketService.init();
        
        // Set up event handlers
        socketService.onConnect(() => {
            this.updateConnectionStatus(true);
        });
        
        socketService.onDisconnect(() => {
            this.updateConnectionStatus(false);
        });
        
        socketService.onTrainingUpdate((data) => {
            this.updateTrainingStatus(data);
        });
        
        socketService.onTrainingFinished((data) => {
            this.handleTrainingFinished(data);
        });
        
        socketService.onError((data) => {
            this.showAlert(data.message || 'An error occurred', 'error');
        });
    }

    /**
     * Initialize components
     */
    initComponents() {
        // Initialize dashboard component
        if (typeof dashboardComponent !== 'undefined') {
            dashboardComponent.init();
        }
        
        // Initialize training component
        if (typeof trainingComponent !== 'undefined') {
            trainingComponent.init();
        }
        
        // Initialize models component
        if (typeof modelsComponent !== 'undefined') {
            modelsComponent.init();
        }
        
        // Initialize quantization component
        if (typeof quantizationComponent !== 'undefined') {
            quantizationComponent.init();
        }
        
        // Initialize generation component
        if (typeof generationComponent !== 'undefined') {
            generationComponent.init();
        }
    }

    /**
     * Initialize navigation
     */
    initNavigation() {
        // Set up navigation
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Get target view
                const target = link.getAttribute('data-target');
                
                // Show view
                this.showView(target);
            });
        });
        
        // Show default view
        this.showView('dashboard');
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Add event listeners here
    }

    /**
     * Show a specific view
     * @param {string} viewId - View ID
     */
    showView(viewId) {
        // Hide all views
        const views = document.querySelectorAll('.view');
        views.forEach(view => {
            view.classList.add('d-none');
        });
        
        // Show target view
        const targetView = document.getElementById(`${viewId}-view`);
        if (targetView) {
            targetView.classList.remove('d-none');
        }
        
        // Update active navigation link
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.classList.remove('active');
            
            if (link.getAttribute('data-target') === viewId) {
                link.classList.add('active');
            }
        });
        
        // Call onActivate method of the component
        const componentName = `${viewId}Component`;
        if (typeof window[componentName] !== 'undefined' && typeof window[componentName].onActivate === 'function') {
            window[componentName].onActivate();
        }
    }

    /**
     * Load initial data
     */
    async loadInitialData() {
        // Check training status
        await this.checkTrainingStatus();
    }

    /**
     * Start periodic updates
     * DISABLED - Now using main app single consolidated update
     */
    startPeriodicUpdates() {
        console.log('🚫 js/app.js periodic updates DISABLED - using main app single update');
        // Periodic updates now handled by main app.js performSingleUpdate()
        // This prevents duplicate API calls
    }

    /**
     * Update connection status
     * @param {boolean} connected - Connection status
     */
    updateConnectionStatus(connected) {
        // Update connection status UI
        const statusElement = document.getElementById('connection-status');
        
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.classList.toggle('text-success', connected);
            statusElement.classList.toggle('text-danger', !connected);
        }
    }

    /**
     * Check training status
     * DISABLED - Now handled by main app.js performSingleUpdate()
     */
    async checkTrainingStatus() {
        console.log('🚫 js/app.js checkTrainingStatus DISABLED - using main app single update');
        // Training status now handled by main app.js performSingleUpdate()
        // This prevents duplicate API calls
    }

    /**
     * Update training buttons
     * @param {boolean} isTraining - Whether training is active
     */
    updateTrainingButtons(isTraining) {
        // Update training buttons UI
    }

    /**
     * Update training status
     * @param {object} data - Training status data
     */
    updateTrainingStatus(data) {
        // Update training status UI
    }

    /**
     * Handle training finished
     * @param {object} data - Training finished data
     */
    handleTrainingFinished(data) {
        // Handle training finished
    }

    /**
     * Show an alert
     * @param {string} message - Alert message
     * @param {string} type - Alert type (success, info, warning, error)
     */
    showAlert(message, type = 'info') {
        // Show an alert
        console.log(`[${type}] ${message}`);
    }
}

// Create a singleton instance
const trainingInterface = new TrainingInterface();

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize training interface
    if (typeof trainingInterface !== 'undefined') trainingInterface.init();
    
    // Initialize components
    if (typeof trainingComponent !== 'undefined') trainingComponent.init();
    if (typeof modelsComponent !== 'undefined') modelsComponent.init();
    if (typeof quantizationComponent !== 'undefined') quantizationComponent.init();
    if (typeof generationComponent !== 'undefined') generationComponent.init();
    if (typeof dashboardComponent !== 'undefined') dashboardComponent.init();
    
    // Fix for model loading form
    const modelLoadForm = document.getElementById('model-load-form');
    if (modelLoadForm) {
        modelLoadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const modelName = document.getElementById('test-model-select').value;
            const adapterPath = document.getElementById('adapter-path').value;
            
            // Show loading overlay
            document.getElementById('loading-overlay').classList.remove('d-none');
            document.getElementById('loading-message').textContent = 'Loading model...';
            
            // Make the API request directly
            fetch('/api/model/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: modelName,
                    adapter_path: adapterPath || null
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update UI
                    document.getElementById('unload-model-btn').disabled = false;
                    document.getElementById('generate-btn').disabled = false;
                    
                    // Update model status
                    const modelStatus = document.getElementById('model-status');
                    if (modelStatus) {
                        modelStatus.innerHTML = `
                            <div class="d-flex align-items-center">
                                <div class="status-indicator status-running me-2"></div>
                                <div>
                                    <h5 class="mb-0">${modelName}</h5>
                                    <small class="text-muted">${adapterPath ? `Adapter: ${adapterPath}` : 'No adapter'}</small>
                                </div>
                            </div>
                        `;
                    }
                    
                    console.log(`Model ${modelName} loaded successfully`);
                } else {
                    // Show error message
                    alert(`Failed to load model: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error loading model:', error);
                alert(`Failed to load model: ${error.message}`);
            })
            .finally(() => {
                // Hide loading overlay
                document.getElementById('loading-overlay').classList.add('d-none');
            });
        });
    }
}); 