/**
 * Training component
 */
class TrainingComponent {
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
        
        // Check training status
        this.checkTrainingStatus();
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Add event listeners here
    }

    /**
     * Check training status - DISABLED to prevent duplicate API calls
     */
    async checkTrainingStatus() {
        console.log('🚫 TrainingComponent checkTrainingStatus DISABLED - using main app single update');
        // Training status now handled by main app.js performSingleUpdate()
        // This prevents duplicate API calls
    }

    /**
     * Start training
     */
    async startTraining() {
        try {
            // Get training configuration from UI
            const config = this.getTrainingConfig();
            
            // Start training
            const response = await apiService.startTraining(config);
            
            // Update UI
            if (response.success) {
                // Training started successfully - NO longer calling checkTrainingStatus()
                console.log('✅ Training started - status updates handled by main app');
            } else {
                // Training failed to start
                console.error('Failed to start training:', response.error);
            }
        } catch (error) {
            console.error('Failed to start training:', error);
        }
    }

    /**
     * Stop training
     */
    async stopTraining() {
        try {
            const response = await apiService.stopTraining();
            
            // Update UI
            if (response.success) {
                // Training stopped successfully - NO longer calling checkTrainingStatus()
                console.log('✅ Training stopped - status updates handled by main app');
            } else {
                // Training failed to stop
                console.error('Failed to stop training:', response.error);
            }
        } catch (error) {
            console.error('Failed to stop training:', error);
        }
    }

    /**
     * Get training configuration from UI
     * @returns {object} - Training configuration
     */
    getTrainingConfig() {
        // Get training configuration from UI
        return {};
    }

    /**
     * Called when the training tab is activated - DISABLED to prevent duplicate API calls
     */
    onActivate() {
        console.log('🚫 TrainingComponent onActivate DISABLED - using main app single update');
        // Training status now handled by main app.js performSingleUpdate()
        // This prevents duplicate API calls
    }
}

// Create a singleton instance
const trainingComponent = new TrainingComponent(); 