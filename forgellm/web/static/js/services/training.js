/**
 * Training Service
 * 
 * Handles all training-related API calls and Socket.IO events.
 */

const trainingService = {
    /**
     * Current training state
     */
    currentTraining: null,
    
    /**
     * Training metrics history
     */
    metricsHistory: {
        iterations: [],
        trainLoss: [],
        valLoss: [],
        learningRate: [],
        tokensPerSec: [],
        memoryUsage: []
    },
    
    /**
     * Initialize training service
     */
    init() {
        // Set up Socket.IO event listeners
        socketService.onTrainingUpdate(this.handleTrainingUpdate.bind(this));
        socketService.onTrainingFinished(this.handleTrainingFinished.bind(this));
        
        // Check for active training on startup
        this.checkActiveTraining();
    },
    
    /**
     * Check if there's an active training session
     * 
     * @returns {Promise<boolean>} True if there's an active training session
     */
    async checkActiveTraining() {
        try {
            const response = await apiService.get('training/status');
            return response.data?.active || false;
        } catch (error) {
            console.error('Failed to check active training:', error);
            return false;
        }
    },
    
    /**
     * Start a new training session
     * 
     * @param {Object} config Training configuration
     * @returns {Promise<Object>} Response data
     */
    async startTraining(config) {
        try {
            const response = await apiService.post('training/start', config);
            return response.data;
        } catch (error) {
            console.error('Failed to start training:', error);
            throw error;
        }
    },
    
    /**
     * Stop the current training session
     * 
     * @returns {Promise<Object>} Response data
     */
    async stopTraining() {
        try {
            const response = await apiService.post('training/stop');
            return response.data;
        } catch (error) {
            console.error('Failed to stop training:', error);
            throw error;
        }
    },
    
    /**
     * Get training dashboard data
     * 
     * @returns {Promise<Object>} Dashboard data
     */
    async getDashboardData() {
        try {
            const response = await apiService.get('dashboard/data');
            return response.data;
        } catch (error) {
            console.error('Failed to get dashboard data:', error);
            throw error;
        }
    },
    
    /**
     * Get historical dashboard data for a specific log file
     * 
     * @param {string} logFile Path to the log file
     * @returns {Promise<Object>} Dashboard data
     */
    async getHistoricalDashboardData(logFile) {
        try {
            const response = await apiService.post('dashboard/historical', { log_file: logFile });
            return response.data;
        } catch (error) {
            console.error('Failed to get historical dashboard data:', error);
            throw error;
        }
    },
    
    /**
     * Get raw training logs
     * 
     * @param {string} logFile Path to the log file
     * @returns {Promise<Object>} Raw log data
     */
    async getRawLogs(logFile) {
        try {
            const response = await apiService.post('logs/raw', { log_file: logFile });
            return response.data;
        } catch (error) {
            console.error('Failed to get raw logs:', error);
            throw error;
        }
    },
    
    /**
     * Get available checkpoints
     * 
     * @returns {Promise<Object>} Checkpoints data
     */
    async getCheckpoints() {
        try {
            const response = await apiService.get('checkpoints');
            return response.data;
        } catch (error) {
            console.error('Failed to get checkpoints:', error);
            throw error;
        }
    },
    
    /**
     * Publish a checkpoint
     * 
     * @param {string} checkpointPath Path to the checkpoint file
     * @returns {Promise<Object>} Response data
     */
    async publishCheckpoint(checkpointPath) {
        try {
            const response = await apiService.post('training/publish_checkpoint', { path: checkpointPath });
            return response.data;
        } catch (error) {
            console.error('Failed to publish checkpoint:', error);
            throw error;
        }
    },
    
    /**
     * Get dataset information
     * 
     * @param {string} directory Dataset directory
     * @returns {Promise<Object>} Dataset information
     */
    async getDatasetInfo(directory = 'dataset') {
        try {
            const response = await apiService.get('dataset/info', { dir: directory });
            return response.data;
        } catch (error) {
            console.error('Failed to get dataset info:', error);
            throw error;
        }
    },
    
    /**
     * Handle training update from Socket.IO
     * 
     * @param {Object} data - Training update data
     */
    handleTrainingUpdate(data) {
        if (!data) {
            return;
        }
        
        // Update current training status
        if (this.currentTraining) {
            this.currentTraining.status = 'running';
        } else {
            this.currentTraining = {
                status: 'running',
                config: {}
            };
        }
        
        // Update metrics history
        this.updateMetricsHistory(data);
        
        // Trigger event for UI updates
        document.dispatchEvent(new CustomEvent('training-update', { detail: data }));
    },
    
    /**
     * Handle training finished event from Socket.IO
     * 
     * @param {Object} data - Training finished data
     */
    handleTrainingFinished(data) {
        if (!data) {
            return;
        }
        
        // Update current training status
        if (this.currentTraining) {
            this.currentTraining.status = 'completed';
        }
        
        // Stop polling
        this.stopPolling();
        
        // Trigger event for UI updates
        document.dispatchEvent(new CustomEvent('training-finished', { detail: data }));
    },
    
    /**
     * Start polling for training updates
     * DISABLED - Now using single consolidated update in main app
     */
    startPolling() {
        console.log('🚫 Training service polling disabled - using main app single update');
        // Polling is now handled by the main app's performSingleUpdate()
        // This prevents duplicate API calls
    },
    
    /**
     * Stop polling for training updates
     */
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    },
    
    /**
     * Reset metrics history
     */
    resetMetricsHistory() {
        this.metricsHistory = {
            iterations: [],
            trainLoss: [],
            valLoss: [],
            learningRate: [],
            tokensPerSec: [],
            memoryUsage: []
        };
    },
    
    /**
     * Update metrics history with new data
     * 
     * @param {Object} data - Training metrics data
     */
    updateMetricsHistory(data) {
        if (!data) {
            return;
        }
        
        // Add current iteration
        if (data.current_iteration !== undefined) {
            this.metricsHistory.iterations.push(data.current_iteration);
        }
        
        // Add train loss
        if (data.train_loss !== undefined) {
            this.metricsHistory.trainLoss.push(data.train_loss);
        }
        
        // Add validation loss
        if (data.val_loss !== undefined) {
            this.metricsHistory.valLoss.push(data.val_loss);
        }
        
        // Add learning rate
        if (data.learning_rate !== undefined) {
            this.metricsHistory.learningRate.push(data.learning_rate);
        }
        
        // Add tokens per second
        if (data.tokens_per_sec !== undefined) {
            this.metricsHistory.tokensPerSec.push(data.tokens_per_sec);
        }
        
        // Add memory usage
        if (data.peak_memory_gb !== undefined) {
            this.metricsHistory.memoryUsage.push(data.peak_memory_gb);
        }
        
        // Limit history size to prevent memory issues
        const maxHistorySize = 1000;
        if (this.metricsHistory.iterations.length > maxHistorySize) {
            this.metricsHistory.iterations = this.metricsHistory.iterations.slice(-maxHistorySize);
            this.metricsHistory.trainLoss = this.metricsHistory.trainLoss.slice(-maxHistorySize);
            this.metricsHistory.valLoss = this.metricsHistory.valLoss.slice(-maxHistorySize);
            this.metricsHistory.learningRate = this.metricsHistory.learningRate.slice(-maxHistorySize);
            this.metricsHistory.tokensPerSec = this.metricsHistory.tokensPerSec.slice(-maxHistorySize);
            this.metricsHistory.memoryUsage = this.metricsHistory.memoryUsage.slice(-maxHistorySize);
        }
    },
    
    /**
     * Set up Socket.IO event listeners
     * 
     * @param {Object} socket Socket.IO instance
     * @param {Function} onTrainingUpdate Callback for training updates
     * @param {Function} onTrainingFinished Callback for training finished event
     */
    setupSocketListeners(socket, onTrainingUpdate, onTrainingFinished) {
        if (!socket) return;
        
        // Listen for training updates
        socket.on('training_update', (data) => {
            if (onTrainingUpdate) {
                onTrainingUpdate(data);
            }
            
            // Dispatch a custom event for other components
            document.dispatchEvent(new CustomEvent('training-update', { detail: data }));
        });
        
        // Listen for training finished event
        socket.on('training_finished', (data) => {
            if (onTrainingFinished) {
                onTrainingFinished(data);
            }
            
            // Dispatch a custom event for other components
            document.dispatchEvent(new CustomEvent('training-finished', { detail: data }));
        });
        
        // Request initial training update - DISABLED to prevent 404 errors
        console.log('🚫 Socket emit disabled - using main app single update approach');
    }
}; 