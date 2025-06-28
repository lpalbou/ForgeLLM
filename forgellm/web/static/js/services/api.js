/**
 * API service for communicating with the server
 */
class ApiService {
    constructor() {
        this.baseUrl = '';
    }

    /**
     * Make a GET request to the API
     * @param {string} endpoint - API endpoint
     * @param {object} params - Query parameters
     * @returns {Promise<object>} - Response data
     */
    async get(endpoint, params = {}) {
        try {
            // Build query string
            const queryString = Object.keys(params)
                .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
                .join('&');

            // Build URL
            const url = `${this.baseUrl}/api/${endpoint}${queryString ? `?${queryString}` : ''}`;

            // Make request
            const response = await fetch(url);

            // Check if response is OK
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            // Parse response
            return await response.json();
        } catch (error) {
            console.error(`GET ${endpoint} failed:`, error);
            throw error;
        }
    }

    /**
     * Make a POST request to the API
     * @param {string} endpoint - API endpoint
     * @param {object} data - Request body
     * @returns {Promise<object>} - Response data
     */
    async post(endpoint, data = {}) {
        try {
            // Build URL
            const url = `${this.baseUrl}/api/${endpoint}`;

            // Make request
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            // Check if response is OK
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            // Parse response
            return await response.json();
        } catch (error) {
            console.error(`POST ${endpoint} failed:`, error);
            throw error;
        }
    }

    /**
     * Get available base models
     * @returns {Promise<object>} - Response data
     */
    async getBaseModels() {
        return this.get('base_models');
    }

    /**
     * Get available CPT models
     * @returns {Promise<object>} - Response data
     */
    async getCPTModels() {
        return this.get('cpt_models');
    }

    /**
     * Get available IFT models
     * @returns {Promise<object>} - Response data
     */
    async getIFTModels() {
        return this.get('ift_models');
    }

    /**
     * Get dataset information
     * @param {string} dir - Dataset directory
     * @returns {Promise<object>} - Response data
     */
    async getDatasetInfo(dir = 'mnemosyne') {
        return this.get('dataset/info', { dir });
    }

    /**
     * Start training
     * @param {object} config - Training configuration
     * @returns {Promise<object>} - Response data
     */
    async startTraining(config) {
        return this.post('training/start', config);
    }

    /**
     * Stop training
     * @returns {Promise<object>} - Response data
     */
    async stopTraining() {
        return this.post('training/stop');
    }

    /**
     * Get training status
     * @returns {Promise<object>} - Response data
     */
    async getTrainingStatus() {
        return this.get('training/status');
    }

    /**
     * Get dashboard data
     * @returns {Promise<object>} - Response data
     */
    async getDashboardData() {
        return this.get('dashboard/data');
    }

    /**
     * Get checkpoints
     * @returns {Promise<object>} - Response data
     */
    async getCheckpoints() {
        return this.get('checkpoints');
    }

    /**
     * Load a model
     * @param {string} modelName - Model name
     * @param {string} adapterPath - Adapter path
     * @returns {Promise<object>} - Response data
     */
    async loadModel(modelName, adapterPath = null) {
        return this.post('model/load', { model_name: modelName, adapter_path: adapterPath });
    }

    /**
     * Unload the current model
     * @returns {Promise<object>} - Response data
     */
    async unloadModel() {
        return this.post('model/unload');
    }

    /**
     * Generate text
     * @param {object} params - Generation parameters
     * @returns {Promise<object>} - Response data
     */
    async generateText(params) {
        return this.post('model/generate', params);
    }

    /**
     * Check if a dashboard exists for a model
     * @param {string} path - Model path
     * @returns {Promise<object>} - Response data
     */
    async checkDashboard(path) {
        return this.get('check_dashboard', { path });
    }

    /**
     * Publish a checkpoint
     * @param {string} path - Checkpoint path
     * @returns {Promise<object>} - Response data
     */
    async publishCheckpoint(path) {
        return this.post('training/publish_checkpoint', { path });
    }

    /**
     * Get raw logs
     * @param {string} logFile - Log file path
     * @returns {Promise<object>} - Response data
     */
    async getRawLogs(logFile) {
        return this.post('logs/raw', { log_file: logFile });
    }

    /**
     * Get historical dashboard data
     * @param {string} logFile - Log file path
     * @returns {Promise<object>} - Response data
     */
    async getHistoricalDashboard(logFile) {
        return this.post('dashboard/historical', { log_file: logFile });
    }
}

// Create a singleton instance
const apiService = new ApiService(); 