/**
 * API Service
 * 
 * Handles all communication with the backend API.
 */

const apiService = {
    /**
     * Base API URL
     */
    baseUrl: '/api',
    
    /**
     * Make a GET request to the API
     * 
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise} - Promise with response data
     */
    async get(endpoint, params = {}) {
        try {
            const url = new URL(`${this.baseUrl}/${endpoint}`, window.location.origin);
            
            // Add query parameters
            Object.keys(params).forEach(key => {
                url.searchParams.append(key, params[key]);
            });
            
            const response = await fetch(url.toString());
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`GET ${endpoint} failed:`, error);
            throw error;
        }
    },
    
    /**
     * Make a POST request to the API
     * 
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @returns {Promise} - Promise with response data
     */
    async post(endpoint, data = {}) {
        try {
            const url = `${this.baseUrl}/${endpoint}`;
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`POST ${endpoint} failed:`, error);
            throw error;
        }
    },
    
    /**
     * Make a PUT request to the API
     * 
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @returns {Promise} - Promise with response data
     */
    async put(endpoint, data = {}) {
        try {
            const url = `${this.baseUrl}/${endpoint}`;
            
            const response = await fetch(url, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`PUT ${endpoint} failed:`, error);
            throw error;
        }
    },
    
    /**
     * Make a DELETE request to the API
     * 
     * @param {string} endpoint - API endpoint
     * @returns {Promise} - Promise with response data
     */
    async delete(endpoint) {
        try {
            const url = `${this.baseUrl}/${endpoint}`;
            
            const response = await fetch(url, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`DELETE ${endpoint} failed:`, error);
            throw error;
        }
    },
    
    // Training API methods
    
    /**
     * Start a new training job
     * 
     * @param {Object} config - Training configuration
     * @returns {Promise} - Promise with response data
     */
    async startTraining(config) {
        return this.post('training/start', config);
    },
    
    /**
     * Stop the current training job
     * 
     * @returns {Promise} - Promise with response data
     */
    async stopTraining() {
        return this.post('training/stop');
    },
    
    /**
     * Get the current training status
     * 
     * @returns {Promise} - Promise with response data
     */
    async getTrainingStatus() {
        return this.get('training/status');
    },
    
    /**
     * Get dashboard data for the current training
     * 
     * @returns {Promise} - Promise with response data
     */
    async getDashboardData() {
        return this.get('dashboard/data');
    },
    
    /**
     * Get historical dashboard data
     * 
     * @param {string} logFile - Path to the log file
     * @returns {Promise} - Promise with response data
     */
    async getHistoricalDashboard(logFile) {
        return this.post('dashboard/historical', { log_file: logFile });
    },
    
    /**
     * Publish a checkpoint to a shareable format
     * 
     * @param {string} path - Path to the checkpoint
     * @returns {Promise} - Promise with response data
     */
    async publishCheckpoint(path) {
        return this.post('training/publish_checkpoint', { path });
    },
    
    // Model API methods
    
    /**
     * Get available base models
     * 
     * @returns {Promise} - Promise with response data
     */
    async getBaseModels() {
        return this.get('base_models');
    },
    
    /**
     * Get available CPT models
     * 
     * @returns {Promise} - Promise with response data
     */
    async getCPTModels() {
        return this.get('cpt_models');
    },
    
    /**
     * Get available instruction-tuned models
     * 
     * @returns {Promise} - Promise with response data
     */
    async getIFTModels() {
        return this.get('ift_models');
    },
    
    /**
     * Load a model for generation
     * 
     * @param {string} model - Model name or path
     * @param {string} adapterPath - Optional adapter path
     * @returns {Promise} - Promise with response data
     */
    async loadModel(model, adapterPath = null) {
        return this.post('models/load', { 
            model, 
            adapter_path: adapterPath 
        });
    },
    
    /**
     * Unload the current model
     * 
     * @returns {Promise} - Promise with response data
     */
    async unloadModel() {
        return this.post('models/unload');
    },
    
    /**
     * Generate text with the loaded model
     * 
     * @param {Object} params - Generation parameters
     * @returns {Promise} - Promise with response data
     */
    async generateText(params) {
        return this.post('models/generate', params);
    },
    
    /**
     * Get dataset information
     * 
     * @param {string} dir - Dataset directory
     * @returns {Promise} - Promise with response data
     */
    async getDatasetInfo(dir = 'mnemosyne') {
        return this.get('dataset/info', { dir });
    },
    
    /**
     * Get memory usage information
     * 
     * @returns {Promise} - Promise with response data
     */
    async getMemoryUsage() {
        return this.get('memory');
    },
    
    /**
     * Check if a training dashboard exists for a model
     * 
     * @param {string} path - Model path
     * @returns {Promise} - Promise with response data
     */
    async checkDashboard(path) {
        return this.get('check_dashboard', { path });
    }
}; 