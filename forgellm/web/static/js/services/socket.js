/**
 * Socket.IO service for real-time communication with the server
 */
class SocketService {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.callbacks = {
            connect: [],
            disconnect: [],
            training_update: [],
            training_finished: [],
            error: []
        };
    }

    /**
     * Initialize the Socket.IO connection
     * DISABLED - Preventing 404 errors by using HTTP API approach instead
     */
    init() {
        console.log('🚫 Socket.IO connection DISABLED - using HTTP API single update approach');
        
        // Simulate connected state for compatibility
        this.connected = true;
        this.socket = null;
        
        // Trigger connect callbacks for compatibility
        setTimeout(() => {
            this._triggerCallbacks('connect');
        }, 100);
    }

    /**
     * Register a callback for a specific event
     * @param {string} event - Event name
     * @param {function} callback - Callback function
     */
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }

    /**
     * Trigger callbacks for a specific event
     * @param {string} event - Event name
     * @param {object} data - Event data
     * @private
     */
    _triggerCallbacks(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => {
                try {
                    callback(data);
                } catch (e) {
                    console.error(`Error in ${event} callback:`, e);
                }
            });
        }
    }

    /**
     * Check if the socket is connected
     * @returns {boolean} - Connection status
     */
    isConnected() {
        return this.connected;
    }

    /**
     * Request a training update from the server
     * DISABLED - Using HTTP API approach instead
     */
    requestUpdate() {
        console.log('🚫 Socket requestUpdate disabled - using HTTP API');
    }

    /**
     * Check training status
     * DISABLED - Using HTTP API approach instead
     */
    checkTrainingStatus() {
        console.log('🚫 Socket checkTrainingStatus disabled - using HTTP API');
    }

    /**
     * Load a training log file
     * DISABLED - Using HTTP API approach instead
     * @param {string} logFile - Path to the log file
     */
    loadTrainingLog(logFile) {
        console.log('🚫 Socket loadTrainingLog disabled - using HTTP API');
    }

    /**
     * Start text generation
     * DISABLED - Using HTTP API approach instead
     * @param {object} params - Generation parameters
     */
    startGeneration(params) {
        console.log('🚫 Socket startGeneration disabled - using HTTP API');
    }

    /**
     * Stop text generation
     * DISABLED - Using HTTP API approach instead
     */
    stopGeneration() {
        console.log('🚫 Socket stopGeneration disabled - using HTTP API');
    }
}

// Create a singleton instance
const socketService = new SocketService();

// Legacy compatibility functions
socketService.onConnect = function(callback) {
    this.on('connect', callback);
};

socketService.onDisconnect = function(callback) {
    this.on('disconnect', callback);
};

socketService.onTrainingUpdate = function(callback) {
    this.on('training_update', callback);
};

socketService.onTrainingFinished = function(callback) {
    this.on('training_finished', callback);
};

socketService.onError = function(callback) {
    this.on('error', callback);
}; 