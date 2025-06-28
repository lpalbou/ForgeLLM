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
     */
    init() {
        // Connect to the Socket.IO server
        this.socket = io();

        // Set up event handlers
        this.socket.on('connect', () => {
            console.log('Socket.IO connected');
            this.connected = true;
            this._triggerCallbacks('connect');
        });

        this.socket.on('disconnect', () => {
            console.log('Socket.IO disconnected');
            this.connected = false;
            this._triggerCallbacks('disconnect');
        });

        this.socket.on('connected', (data) => {
            console.log('Server connection acknowledged: ', data);
        });

        this.socket.on('training_update', (data) => {
            this._triggerCallbacks('training_update', data);
        });

        this.socket.on('training_finished', (data) => {
            this._triggerCallbacks('training_finished', data);
        });

        this.socket.on('error', (data) => {
            console.error('Socket error:', data);
            this._triggerCallbacks('error', data);
        });
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
     */
    requestUpdate() {
        if (this.connected) {
            this.socket.emit('request_update');
        }
    }

    /**
     * Check training status
     */
    checkTrainingStatus() {
        if (this.connected) {
            this.socket.emit('check_training_status');
        }
    }

    /**
     * Load a training log file
     * @param {string} logFile - Path to the log file
     */
    loadTrainingLog(logFile) {
        if (this.connected) {
            this.socket.emit('load_training_log', { log_file: logFile });
        }
    }

    /**
     * Start text generation
     * @param {object} params - Generation parameters
     */
    startGeneration(params) {
        if (this.connected) {
            this.socket.emit('start_generation', params);
        }
    }

    /**
     * Stop text generation
     */
    stopGeneration() {
        if (this.connected) {
            this.socket.emit('stop_generation');
        }
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