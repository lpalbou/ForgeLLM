/**
 * Socket Service
 * 
 * Handles Socket.IO connections and events.
 */

const socketService = {
    /**
     * Socket.IO instance
     */
    socket: null,
    
    /**
     * Connection status
     */
    connected: false,
    
    /**
     * Initialize Socket.IO connection
     */
    init() {
        // Create Socket.IO connection
        this.socket = io({
            transports: ['websocket'],
            upgrade: false
        });
        
        // Set up event listeners
        this.setupEventListeners();
    },
    
    /**
     * Set up Socket.IO event listeners
     */
    setupEventListeners() {
        if (!this.socket) return;
        
        // Connection events
        this.socket.on('connect', () => {
            console.log('Socket.IO connected');
            this.connected = true;
            document.dispatchEvent(new CustomEvent('socket-connected'));
        });
        
        this.socket.on('disconnect', () => {
            console.log('Socket.IO disconnected');
            this.connected = false;
            document.dispatchEvent(new CustomEvent('socket-disconnected'));
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Socket.IO connection error:', error);
            this.connected = false;
            document.dispatchEvent(new CustomEvent('socket-error', { detail: error }));
        });
        
        // Server events
        this.socket.on('connected', (data) => {
            console.log('Server connection acknowledged:', data);
        });
        
        this.socket.on('error', (data) => {
            console.error('Server error:', data);
            document.dispatchEvent(new CustomEvent('server-error', { detail: data }));
        });
    },
    
    /**
     * Request training update
     */
    requestTrainingUpdate() {
        if (!this.socket || !this.connected) return;
        this.socket.emit('request_training_update');
    },
    
    /**
     * Stop training
     */
    stopTraining() {
        if (!this.socket || !this.connected) return;
        this.socket.emit('stop_training');
    },
    
    /**
     * Add event listener
     * 
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (!this.socket) return;
        this.socket.on(event, callback);
    },
    
    /**
     * Remove event listener
     * 
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    off(event, callback) {
        if (!this.socket) return;
        this.socket.off(event, callback);
    },
    
    /**
     * Emit event
     * 
     * @param {string} event - Event name
     * @param {*} data - Event data
     */
    emit(event, data) {
        if (!this.socket || !this.connected) return;
        this.socket.emit(event, data);
    }
}; 