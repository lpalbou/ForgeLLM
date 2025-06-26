/**
 * ForgeLLM - Main Application
 * 
 * This file initializes the application and handles navigation between views.
 */

class ForgeLLMApp {
    constructor() {
        // Current view
        this.currentView = 'dashboard';
        
        // Components
        this.components = {
            dashboard: null,
            training: null,
            models: null,
            generation: null,
            settings: null
        };
        
        // Navigation elements
        this.navItems = {
            dashboard: document.getElementById('nav-dashboard'),
            training: document.getElementById('nav-training'),
            models: document.getElementById('nav-models'),
            generate: document.getElementById('nav-generate'),
            settings: document.getElementById('nav-settings')
        };
        
        // View containers
        this.viewContainers = {
            dashboard: document.getElementById('dashboard-view'),
            training: document.getElementById('training-view'),
            models: document.getElementById('models-view'),
            generate: document.getElementById('generate-view'),
            settings: document.getElementById('settings-view')
        };
        
        // Status elements
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
        this.memoryUsage = document.getElementById('memory-usage');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingMessage = document.getElementById('loading-message');
        
        // Alert container
        this.alertContainer = document.getElementById('alert-container');
        
        // Socket connection status
        this.socketConnected = false;
    }
    
    /**
     * Initialize the application
     */
    init() {
        // Initialize components
        this.initComponents();
        
        // Set up navigation
        this.initNavigation();
        
        // Initialize socket connection
        this.initSocket();
        
        // Check memory usage periodically
        this.startMemoryMonitor();
    }
    
    /**
     * Initialize components
     */
    initComponents() {
        // Initialize dashboard component
        this.components.dashboard = new DashboardComponent();
        this.components.dashboard.init();
        
        // Initialize training component
        this.components.training = new TrainingComponent();
        this.components.training.init();
        
        // Initialize models component
        this.components.models = new ModelsComponent();
        this.components.models.init();
        
        // Initialize generation component
        this.components.generation = new GenerationComponent();
        this.components.generation.init();
    }
    
    /**
     * Initialize navigation
     */
    initNavigation() {
        // Set up navigation event listeners
        Object.keys(this.navItems).forEach(view => {
            const navItem = this.navItems[view];
            if (navItem) {
                navItem.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.showView(view);
                });
            }
        });
        
        // Show initial view
        this.showView(this.currentView);
    }
    
    /**
     * Initialize socket connection
     */
    initSocket() {
        // Initialize Socket.IO service
        socketService.init();
        
        // Set up event listeners
        socketService.onConnect(() => {
            this.setConnected(true);
        });
        
        socketService.onDisconnect(() => {
            this.setConnected(false);
        });
        
        socketService.onTrainingUpdate((data) => {
            // Dispatch training update event
            document.dispatchEvent(new CustomEvent('training-update', { detail: data }));
            
            // Update memory usage
            if (data.peak_memory_gb !== undefined) {
                this.updateMemoryUsage(data.peak_memory_gb);
            }
        });
        
        socketService.onTrainingFinished((data) => {
            // Dispatch training finished event
            document.dispatchEvent(new CustomEvent('training-finished', { detail: data }));
        });
        
        socketService.onGenerationStart((data) => {
            // Set status to working
            this.setWorking(true);
        });
        
        socketService.onGenerationComplete((data) => {
            // Set status to connected
            this.setWorking(false);
            
            // Update memory usage
            if (data.memory_gb !== undefined) {
                this.updateMemoryUsage(data.memory_gb);
            }
            
            // Handle generation response
            if (this.components.generation) {
                this.components.generation.handleGenerationResponse(data);
            }
        });
        
        socketService.onError((error) => {
            // Show error message
            this.showAlert(error.message, 'danger');
            
            // Set status to connected
            this.setWorking(false);
        });
    }
    
    /**
     * Start memory monitor
     */
    startMemoryMonitor() {
        // Check memory usage every 30 seconds
        setInterval(async () => {
            try {
                // Get memory usage from API
                const response = await apiService.getMemoryUsage();
                
                if (response.memory_gb !== undefined) {
                    this.updateMemoryUsage(response.memory_gb);
                }
            } catch (error) {
                console.error('Failed to get memory usage:', error);
            }
        }, 30000);
    }
    
    /**
     * Show a view
     * 
     * @param {string} view - View name
     */
    showView(view) {
        // Validate view
        if (!this.viewContainers[view]) {
            console.error(`View not found: ${view}`);
            return;
        }
        
        // Hide all views
        Object.values(this.viewContainers).forEach(container => {
            if (container) {
                container.classList.remove('active');
            }
        });
        
        // Show selected view
        this.viewContainers[view].classList.add('active');
        
        // Update navigation
        Object.keys(this.navItems).forEach(key => {
            const navItem = this.navItems[key];
            if (navItem) {
                navItem.classList.remove('active');
                navItem.classList.add('text-white');
            }
        });
        
        if (this.navItems[view]) {
            this.navItems[view].classList.add('active');
            this.navItems[view].classList.remove('text-white');
        }
        
        // Update current view
        this.currentView = view;
        
        // Notify component
        if (this.components[view] && typeof this.components[view].onActivate === 'function') {
            this.components[view].onActivate();
        }
    }
    
    /**
     * Set connection status
     * 
     * @param {boolean} connected - Whether the socket is connected
     */
    setConnected(connected) {
        this.socketConnected = connected;
        
        if (this.statusIndicator) {
            this.statusIndicator.className = 'bi bi-circle-fill me-2';
            this.statusIndicator.classList.add(connected ? 'connected' : 'disconnected');
        }
        
        if (this.statusText) {
            this.statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }
    
    /**
     * Set working status
     * 
     * @param {boolean} working - Whether the application is working
     */
    setWorking(working) {
        if (this.statusIndicator) {
            this.statusIndicator.className = 'bi bi-circle-fill me-2';
            this.statusIndicator.classList.add(working ? 'working' : (this.socketConnected ? 'connected' : 'disconnected'));
        }
        
        if (this.statusText) {
            this.statusText.textContent = working ? 'Working...' : (this.socketConnected ? 'Connected' : 'Disconnected');
        }
    }
    
    /**
     * Update memory usage
     * 
     * @param {number} memoryGB - Memory usage in GB
     */
    updateMemoryUsage(memoryGB) {
        if (this.memoryUsage) {
            this.memoryUsage.textContent = `${formatUtil.formatDecimal(memoryGB, 1)} GB`;
        }
    }
    
    /**
     * Show loading overlay
     * 
     * @param {string} message - Loading message
     */
    showLoading(message = 'Loading...') {
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
        }
        
        if (loadingMessage) {
            loadingMessage.textContent = message;
        }
    }
    
    /**
     * Hide loading overlay
     */
    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
        }
    }
    
    /**
     * Show alert message
     * 
     * @param {string} message - Alert message
     * @param {string} type - Alert type (success, info, warning, danger)
     * @param {number} duration - Duration in milliseconds (0 for no auto-hide)
     */
    showAlert(message, type = 'info', duration = 5000) {
        if (!this.alertContainer) {
            return;
        }
        
        // Create alert element
        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        alertElement.role = 'alert';
        alertElement.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add alert to container
        this.alertContainer.appendChild(alertElement);
        
        // Auto-hide alert
        if (duration > 0) {
            setTimeout(() => {
                alertElement.classList.remove('show');
                setTimeout(() => {
                    alertElement.remove();
                }, 150);
            }, duration);
        }
    }
    
    /**
     * Show confirmation dialog
     * 
     * @param {string} title - Dialog title
     * @param {string} message - Dialog message
     * @param {Function} onConfirm - Callback function when confirmed
     * @param {string} confirmText - Confirm button text
     * @param {string} cancelText - Cancel button text
     */
    showConfirm(title, message, onConfirm, confirmText = 'Confirm', cancelText = 'Cancel') {
        const modal = document.getElementById('confirm-modal');
        if (!modal) {
            return;
        }
        
        const modalTitle = document.getElementById('confirm-modal-title');
        const modalBody = document.getElementById('confirm-modal-body');
        const confirmBtn = document.getElementById('confirm-modal-confirm');
        
        if (modalTitle) {
            modalTitle.textContent = title;
        }
        
        if (modalBody) {
            modalBody.textContent = message;
        }
        
        if (confirmBtn) {
            confirmBtn.textContent = confirmText;
            
            // Remove existing event listeners
            const newConfirmBtn = confirmBtn.cloneNode(true);
            confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
            
            // Add new event listener
            newConfirmBtn.addEventListener('click', () => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
                
                if (typeof onConfirm === 'function') {
                    onConfirm();
                }
            });
        }
        
        // Show modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create app instance
    window.app = new ForgeLLMApp();
    
    // Initialize app
    window.app.init();
}); 