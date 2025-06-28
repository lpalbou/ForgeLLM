/**
 * Dashboard component
 */
class DashboardComponent {
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
        
        // Load dashboard data
        this.loadDashboardData();
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Add event listeners here
    }

    /**
     * Load dashboard data
     */
    async loadDashboardData() {
        try {
            // Load dashboard data
            const response = await apiService.getDashboardData();
            
            // Update dashboard UI
            this.updateDashboard(response);
            
            // Load recent models
            this.loadRecentModels();
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }

    /**
     * Update dashboard UI
     * @param {object} data - Dashboard data
     */
    updateDashboard(data) {
        const dashboardContainer = document.getElementById('dashboard-container');
        if (!dashboardContainer) return;
        
        // Check if training is active
        const isTrainingActive = data.active === true;
        
        // Clear previous content
        dashboardContainer.innerHTML = '';
        
        if (isTrainingActive) {
            // Training is active, show real-time metrics
            this.renderActiveTrainingDashboard(data, dashboardContainer);
        } else {
            // No active training, show summary
            this.renderInactiveTrainingDashboard(dashboardContainer);
        }
    }
    
    /**
     * Render dashboard for active training
     * @param {object} data - Dashboard data
     * @param {HTMLElement} container - Container element
     */
    renderActiveTrainingDashboard(data, container) {
        // Create header
        const header = document.createElement('div');
        header.className = 'dashboard-header';
        header.innerHTML = `
            <h3>Active Training Session</h3>
            <div class="dashboard-status">
                <span class="status-badge active">Active</span>
                <span class="model-name">${data.config?.model_name || 'Unknown Model'}</span>
            </div>
        `;
        container.appendChild(header);
        
        // Create metrics section
        const metricsSection = document.createElement('div');
        metricsSection.className = 'dashboard-metrics';
        
        // Progress
        const progress = (data.current_iteration / data.max_iterations) * 100 || 0;
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-container';
        progressBar.innerHTML = `
            <div class="progress-label">Progress: ${data.current_iteration || 0} / ${data.max_iterations || 0} iterations (${progress.toFixed(1)}%)</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${progress}%"></div>
            </div>
        `;
        metricsSection.appendChild(progressBar);
        
        // Key metrics
        const metricsGrid = document.createElement('div');
        metricsGrid.className = 'metrics-grid';
        
        // Helper function to format metric value
        const formatMetric = (value, type = 'number') => {
            if (value === null || value === undefined) return 'N/A';
            if (type === 'number') return typeof value === 'number' ? value.toFixed(4) : value;
            if (type === 'time') return typeof value === 'number' ? `${value.toFixed(1)} min` : value;
            return value;
        };
        
        // Add metrics
        const metrics = [
            { label: 'Train Loss', value: formatMetric(data.train_loss) },
            { label: 'Val Loss', value: formatMetric(data.val_loss) },
            { label: 'Train PPL', value: formatMetric(data.train_perplexity) },
            { label: 'Val PPL', value: formatMetric(data.val_perplexity) },
            { label: 'Elapsed', value: formatMetric(data.elapsed_minutes, 'time') },
            { label: 'ETA', value: formatMetric(data.eta_minutes, 'time') },
            { label: 'Tokens/sec', value: formatMetric(data.tokens_per_sec) },
            { label: 'Memory', value: formatMetric(data.peak_memory_gb) ? `${formatMetric(data.peak_memory_gb)} GB` : 'N/A' }
        ];
        
        metrics.forEach(metric => {
            const metricEl = document.createElement('div');
            metricEl.className = 'metric-item';
            metricEl.innerHTML = `
                <div class="metric-label">${metric.label}</div>
                <div class="metric-value">${metric.value}</div>
            `;
            metricsGrid.appendChild(metricEl);
        });
        
        metricsSection.appendChild(metricsGrid);
        container.appendChild(metricsSection);
        
        // Add config section
        if (data.config) {
            const configSection = document.createElement('div');
            configSection.className = 'dashboard-config';
            configSection.innerHTML = `
                <h4>Training Configuration</h4>
                <div class="config-grid">
                    <div class="config-item">
                        <div class="config-label">Model</div>
                        <div class="config-value">${data.config.model_name || 'N/A'}</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Batch Size</div>
                        <div class="config-value">${data.config.batch_size || 'N/A'}</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">Learning Rate</div>
                        <div class="config-value">${data.config.learning_rate || 'N/A'}</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">LR Schedule</div>
                        <div class="config-value">${data.config.lr_schedule || 'N/A'}</div>
                    </div>
                </div>
            `;
            container.appendChild(configSection);
        }
        
        // Add stop button
        const actionsSection = document.createElement('div');
        actionsSection.className = 'dashboard-actions';
        const stopButton = document.createElement('button');
        stopButton.className = 'btn btn-danger';
        stopButton.textContent = 'Stop Training';
        stopButton.onclick = () => this.stopTraining();
        actionsSection.appendChild(stopButton);
        container.appendChild(actionsSection);
    }
    
    /**
     * Render dashboard when no training is active
     * @param {HTMLElement} container - Container element
     */
    renderInactiveTrainingDashboard(container) {
        container.innerHTML = `
            <div class="dashboard-inactive">
                <h3>No Active Training</h3>
                <p>There is no active training session. Start a new training session from the Training tab.</p>
            </div>
        `;
    }
    
    /**
     * Stop the current training session
     */
    async stopTraining() {
        try {
            const confirmed = confirm('Are you sure you want to stop the current training session?');
            if (!confirmed) return;
            
            await apiService.stopTraining();
            alert('Training stopped successfully');
            this.loadDashboardData();
        } catch (error) {
            console.error('Failed to stop training:', error);
            alert('Failed to stop training: ' + error.message);
        }
    }

    /**
     * Load recent models
     */
    async loadRecentModels() {
        try {
            const response = await apiService.getCPTModels();
            
            // Update recent models UI
        } catch (error) {
            console.error('Failed to load recent models:', error);
        }
    }

    /**
     * Called when the dashboard tab is activated
     */
    onActivate() {
        // Refresh dashboard data
        this.loadDashboardData();
    }
}

// Create a singleton instance
const dashboardComponent = new DashboardComponent(); 