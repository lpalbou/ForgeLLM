/**
 * Training Component
 * 
 * Handles training configuration, monitoring, and visualization.
 */

class TrainingComponent {
    constructor() {
        // Configuration form
        this.configForm = document.getElementById('training-config-form');
        this.modelNameInput = document.getElementById('model-name');
        this.datasetInput = document.getElementById('dataset');
        this.batchSizeInput = document.getElementById('batch-size');
        this.learningRateInput = document.getElementById('learning-rate');
        this.epochsInput = document.getElementById('epochs');
        this.validationSplitInput = document.getElementById('validation-split');
        this.startTrainingBtn = document.getElementById('start-training-btn');
        
        // Monitoring elements
        this.monitoringSection = document.getElementById('training-monitoring');
        this.progressBar = document.getElementById('training-progress');
        this.statusText = document.getElementById('training-status');
        this.iterationText = document.getElementById('training-iteration');
        this.lossText = document.getElementById('training-loss');
        this.valLossText = document.getElementById('training-val-loss');
        this.elapsedText = document.getElementById('training-elapsed');
        this.etaText = document.getElementById('training-eta');
        this.speedText = document.getElementById('training-speed');
        this.memoryText = document.getElementById('training-memory');
        this.stopTrainingBtn = document.getElementById('stop-training-btn');
        
        // Charts
        this.lossChartCanvas = document.getElementById('loss-chart');
        this.perplexityChartCanvas = document.getElementById('perplexity-chart');
        this.learningRateChartCanvas = document.getElementById('learning-rate-chart');
        this.performanceChartCanvas = document.getElementById('performance-chart');
        this.memoryChartCanvas = document.getElementById('memory-chart');
        
        // Training data
        this.trainingData = {
            iterations: [],
            trainLoss: [],
            valLoss: [],
            learningRate: [],
            tokensPerSec: [],
            memoryUsage: []
        };
        
        // Charts
        this.charts = {};
    }
    
    /**
     * Initialize the training component
     */
    init() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Check for active training
        this.checkActiveTraining();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Start training button
        if (this.startTrainingBtn) {
            this.startTrainingBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.startTraining();
            });
        }
        
        // Stop training button
        if (this.stopTrainingBtn) {
            this.stopTrainingBtn.addEventListener('click', () => {
                this.stopTraining();
            });
        }
        
        // Listen for training updates
        document.addEventListener('training-update', (event) => {
            this.updateTrainingMonitor(event.detail);
        });
        
        document.addEventListener('training-finished', (event) => {
            this.handleTrainingFinished(event.detail);
        });
    }
    
    /**
     * Check for active training
     */
    async checkActiveTraining() {
        try {
            const hasActiveTraining = await trainingService.checkActiveTraining();
            
            if (hasActiveTraining) {
                this.showTrainingMonitor();
                this.hideConfigForm();
            } else {
                this.hideTrainingMonitor();
                this.showConfigForm();
            }
        } catch (error) {
            console.error('Failed to check active training:', error);
            window.app.showAlert('Failed to check active training status', 'danger');
        }
    }
    
    /**
     * Start training
     */
    async startTraining() {
        if (!this.configForm) {
            return;
        }
        
        // Validate form
        if (!this.configForm.checkValidity()) {
            this.configForm.reportValidity();
            return;
        }
        
        // Get form data
        const formData = new FormData(this.configForm);
        
        // Map form fields to expected TrainingConfig parameters
        const config = {
            model_name: formData.get('model_name'),
            input_dir: formData.get('dataset'),  // Changed from 'dataset' to 'input_dir'
            batch_size: parseInt(formData.get('batch_size'), 10),
            learning_rate: parseFloat(formData.get('learning_rate')),
            max_iterations: parseInt(formData.get('epochs'), 10),  // Changed from 'epochs' to 'max_iterations'
            validation_split: parseFloat(formData.get('validation_split')),
            
            // Changed from 'use_lora' to 'fine_tune_type'
            fine_tune_type: formData.get('use_lora') === 'on' ? 'lora' : 'full',
            
            // LoRA parameters
            lora_r: parseInt(formData.get('lora_r') || '8', 10),
            lora_alpha: parseInt(formData.get('lora_alpha') || '16', 10),
            lora_dropout: parseFloat(formData.get('lora_dropout') || '0.05'),
            
            // Changed from 'sequence_length' to 'max_seq_length'
            max_seq_length: parseInt(formData.get('sequence_length') || '2048', 10),
            
            gradient_accumulation_steps: parseInt(formData.get('gradient_accumulation_steps') || '1', 10),
            warmup_steps: parseInt(formData.get('warmup_steps') || '100', 10),
            
            // Changed from 'eval_interval' to 'steps_per_eval'
            steps_per_eval: parseInt(formData.get('eval_interval') || '100', 10),
            
            // Changed from 'save_interval' to 'save_every'
            save_every: parseInt(formData.get('save_interval') || '500', 10),
            
            // Add default values for required parameters that might be missing
            output_dir: 'models',
            data_dir: 'data/pretraining'
        };
        
        // Log the config being sent to the API
        console.log('Training config being sent to API:', config);
        
        try {
            // Show loading
            window.app.showLoading('Starting training...');
            
            // Start training
            const response = await trainingService.startTraining(config);
            
            // Hide loading
            window.app.hideLoading();
            
            // Show success message
            window.app.showAlert('Training started successfully!', 'success');
            
            // Show training monitor
            this.showTrainingMonitor();
            this.hideConfigForm();
            
            // Reset training data
            this.resetTrainingData();
            
        } catch (error) {
            // Hide loading
            window.app.hideLoading();
            
            // Log detailed error information
            console.error('Training start error:', error);
            
            // Show error message
            window.app.showAlert(`Failed to start training: ${error.message}`, 'danger');
        }
    }
    
    /**
     * Stop training
     */
    async stopTraining() {
        if (confirm('Are you sure you want to stop the training?')) {
            try {
                // Show loading
                window.app.showLoading('Stopping training...');
                
                // Stop training
                await trainingService.stopTraining();
                
                // Hide loading
                window.app.hideLoading();
                
                // Show success message
                window.app.showAlert('Training stopped successfully', 'success');
                
                // Hide training monitor
                this.hideTrainingMonitor();
                this.showConfigForm();
                
            } catch (error) {
                // Hide loading
                window.app.hideLoading();
                
                // Show error message
                window.app.showAlert(`Failed to stop training: ${error.message}`, 'danger');
            }
        }
    }
    
    /**
     * Show training monitor
     */
    showTrainingMonitor() {
        if (this.monitoringSection) {
            this.monitoringSection.classList.remove('d-none');
        }
        
        // Initialize charts if they don't exist
        this.initCharts();
    }
    
    /**
     * Hide training monitor
     */
    hideTrainingMonitor() {
        if (this.monitoringSection) {
            this.monitoringSection.classList.add('d-none');
        }
    }
    
    /**
     * Show configuration form
     */
    showConfigForm() {
        if (this.configForm) {
            this.configForm.classList.remove('d-none');
        }
    }
    
    /**
     * Hide configuration form
     */
    hideConfigForm() {
        if (this.configForm) {
            this.configForm.classList.add('d-none');
        }
    }
    
    /**
     * Initialize charts
     */
    initCharts() {
        if (this.lossChartCanvas && !this.charts.lossChart) {
            this.charts.lossChart = chartsUtil.createLossChart('loss-chart', this.trainingData);
        }
        
        if (this.perplexityChartCanvas && !this.charts.perplexityChart) {
            this.charts.perplexityChart = chartsUtil.createPerplexityChart('perplexity-chart', this.trainingData);
        }
        
        if (this.learningRateChartCanvas && !this.charts.learningRateChart) {
            this.charts.learningRateChart = chartsUtil.createLearningRateChart('learning-rate-chart', this.trainingData);
        }
        
        if (this.performanceChartCanvas && !this.charts.performanceChart) {
            this.charts.performanceChart = chartsUtil.createPerformanceChart('performance-chart', this.trainingData);
        }
        
        if (this.memoryChartCanvas && !this.charts.memoryChart) {
            this.charts.memoryChart = chartsUtil.createMemoryChart('memory-chart', this.trainingData);
        }
    }
    
    /**
     * Reset training data
     */
    resetTrainingData() {
        this.trainingData = {
            iterations: [],
            trainLoss: [],
            valLoss: [],
            learningRate: [],
            tokensPerSec: [],
            memoryUsage: []
        };
        
        // Update charts
        if (this.charts.lossChart) {
            chartsUtil.updateChart('loss-chart', this.trainingData);
        }
        
        if (this.charts.perplexityChart) {
            chartsUtil.updateChart('perplexity-chart', this.trainingData);
        }
        
        if (this.charts.learningRateChart) {
            chartsUtil.updateChart('learning-rate-chart', this.trainingData);
        }
        
        if (this.charts.performanceChart) {
            chartsUtil.updateChart('performance-chart', this.trainingData);
        }
        
        if (this.charts.memoryChart) {
            chartsUtil.updateChart('memory-chart', this.trainingData);
        }
    }
    
    /**
     * Update training monitor with new data
     * 
     * @param {Object} data - Training data
     */
    updateTrainingMonitor(data) {
        if (!data) {
            return;
        }
        
        // Update progress bar
        if (this.progressBar) {
            const progress = data.progress || 0;
            this.progressBar.style.width = `${progress}%`;
            this.progressBar.setAttribute('aria-valuenow', progress);
            this.progressBar.textContent = `${Math.round(progress)}%`;
            
            // Update progress bar color
            this.progressBar.className = 'progress-bar';
            if (progress < 33) {
                this.progressBar.classList.add('bg-info');
            } else if (progress < 66) {
                this.progressBar.classList.add('bg-primary');
            } else {
                this.progressBar.classList.add('bg-success');
            }
        }
        
        // Update status text
        if (this.statusText) {
            this.statusText.textContent = data.status || 'Training';
        }
        
        // Update iteration text
        if (this.iterationText) {
            this.iterationText.textContent = `${formatUtil.formatNumber(data.current_iteration || 0)} / ${formatUtil.formatNumber(data.max_iterations || 0)}`;
        }
        
        // Update loss text
        if (this.lossText) {
            const trainLoss = data.train_loss !== undefined ? formatUtil.formatDecimal(data.train_loss, 4) : '—';
            const trainPpl = data.train_perplexity !== undefined ? formatUtil.formatDecimal(data.train_perplexity, 2) : '—';
            this.lossText.textContent = `${trainLoss} (ppl: ${trainPpl})`;
        }
        
        // Update validation loss text
        if (this.valLossText) {
            const valLoss = data.val_loss !== undefined ? formatUtil.formatDecimal(data.val_loss, 4) : '—';
            const valPpl = data.val_perplexity !== undefined ? formatUtil.formatDecimal(data.val_perplexity, 2) : '—';
            this.valLossText.textContent = `${valLoss} (ppl: ${valPpl})`;
        }
        
        // Update elapsed text
        if (this.elapsedText && data.elapsed_minutes !== undefined) {
            this.elapsedText.textContent = formatUtil.formatDuration(data.elapsed_minutes * 60);
        }
        
        // Update ETA text
        if (this.etaText && data.eta_minutes !== undefined) {
            this.etaText.textContent = formatUtil.formatDuration(data.eta_minutes * 60);
        }
        
        // Update speed text
        if (this.speedText && data.tokens_per_sec !== undefined) {
            this.speedText.textContent = `${formatUtil.formatNumber(Math.round(data.tokens_per_sec))} tokens/sec`;
        }
        
        // Update memory text
        if (this.memoryText && data.peak_memory_gb !== undefined) {
            this.memoryText.textContent = `${formatUtil.formatDecimal(data.peak_memory_gb, 1)} GB`;
        }
        
        // Update training data
        if (data.current_iteration !== undefined) {
            // Add data point
            this.trainingData.iterations.push(data.current_iteration);
            this.trainingData.trainLoss.push(data.train_loss !== undefined ? data.train_loss : null);
            this.trainingData.valLoss.push(data.val_loss !== undefined ? data.val_loss : null);
            this.trainingData.learningRate.push(data.learning_rate !== undefined ? data.learning_rate : null);
            this.trainingData.tokensPerSec.push(data.tokens_per_sec !== undefined ? data.tokens_per_sec : null);
            this.trainingData.memoryUsage.push(data.peak_memory_gb !== undefined ? data.peak_memory_gb : null);
            
            // Update charts
            if (this.charts.lossChart) {
                chartsUtil.updateChart('loss-chart', this.trainingData);
            }
            
            if (this.charts.perplexityChart) {
                chartsUtil.updateChart('perplexity-chart', this.trainingData);
            }
            
            if (this.charts.learningRateChart) {
                chartsUtil.updateChart('learning-rate-chart', this.trainingData);
            }
            
            if (this.charts.performanceChart) {
                chartsUtil.updateChart('performance-chart', this.trainingData);
            }
            
            if (this.charts.memoryChart) {
                chartsUtil.updateChart('memory-chart', this.trainingData);
            }
        }
    }
    
    /**
     * Handle training finished event
     * 
     * @param {Object} data - Training finished data
     */
    handleTrainingFinished(data) {
        // Show success message
        window.app.showAlert('Training completed successfully!', 'success');
        
        // Hide training monitor
        this.hideTrainingMonitor();
        this.showConfigForm();
    }
    
    /**
     * Called when the training view is activated
     */
    onActivate() {
        // Check for active training
        this.checkActiveTraining();
    }
} 