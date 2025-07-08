/**
 * Training Session Comparison Component
 * Handles comparison of multiple training sessions with advanced visualization
 */

// SIMPLE GLOBAL APPROACH - NO CLASSES, NO COMPLEX INITIALIZATION
let selectedSessions = new Map();
let comparisonData = null;

// Simple function to check if elements exist
function elementsExist() {
    const required = ['comparison-placeholder', 'comparison-charts-grid', 'compare-sessions-list'];
    return required.every(id => document.getElementById(id) !== null);
}

// Function to sort sessions by model name and size
function sortSessions(sessions) {
    return sessions.sort((a, b) => {
        // Extract model base name (remove size info)
        const getModelBase = (name) => name.replace(/-(7B|9B|13B|32B|70B)/gi, '').toLowerCase();
        const getModelSize = (name) => {
            const match = name.match(/(7B|9B|13B|32B|70B)/gi);
            return match ? match[0] : 'Unknown';
        };
        
        const aBase = getModelBase(a.model_name || '');
        const bBase = getModelBase(b.model_name || '');
        const aSize = getModelSize(a.model_name || '');
        const bSize = getModelSize(b.model_name || '');
        
        // First sort by model base name
        if (aBase !== bBase) {
            return aBase.localeCompare(bBase);
        }
        
        // Then by size (convert to numbers for proper sorting)
        const sizeOrder = {'7B': 1, '9B': 2, '13B': 3, '32B': 4, '70B': 5, 'Unknown': 6};
        return (sizeOrder[aSize] || 6) - (sizeOrder[bSize] || 6);
    });
}

// Function to extract training parameters for tooltip
function getTrainingParameters(session) {
    const sessionName = session.session_name || '';
    const params = {
        learningRate: 'Unknown',
        batchSize: 'Unknown',
        iterations: session.latest_iteration || 'Unknown',
        sequenceLength: 'Unknown',
        trainingType: 'Unknown'
    };
    
    // Extract from session name (pattern: model_lr_bs_iter_seq_date)
    const parts = sessionName.split('_');
    
    // Look for learning rate (lr1e_05 -> 1e-05)
    const lrMatch = sessionName.match(/lr(\d+e?_?\d*)/i);
    if (lrMatch) {
        params.learningRate = lrMatch[1].replace('_', '-');
    }
    
    // Look for batch size (bs5 -> 5)
    const bsMatch = sessionName.match(/bs(\d+)/i);
    if (bsMatch) {
        params.batchSize = bsMatch[1];
    }
    
    // Look for sequence length (seq3072 -> 3072)
    const seqMatch = sessionName.match(/seq(\d+)/i);
    if (seqMatch) {
        params.sequenceLength = seqMatch[1];
    }
    
    // Determine training type based on folder structure
    if (session.log_file) {
        if (session.log_file.includes('/cpt/')) {
            params.trainingType = 'CPT (Continued Pre-training)';
        } else if (session.log_file.includes('/ift/')) {
            params.trainingType = 'IFT (Instruction Fine-tuning)';
        } else if (session.log_file.includes('lora')) {
            params.trainingType = 'LoRA';
        } else if (session.log_file.includes('dora')) {
            params.trainingType = 'DoRA';
        }
    }
    
    return params;
}

// Load and display sessions
async function loadSessions() {
    try {
        const response = await fetch('/api/training/sessions');
        const data = await response.json();
        console.log('Sessions API response:', data);
        
        // Handle different API response formats
        let sessions = data.training_sessions || data.sessions || data || [];
        
        const container = document.getElementById('compare-sessions-list');
        if (!container) return;
        
        if (!Array.isArray(sessions)) {
            container.innerHTML = '<div class="text-muted">No sessions found</div>';
            return;
        }
        
        // Sort sessions by model name and size
        sessions = sortSessions(sessions);
        
        // Create compact session items with better layout and tooltips
        container.innerHTML = sessions.map(session => {
            // Clean up model name by removing "dataset_cpt_" prefix
            const cleanModelName = (session.model_name || 'Unknown').replace(/^dataset_cpt_/, '');
            
            // Use start_time instead of started_at
            const startDate = session.start_time ? new Date(session.start_time).toLocaleDateString() : 'Unknown';
            
            // Get training parameters for tooltip
            const params = getTrainingParameters(session);
            
            const tooltipContent = `Training Parameters:
• Type: ${params.trainingType}
• Learning Rate: ${params.learningRate}
• Batch Size: ${params.batchSize}
• Iterations: ${params.iterations}
• Sequence Length: ${params.sequenceLength}`;
            
            return `
                <div class="session-item mb-2" title="${tooltipContent}">
                    <div class="form-check">
                        <input class="form-check-input session-checkbox" type="checkbox" 
                               id="session-${session.session_id || session.id}" 
                               value="${session.session_id || session.id}"
                               onchange="handleSessionChange('${session.session_id || session.id}', this.checked)">
                        <label class="form-check-label w-100" for="session-${session.session_id || session.id}">
                            <div class="session-card">
                                <div class="session-header">
                                    <div class="session-name">${session.session_name || session.name || 'Unnamed Session'}</div>
                                    <div class="session-status">
                                        <span class="badge bg-secondary">${session.latest_iteration || session.iterations || 'N/A'} iter</span>
                                    </div>
                                </div>
                                <div class="session-details">
                                    <div class="detail-row">
                                        <span class="detail-label">Model:</span>
                                        <span class="detail-value">${cleanModelName}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">Started:</span>
                                        <span class="detail-value">${startDate}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">LR:</span>
                                        <span class="detail-value">${params.learningRate}</span>
                                    </div>
                                </div>
                            </div>
                        </label>
                    </div>
                </div>
            `;
        }).join('');
        
        console.log(`Loaded ${sessions.length} sessions`);
        
    } catch (error) {
        console.error('Error loading sessions:', error);
        const container = document.getElementById('compare-sessions-list');
        if (container) {
            container.innerHTML = '<div class="text-danger">Error loading sessions</div>';
        }
    }
}

// Handle session selection change
async function handleSessionChange(sessionId, isSelected) {
    if (isSelected) {
        // Add to selection
        try {
            // First get session details to find log file
            const sessionsResponse = await fetch('/api/training/sessions');
            const sessionsData = await sessionsResponse.json();
            const sessions = sessionsData.training_sessions || [];
            const session = sessions.find(s => s.session_id === sessionId);
            
            if (!session || !session.log_file) {
                throw new Error('Session log file not found');
            }
            
            const response = await fetch(`/api/dashboard/historical`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: session.log_file })
            });
            const sessionData = await response.json();
            
            // Store both the API data and session metadata
            selectedSessions.set(sessionId, {
                ...sessionData,
                session_name: session.session_name,
                session_id: sessionId
            });
            console.log(`Added session ${sessionId} to comparison`);
        } catch (error) {
            console.error(`Error loading session ${sessionId}:`, error);
            // Uncheck the checkbox
            document.getElementById(`session-${sessionId}`).checked = false;
            return;
        }
    } else {
        // Remove from selection
        selectedSessions.delete(sessionId);
        console.log(`Removed session ${sessionId} from comparison`);
    }
    
    updateSelectionSummary();
    
    // Auto-generate comparison if 2+ sessions selected
    if (selectedSessions.size >= 2) {
        generateComparison();
    } else {
        hideComparison();
    }
}

function updateSelectionSummary() {
    const summary = document.getElementById('selected-sessions-summary');
    const count = document.getElementById('selected-sessions-count');
    
    if (!summary || !count) return;
    
    if (selectedSessions.size > 0) {
        summary.style.display = 'block';
        count.textContent = `${selectedSessions.size} session${selectedSessions.size === 1 ? '' : 's'} selected`;
    } else {
        summary.style.display = 'none';
    }
}

function hideComparison() {
    const placeholder = document.getElementById('comparison-placeholder');
    const chartsGrid = document.getElementById('comparison-charts-grid');
    
    if (placeholder) placeholder.style.display = 'block';
    if (chartsGrid) chartsGrid.style.display = 'none';
}

async function generateComparison() {
    try {
        console.log(`Generating comparison for ${selectedSessions.size} sessions`);
        
        const placeholder = document.getElementById('comparison-placeholder');
        const chartsGrid = document.getElementById('comparison-charts-grid');
        
        if (!placeholder || !chartsGrid) {
            throw new Error('Chart containers not found');
        }
        
        // Show charts grid
        placeholder.style.display = 'none';
        chartsGrid.style.display = 'block';
        
        // Wait for DOM to update, then render charts
        setTimeout(async () => {
            await Promise.all([
                renderLossComparison(),
                renderPerplexityComparison(), 
                renderStabilityComparison(),
                renderGeneralizationComparison()
            ]);
        }, 100);
        
        console.log('Comparison charts generated successfully');
        
    } catch (error) {
        console.error('Error generating comparison:', error);
        hideComparison();
    }
}

// Function to get actual container dimensions and calculate proper chart size
function getChartDimensions(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return { width: 600, height: 400 };
    
    // Get the parent card-body
    const cardBody = container.parentElement;
    if (!cardBody) return { width: 600, height: 400 };
    
    // Calculate available space (accounting for padding/margins)
    const containerRect = cardBody.getBoundingClientRect();
    const width = Math.max(400, containerRect.width - 20); // Subtract padding
    const height = 400;
    
    console.log(`Chart dimensions for ${containerId}: ${width}x${height}`);
    return { width, height };
}

// Function to get dark theme compatible layout with explicit dimensions
function getDarkThemeLayout(title, width, height) {
    return {
        title: {
            text: title,
            font: { color: '#ffffff', size: 16 }
        },
        xaxis: { 
            title: { text: 'Iterations', font: { color: '#ffffff' } },
            color: '#ffffff',
            gridcolor: '#404040'
        },
        yaxis: { 
            title: { text: 'Loss', font: { color: '#ffffff' } },
            color: '#ffffff',
            gridcolor: '#404040'
        },
        margin: { l: 60, r: 30, t: 50, b: 50 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        legend: { font: { color: '#ffffff' } },
        width: width,
        height: height,
        autosize: false
    };
}

// Plotly config for full width usage
const plotlyConfig = {
    responsive: true,
    displayModeBar: false
};

async function renderLossComparison() {
    const chartContainer = document.getElementById('loss-comparison-chart');
    if (!chartContainer) return;
    
    // Get actual container dimensions
    const { width, height } = getChartDimensions('loss-comparison-chart');
    
    const traces = [];
    const colors = ['#007AFF', '#FF3B30', '#34C759', '#FF9500', '#AF52DE', '#00C7BE'];
    let colorIndex = 0;
    
    for (const [sessionId, sessionData] of selectedSessions) {
        const color = colors[colorIndex % colors.length];
        colorIndex++;
        
        // Access the correct data structure: charts.loss.data
        if (sessionData.charts && sessionData.charts.loss && sessionData.charts.loss.data) {
            const lossCharts = sessionData.charts.loss.data;
            
            // Find training loss data
            const trainingLoss = lossCharts.find(chart => chart.name === 'Training Loss');
            if (trainingLoss && trainingLoss.x && trainingLoss.y) {
                traces.push({
                    x: trainingLoss.x,
                    y: trainingLoss.y,
                    type: 'scatter',
                    mode: 'lines',
                    name: sessionData.session_name || `Session ${sessionId}`,
                    line: { color: color, width: 2 }
                });
            }
        }
    }
    
    const layout = getDarkThemeLayout('Training Loss Comparison', width, height);
    layout.yaxis.title.text = 'Loss';
    
    // Clear and set container dimensions
    chartContainer.style.width = width + 'px';
    chartContainer.style.height = height + 'px';
    chartContainer.innerHTML = '';
    
    Plotly.newPlot(chartContainer, traces, layout, plotlyConfig);
}

async function renderPerplexityComparison() {
    const chartContainer = document.getElementById('perplexity-comparison-chart');
    if (!chartContainer) return;
    
    // Get actual container dimensions
    const { width, height } = getChartDimensions('perplexity-comparison-chart');
    
    const traces = [];
    const colors = ['#007AFF', '#FF3B30', '#34C759', '#FF9500', '#AF52DE', '#00C7BE'];
    let colorIndex = 0;
    
    for (const [sessionId, sessionData] of selectedSessions) {
        const color = colors[colorIndex % colors.length];
        colorIndex++;
        
        // Access the correct data structure: charts.perplexity.data
        if (sessionData.charts && sessionData.charts.perplexity && sessionData.charts.perplexity.data) {
            const perplexityCharts = sessionData.charts.perplexity.data;
            
            // Find training perplexity data
            const trainingPerplexity = perplexityCharts.find(chart => chart.name === 'Training Perplexity');
            if (trainingPerplexity && trainingPerplexity.x && trainingPerplexity.y) {
                traces.push({
                    x: trainingPerplexity.x,
                    y: trainingPerplexity.y,
                    type: 'scatter',
                    mode: 'lines',
                    name: sessionData.session_name || `Session ${sessionId}`,
                    line: { color: color, width: 2 }
                });
            }
        }
    }
    
    const layout = getDarkThemeLayout('Perplexity Comparison', width, height);
    layout.yaxis.title.text = 'Perplexity';
    
    // Clear and set container dimensions
    chartContainer.style.width = width + 'px';
    chartContainer.style.height = height + 'px';
    chartContainer.innerHTML = '';
    
    Plotly.newPlot(chartContainer, traces, layout, plotlyConfig);
}

async function renderStabilityComparison() {
    const chartContainer = document.getElementById('stability-comparison-chart');
    if (!chartContainer) return;
    
    // Get actual container dimensions
    const { width, height } = getChartDimensions('stability-comparison-chart');
    
    const traces = [];
    const colors = ['#007AFF', '#FF3B30', '#34C759', '#FF9500', '#AF52DE', '#00C7BE'];
    let colorIndex = 0;
    
    let minX = Infinity, maxX = -Infinity;
    
    for (const [sessionId, sessionData] of selectedSessions) {
        const color = colors[colorIndex % colors.length];
        colorIndex++;
        
        // Calculate loss variance/stability from training loss data
        if (sessionData.charts && sessionData.charts.loss && sessionData.charts.loss.data) {
            const lossCharts = sessionData.charts.loss.data;
            const trainingLoss = lossCharts.find(chart => chart.name === 'Training Loss');
            
            if (trainingLoss && trainingLoss.x && trainingLoss.y) {
                const windowSize = 10; // Calculate rolling variance
                const variances = [];
                const xValues = [];
                
                for (let i = windowSize; i < trainingLoss.y.length; i++) {
                    const window = trainingLoss.y.slice(i - windowSize, i);
                    const mean = window.reduce((a, b) => a + b, 0) / window.length;
                    const variance = window.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / window.length;
                    variances.push(variance);
                    xValues.push(trainingLoss.x[i]);
                }
                
                // Track min/max X for background zones
                if (xValues.length > 0) {
                    minX = Math.min(minX, Math.min(...xValues));
                    maxX = Math.max(maxX, Math.max(...xValues));
                }
                
                traces.push({
                    x: xValues,
                    y: variances,
                    type: 'scatter',
                    mode: 'lines',
                    name: sessionData.session_name || `Session ${sessionId}`,
                    line: { color: color, width: 2 }
                });
            }
        }
    }
    
    const layout = getDarkThemeLayout('Loss Stability (Variance)', width, height);
    layout.yaxis.title.text = 'Loss Variance';
    
    // Add colored background zones for stability levels
    if (minX !== Infinity && maxX !== -Infinity) {
        layout.shapes = [
            // Excellent zone (0 - 0.01)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: minX,
                y0: 0,
                x1: maxX,
                y1: 0.01,
                fillcolor: 'rgba(52, 199, 89, 0.2)', // Green
                line: { width: 0 },
                layer: 'below'
            },
            // Good zone (0.01 - 0.05)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: minX,
                y0: 0.01,
                x1: maxX,
                y1: 0.05,
                fillcolor: 'rgba(255, 204, 0, 0.2)', // Yellow
                line: { width: 0 },
                layer: 'below'
            },
            // Unstable zone (0.05+)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: minX,
                y0: 0.05,
                x1: maxX,
                y1: 0.2, // Cap at reasonable value for visualization
                fillcolor: 'rgba(255, 59, 48, 0.2)', // Red
                line: { width: 0 },
                layer: 'below'
            }
        ];
        
        // Add annotations for zones
        layout.annotations = [
            {
                x: minX + (maxX - minX) * 0.02,
                y: 0.005,
                text: 'Excellent',
                showarrow: false,
                font: { size: 10, color: '#ffffff' },
                bgcolor: 'rgba(52, 199, 89, 0.8)',
                bordercolor: 'rgba(52, 199, 89, 1)',
                borderwidth: 1
            },
            {
                x: minX + (maxX - minX) * 0.02,
                y: 0.03,
                text: 'Good',
                showarrow: false,
                font: { size: 10, color: '#ffffff' },
                bgcolor: 'rgba(255, 204, 0, 0.8)',
                bordercolor: 'rgba(255, 204, 0, 1)',
                borderwidth: 1
            },
            {
                x: minX + (maxX - minX) * 0.02,
                y: 0.1,
                text: 'Unstable',
                showarrow: false,
                font: { size: 10, color: '#ffffff' },
                bgcolor: 'rgba(255, 59, 48, 0.8)',
                bordercolor: 'rgba(255, 59, 48, 1)',
                borderwidth: 1
            }
        ];
    }
    
    // Clear and set container dimensions
    chartContainer.style.width = width + 'px';
    chartContainer.style.height = height + 'px';
    chartContainer.innerHTML = '';
    
    Plotly.newPlot(chartContainer, traces, layout, plotlyConfig);
}

async function renderGeneralizationComparison() {
    const chartContainer = document.getElementById('generalization-comparison-chart');
    if (!chartContainer) return;
    
    // Get actual container dimensions
    const { width, height } = getChartDimensions('generalization-comparison-chart');
    
    const traces = [];
    const colors = ['#007AFF', '#FF3B30', '#34C759', '#FF9500', '#AF52DE', '#00C7BE'];
    let colorIndex = 0;
    
    let minX = Infinity, maxX = -Infinity;
    let hasValidationData = false;
    
    for (const [sessionId, sessionData] of selectedSessions) {
        const color = colors[colorIndex % colors.length];
        colorIndex++;
        
        // Calculate generalization gap (validation - training loss)
        if (sessionData.charts && sessionData.charts.loss && sessionData.charts.loss.data) {
            const lossCharts = sessionData.charts.loss.data;
            const trainingLoss = lossCharts.find(chart => chart.name === 'Training Loss');
            const validationLoss = lossCharts.find(chart => chart.name === 'Validation Loss');
            
            console.log(`Session ${sessionId}: Training=${!!trainingLoss}, Validation=${!!validationLoss}`);
            
            if (trainingLoss && trainingLoss.x && trainingLoss.y) {
                if (validationLoss && validationLoss.x && validationLoss.y && validationLoss.y.length > 0) {
                    // Both training and validation data available
                    hasValidationData = true;
                    const gaps = [];
                    const xValues = [];
                    
                    // Create a map of validation data by iteration for easier lookup
                    const valMap = new Map();
                    for (let i = 0; i < validationLoss.x.length; i++) {
                        valMap.set(validationLoss.x[i], validationLoss.y[i]);
                    }
                    
                    // For each training point, find corresponding validation point
                    for (let i = 0; i < trainingLoss.x.length; i++) {
                        const iter = trainingLoss.x[i];
                        const trainLossVal = trainingLoss.y[i];
                        const valLossVal = valMap.get(iter);
                        
                        if (valLossVal !== undefined) {
                            gaps.push(valLossVal - trainLossVal);
                            xValues.push(iter);
                        }
                    }
                    
                    if (gaps.length > 0) {
                        // Track min/max X for background zones
                        minX = Math.min(minX, Math.min(...xValues));
                        maxX = Math.max(maxX, Math.max(...xValues));
                        
                        traces.push({
                            x: xValues,
                            y: gaps,
                            type: 'scatter',
                            mode: 'lines',
                            name: sessionData.session_name || `Session ${sessionId}`,
                            line: { color: color, width: 2 }
                        });
                    }
                } else {
                    // Only training data - show as flat line at 0 (no gap available)
                    if (trainingLoss.x.length > 0) {
                        minX = Math.min(minX, Math.min(...trainingLoss.x));
                        maxX = Math.max(maxX, Math.max(...trainingLoss.x));
                    }
                    
                    traces.push({
                        x: trainingLoss.x,
                        y: trainingLoss.x.map(() => 0),
                        type: 'scatter',
                        mode: 'lines',
                        name: `${sessionData.session_name || `Session ${sessionId}`} (Training only)`,
                        line: { color: color, width: 2, dash: 'dash' }
                    });
                }
            }
        }
    }
    
    const layout = getDarkThemeLayout('Generalization Gap', width, height);
    layout.yaxis.title.text = 'Validation Loss - Training Loss';
    
    // Add colored background zones for generalization assessment (only if we have validation data)
    if (minX !== Infinity && maxX !== -Infinity && hasValidationData) {
        layout.shapes = [
            // Underfitting zone (gap < -0.1)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: minX,
                y0: -0.5,
                x1: maxX,
                y1: -0.1,
                fillcolor: 'rgba(52, 199, 89, 0.2)', // Green
                line: { width: 0 },
                layer: 'below'
            },
            // Good fit zone (-0.1 to 0.1)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: minX,
                y0: -0.1,
                x1: maxX,
                y1: 0.1,
                fillcolor: 'rgba(100, 150, 255, 0.2)', // Blue
                line: { width: 0 },
                layer: 'below'
            },
            // Overfitting zone (gap > 0.1)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: minX,
                y0: 0.1,
                x1: maxX,
                y1: 0.5,
                fillcolor: 'rgba(255, 59, 48, 0.2)', // Red
                line: { width: 0 },
                layer: 'below'
            }
        ];
        
        // Add reference line at y=0
        layout.shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'y',
            x0: minX,
            y0: 0,
            x1: maxX,
            y1: 0,
            line: { color: '#666666', width: 1, dash: 'dot' },
            layer: 'below'
        });
        
        // Add annotations for zones
        layout.annotations = [
            {
                x: minX + (maxX - minX) * 0.02,
                y: -0.3,
                text: 'Underfitting',
                showarrow: false,
                font: { size: 10, color: '#ffffff' },
                bgcolor: 'rgba(52, 199, 89, 0.8)',
                bordercolor: 'rgba(52, 199, 89, 1)',
                borderwidth: 1
            },
            {
                x: minX + (maxX - minX) * 0.02,
                y: 0,
                text: 'Good fit',
                showarrow: false,
                font: { size: 10, color: '#ffffff' },
                bgcolor: 'rgba(100, 150, 255, 0.8)',
                bordercolor: 'rgba(100, 150, 255, 1)',
                borderwidth: 1
            },
            {
                x: minX + (maxX - minX) * 0.02,
                y: 0.3,
                text: 'Overfitting',
                showarrow: false,
                font: { size: 10, color: '#ffffff' },
                bgcolor: 'rgba(255, 59, 48, 0.8)',
                bordercolor: 'rgba(255, 59, 48, 1)',
                borderwidth: 1
            }
        ];
    }
    
    // Clear and set container dimensions
    chartContainer.style.width = width + 'px';
    chartContainer.style.height = height + 'px';
    chartContainer.innerHTML = '';
    
    Plotly.newPlot(chartContainer, traces, layout, plotlyConfig);
}

// Clear all selections
function clearAllSelections() {
    selectedSessions.clear();
    document.querySelectorAll('.session-checkbox').forEach(checkbox => {
        checkbox.checked = false;
    });
    updateSelectionSummary();
    hideComparison();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Compare tab initialization started');
    
    // Check if we're on the compare tab and elements exist
    if (elementsExist()) {
        console.log('Compare tab elements found, loading sessions');
        loadSessions();
        
        // Add event listener for clear button
        const clearBtn = document.getElementById('clear-selection-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', clearAllSelections);
        }
        
        // Add event listener for refresh button
        const refreshBtn = document.getElementById('refresh-sessions-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', loadSessions);
        }
    } else {
        console.log('Compare tab elements not found, will retry when tab becomes active');
    }
});

// Also listen for Bootstrap tab shown event
document.addEventListener('shown.bs.tab', function(event) {
    if (event.target.getAttribute('data-bs-target') === '#compare') {
        console.log('Compare tab activated, loading sessions');
        setTimeout(() => {
            if (elementsExist()) {
                loadSessions();
            }
        }, 100);
    }
});

console.log('Compare.js simple version loaded');

// Inject CSS for improved session layout and FULL WIDTH charts
const style = document.createElement('style');
style.textContent = `
/* Session List Container */
#compare-sessions-list {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 5px;
}

/* Custom scrollbar for session list */
#compare-sessions-list::-webkit-scrollbar {
    width: 6px;
}

#compare-sessions-list::-webkit-scrollbar-track {
    background: var(--surface-color);
    border-radius: 3px;
}

#compare-sessions-list::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 3px;
}

#compare-sessions-list::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Session Card Styling */
.session-card {
    background: var(--surface-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
}

.session-card:hover {
    border-color: #007AFF !important;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.1) !important;
    transform: translateY(-1px);
}

/* Session Header */
.session-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}

.session-name {
    font-weight: 600 !important;
    font-size: 14px !important;
    color: var(--text-color) !important;
    line-height: 1.3;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.session-status {
    flex-shrink: 0;
    margin-left: 8px;
}

.session-status .badge {
    font-size: 10px !important;
    padding: 2px 6px !important;
}

/* Session Details */
.session-details {
    font-size: 11px !important;
    color: var(--text-muted) !important;
}

.detail-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}

.detail-label {
    font-weight: 500;
    min-width: 50px;
}

.detail-value {
    text-align: right;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Checkbox Styling */
.session-checkbox {
    margin-top: 2px !important;
}

/* Selected State */
.form-check-input:checked + .form-check-label .session-card {
    background: rgba(0, 122, 255, 0.1) !important;
    border-color: #007AFF !important;
}

/* Session Item Spacing */
.session-item {
    margin-bottom: 8px !important;
}

.session-item:last-child {
    margin-bottom: 0 !important;
}

/* Form Check Label */
.form-check-label {
    margin-bottom: 0 !important;
    cursor: pointer;
}

/* Dark Mode Enhancements */
[data-theme="dark"] .session-card {
    background: #2a2a2a !important;
    border-color: #404040 !important;
}

[data-theme="dark"] .session-card:hover {
    background: #333333 !important;
    border-color: #007AFF !important;
}

[data-theme="dark"] .form-check-input:checked + .form-check-label .session-card {
    background: rgba(0, 122, 255, 0.15) !important;
}

/* Selection Summary Styling */
#selected-sessions-summary {
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
}

#selected-sessions-summary h6 {
    margin-bottom: 8px;
    font-size: 14px;
    color: var(--text-color);
}

#selected-sessions-count {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 10px;
}

/* CRITICAL: Force chart containers to use full available space */
#loss-comparison-chart,
#perplexity-comparison-chart,
#stability-comparison-chart,
#generalization-comparison-chart {
    min-height: 400px !important;
}

/* Ensure card bodies use full width */
.card-body {
    padding: 8px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Force Bootstrap columns to use full allocated space */
.comparison-charts-grid .col-lg-6 {
    width: 50% !important;
    max-width: 50% !important;
    flex: 0 0 50% !important;
    padding: 0 8px !important;
}

/* Tooltip styling */
.session-item[title] {
    position: relative;
}

.session-item[title]:hover::after {
    content: attr(title);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 11px;
    white-space: pre-line;
    z-index: 1000;
    max-width: 200px;
    word-wrap: break-word;
}

.session-item[title]:hover::before {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(100%);
    border: 5px solid transparent;
    border-top-color: rgba(0, 0, 0, 0.9);
    z-index: 1000;
}
`;
document.head.appendChild(style); 