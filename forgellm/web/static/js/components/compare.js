// COMPARE TAB - USING PLOTLY (SAME AS MONITORING TAB)
let selectedSessions = new Map();

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
        try {
            const sessionsResponse = await fetch('/api/training/sessions');
            const sessionsData = await sessionsResponse.json();
            const sessions = sessionsData.training_sessions || [];
            const session = sessions.find(s => s.session_id === sessionId);
            if (!session || !session.log_file) throw new Error('Session log file not found');

            const response = await fetch(`/api/dashboard/historical`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: session.log_file })
            });
            const sessionData = await response.json();
            
            selectedSessions.set(sessionId, {
                ...sessionData,
                session_name: session.session_name,
                session_id: sessionId
            });
        } catch (error) {
            console.error(`Error loading session ${sessionId}:`, error);
            document.getElementById(`session-${sessionId}`).checked = false;
            return;
        }
    } else {
        selectedSessions.delete(sessionId);
    }
    
    updateSelectionSummary();
    updateSessionColorsAndUI(); // Centralized function to update colors and UI

    if (selectedSessions.size >= 2) {
        generateComparison();
    } else {
        hideComparison();
    }
}

function updateSessionColorsAndUI() {
    const colors = getSessionColors();
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark' || document.body.getAttribute('data-theme') === 'dark';
    
    // Reset all session cards to their default, non-selected state
    document.querySelectorAll('.session-card').forEach(card => {
        card.style.backgroundColor = '';
        card.style.borderLeftColor = 'transparent';
        card.classList.remove('selected-session-card');
        // Reset text color to default
        card.querySelectorAll('.session-card-title, .session-card-text, .session-card-text-label').forEach(el => el.style.color = '');
    });

    let colorIndex = 0;
    for (const [sessionId, sessionData] of selectedSessions) {
        const color = colors[colorIndex % colors.length];
        sessionData.color = color; // Assign/update color in the session data map

        const sessionCard = document.querySelector(`#session-${sessionId} + label .session-card`);
        if (sessionCard) {
            sessionCard.classList.add('selected-session-card');
            sessionCard.style.borderLeftColor = color;
            sessionCard.style.backgroundColor = `${color}33`; // Use color with ~20% alpha

            // Change text to be readable on the new background
            const textColor = isDarkMode ? '#FFFFFF' : '#000000';
            sessionCard.querySelectorAll('.session-card-title, .session-card-text, .session-card-text-label').forEach(el => {
                el.style.color = textColor;
            });
        }
        colorIndex++;
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

// Function to wrap long legend text for Plotly by inserting <br> tags
function wrapText(text, maxLength = 40) {
    if (!text || text.length <= maxLength) {
        return text;
    }
    // A more robust way to split, handling long segments without underscores
    const parts = text.split(/([_])/).flatMap(part => {
        if (part.length > maxLength) {
            return part.match(new RegExp(`.{1,${maxLength}}`, 'g')) || [];
        }
        return part;
    });

    let wrappedText = '';
    let currentLine = '';

    parts.forEach((part, index) => {
        if (currentLine.length + part.length > maxLength) {
            wrappedText += currentLine + '<br>';
            currentLine = part;
        } else {
            currentLine += part;
        }
    });
    wrappedText += currentLine;

    // Clean up to avoid leading/trailing underscores on lines
    return wrappedText.replace(/<br>_/g, '<br>').replace(/_<br>/g, '<br>');
}

// Generic function to render any comparison chart
function renderComparisonChart(containerId, traces, layoutOptions) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const containerWidth = container.offsetWidth;
    const containerHeight = container.offsetHeight;
    if (containerWidth < 50 || containerHeight < 50) return;

    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark' || document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDarkMode ? '#F5F5F5' : '#333333';
    const borderColor = isDarkMode ? '#555555' : '#DDDDDD';

    const layout = {
        width: containerWidth,
        height: containerHeight,
        autosize: false,
        margin: { l: 60, r: 20, t: 50, b: 60 }, // Clean bottom margin, legend is removed
        showlegend: false, // --- LEGEND IS NOW REMOVED ---
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textColor },
        title: {
            text: layoutOptions.title,
            x: 0.05,
            font: { color: textColor, size: 16 }
        },
        xaxis: {
            ...layoutOptions.xaxis,
            title: {
                text: layoutOptions.xaxis.title,
                font: { color: textColor },
                standoff: 20
            },
            gridcolor: borderColor,
            linecolor: borderColor,
            zerolinecolor: borderColor,
            ticks: 'outside',
            tickcolor: borderColor,
            tickfont: { color: textColor }
        },
        yaxis: {
            ...layoutOptions.yaxis,
            title: { text: layoutOptions.yaxis.title, font: { color: textColor } },
            gridcolor: borderColor,
            linecolor: borderColor,
            zerolinecolor: borderColor,
            ticks: 'outside',
            tickcolor: borderColor,
            tickfont: { color: textColor },
            automargin: true
        },
        shapes: layoutOptions.shapes || [],
        annotations: layoutOptions.annotations || []
    };

    Plotly.react(container, traces, layout, {
        responsive: false,
        displayModeBar: false
    });
}

async function generateComparison() {
    const placeholder = document.getElementById('comparison-placeholder');
    const chartsGrid = document.getElementById('comparison-charts-grid');
    if (!placeholder || !chartsGrid) return;

    placeholder.style.display = 'none';
    chartsGrid.style.display = 'block';
    
    setTimeout(() => {
        try {
            // --- 1. Loss Comparison (VALIDATION) ---
            const lossTraces = [];
            for (const [sessionId, sessionData] of selectedSessions) {
                if (sessionData.charts?.loss?.data) {
                    const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
                    if (validationLoss?.x && validationLoss?.y) {
                        lossTraces.push({
                            x: validationLoss.x, y: validationLoss.y, type: 'scatter', mode: 'lines',
                            name: sessionData.session_name, // Name for hover data
                            line: { color: sessionData.color, width: 2 } // Use assigned color
                        });
                    }
                }
            }
            renderComparisonChart('loss-comparison-chart', lossTraces, {
                title: 'Validation Loss',
                xaxis: { title: 'Iterations' },
                yaxis: { title: 'Validation Loss' }
            });

            // --- 2. Perplexity Comparison (VALIDATION) ---
            const perplexityTraces = [];
            for (const [sessionId, sessionData] of selectedSessions) {
                 if (sessionData.charts?.perplexity?.data) {
                    const validationPerplexity = sessionData.charts.perplexity.data.find(c => c.name === 'Validation Perplexity');
                    if (validationPerplexity?.x && validationPerplexity?.y) {
                        perplexityTraces.push({
                            x: validationPerplexity.x, y: validationPerplexity.y, type: 'scatter', mode: 'lines',
                            name: sessionData.session_name,
                            line: { color: sessionData.color, width: 2 } // Use assigned color
                        });
                    }
                }
            }
            renderComparisonChart('perplexity-comparison-chart', perplexityTraces, {
                title: 'Validation Perplexity',
                xaxis: { title: 'Iterations' },
                yaxis: { title: 'Validation Perplexity' }
            });

            // --- 3. Stability Comparison (VALIDATION LOSS) ---
            const stabilityTraces = [];
            for (const [sessionId, sessionData] of selectedSessions) {
                if (sessionData.charts?.loss?.data) {
                    const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
                    if (validationLoss?.x && validationLoss?.y) {
                        const windowSize = 10;
                        const varianceX = [], varianceY = [];
                        // Ensure there are enough points to calculate variance
                        if (validationLoss.y.length >= windowSize) {
                            for (let i = windowSize; i < validationLoss.y.length; i++) {
                                const window = validationLoss.y.slice(i - windowSize, i);
                                const mean = window.reduce((a, b) => a + b, 0) / window.length;
                                const variance = window.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / window.length;
                                varianceX.push(validationLoss.x[i]);
                                varianceY.push(variance);
                            }
                        }
                        if (varianceX.length > 0) {
                            stabilityTraces.push({
                                x: varianceX, y: varianceY, type: 'scatter', mode: 'lines',
                                name: sessionData.session_name,
                                line: { color: sessionData.color, width: 2 } // Use assigned color
                            });
                        }
                    }
                }
            }
            renderComparisonChart('stability-comparison-chart', stabilityTraces, {
                title: 'Validation Loss Stability',
                xaxis: { title: 'Iterations' },
                yaxis: { title: 'Loss Variance', range: [0, 0.1] },
                shapes: [
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0, x1: 1, y1: 0.005, fillcolor: 'rgba(40, 167, 69, 0.2)', line: { width: 0 }, layer: 'below' },
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0.005, x1: 1, y1: 0.02, fillcolor: 'rgba(255, 193, 7, 0.2)', line: { width: 0 }, layer: 'below' },
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0.02, x1: 1, y1: 0.1, fillcolor: 'rgba(220, 53, 69, 0.2)', line: { width: 0 }, layer: 'below' }
                ],
                annotations: [
                    { text: 'Excellent', x: 0.95, y: 0.0025, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(40, 167, 69, 0.9)', size: 10 }, xanchor: 'right' },
                    { text: 'Good', x: 0.95, y: 0.0125, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(255, 193, 7, 0.9)', size: 10 }, xanchor: 'right' },
                    { text: 'Unstable', x: 0.95, y: 0.06, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(220, 53, 69, 0.9)', size: 10 }, xanchor: 'right' }
                ]
            });

            // --- 4. Generalization Gap (Correct by definition) ---
            const gapTraces = [];
            for (const [sessionId, sessionData] of selectedSessions) {
                if (sessionData.charts?.loss?.data) {
                    const trainingLoss = sessionData.charts.loss.data.find(c => c.name === 'Training Loss');
                    const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
                    if (trainingLoss?.x && trainingLoss?.y) {
                        if (validationLoss?.x?.length > 0) {
                            const valMap = new Map(validationLoss.x.map((iter, i) => [iter, validationLoss.y[i]]));
                            const gapX = [], gapY = [];
                            trainingLoss.x.forEach((iter, i) => {
                                if (valMap.has(iter)) {
                                    gapX.push(iter);
                                    gapY.push(valMap.get(iter) - trainingLoss.y[i]);
                                }
                            });
                            if (gapX.length > 0) {
                                gapTraces.push({
                                    x: gapX, y: gapY, type: 'scatter', mode: 'lines',
                                    name: sessionData.session_name,
                                    line: { color: sessionData.color, width: 2 } // Use assigned color
                                });
                            }
                        } else {
                             gapTraces.push({
                                x: trainingLoss.x, y: trainingLoss.x.map(() => 0), type: 'scatter', mode: 'lines',
                                name: `${sessionData.session_name} (No Val)`,
                                line: { color: sessionData.color, width: 2, dash: 'dash' } // Use assigned color
                            });
                        }
                    }
                }
            }
            renderComparisonChart('generalization-comparison-chart', gapTraces, {
                title: 'Generalization Gap',
                xaxis: { title: 'Iterations' },
                yaxis: { title: 'Val Loss - Train Loss', range: [-0.5, 0.5] },
                shapes: [
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0.1, x1: 1, y1: 0.5, fillcolor: 'rgba(255, 193, 7, 0.2)', line: { width: 0 }, layer: 'below' },
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: -0.1, x1: 1, y1: 0.1, fillcolor: 'rgba(40, 167, 69, 0.2)', line: { width: 0 }, layer: 'below' },
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: -0.5, x1: 1, y1: -0.1, fillcolor: 'rgba(220, 53, 69, 0.2)', line: { width: 0 }, layer: 'below' }
                ],
                annotations: [
                    { text: 'Underfitting', x: 0.95, y: 0.3, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(255, 193, 7, 0.9)', size: 10 }, xanchor: 'right' },
                    { text: 'Good Fit', x: 0.95, y: 0, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(40, 167, 69, 0.9)', size: 10 }, xanchor: 'right' },
                    { text: 'Overfitting', x: 0.95, y: -0.3, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(220, 53, 69, 0.9)', size: 10 }, xanchor: 'right' }
                ]
            });
        } catch (error) {
            console.error('Error generating comparison:', error);
            hideComparison();
        }
    }, 100);
}

// Generate distinct colors for sessions
function getSessionColors() {
    // Standard color palette for charts and UI elements
    return ['#0d6efd', '#dc3545', '#198754', '#fd7e14', '#6f42c1', '#d63384', '#20c997'];
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

console.log('Compare.js Plotly version loaded');

// Inject CSS for clean styling
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
    transition: background-color 0.2s ease-in-out, border-left-color 0.2s ease-in-out;
    border-left: 4px solid transparent;
}

.session-card-title,
.session-card-text,
.session-card-text-label {
    transition: color 0.2s ease-in-out;
}

.session-card:hover {
    background-color: var(--bs-tertiary-bg);
}

.btn-check:checked + label .session-card {
    border-left-color: var(--bs-primary); /* Default border color */
    background-color: var(--bs-tertiary-bg);
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

/* Chart containers using Plotly */
#loss-comparison-chart,
#perplexity-comparison-chart,
#stability-comparison-chart,
#generalization-comparison-chart {
    width: 100%;
    height: 300px;
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

/* Selected State for Session Card */
.session-card.selected-session-card {
    border-left-width: 4px !important;
    border-left-style: solid !important;
}

#selection-summary {
    position: sticky;
    bottom: 0;
    background: var(--bs-body-bg);
    padding: 0.75rem;
    border-top: 1px solid var(--bs-border-color);
    z-index: 1000;
}
`;
document.head.appendChild(style); 