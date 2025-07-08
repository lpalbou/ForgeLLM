// COMPARE TAB - USING PLOTLY (SAME AS MONITORING TAB)
let selectedSessions = new Map();

// Simple function to check if elements exist
function elementsExist() {
    const required = ['comparison-placeholder', 'comparison-charts-grid', 'compare-sessions-list'];
    return required.every(id => document.getElementById(id) !== null);
}

// Function to store selected sessions in localStorage
function storeSelectedSessions() {
    const sessionsArray = Array.from(selectedSessions.keys());
    localStorage.setItem('compare-selected-sessions', JSON.stringify(sessionsArray));
}

// Function to restore selected sessions from localStorage
async function restoreSelectedSessions() {
    try {
        const storedSessions = localStorage.getItem('compare-selected-sessions');
        if (storedSessions) {
            const sessionsArray = JSON.parse(storedSessions);
            if (Array.isArray(sessionsArray) && sessionsArray.length > 0) {
                // First clear any existing selections to avoid duplicates
                clearAllSelections();
                
                // Then check each stored session ID and select it if available
                for (const sessionId of sessionsArray) {
                    const checkbox = document.getElementById(`session-${sessionId}`);
                    if (checkbox) {
                        checkbox.checked = true;
                        await handleSessionChange(sessionId, true);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error restoring selected sessions:', error);
    }
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
        
        // Debug session structure
        if (sessions.length > 0) {
            console.log('First session structure:', sessions[0]);
            console.log('Session ID format:', sessions[0].session_id || sessions[0].id);
        }
        
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
            // The session ID is the full directory name
            const sessionId = session.session_id || session.id || '';
            
            // Debug session ID
            console.log(`Session ${session.session_name || 'unnamed'} has ID: ${sessionId}`);
            
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
                <div class="session-item mb-2">
                    <div class="form-check">
                        <input class="form-check-input session-checkbox" type="checkbox" 
                               id="session-${sessionId}" 
                               value="${sessionId}"
                               onchange="handleSessionChange('${sessionId}', this.checked)">
                        <label class="form-check-label w-100" for="session-${sessionId}">
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
                                <div class="session-actions mt-2 pt-2 border-top">
                                    <button class="btn btn-sm btn-outline-secondary view-params-btn" 
                                            onclick="showSessionParameters('${sessionId}'); event.preventDefault(); event.stopPropagation();"
                                            title="View Parameters">
                                        <i class="fas fa-file-code"></i>
                                    </button>
                                    <button class="btn btn-sm btn-outline-secondary test-session-btn" 
                                            onclick="testSessionInPlayground('${sessionId}'); event.preventDefault(); event.stopPropagation();"
                                            title="Test in Playground">
                                        <i class="fas fa-vial"></i>
                                    </button>
                                </div>
                            </div>
                        </label>
                    </div>
                </div>
            `;
        }).join('');
        
        console.log(`Loaded ${sessions.length} sessions`);
        
        // Add event listener for tab changes to store/restore selections
        document.addEventListener('shown.bs.tab', function(event) {
            // If we're leaving the compare tab, store the selections
            if (event.relatedTarget && event.relatedTarget.id === 'compare-tab') {
                storeSelectedSessions();
            }
            
            // If we're entering the compare tab, restore the selections
            if (event.target && event.target.id === 'compare-tab') {
                setTimeout(() => {
                    restoreSelectedSessions();
                }, 100); // Short delay to ensure DOM is ready
            }
        });
        
        // Restore selections when the page loads
        if (document.querySelector('#compare-tab.active')) {
            setTimeout(() => {
                restoreSelectedSessions();
            }, 100);
        }
        
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

    // Store the updated selections
    storeSelectedSessions();
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

// Add these functions to handle the button clicks

// Function to show session parameters in a modal
async function showSessionParameters(sessionId) {
    try {
        console.log('Showing parameters for session ID:', sessionId);
        
        // Show loading indicator
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        if (loadingOverlay && loadingMessage) {
            loadingMessage.textContent = 'Loading training logs...';
            loadingOverlay.classList.remove('d-none');
        }
        
        // Set a timeout to prevent infinite loading
        const timeoutId = setTimeout(() => {
            if (loadingOverlay) loadingOverlay.classList.add('d-none');
            alert('Loading training logs is taking too long. Please try again.');
        }, 30000); // 30 second timeout
        
        // Find the log file for this session
        const sessionsResponse = await fetch('/api/training/sessions');
        if (!sessionsResponse.ok) {
            throw new Error(`Failed to load training sessions: ${sessionsResponse.status}`);
        }
        
        const sessionsData = await sessionsResponse.json();
        if (!sessionsData.success || !sessionsData.training_sessions) {
            throw new Error('Failed to load training sessions data');
        }
        
        // Find the matching session
        const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        
        console.log('Found session:', session);
        const logFile = session.log_file;
        
        // Use the /api/logs/raw endpoint with the exact log file path
        const response = await fetch('/api/logs/raw', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ log_file: logFile })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to load session parameters (${response.status}): ${response.statusText}`);
        }
        
        const sessionDetails = await response.json();
        console.log('Session details:', sessionDetails);
        
        if (!sessionDetails.success) {
            throw new Error(sessionDetails.error || 'Failed to load logs');
        }
        
        // Parse the raw logs content
        let rawData;
        try {
            rawData = JSON.parse(sessionDetails.logs);
        } catch (e) {
            // If parsing fails, treat as plain text
            rawData = { raw_content: sessionDetails.logs };
        }
        
        // Extract metadata for display
        let metadataHtml = '';
        if (rawData.config || rawData.base_model) {
            const config = rawData.config || {};
            
            // Get model name from top level or config
            const modelName = rawData.base_model || config.base_model || rawData.model_name || config.model_name || 'N/A';
            // Clean up model name (remove mlx-community/ prefix if present)
            const cleanModelName = modelName.replace(/^mlx-community\//, '');
            
            metadataHtml = `
                <div class="alert alert-info mb-3">
                    <h6 class="mb-2"><i class="fas fa-info-circle me-2"></i>Training Metadata</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <small><strong>Model:</strong> ${cleanModelName}</small><br>
                            <small><strong>Type:</strong> ${rawData.training_type || config.training_type || 'N/A'}</small><br>
                            <small><strong>Fine-tune Type:</strong> ${config.fine_tune_type || 'N/A'}</small>
                        </div>
                        <div class="col-md-6">
                            <small><strong>Batch Size:</strong> ${config.batch_size || 'N/A'}</small><br>
                            <small><strong>Learning Rate:</strong> ${config.learning_rate || 'N/A'}</small><br>
                            <small><strong>Max Iterations:</strong> ${config.max_iterations || 'N/A'}</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Show the formatted logs in the modal with metadata
        const logsContent = document.getElementById('logs-content');
        
        // Clear existing content
        if (logsContent) {
            logsContent.innerHTML = '';
            
            // Add metadata section if available
            if (metadataHtml) {
                const metadataDiv = document.createElement('div');
                metadataDiv.innerHTML = metadataHtml;
                logsContent.appendChild(metadataDiv);
            }
            
            // Add formatted JSON content
            const jsonPre = document.createElement('pre');
            jsonPre.className = 'json-content bg-dark text-light p-3 rounded';
            jsonPre.style.cssText = 'white-space: pre-wrap; word-wrap: break-word; font-family: "Courier New", monospace; font-size: 12px; line-height: 1.4; max-height: 60vh; overflow-y: auto; border: 1px solid #444;';
            
            if (rawData.raw_content) {
                // Plain text content
                jsonPre.textContent = rawData.raw_content;
            } else {
                // Formatted JSON content with syntax highlighting
                const formattedJson = JSON.stringify(rawData, null, 2);
                jsonPre.innerHTML = syntaxHighlightJson(formattedJson);
            }
            
            logsContent.appendChild(jsonPre);
            
            // Setup copy button (store rawData in closure)
            const originalData = rawData;
            document.getElementById('copy-logs-btn').onclick = () => {
                // Get the JSON content from the pre element
                const jsonElement = logsContent.querySelector('.json-content');
                let textToCopy;
                
                if (jsonElement) {
                    // For syntax highlighted JSON, we need to get the original text
                    if (originalData.raw_content) {
                        textToCopy = originalData.raw_content;
                    } else {
                        textToCopy = JSON.stringify(originalData, null, 2);
                    }
                } else {
                    textToCopy = logsContent.textContent;
                }
                
                navigator.clipboard.writeText(textToCopy)
                    .then(() => {
                        // Show success feedback
                        const copyBtn = document.getElementById('copy-logs-btn');
                        const originalText = copyBtn.innerHTML;
                        copyBtn.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
                        setTimeout(() => {
                            copyBtn.innerHTML = originalText;
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Failed to copy logs:', err);
                        alert('Failed to copy logs: ' + err);
                    });
            };
            
            // Update the modal title
            const modalTitle = document.querySelector('#logsModal .modal-title');
            if (modalTitle) {
                modalTitle.textContent = `Training Logs: ${sessionId}`;
            }
            
            // Show the modal
            const logsModal = new bootstrap.Modal(document.getElementById('logsModal'));
            logsModal.show();
            
            // Hide loading and show success message
            clearTimeout(timeoutId);
            if (loadingOverlay) loadingOverlay.classList.add('d-none');
            console.log(`Successfully loaded training logs from ${logFile}`);
        } else {
            console.error('Logs content element not found');
            if (loadingOverlay) loadingOverlay.classList.add('d-none');
        }
    } catch (error) {
        console.error('Error fetching session details:', error);
        alert('Failed to load session parameters: ' + error.message);
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) loadingOverlay.classList.add('d-none');
    }
}

// Function to test a session in the playground tab
async function testSessionInPlayground(sessionId) {
    try {
        console.log('Testing session in playground:', sessionId);
        
        // Find the log file for this session
        const sessionsResponse = await fetch('/api/training/sessions');
        if (!sessionsResponse.ok) {
            throw new Error(`Failed to load training sessions: ${sessionsResponse.status}`);
        }
        
        const sessionsData = await sessionsResponse.json();
        if (!sessionsData.success || !sessionsData.training_sessions) {
            throw new Error('Failed to load training sessions data');
        }
        
        // Find the matching session
        const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        
        console.log('Found session:', session);
        
        // Determine if this is a CPT or IFT session
        const sessionType = sessionId.includes('_it_') ? 'ift' : 'cpt';
        
        // Format the model name as seen in the screenshot
        // Use the model_name from the session if available
        const modelName = session.model_name || `dataset_${sessionType}_${sessionId.split('_')[0]}`;
        
        console.log('Using model name:', modelName);
        
        // For the adapter path, use the session directory
        const adapterPath = session.log_file.split('/').slice(0, -1).join('/');
        
        console.log('Adapter path:', adapterPath);
        
        // Store the session configuration in localStorage for the testing tab to pick up
        const testConfig = {
            model: modelName,
            adapter: adapterPath
        };
        
        console.log('Test configuration:', testConfig);
        localStorage.setItem('forge-test-session', JSON.stringify(testConfig));
        
        // Save current scroll position
        const scrollPosition = window.scrollY;
        
        // Switch to the testing tab
        const testingTab = document.querySelector('[data-bs-target="#testing"]');
        if (testingTab) {
            console.log('Clicking testing tab');
            testingTab.click();
            
            // Wait for tab to be shown before setting values
            setTimeout(() => {
                // Restore scroll position to top
                window.scrollTo(0, 0);
                
                // Set the model and adapter values directly
                const modelSelect = document.getElementById('test-model-select');
                const adapterSelect = document.getElementById('adapter-path');
                
                if (modelSelect) {
                    // Find the option that contains the model name
                    const modelOptions = Array.from(modelSelect.options);
                    const modelOption = modelOptions.find(option => 
                        option.textContent.toLowerCase().includes(modelName.toLowerCase())
                    );
                    
                    if (modelOption) {
                        modelSelect.value = modelOption.value;
                        // Trigger change event
                        modelSelect.dispatchEvent(new Event('change'));
                    }
                }
                
                if (adapterSelect) {
                    // Find the option that contains the adapter path
                    const adapterOptions = Array.from(adapterSelect.options);
                    const adapterOption = adapterOptions.find(option => 
                        option.value === adapterPath || 
                        option.textContent.includes(adapterPath.split('/').pop())
                    );
                    
                    if (adapterOption) {
                        adapterSelect.value = adapterOption.value;
                        // Trigger change event
                        adapterSelect.dispatchEvent(new Event('change'));
                    } else {
                        // If adapter option not found, we need to wait for the adapter dropdown to be populated
                        console.log('Adapter option not found, waiting for dropdown to populate...');
                        
                        // Add a new option if it doesn't exist
                        const newOption = document.createElement('option');
                        newOption.value = adapterPath;
                        newOption.textContent = adapterPath.split('/').pop();
                        adapterSelect.appendChild(newOption);
                        adapterSelect.value = adapterPath;
                        adapterSelect.dispatchEvent(new Event('change'));
                    }
                }
                
                // Focus the load model button for better UX
                const loadModelBtn = document.getElementById('load-model-btn');
                if (loadModelBtn) {
                    loadModelBtn.focus();
                }
            }, 500); // Give the tab time to initialize
        } else {
            console.error('Testing tab button not found');
        }
    } catch (error) {
        console.error('Error preparing test session:', error);
        alert('Failed to prepare test session: ' + error.message);
    }
}

// Syntax highlight function for JSON
function syntaxHighlightJson(json) {
    if (!json) return '';
    
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'text-warning'; // number
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'text-info'; // key
            } else {
                cls = 'text-success'; // string
            }
        } else if (/true|false/.test(match)) {
            cls = 'text-primary'; // boolean
        } else if (/null/.test(match)) {
            cls = 'text-danger'; // null
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

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

.session-actions {
    display: flex;
    gap: 5px;
}

.session-actions .btn {
    padding: 0.25rem 0.5rem;
}
`;
document.head.appendChild(style); 