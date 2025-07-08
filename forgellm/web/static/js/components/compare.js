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
                    const sessionCard = document.querySelector(`#session-card-${sessionId}`);
                    if (sessionCard) {
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
    
    // Extract values directly from session name using regex
    const lrMatch = sessionName.match(/lr(\d+e?_?\d*)/i);
    const bsMatch = sessionName.match(/bs(\d+)/i);
    const seqMatch = sessionName.match(/seq(\d+)/i);
    
    // Determine training type based on folder structure
    let trainingType = '';
    let fineTuneType = '';
    
    if (session.log_file) {
        if (session.log_file.includes('/cpt/')) {
            trainingType = 'CPT (Continued Pre-training)';
        } else if (session.log_file.includes('/ift/')) {
            trainingType = 'IFT (Instruction Fine-tuning)';
        }
        
        if (session.log_file.includes('lora')) {
            fineTuneType = 'LoRA';
        } else if (session.log_file.includes('dora')) {
            fineTuneType = 'DoRA';
        } else {
            fineTuneType = 'Full';
        }
    }
    
    return {
        learningRate: lrMatch ? lrMatch[1].replace('_', '-') : '',
        batchSize: bsMatch ? bsMatch[1] : '',
        iterations: session.latest_iteration || '',
        sequenceLength: seqMatch ? seqMatch[1] : '',
        trainingType: trainingType,
        fineTuneType: fineTuneType
    };
}

// Format values to avoid showing empty strings
function formatValue(value) {
    return value === '' ? '-' : value;
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
            // The session ID is the full directory name
            const sessionId = session.session_id || session.id || '';
            
            // Clean up model name by removing "dataset_cpt_" prefix
            const cleanModelName = (session.model_name || 'Unknown').replace(/^dataset_cpt_/, '');
            
            // Use start_time instead of started_at
            const startDate = session.start_time ? new Date(session.start_time).toLocaleDateString() : 'Unknown';
            
            // Get training parameters for tooltip
            const params = getTrainingParameters(session);
            
            // Build a simple tooltip with parameters
            const tooltipContent = `Training Parameters:
• Type: ${formatValue(params.trainingType)}
• Fine-tune Type: ${formatValue(params.fineTuneType)}
• Learning Rate: ${formatValue(params.learningRate)}
• Batch Size: ${formatValue(params.batchSize)}
• Iterations: ${formatValue(params.iterations)}
• Sequence Length: ${formatValue(params.sequenceLength)}`;
            
            // Check if this session is selected
            const isSelected = selectedSessions.has(sessionId);
            const selectedClass = isSelected ? 'selected-session-card' : '';
            
            // ENABLE FUSE BUTTON FOR ALL MODELS - no detection logic needed
            // All models can be fused, regardless of type
            
            return `
                <div class="session-item mb-2">
                    <div class="session-card ${selectedClass}" 
                         id="session-card-${sessionId}" 
                         data-session-id="${sessionId}"
                         data-bs-toggle="tooltip" 
                         data-bs-placement="right" 
                         data-bs-html="false"
                         title="${tooltipContent}"
                         onclick="handleSessionChange('${sessionId}', !selectedSessions.has('${sessionId}'))">
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
                                <span class="detail-label">Type:</span>
                                <span class="detail-value">${formatValue(params.fineTuneType)}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Started:</span>
                                <span class="detail-value">${startDate}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">LR:</span>
                                <span class="detail-value">${formatValue(params.learningRate)}</span>
                            </div>
                        </div>
                        <div class="session-actions mt-2 pt-2 border-top">
                            <button class="btn btn-sm btn-outline-secondary view-params-btn" 
                                    onclick="showSessionParameters('${sessionId}'); event.preventDefault(); event.stopPropagation();"
                                    title="View Parameters">
                                <i class="fas fa-file-code"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-secondary fuse-adapter-btn" 
                                    onclick="fuseSessionAdapter('${sessionId}'); event.preventDefault(); event.stopPropagation();"
                                    title="Fuse this adapter with base model">
                                <i class="fas fa-layer-group"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-secondary test-session-btn" 
                                    onclick="testSessionInPlayground('${sessionId}'); event.preventDefault(); event.stopPropagation();"
                                    title="Test in Playground">
                                <i class="fas fa-vial"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        console.log(`Loaded ${sessions.length} sessions`);
        
        // Initialize Bootstrap tooltips
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(tooltip => {
            new bootstrap.Tooltip(tooltip, {
                html: false,
                placement: 'right',
                trigger: 'hover'
            });
        });
        
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
    // Check if dark mode is enabled
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark' || document.body.getAttribute('data-theme') === 'dark';
    const defaultBgColor = isDarkMode ? '#2d2d2d' : '#f8f9fa';

    // First update the visual state immediately for better UX
    const sessionCard = document.querySelector(`#session-card-${sessionId}`);
    if (sessionCard) {
        if (isSelected) {
            sessionCard.classList.add('selected-session-card');
            sessionCard.style.backgroundColor = 'rgba(13, 110, 253, 0.25)';
            sessionCard.style.borderLeftColor = '#0d6efd';
        } else {
            sessionCard.classList.remove('selected-session-card');
            sessionCard.style.backgroundColor = defaultBgColor;
            sessionCard.style.borderLeftColor = 'transparent';
        }
    }

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
    // Check if dark mode is enabled
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark' || document.body.getAttribute('data-theme') === 'dark';
    const defaultBgColor = isDarkMode ? '#2d2d2d' : '#f8f9fa';
    
    // Reset all session cards to their default, non-selected state
    document.querySelectorAll('.session-card').forEach(card => {
        card.classList.remove('selected-session-card');
        card.style.backgroundColor = defaultBgColor;
        card.style.borderLeftColor = 'transparent';
    });

    // Apply selected state to selected session cards
    for (const [sessionId, sessionData] of selectedSessions) {
        const sessionCard = document.querySelector(`#session-card-${sessionId}`);
        if (sessionCard) {
            sessionCard.classList.add('selected-session-card');
            // Apply inline styles as well for maximum compatibility
            sessionCard.style.backgroundColor = 'rgba(13, 110, 253, 0.25)';
            sessionCard.style.borderLeftColor = '#0d6efd';
        }
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
            // Get colors for sessions
            const colors = getSessionColors();
            let colorIndex = 0;
            
            // Assign colors to sessions
            for (const [sessionId, sessionData] of selectedSessions) {
                sessionData.color = colors[colorIndex % colors.length];
                colorIndex++;
            }
            
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
                xaxis: { title: 'Iterations', range: [0, null] },
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
    updateSelectionSummary();
    updateSessionColorsAndUI(); // Reset the UI for all cards
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
    
    // Add event listener for fuse tab shown to handle adapter selection from localStorage
    document.addEventListener('shown.bs.tab', function(event) {
        if (event.target.getAttribute('data-bs-target') === '#fuse') {
            setTimeout(() => {
                const storedAdapter = localStorage.getItem('forge-fuse-adapter');
                if (storedAdapter) {
                    console.log('Found stored adapter path:', storedAdapter);
                    const adapterSelect = document.getElementById('fuse-adapter-select');
                    if (adapterSelect) {
                        // Try to select the adapter
                        for (let i = 0; i < adapterSelect.options.length; i++) {
                            if (adapterSelect.options[i].value === storedAdapter) {
                                adapterSelect.selectedIndex = i;
                                adapterSelect.dispatchEvent(new Event('change'));
                                // Clear the storage after use
                                localStorage.removeItem('forge-fuse-adapter');
                                break;
                            }
                        }
                    }
                }
            }, 300);
        }
    });
});

// Function to ensure the fuse adapter dropdown is populated
async function ensureFuseAdapterDropdownPopulated() {
    const adapterSelect = document.getElementById('fuse-adapter-select');
    if (!adapterSelect) {
        console.error('Adapter select dropdown not found!');
        return false;
    }
    
    // If dropdown is empty or only has the placeholder option, populate it
    if (adapterSelect.options.length <= 1) {
        console.log('Dropdown is empty, populating it manually');
        
        try {
            // Get adapters directly from API
            const response = await fetch('/api/checkpoints');
            const data = await response.json();
            
            if (data.success && data.checkpoints && data.checkpoints.length > 0) {
                console.log(`Got ${data.checkpoints.length} adapters from API`);
                
                // Clear existing options except the first one
                const firstOption = adapterSelect.firstElementChild;
                adapterSelect.innerHTML = '';
                if (firstOption) {
                    adapterSelect.appendChild(firstOption);
                }
                
                // Add the adapters to the dropdown
                data.checkpoints.forEach(checkpoint => {
                    const option = document.createElement('option');
                    option.value = checkpoint.path;
                    
                    // Format the display name
                    const modelName = checkpoint.model || 'Unknown';
                    const iteration = checkpoint.iteration || 0;
                    const size = checkpoint.size ? `${checkpoint.size.toFixed(1)}MB` : '';
                    
                    option.textContent = `${modelName} - iter ${iteration} ${size}`;
                    adapterSelect.appendChild(option);
                });
                
                console.log(`Populated dropdown with ${adapterSelect.options.length} options`);
                return true;
            }
        } catch (error) {
            console.error('Error populating dropdown:', error);
        }
    }
    
    return adapterSelect.options.length > 1;
}

// Function to extract iteration number from option text or value
function extractIterationNumber(text, value) {
    // First try to extract from text which has format like "Model Name [CPT] - iter 300 504.1MB"
    const iterTextMatch = text.match(/iter\s+(\d+)/i);
    if (iterTextMatch && iterTextMatch[1]) {
        return parseInt(iterTextMatch[1], 10);
    }
    
    // If not found in text, try to extract from the value path
    // Format like "models/cpt/model_name_iter300_seq3072_date/000300_adapters.safetensors"
    const iterValueMatch = value.match(/iter(\d+)_|\/0*(\d+)_adapters\.safetensors$/);
    if (iterValueMatch) {
        return parseInt(iterValueMatch[1] || iterValueMatch[2], 10);
    }
    
    // Last attempt - look for any number in the path that might be an iteration
    const lastNumberMatch = value.match(/\/0*(\d+)_/);
    if (lastNumberMatch && lastNumberMatch[1]) {
        return parseInt(lastNumberMatch[1], 10);
    }
    
    return 0; // Default if no iteration number found
}

// Function to get the best checkpoint iteration (lowest val loss) for a session
async function getBestCheckpointIteration(sessionId) {
    try {
        // Get session data
        const sessionsResponse = await fetch('/api/training/sessions');
        const sessionsData = await sessionsResponse.json();
        const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
        
        if (!session || !session.log_file) {
            console.error('Session not found or missing log file');
            return null;
        }
        
        // Get validation loss data for this session
        const response = await fetch(`/api/dashboard/historical`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ log_file: session.log_file })
        });
        
        const sessionData = await response.json();
        
        if (sessionData.charts && sessionData.charts.loss && sessionData.charts.loss.data) {
            const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
            
            if (validationLoss && validationLoss.x && validationLoss.y) {
                console.log('Found validation loss data:', validationLoss);
                
                // Create pairs of [iteration, val_loss]
                const iterLossPairs = [];
                for (let i = 0; i < validationLoss.x.length; i++) {
                    // Only include non-null values
                    if (validationLoss.y[i] !== null && validationLoss.y[i] !== undefined) {
                        iterLossPairs.push({
                            iteration: validationLoss.x[i],
                            valLoss: validationLoss.y[i]
                        });
                    }
                }
                
                if (iterLossPairs.length > 0) {
                    // Sort by validation loss (ascending - lower is better)
                    iterLossPairs.sort((a, b) => a.valLoss - b.valLoss);
                    
                    // Get the iteration with lowest validation loss
                    const bestIteration = iterLossPairs[0].iteration;
                    console.log(`Best checkpoint is iteration ${bestIteration} with val loss ${iterLossPairs[0].valLoss}`);
                    return bestIteration;
                }
            }
        }
        
        console.warn('No validation loss data found');
        return null;
    } catch (error) {
        console.error('Error getting best checkpoint iteration:', error);
        return null;
    }
}

// Helper function to select the best checkpoint in a dropdown
async function selectBestCheckpoint(dropdownId, adapterInfo) {
    const adapterSelect = document.getElementById(dropdownId);
    if (!adapterSelect) {
        console.error(`Dropdown with ID ${dropdownId} not found!`);
        return false;
    }
    
    console.log(`Dropdown ${dropdownId} has ${adapterSelect.options.length} options`);
    
    // If we have a specific target iteration, try to find it
    if (adapterInfo.bestIteration) {
        console.log(`Looking for specific target iteration: ${adapterInfo.bestIteration}`);
        
        // Format the target iteration with leading zeros (both formats)
        const targetIterStr = adapterInfo.bestIteration.toString();
        const targetIterPadded = targetIterStr.padStart(6, '0');
        
        // Look for the exact iteration in the dropdown
        for (let i = 0; i < adapterSelect.options.length; i++) {
            const optionValue = adapterSelect.options[i].value;
            const optionText = adapterSelect.options[i].text;
            
            // Check for the target iteration in the option value
            if ((optionValue.includes(`/${targetIterPadded}_`) || 
                 optionValue.includes(`/${targetIterStr}_`) ||
                 optionValue.includes(`_iter${targetIterStr}_`)) && 
                optionValue.includes(adapterInfo.path)) {
                
                console.log(`Found target iteration ${adapterInfo.bestIteration} at index ${i}: ${optionValue}`);
                adapterSelect.selectedIndex = i;
                adapterSelect.dispatchEvent(new Event('change'));
                return true;
            }
            
            // Also check the option text for the iteration
            if (optionText.includes(`iter ${targetIterStr}`) && 
                optionValue.includes(adapterInfo.path)) {
                
                console.log(`Found target iteration ${adapterInfo.bestIteration} in text at index ${i}: ${optionText}`);
                adapterSelect.selectedIndex = i;
                adapterSelect.dispatchEvent(new Event('change'));
                return true;
            }
        }
        
        console.warn(`Could not find target iteration ${adapterInfo.bestIteration}, falling back to path matching`);
    }
    
    // Find all matching options for this adapter path
    const matchingOptions = [];
    for (let i = 0; i < adapterSelect.options.length; i++) {
        const optionValue = adapterSelect.options[i].value;
        const optionText = adapterSelect.options[i].text;
        
        // Check if the option value contains the adapter path
        if (optionValue.includes(adapterInfo.path)) {
            console.log(`Found matching option at index ${i}: ${optionValue}`);
            
            // Extract iteration number using improved function
            const iterNumber = extractIterationNumber(optionText, optionValue);
            console.log(`Extracted iteration number: ${iterNumber}`);
            
            matchingOptions.push({
                index: i,
                value: optionValue,
                text: optionText,
                iteration: iterNumber
            });
        }
    }
    
    // If we found matching options but couldn't find the exact target iteration,
    // try to find the closest one
    if (matchingOptions.length > 0 && adapterInfo.bestIteration) {
        console.log(`Looking for closest iteration to target: ${adapterInfo.bestIteration}`);
        
        // Find the option with iteration closest to the target
        let closestOption = matchingOptions[0];
        let minDiff = Math.abs(closestOption.iteration - adapterInfo.bestIteration);
        
        for (let i = 1; i < matchingOptions.length; i++) {
            const diff = Math.abs(matchingOptions[i].iteration - adapterInfo.bestIteration);
            if (diff < minDiff) {
                closestOption = matchingOptions[i];
                minDiff = diff;
            }
        }
        
        console.log(`Selecting closest iteration: ${closestOption.iteration} at index ${closestOption.index}`);
        adapterSelect.selectedIndex = closestOption.index;
        adapterSelect.dispatchEvent(new Event('change'));
        return true;
    } else if (matchingOptions.length > 0) {
        // If we don't have a target iteration, sort by iteration number (descending)
        matchingOptions.sort((a, b) => b.iteration - a.iteration);
        
        // Select the highest iteration
        const bestOption = matchingOptions[0];
        console.log(`No target iteration, selecting highest iteration ${bestOption.iteration} at index ${bestOption.index}`);
        adapterSelect.selectedIndex = bestOption.index;
        adapterSelect.dispatchEvent(new Event('change'));
        return true;
    }
    
    // If no direct matches, try more flexible matching
    console.log('No direct matches found, trying more flexible matching');
    
    // Extract model name from stored path
    const pathParts = adapterInfo.path.split('/');
    const modelFileName = pathParts[pathParts.length - 1];
    
    const flexibleMatches = [];
    for (let i = 0; i < adapterSelect.options.length; i++) {
        const optionValue = adapterSelect.options[i].value;
        const optionText = adapterSelect.options[i].text;
        
        // Check for partial matches
        if (optionValue.includes(modelFileName) || 
            optionText.includes(modelFileName) ||
            (adapterInfo.model && optionText.includes(adapterInfo.model))) {
            
            // Extract iteration number using improved function
            const iterNumber = extractIterationNumber(optionText, optionValue);
            console.log(`Extracted iteration number from flexible match: ${iterNumber}`);
            
            flexibleMatches.push({
                index: i,
                value: optionValue,
                text: optionText,
                iteration: iterNumber
            });
        }
    }
    
    // Select the best match if any were found
    if (flexibleMatches.length > 0) {
        // Sort by iteration number (descending) as fallback
        flexibleMatches.sort((a, b) => b.iteration - a.iteration);
        
        // Select the highest iteration
        const bestOption = flexibleMatches[0];
        console.log(`Selecting best flexible match at index ${bestOption.index} with iteration ${bestOption.iteration}`);
        adapterSelect.selectedIndex = bestOption.index;
        adapterSelect.dispatchEvent(new Event('change'));
        return true;
    }
    
    console.error('Could not find adapter in dropdown after multiple attempts');
    return false;
}

// Function to fuse a session adapter
async function fuseSessionAdapter(sessionId) {
    try {
        console.log('Fusing adapter for session ID:', sessionId);
        
        // Get session data
        const sessionsResponse = await fetch('/api/training/sessions');
        const sessionsData = await sessionsResponse.json();
        const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
        
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        
        // Get adapter path from log file path - use just the directory
        const logFilePath = session.log_file;
        const adapterPath = logFilePath.substring(0, logFilePath.lastIndexOf('/'));
        
        console.log('Adapter path:', adapterPath);
        
        // Get the best checkpoint iteration based on validation loss
        const bestIteration = await getBestCheckpointIteration(sessionId);
        console.log('Best checkpoint iteration:', bestIteration);
        
        // Store adapter info in localStorage for the fuse tab
        const adapterInfo = {
            path: adapterPath,
            model: session.model_name,
            session_name: session.session_name,
            bestIteration: bestIteration || null
        };
        
        localStorage.setItem('forge-fuse-adapter', JSON.stringify(adapterInfo));
        console.log('Stored adapter info in localStorage:', adapterInfo);
        
        // Switch to fuse tab
        const fuseTab = document.querySelector('[data-bs-target="#fuse"]');
        if (fuseTab) {
            console.log('Switching to Fuse tab');
            fuseTab.click();
            
            // Wait for tab to be shown before setting values
            setTimeout(async () => {
                // Try to select the best checkpoint in the adapter dropdown
                const storedAdapterJson = localStorage.getItem('forge-fuse-adapter');
                if (storedAdapterJson) {
                    try {
                        const adapterInfo = JSON.parse(storedAdapterJson);
                        await selectBestCheckpoint('fuse-adapter-select', adapterInfo);
                        // Clear the storage after attempting selection
                        localStorage.removeItem('forge-fuse-adapter');
                    } catch (error) {
                        console.error('Error parsing stored adapter info:', error);
                        localStorage.removeItem('forge-fuse-adapter');
                    }
                }
            }, 1000);
        } else {
            console.error('Fuse tab not found');
        }
    } catch (error) {
        console.error('Error fusing adapter:', error);
        alert('Failed to fuse adapter: ' + error.message);
    }
}

// Function to test a session in the playground tab
async function testSessionInPlayground(sessionId) {
    try {
        // Get session data
        const sessionsResponse = await fetch('/api/training/sessions');
        const sessionsData = await sessionsResponse.json();
        const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
        
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        
        // Get adapter path from log file path
        const logFilePath = session.log_file;
        const adapterPath = logFilePath.substring(0, logFilePath.lastIndexOf('/'));
        
        console.log('Testing adapter path:', adapterPath);
        
        // Get the best checkpoint iteration based on validation loss
        const bestIteration = await getBestCheckpointIteration(sessionId);
        console.log('Best checkpoint iteration:', bestIteration);
        
        // Store adapter info in localStorage for the testing tab
        const adapterInfo = {
            path: adapterPath,
            model: session.model_name,
            session_name: session.session_name,
            bestIteration: bestIteration || null
        };
        
        localStorage.setItem('forge-test-session', JSON.stringify(adapterInfo));
        console.log('Stored test session info:', adapterInfo);
        
        // Switch to testing tab
        const testingTab = document.querySelector('[data-bs-target="#testing"]');
        if (testingTab) {
            testingTab.click();
            
            // Wait for tab to be shown before setting values
            setTimeout(async () => {
                // Reset scroll position
                window.scrollTo(0, 0);
                
                // Try to select the best checkpoint in the adapter dropdown
                try {
                    const success = await selectBestCheckpoint('adapter-path', adapterInfo);
                    if (!success) {
                        // If no matching options found, add the adapter path as a new option
                        console.log('No matching options found, adding adapter path as new option');
                        const adapterSelect = document.getElementById('adapter-path');
                        if (adapterSelect) {
                            const option = document.createElement('option');
                            option.value = adapterPath;
                            option.text = adapterPath.split('/').pop();
                            adapterSelect.add(option);
                            adapterSelect.value = adapterPath;
                            adapterSelect.dispatchEvent(new Event('change'));
                        }
                    }
                } catch (error) {
                    console.error('Error selecting best checkpoint:', error);
                }
            }, 500);
        }
    } catch (error) {
        console.error('Error preparing test session:', error);
        alert('Failed to prepare test session: ' + error.message);
    }
}

// Function to show session parameters in a modal
async function showSessionParameters(sessionId) {
    try {
        console.log('Showing parameters for session ID:', sessionId);
        
        // Show loading indicator
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
        }
        
        // Find the log file for this session
        const sessionsResponse = await fetch('/api/training/sessions');
        const sessionsData = await sessionsResponse.json();
        
        // Find the matching session
        const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        
        const logFile = session.log_file;
        
        // Get the raw logs
        const response = await fetch('/api/logs/raw', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ log_file: logFile })
        });
        
        const sessionDetails = await response.json();
        
        // Parse the raw logs content
        let rawData;
        try {
            rawData = JSON.parse(sessionDetails.logs);
        } catch (e) {
            rawData = { raw_content: sessionDetails.logs };
        }
        
        // Create a simple metadata display
        const config = rawData.config || {};
        const modelName = rawData.base_model || config.model_name || 'N/A';
        const trainingType = rawData.training_type || config.training_type || 'N/A';
        const fineTuneType = config.fine_tune_type || 'Full';
        
        // Display the parameters in the modal
        const logsContent = document.getElementById('logs-content');
        if (logsContent) {
            logsContent.innerHTML = `
                <div class="alert alert-info mb-3">
                    <h6 class="mb-2"><i class="fas fa-info-circle me-2"></i>Training Parameters</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <small><strong>Model:</strong> ${modelName}</small><br>
                            <small><strong>Type:</strong> ${trainingType}</small><br>
                            <small><strong>Fine-tune Type:</strong> ${fineTuneType}</small><br>
                            <small><strong>Batch Size:</strong> ${config.batch_size || 'N/A'}</small><br>
                            <small><strong>Learning Rate:</strong> ${config.learning_rate || 'N/A'}</small>
                        </div>
                        <div class="col-md-6">
                            <small><strong>Max Iterations:</strong> ${config.max_iterations || 'N/A'}</small><br>
                            <small><strong>Sequence Length:</strong> ${config.max_seq_length || 'N/A'}</small><br>
                            <small><strong>Weight Decay:</strong> ${config.weight_decay || 'N/A'}</small><br>
                            <small><strong>LR Schedule:</strong> ${config.lr_schedule || 'N/A'}</small><br>
                            <small><strong>LR Decay Factor:</strong> ${config.lr_decay_factor || 'N/A'}</small>
                        </div>
                    </div>
                </div>
                <pre class="json-content bg-dark text-light p-3 rounded" style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; max-height: 60vh; overflow-y: auto; border: 1px solid #444;">
${JSON.stringify(rawData, null, 2)}
                </pre>
            `;
            
            // Update the modal title
            const modalTitle = document.querySelector('#logsModal .modal-title');
            if (modalTitle) {
                modalTitle.textContent = `Training Logs: ${sessionId}`;
            }
            
            // Show the modal
            const logsModal = new bootstrap.Modal(document.getElementById('logsModal'));
            logsModal.show();
        }
        
        // Hide loading indicator
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
        }
    } catch (error) {
        console.error('Error fetching session details:', error);
        alert('Failed to load session parameters: ' + error.message);
        
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
        }
    }
}

// Global tab change event listener
document.addEventListener('shown.bs.tab', function(event) {
    // If we're entering the compare tab
    if (event.target.getAttribute('data-bs-target') === '#compare') {
        console.log('Compare tab activated, loading sessions');
        setTimeout(() => {
            if (elementsExist()) {
                loadSessions();
            }
        }, 100);
    }
    
    // If we're entering the fuse tab
    if (event.target.getAttribute('data-bs-target') === '#fuse') {
        const storedAdapterJson = localStorage.getItem('forge-fuse-adapter');
        if (storedAdapterJson) {
            try {
                const adapterInfo = JSON.parse(storedAdapterJson);
                console.log('Found stored adapter info:', adapterInfo);
                
                // Use the helper function to select the best checkpoint
                setTimeout(async () => {
                    try {
                        const success = await selectBestCheckpoint('fuse-adapter-select', adapterInfo);
                        if (success) {
                            console.log('Successfully selected best checkpoint');
                        } else {
                            console.error('Failed to select best checkpoint');
                        }
                    } catch (error) {
                        console.error('Error selecting best checkpoint:', error);
                    }
                    
                    // Clear the storage after attempting selection
                    localStorage.removeItem('forge-fuse-adapter');
                }, 1000);
            } catch (error) {
                console.error('Error parsing stored adapter info:', error);
                localStorage.removeItem('forge-fuse-adapter');
            }
        }
    }
    
    // If we're entering the testing tab
    if (event.target.getAttribute('data-bs-target') === '#testing') {
        const storedTestSessionJson = localStorage.getItem('forge-test-session');
        if (storedTestSessionJson) {
            try {
                const adapterInfo = JSON.parse(storedTestSessionJson);
                console.log('Found stored test session info:', adapterInfo);
                
                // Use the helper function to select the best checkpoint
                setTimeout(async () => {
                    try {
                        const success = await selectBestCheckpoint('adapter-path', adapterInfo);
                        if (success) {
                            console.log('Successfully selected best checkpoint for testing');
                        } else {
                            console.error('Failed to select best checkpoint for testing');
                        }
                    } catch (error) {
                        console.error('Error selecting best checkpoint for testing:', error);
                    }
                    
                    // Clear the storage after attempting selection
                    localStorage.removeItem('forge-test-session');
                }, 1000);
            } catch (error) {
                console.error('Error parsing stored test session info:', error);
                localStorage.removeItem('forge-test-session');
            }
        }
    }
    
    // If we're leaving the compare tab, store the selections
    if (event.relatedTarget && event.relatedTarget.id === 'compare-tab') {
        storeSelectedSessions();
    }
});

console.log('Compare.js Plotly version loaded');

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
    transition: all 0.2s ease-in-out;
    border-left: 4px solid transparent;
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 8px;
    cursor: pointer;
}

/* Light mode */
:root:not([data-theme="dark"]) .session-card {
    background-color: #f8f9fa;
}

/* Dark mode - using proper dark theme background */
[data-theme="dark"] .session-card {
    background-color: #2d2d2d;
    border-color: #404040;
}

.session-card:hover {
    background-color: rgba(0, 123, 255, 0.1);
}

/* Selected state styling - IMPORTANT: Make this very visible with blue background */
.selected-session-card {
    background-color: rgba(13, 110, 253, 0.25) !important;
    border-left-color: #0d6efd !important;
    box-shadow: 0 0 0 1px rgba(13, 110, 253, 0.25) !important;
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
[data-theme="dark"] .session-card:hover {
    background-color: #333333;
}

[data-theme="dark"] .selected-session-card {
    background-color: rgba(13, 110, 253, 0.25) !important;
    border-left-color: #0d6efd !important;
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