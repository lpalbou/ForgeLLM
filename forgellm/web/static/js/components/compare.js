// COMPARE TAB - USING PLOTLY (SAME AS MONITORING TAB)
let selectedSessions = new Map();
let hoveredSessionId = null; // Track which session is currently being hovered

// =============================================================================
// SHARED SESSION CARD COMPONENT
// =============================================================================

/**
 * Creates a professional session card that can be used across different tabs
 * @param {Object} session - Session data object
 * @param {Object} options - Configuration options for the card
 * @returns {string} HTML string for the session card
 */
function createSessionCard(session, options = {}) {
    const {
        showSelection = false,        // Show selection checkbox/click behavior
        showImportantNote = false,    // Show the "Important: only compare..." note
        actions = [],                // Array of action button configs
        context = 'default'          // Context for styling/behavior
    } = options;
    
    const sessionId = session.session_id || session.id || '';
    const escapedSessionId = escapeSelector(sessionId);
    
    // Clean up model name by removing "dataset_cpt_" prefix
    const cleanModelName = (session.model_name || 'Unknown').replace(/^dataset_cpt_/, '');
    
    // Use start_time with time included
    const startDateTime = session.start_time ? new Date(session.start_time) : null;
    const startDate = startDateTime ? startDateTime.toLocaleDateString() : 'Unknown';
    const startTime = startDateTime ? startDateTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '';
    
    // Get training parameters
    const params = getTrainingParameters(session);
    
    // Check if this session is selected (for compare context)
    const isSelected = showSelection && selectedSessions.has(sessionId);
    const selectedClass = isSelected ? 'selected-session-card' : '';
    
    // Create badge combinations for training type and sequence length
    const trainingBadges = [];
    if (params.fineTuneType && params.fineTuneType !== '-') {
        const bgColor = params.fineTuneType === 'Full' ? 'bg-primary' : 
                       params.fineTuneType === 'LoRA' ? 'bg-success' : 'bg-info';
        trainingBadges.push(`<span class="badge ${bgColor}">${params.fineTuneType}</span>`);
    }
    if (params.trainingTypeShort && params.trainingTypeShort !== '-') {
        trainingBadges.push(`<span class="badge bg-warning text-dark">${params.trainingTypeShort}</span>`);
    }
    
    // Add sequence length badge
    const seqLength = params.maxSeqLength || params.sequenceLength;
    if (seqLength && seqLength !== '-' && seqLength !== '') {
        trainingBadges.push(`<span class="badge bg-secondary">${seqLength}</span>`);
    }
    
    const trainingBadgeHtml = trainingBadges.join(' ');

    // Create loss badges (Train Loss and Val Loss) - these replace the iteration count
    const lossBadges = [];
    
    // Try to get loss values from session data
    // API returns: latest_loss (train), latest_val_loss (val)
    const trainLoss = session.train_loss || session.latest_train_loss || session.last_train_loss || session.latest_loss;
    const valLoss = session.val_loss || session.latest_val_loss || session.last_val_loss || session.validation_loss;
    
    if (trainLoss && trainLoss !== 'N/A' && trainLoss !== '-') {
        const lossValue = parseFloat(trainLoss);
        if (!isNaN(lossValue)) {
            lossBadges.push(`<span class="badge bg-warning text-dark" title="Best Training Loss (at lowest val loss iteration)">T: ${lossValue.toFixed(3)}</span>`);
        }
    }
    
    if (valLoss && valLoss !== 'N/A' && valLoss !== '-') {
        const lossValue = parseFloat(valLoss);
        if (!isNaN(lossValue)) {
            lossBadges.push(`<span class="badge bg-success" title="Best Validation Loss (lowest across all iterations)">V: ${lossValue.toFixed(3)}</span>`);
        }
    }
    
    // Fallback to iteration count if no loss data available
    if (lossBadges.length === 0) {
        const iterations = session.latest_iteration || session.iterations || session.current_iteration;
        if (iterations && iterations !== 'N/A') {
            lossBadges.push(`<span class="badge bg-secondary">${iterations}</span>`);
        }
    }
    
    const lossBadgeHtml = lossBadges.join(' ');

    // Create comprehensive learning rate display with LR decay and weight decay
    let lrDisplay = formatScientificNotation(params.learningRate);
    let additionalParams = [];
    
    // Add LR decay factor if available
    if (params.lrDecayFactor && params.lrDecayFactor !== '') {
        additionalParams.push(`LDR ${params.lrDecayFactor}`);
    }
    
    // Add weight decay if available
    if (params.weightDecay && params.weightDecay !== '') {
        additionalParams.push(`WD ${params.weightDecay}`);
    }
    
    // Combine everything
    if (additionalParams.length > 0) {
        lrDisplay = `${lrDisplay} | ${additionalParams.join(' | ')}`;
    }

    // Generate action buttons
    const actionButtonsHtml = actions.map(action => {
        const buttonClass = action.style === 'danger' ? 'btn-outline-danger' : 'btn-outline-secondary';
        const extraClass = action.rightAlign ? 'ms-auto' : '';
        
        // Handle custom handlers (for methods that need different parameters)
        let clickHandler;
        if (action.customHandler) {
            clickHandler = `${action.onclick}(); event.preventDefault(); event.stopPropagation();`;
        } else {
            clickHandler = `${action.onclick}('${sessionId.replace(/'/g, "\\'")}'); event.preventDefault(); event.stopPropagation();`;
        }
        
        return `
            <button class="btn btn-xs ${buttonClass} ${extraClass}" 
                    onclick="${clickHandler}"
                    title="${action.title}">
                <i class="${action.icon}"></i>
            </button>
        `;
    }).join('');

    // Create click handler for selection (compare context)
    const clickHandler = showSelection ? 
        `onclick="handleSessionChange('${sessionId.replace(/'/g, "\\'")}', !selectedSessions.has('${sessionId.replace(/'/g, "\\'")}'))"` : '';

    return `
        <div class="session-item mb-2">
            <div class="session-card ${selectedClass}" 
                 id="session-card-${escapedSessionId}" 
                 data-session-id="${sessionId}"
                 ${clickHandler}>
                
                <!-- Header with model name and loss badges -->
                <div class="session-header">
                    <div class="session-name" title="${cleanModelName}">${cleanModelName}</div>
                    <div class="loss-badges">
                        ${lossBadgeHtml}
                    </div>
                </div>
                
                <!-- Training type badges row -->
                <div class="training-badges mb-2">
                    ${trainingBadgeHtml}
                </div>
                
                <!-- Compact info row -->
                <div class="session-compact-info">
                    <div class="info-item">
                        <i class="fas fa-calendar-alt text-muted"></i>
                        <span class="info-text">${startDate}${startTime ? ` ${startTime}` : ''}</span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-chart-line text-muted"></i>
                        <span class="info-text">LR: ${lrDisplay}</span>
                    </div>
                </div>
                
                <!-- Action buttons -->
                ${actionButtonsHtml ? `
                <div class="session-actions">
                    ${actionButtonsHtml}
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Make the session card component globally available
window.createSessionCard = createSessionCard;

// NEW: Centralized Session Data Manager
class SessionDataManager {
    constructor() {
        this.sessionsCache = new Map(); // Cache for full session data
        this.configCache = new Map();   // Cache for parsed config data
    }

    // Load and cache session data
    async loadSessionData(sessionId) {
        if (this.sessionsCache.has(sessionId)) {
            return this.sessionsCache.get(sessionId);
        }

        try {
            // Get basic session info
            const sessionsResponse = await fetch('/api/training/sessions');
            const sessionsData = await sessionsResponse.json();
            const session = sessionsData.training_sessions.find(s => s.session_id === sessionId);
            
            if (!session) throw new Error(`Session ${sessionId} not found`);

            // Get detailed session data (charts, logs, config)
            const response = await fetch(`/api/dashboard/historical`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: session.log_file })
            });
            const sessionData = await response.json();

            // Get parsed config data
            const configData = await this.loadSessionConfig(session);

            // Combine all data
            const fullSessionData = {
                ...sessionData,
                ...session,
                sessionId: sessionId,
                config: configData,
                raw_session: session
            };

            this.sessionsCache.set(sessionId, fullSessionData);
            return fullSessionData;

        } catch (error) {
            console.error(`Error loading session ${sessionId}:`, error);
            return null;
        }
    }

    // Load and parse session configuration
    async loadSessionConfig(session) {
        if (this.configCache.has(session.session_id)) {
            return this.configCache.get(session.session_id);
        }

        try {
            const response = await fetch('/api/logs/raw', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: session.log_file })
            });

            if (response.ok) {
                const sessionDetails = await response.json();
                const rawData = JSON.parse(sessionDetails.logs);
                const config = rawData.config || {};
                
                this.configCache.set(session.session_id, config);
                return config;
            }
        } catch (error) {
            console.log('Could not parse session config for', session.session_name, ':', error);
        }

        return {};
    }

    // Extract training parameters from session data
    extractTrainingParameters(sessionData) {
        const config = sessionData.config || {};
        const session = sessionData.raw_session || sessionData;

        return {
            max_seq_length: config.max_seq_length || config.max_sequence_length || 'N/A',
            max_iterations: config.max_iterations || config.iters || session.latest_iteration || 'N/A',
            learning_rate: config.learning_rate || config.lr || 'N/A',
            lr_decay_factor: config.lr_decay_factor || 'N/A',
            weight_decay: config.weight_decay || 'N/A',
            batch_size: config.batch_size || 'N/A',
            warmup_steps: config.warmup_steps || 'N/A',
            fine_tune_type: config.fine_tune_type || 'Full'
        };
    }

    // Extract LoRA/DoRA parameters
    extractLoRAParameters(sessionData) {
        const config = sessionData.config || {};
        const params = this.extractTrainingParameters(sessionData);
        
        const isLoRATraining = params.fine_tune_type && (
            params.fine_tune_type.toLowerCase().includes('lora') || 
            params.fine_tune_type.toLowerCase().includes('dora')
        );

        if (isLoRATraining) {
            return {
                rank: config.lora_rank || config.rank || 'N/A',
                scale: config.lora_scale || config.scale || 'N/A',
                dropout: config.lora_dropout || config.dropout || 'N/A',
                layers: this.formatLayersCount(config.num_layers || config.layers),
                target_modules: this.formatTargetModules(config.lora_modules || config.target_modules)
            };
        }

        return {
            rank: '-',
            scale: '-',
            dropout: '-',
            layers: '-',
            target_modules: '-'
        };
    }

    formatTargetModules(modules) {
        if (!modules) return 'N/A';
        if (typeof modules === 'string') return modules;
        if (Array.isArray(modules)) return modules.join(', ');
        return 'Custom';
    }

    formatLayersCount(layers) {
        if (layers === undefined || layers === null) return 'N/A';
        if (layers === -1) return 'All';
        return layers.toString();
    }

    // Get best validation loss and checkpoint info
    getBestValidationLoss(sessionData) {
        if (sessionData.charts?.loss?.data) {
            const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
            if (validationLoss?.y && validationLoss.y.length > 0) {
                const minLoss = Math.min(...validationLoss.y);
                const minIndex = validationLoss.y.indexOf(minLoss);
                const iteration = validationLoss.x ? validationLoss.x[minIndex] : minIndex;
                return { loss: minLoss, iteration };
            }
        }
        return { loss: Infinity, iteration: 'N/A' };
    }

    getBestCheckpointInfo(sessionData) {
        const bestLoss = this.getBestValidationLoss(sessionData);
        if (bestLoss.loss === Infinity) {
            return { checkpoint: 'N/A', iteration: 'N/A' };
        }
        return {
            checkpoint: String(bestLoss.iteration).padStart(6, '0'), // Just the iteration number with padding
            iteration: bestLoss.iteration
        };
    }
}

// Global instance
const sessionDataManager = new SessionDataManager();

// Add this function to escape special characters in CSS selectors
function escapeSelector(selector) {
    // Escape special characters in CSS selectors
    return selector.replace(/[ !"#$%&'()*+,./:;<=>?@[\\\]^`{|}~]/g, '\\$&');
}

// Simple function to check if elements exist
function elementsExist() {
    const required = ['comparison-placeholder', 'comparison-charts-grid', 'compare-sessions-list'];
    return required.every(id => document.getElementById(id) !== null);
}

// Function to store selected sessions in localStorage
function storeSelectedSessions() {
    const sessionsArray = Array.from(selectedSessions.keys());
    localStorage.setItem('compareSelectedSessions', JSON.stringify(sessionsArray));
}

// Function to restore selected sessions from localStorage
async function restoreSelectedSessions() {
    try {
        const storedSessions = localStorage.getItem('compareSelectedSessions');
        if (storedSessions) {
            const sessionsArray = JSON.parse(storedSessions);
            console.log('Restoring selected sessions:', sessionsArray);
            
            // Clear current selections first
            selectedSessions.clear();
            
            // Then check each stored session ID and select it if available
            for (const sessionId of sessionsArray) {
                console.log(`Trying to restore session: ${sessionId}`);
                const escapedSessionId = escapeSelector(sessionId);
                const sessionCard = document.querySelector(`#session-card-${escapedSessionId}`);
                if (sessionCard) {
                    console.log(`Found session card for ${sessionId}, selecting it`);
                    await handleSessionChange(sessionId, true);
                } else {
                    console.warn(`Session card for ${sessionId} not found during restore`);
                }
            }
            
            // Update UI after all sessions are restored
            updateSessionColorsAndUI();
            
            // Generate comparison if we have at least 1 session
            if (selectedSessions.size >= 1) {
                generateComparison();
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
async function getTrainingParametersAsync(session) {
    // If we need to fetch config data from the session logs
    let sessionConfigData = null;
    
    try {
        if (session.log_file && (!session.fine_tune_type && !(session.config && session.config.fine_tune_type))) {
            // Try to get the actual training configuration
            const response = await fetch('/api/logs/raw', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_file: session.log_file })
            });
            
            if (response.ok) {
                const sessionDetails = await response.json();
                try {
                    const rawData = JSON.parse(sessionDetails.logs);
                    sessionConfigData = rawData.config || {};
                } catch (e) {
                    console.log('Could not parse session config for', session.session_name);
                }
            }
        }
    } catch (error) {
        console.log('Error fetching session config for', session.session_name, ':', error);
    }
    
    return getTrainingParameters(session, sessionConfigData);
}

// Synchronous version for immediate use
function getTrainingParameters(session, extraConfig = null) {
    const sessionName = session.session_name || '';
    
    // Extract values directly from session name using regex
    const lrMatch = sessionName.match(/lr(\d+e?_?\d*)/i);
    const bsMatch = sessionName.match(/bs(\d+)/i);
    const seqMatch = sessionName.match(/seq(\d+)/i);
    const decayMatch = sessionName.match(/decay([0-9.]+)/i);
    
    // Try to extract weight decay, LR decay factor, and max sequence length from session config or extraConfig
    let weightDecay = '';
    let lrDecayFactor = '';
    let maxSeqLength = '';
    
    // Check session config first
    if (session.config) {
        weightDecay = session.config.weight_decay || '';
        lrDecayFactor = session.config.lr_decay_factor || '';
        maxSeqLength = session.config.max_seq_length || '';
    }
    
    // Check extraConfig if available
    if (extraConfig) {
        weightDecay = weightDecay || extraConfig.weight_decay || '';
        lrDecayFactor = lrDecayFactor || extraConfig.lr_decay_factor || '';
        maxSeqLength = maxSeqLength || extraConfig.max_seq_length || '';
    }
    
    // Check if session already has fine_tune_type information
    let fineTuneTypeFromSession = '';
    if (session.fine_tune_type) {
        fineTuneTypeFromSession = session.fine_tune_type;
    } else if (session.config && session.config.fine_tune_type) {
        fineTuneTypeFromSession = session.config.fine_tune_type;
    } else if (extraConfig && extraConfig.fine_tune_type) {
        fineTuneTypeFromSession = extraConfig.fine_tune_type;
    }
    
    // Determine training type based on folder structure
    let trainingType = '';
    let fineTuneType = '';
    let trainingTypeShort = '';
    
    if (session.log_file) {
        if (session.log_file.includes('/cpt/')) {
            trainingType = 'CPT (Continued Pre-training)';
            trainingTypeShort = 'CPT';
        } else if (session.log_file.includes('/ift/')) {
            trainingType = 'IFT (Instruction Fine-tuning)';
            trainingTypeShort = 'SFT'; // More commonly known as SFT (Supervised Fine-tuning)
        }
        
        // Enhanced detection logic for fine-tune type
        // 1. First priority: Use fine_tune_type from session data if available
        if (fineTuneTypeFromSession) {
            fineTuneType = fineTuneTypeFromSession;
        } else {
            // 2. Check session name and log file for LoRA/DoRA indicators
            const sessionNameLower = sessionName.toLowerCase();
            const logFileLower = session.log_file ? session.log_file.toLowerCase() : '';
            
            // Check for LoRA indicators
            if (sessionNameLower.includes('lora') || 
                sessionNameLower.includes('_lora_') ||
                logFileLower.includes('lora') ||
                logFileLower.includes('adapters.safetensors') ||
                logFileLower.includes('_adapters.')) {
            fineTuneType = 'LoRA';
            } 
            // Check for DoRA indicators
            else if (sessionNameLower.includes('dora') || 
                     sessionNameLower.includes('_dora_') ||
                     logFileLower.includes('dora')) {
            fineTuneType = 'DoRA';
            } 
            // Check for explicit Full indicators
            else if (sessionNameLower.includes('full') || 
                     sessionNameLower.includes('_full_') ||
                     logFileLower.includes('full')) {
            fineTuneType = 'Full';
            } else {
                // 3. Smart detection based on session patterns
                // Check if this looks like LoRA training based on multiple indicators
                const hasLoRAIndicators = (
                    // Check if folder structure suggests LoRA (most CPT sessions are LoRA)
                    (session.log_file && session.log_file.includes('/cpt/')) ||
                    // Check if session has typical LoRA naming patterns
                    sessionName.includes('iter') || 
                    sessionName.includes('lr') ||
                    sessionName.includes('bs') ||
                    // Check latest iteration count (LoRA typically has more iterations)
                    (session.latest_iteration && session.latest_iteration > 100)
                );
                
                if (hasLoRAIndicators) {
                    fineTuneType = 'LoRA'; // Likely LoRA training
                } else {
                    fineTuneType = 'Full'; // Default for other types
                }
            }
        }
    }
    
    // Debug logging to help identify the issue
    console.log(`Session Debug - Name: "${sessionName}", Log: "${session.log_file}", FromSession: "${fineTuneTypeFromSession}", Detected: "${fineTuneType}"`);
    
    return {
        learningRate: lrMatch ? lrMatch[1].replace('_', '-') : '',
        learningRateDecay: decayMatch ? decayMatch[1] : '',
        lrDecayFactor: lrDecayFactor,
        weightDecay: weightDecay,
        batchSize: bsMatch ? bsMatch[1] : '',
        iterations: session.latest_iteration || '',
        sequenceLength: seqMatch ? seqMatch[1] : '',
        maxSeqLength: maxSeqLength,
        trainingType: trainingType,
        trainingTypeShort: trainingTypeShort,
        fineTuneType: fineTuneType
    };
}

// Format values to avoid showing empty strings
function formatValue(value) {
    return value === '' ? '-' : value;
}

function formatScientificNotation(value) {
    if (value === undefined || value === null || value === '' || value === 'N/A') return value;
    
    const num = parseFloat(value);
    if (isNaN(num)) return value;
    
    // Only convert to scientific notation for small numbers (< 0.01)
    if (num < 0.01 && num > 0) {
        return num.toExponential(2); // 2 decimal places in scientific notation
    }
    
    return value; // Return original for normal numbers
}

// Load and display sessions
async function loadSessions() {
    try {
        const response = await fetch('/api/training/sessions');
        const data = await response.json();
        console.log('Sessions API response:', data);
        
        // Handle different API response formats
        let sessions = data.training_sessions || data.sessions || data || [];
        
        // Debug: log first session structure if available
        if (sessions.length > 0) {
            console.log('Sample session structure:', sessions[0]);
        }
        
        const container = document.getElementById('compare-sessions-list');
        if (!container) return;
        
        if (!Array.isArray(sessions)) {
            container.innerHTML = '<div class="text-muted">No sessions found</div>';
            return;
        }
        
        // Sort sessions by model name and size
        sessions = sortSessions(sessions);
        
        // Create session cards using the shared component with Compare tab configuration
        container.innerHTML = sessions.map(session => {
            // Define actions for Compare tab
            const actions = [
                { icon: 'fas fa-file-code', onclick: 'showSessionParameters', title: 'View Parameters' },
                { icon: 'fas fa-layer-group', onclick: 'fuseSessionAdapter', title: 'Fuse Adapter' },
                { icon: 'fas fa-flask', onclick: 'testSessionInPlayground', title: 'Test in Playground' },
                { icon: 'fas fa-folder-open', onclick: 'showSessionFolder', title: 'View Folder', rightAlign: true },
                { icon: 'fas fa-trash', onclick: 'deleteSession', title: 'Delete Session', style: 'danger' }
            ];
            
            return createSessionCard(session, {
                showSelection: true,
                actions: actions,
                context: 'compare'
            });
        }).join('');
        
        console.log(`Loaded ${sessions.length} sessions`);
        
        // Add hover event listeners to session cards for chart highlighting
        sessions.forEach(session => {
            const sessionCard = document.querySelector(`#session-card-${escapeSelector(session.session_id)}`);
            if (sessionCard) {
                sessionCard.addEventListener('mouseenter', () => handleSessionHover(session.session_id));
                sessionCard.addEventListener('mouseleave', () => handleSessionHoverEnd());
            }
        });
        
        // Background task: Fetch actual fine_tune_types for more accurate badges
        setTimeout(() => {
            updateSessionBadgesWithActualData(sessions);
        }, 1000);
        
        // Tooltips removed - all information is now displayed directly in badges and info lines
        
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
                    initializeSessionSearch(); // Initialize search functionality
                }, 100); // Short delay to ensure DOM is ready
            }
        });
        
        // Restore selections when the page loads
        if (document.querySelector('#compare-tab.active')) {
            setTimeout(() => {
                restoreSelectedSessions();
                initializeSessionSearch(); // Initialize search functionality
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
    // Check if dark mode is enabled (theme is only set on body)
    const isDarkMode = document.body.getAttribute('data-theme') === 'dark';
    const defaultBgColor = isDarkMode ? '#2d2d2d' : '#f8f9fa';

    console.log(`Handling session change for ${sessionId}, isSelected: ${isSelected}`);

    // First update the visual state immediately for better UX
    const sessionCard = document.querySelector(`#session-card-${escapeSelector(sessionId)}`);
    if (sessionCard) {
        if (isSelected) {
            // Apply all selected styles directly
            sessionCard.classList.add('selected-session-card');
            sessionCard.style.backgroundColor = isDarkMode ? 'rgba(13, 110, 253, 0.35)' : 'rgba(13, 110, 253, 0.25)';
            sessionCard.style.borderLeftColor = '#0d6efd';
            sessionCard.style.borderLeftWidth = '4px';
            sessionCard.style.borderLeftStyle = 'solid';
            sessionCard.style.border = isDarkMode ? '2px solid rgba(13, 110, 253, 0.6)' : '2px solid rgba(13, 110, 253, 0.5)';
            sessionCard.style.borderRadius = '8px';
            sessionCard.style.boxShadow = isDarkMode ? 
                '0 0 0 2px rgba(13, 110, 253, 0.4)' : 
                '0 0 0 2px rgba(13, 110, 253, 0.3)';
            sessionCard.style.position = 'relative';
            sessionCard.style.zIndex = '1';
        } else {
            // Remove all selected styles
            sessionCard.classList.remove('selected-session-card');
            sessionCard.style.backgroundColor = defaultBgColor;
            sessionCard.style.borderLeftColor = 'transparent';
            sessionCard.style.borderLeftWidth = '4px';
            sessionCard.style.border = `1px solid ${isDarkMode ? '#404040' : '#e9ecef'}`;
            sessionCard.style.borderRadius = '8px';
            sessionCard.style.boxShadow = 'none';
            sessionCard.style.position = '';
            sessionCard.style.zIndex = '';
        }
    } else {
        console.warn(`Session card for ${sessionId} not found in DOM`);
    }

    if (isSelected) {
        try {
            // Use the new SessionDataManager to load all session data
            const sessionData = await sessionDataManager.loadSessionData(sessionId);
            if (!sessionData) {
                throw new Error('Failed to load session data');
            }
            
            selectedSessions.set(sessionId, sessionData);
            console.log(`Added session ${sessionId} to selectedSessions map with full data`);
        } catch (error) {
            console.error(`Error loading session ${sessionId}:`, error);
            // Remove the checkbox reference that doesn't exist
            // Instead, just update the UI to reflect that selection failed
            if (sessionCard) {
                sessionCard.classList.remove('selected-session-card');
                sessionCard.style.backgroundColor = defaultBgColor;
                sessionCard.style.borderLeftColor = 'transparent';
                sessionCard.style.boxShadow = 'none';
                sessionCard.style.position = '';
                sessionCard.style.zIndex = '';
            }
            return;
        }
    } else {
        selectedSessions.delete(sessionId);
        console.log(`Removed session ${sessionId} from selectedSessions map`);
    }
    
    updateSelectionSummary();
    updateSessionColorsAndUI(); // Centralized function to update colors and UI

    if (selectedSessions.size >= 1) {
        generateComparison();
    } else {
        hideComparison();
    }

    // Store the updated selections
    storeSelectedSessions();
    
    // Debug: log the current selected sessions
    console.log('Current selected sessions:');
    for (const [id, data] of selectedSessions.entries()) {
        console.log(`- ${id}: ${data.session_name}`);
    }
}

function updateSessionColorsAndUI() {
    // Check if dark mode is enabled (theme is only set on body)
    const isDarkMode = document.body.getAttribute('data-theme') === 'dark';
    const defaultBgColor = isDarkMode ? '#2d2d2d' : '#f8f9fa';
    
    console.log('Updating session colors and UI');
    console.log(`Dark mode: ${isDarkMode}, Default bg color: ${defaultBgColor}`);
    console.log(`Selected sessions count: ${selectedSessions.size}`);
    
    // Reset all session cards to their default, non-selected state
    document.querySelectorAll('.session-card').forEach(card => {
        card.classList.remove('selected-session-card');
        card.style.backgroundColor = defaultBgColor;
        card.style.borderLeftColor = 'transparent';
        card.style.border = `1px solid ${isDarkMode ? '#404040' : '#e9ecef'}`;
        card.style.borderRadius = '8px';
        card.style.boxShadow = 'none';
    });

    // Apply selected state to selected session cards
    for (const [sessionId, sessionData] of selectedSessions) {
        console.log(`Applying selected state to session: ${sessionId}`);
        const sessionCard = document.querySelector(`#session-card-${escapeSelector(sessionId)}`);
        if (sessionCard) {
            console.log(`Found session card for ${sessionId}`);
            sessionCard.classList.add('selected-session-card');
            // Apply inline styles as well for maximum compatibility
            sessionCard.style.backgroundColor = isDarkMode ? 'rgba(13, 110, 253, 0.35)' : 'rgba(13, 110, 253, 0.25)';
            sessionCard.style.borderLeftColor = '#0d6efd';
            sessionCard.style.borderLeftWidth = '4px';
            sessionCard.style.borderLeftStyle = 'solid';
            sessionCard.style.border = isDarkMode ? '2px solid rgba(13, 110, 253, 0.6)' : '2px solid rgba(13, 110, 253, 0.5)';
            sessionCard.style.borderRadius = '8px';
            sessionCard.style.boxShadow = isDarkMode ? 
                '0 0 0 2px rgba(13, 110, 253, 0.4)' : 
                '0 0 0 2px rgba(13, 110, 253, 0.3)';
            sessionCard.style.position = 'relative';
            sessionCard.style.zIndex = '1';
        } else {
            console.warn(`Session card for ${sessionId} not found when applying selected state`);
        }
    }
}

function updateSelectionSummary() {
    // Only manage the summary table now (Selected Sessions panel has been removed)
    if (selectedSessions.size > 0) {
        generateSummaryTable();
    } else {
        hideSummaryTable();
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

    const isDarkMode = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDarkMode ? '#F5F5F5' : '#333333';
    const borderColor = isDarkMode ? '#555555' : '#DDDDDD';

    // Apply hover highlighting to traces if a session is being hovered
    if (hoveredSessionId) {
        traces = traces.map(trace => {
            // Find the session that matches this trace
            const hoveredSession = selectedSessions.get(hoveredSessionId);
            const isHovered = hoveredSession && trace.name === hoveredSession.session_name;
            
            if (isHovered) {
                // Highlight the hovered trace
                return {
                    ...trace,
                    line: {
                        ...trace.line,
                        width: 4, // Thicker line
                        color: trace.line.color // Keep the same color but thicker
                    },
                    opacity: 1.0 // Full opacity
                };
            } else {
                // Dim other traces
                return {
                    ...trace,
                    line: {
                        ...trace.line,
                        width: 1.5 // Thinner line
                    },
                    opacity: 0.4 // Reduced opacity
                };
            }
        });
    }

    // Debug the traces for stability chart
    if (containerId === 'stability-comparison-chart') {
        console.log('Stability Chart Traces:', JSON.stringify(traces));
        console.log('X-axis values:', traces.flatMap(t => t.x));
    }

    // Ensure all charts have x-axis starting at 0 for consistency
    if (!layoutOptions.xaxis) layoutOptions.xaxis = {};
    
    // Force x-axis to start at 0 and be linear
    layoutOptions.xaxis.range = [0, layoutOptions.xaxis.range?.[1] || null];
    layoutOptions.xaxis.type = 'linear';
    layoutOptions.xaxis.fixedrange = true; // Prevent user from zooming/panning
    
    console.log(`Chart ${containerId} x-axis range:`, layoutOptions.xaxis.range);

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
        annotations: (layoutOptions.annotations || []).map(annotation => ({
            ...annotation,
            font: { ...annotation.font, color: annotation.font?.color || textColor }
        }))
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
            let maxLossIteration = 0; // Track maximum iteration
            
            for (const [sessionId, sessionData] of selectedSessions) {
                if (sessionData.charts?.loss?.data) {
                    const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
                    if (validationLoss?.x && validationLoss?.y) {
                        // Track maximum iteration
                        if (validationLoss.x.length > 0) {
                            maxLossIteration = Math.max(maxLossIteration, Math.max(...validationLoss.x));
                        }
                        
                        lossTraces.push({
                            x: validationLoss.x, y: validationLoss.y, type: 'scatter', mode: 'lines',
                            name: sessionData.session_name, // Name for hover data
                            line: { color: sessionData.color, width: 2 } // Use assigned color
                        });
                    }
                }
            }
            renderComparisonChart('loss-comparison-chart', lossTraces, {
                title: selectedSessions.size === 1 ? 'Validation Loss' : 'Validation Loss Comparison',
                xaxis: { 
                    title: 'Iterations',
                    range: [0, maxLossIteration > 0 ? maxLossIteration : null]
                },
                yaxis: { title: 'Validation Loss' }
            });

            // --- 2. Perplexity Comparison (VALIDATION) ---
            const perplexityTraces = [];
            let maxPerplexityIteration = 0; // Track maximum iteration
            
            for (const [sessionId, sessionData] of selectedSessions) {
                 if (sessionData.charts?.perplexity?.data) {
                    const validationPerplexity = sessionData.charts.perplexity.data.find(c => c.name === 'Validation Perplexity');
                    if (validationPerplexity?.x && validationPerplexity?.y) {
                        // Track maximum iteration
                        if (validationPerplexity.x.length > 0) {
                            maxPerplexityIteration = Math.max(maxPerplexityIteration, Math.max(...validationPerplexity.x));
                        }
                        
                        perplexityTraces.push({
                            x: validationPerplexity.x, y: validationPerplexity.y, type: 'scatter', mode: 'lines',
                            name: sessionData.session_name,
                            line: { color: sessionData.color, width: 2 } // Use assigned color
                        });
                    }
                }
            }
            renderComparisonChart('perplexity-comparison-chart', perplexityTraces, {
                title: selectedSessions.size === 1 ? 'Validation Perplexity' : 'Validation Perplexity Comparison',
                xaxis: { 
                    title: 'Iterations',
                    range: [0, maxPerplexityIteration > 0 ? maxPerplexityIteration : null]
                },
                yaxis: { title: 'Validation Perplexity' }
            });

            // --- 3. Stability Comparison (VALIDATION LOSS) - Using Coefficient of Variation (CV) ---
            // Completely rewritten implementation using CV instead of variance
            const stabilityContainer = document.getElementById('stability-comparison-chart');
            if (!stabilityContainer) {
                console.error('Stability chart container not found');
                return;
            }
            
            // Clear previous chart if it exists
            Plotly.purge(stabilityContainer);
            
            const stabilityTraces = [];
            let maxIteration = 0;
            
            // Process each selected session
            for (const [sessionId, sessionData] of selectedSessions) {
                try {
                    // Skip if no validation loss data available
                    if (!sessionData.charts?.loss?.data) continue;
                    
                    const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
                    if (!validationLoss?.x || !validationLoss?.y || validationLoss.x.length < 5 || validationLoss.y.length < 5) continue;
                    
                    console.log(`Processing stability data for ${sessionId}, points: ${validationLoss.x.length}`);
                    
                    // Calculate Coefficient of Variation (CV) with a sliding window
                    const windowSize = Math.min(5, Math.floor(validationLoss.x.length / 2));
                    if (windowSize < 3) continue; // Need at least 3 points for meaningful CV
                    
                    const points = [];
                    
                    // Always start with zero CV at iteration 0
                    if (validationLoss.x[0] > 0) {
                        points.push({ x: 0, y: 0 });
                    }
                    
                    // Calculate CV for each window
                    for (let i = windowSize; i < validationLoss.y.length; i++) {
                        const window = validationLoss.y.slice(i - windowSize, i);
                        const mean = window.reduce((a, b) => a + b, 0) / windowSize;
                        
                        // Calculate standard deviation
                        const variance = window.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / windowSize;
                        const stdDev = Math.sqrt(variance);
                        
                        // Calculate Coefficient of Variation (CV) as percentage
                        const cv = (stdDev / mean) * 100; // Express as percentage
                        
                        const x = validationLoss.x[i];
                        if (x >= 0) { // Only use non-negative x values
                            points.push({ x, y: cv });
                            maxIteration = Math.max(maxIteration, x);
                        }
                    }
                    
                    // Only add trace if we have points
                    if (points.length > 0) {
                        // Apply hover highlighting logic for stability chart
                        let lineWidth = 2;
                        let opacity = 1.0;
                        
                        if (hoveredSessionId) {
                            if (sessionId === hoveredSessionId) {
                                lineWidth = 4; // Thicker line for hovered session
                                opacity = 1.0;
                            } else {
                                lineWidth = 1.5; // Thinner line for non-hovered sessions
                                opacity = 0.4; // Dimmed for non-hovered sessions
                            }
                        }
                        
                        stabilityTraces.push({
                            x: points.map(p => p.x),
                            y: points.map(p => p.y),
                            type: 'scatter',
                            mode: 'lines',
                            name: sessionData.session_name,
                            line: { color: sessionData.color, width: lineWidth },
                            opacity: opacity
                        });
                        
                        console.log(`Added stability trace for ${sessionId} with ${points.length} points, CV range: [${Math.min(...points.map(p => p.y))}, ${Math.max(...points.map(p => p.y))}]%`);
                    }
                } catch (error) {
                    console.error(`Error processing stability data for session ${sessionId}:`, error);
                }
            }
            
            // If no traces, show empty chart with message
            if (stabilityTraces.length === 0) {
                console.log('No stability data available');
                
                // Get theme colors
                const isDarkMode = document.body.getAttribute('data-theme') === 'dark';
                const textColor = isDarkMode ? '#F5F5F5' : '#333333';
                const borderColor = isDarkMode ? '#555555' : '#DDDDDD';
                
                // Create empty chart with message
                Plotly.newPlot(stabilityContainer, [], {
                    title: {
                        text: 'Validation Loss Stability - No Data Available',
                        font: { color: textColor }
                    },
                    xaxis: { 
                        title: 'Iterations', 
                        range: [0, 100],
                        tickfont: { color: textColor }
                    },
                    yaxis: { 
                        title: 'Coefficient of Variation (%)', 
                        range: [0, 20],
                        tickfont: { color: textColor }
                    },
                    annotations: [{
                        text: 'No stability data available for selected models',
                        showarrow: false,
                        font: { color: textColor, size: 14 },
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.5,
                        y: 0.5
                    }],
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: textColor }
                }, {
                    responsive: true,
                    displayModeBar: false
                });
                
                return;
            }
            
            // Calculate appropriate y-axis range
            const allCVValues = stabilityTraces.flatMap(trace => trace.y);
            const maxCV = Math.max(20, ...allCVValues); // At least 20% to show all stability zones
            
            console.log(`Stability chart: ${stabilityTraces.length} traces, max iteration: ${maxIteration}, max CV: ${maxCV}%`);
            
            // Create the chart with proper ranges
            const isDarkMode = document.body.getAttribute('data-theme') === 'dark';
            const textColor = isDarkMode ? '#F5F5F5' : '#333333';
            const borderColor = isDarkMode ? '#555555' : '#DDDDDD';
            
            Plotly.newPlot(stabilityContainer, stabilityTraces, {
                title: {
                    text: selectedSessions.size === 1 ? 'Validation Loss Stability' : 'Validation Loss Stability Comparison',
                    x: 0.05,
                    font: { color: textColor, size: 16 }
                },
                xaxis: {
                    title: { text: 'Iterations', font: { color: textColor } },
                    range: [0, maxIteration > 0 ? maxIteration : 100],
                    type: 'linear',
                    gridcolor: borderColor,
                    linecolor: borderColor,
                    zerolinecolor: borderColor,
                    tickfont: { color: textColor }
                },
                yaxis: {
                    title: { text: 'Coefficient of Variation (%)', font: { color: textColor } },
                    range: [0, maxCV * 1.1], // Add 10% padding
                    gridcolor: borderColor,
                    linecolor: borderColor,
                    zerolinecolor: borderColor,
                    tickfont: { color: textColor }
                },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { l: 60, r: 20, t: 50, b: 60 },
                shapes: [
                    // Excellent stability zone: CV < 5%
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0, x1: 1, y1: 5, fillcolor: 'rgba(40, 167, 69, 0.2)', line: { width: 0 }, layer: 'below' },
                    // Good stability zone: CV between 5% and 15%
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 5, x1: 1, y1: 15, fillcolor: 'rgba(255, 193, 7, 0.2)', line: { width: 0 }, layer: 'below' },
                    // Unstable zone: CV >= 15%
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 15, x1: 1, y1: maxCV * 1.1, fillcolor: 'rgba(220, 53, 69, 0.2)', line: { width: 0 }, layer: 'below' }
                ],
                annotations: [
                    { text: 'Excellent (<5%)', x: 0.95, y: 2.5, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(40, 167, 69, 0.9)', size: 10 }, xanchor: 'right' },
                    { text: 'Good (5-15%)', x: 0.95, y: 10, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(255, 193, 7, 0.9)', size: 10 }, xanchor: 'right' },
                    { text: 'Unstable (>15%)', x: 0.95, y: Math.min(maxCV * 0.6, 30), xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(220, 53, 69, 0.9)', size: 10 }, xanchor: 'right' }
                ],
                showlegend: false,
                font: { color: textColor } // Ensure all text uses the correct color
            }, {
                responsive: true,
                displayModeBar: false
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
                            
                            // Add a zero point at iteration 0 if the data doesn't start at 0
                            if (trainingLoss.x.length > 0 && trainingLoss.x[0] > 0) {
                                gapX.push(0);
                                // Use the first available gap value or 0
                                if (valMap.has(trainingLoss.x[0])) {
                                    gapY.push(valMap.get(trainingLoss.x[0]) - trainingLoss.y[0]);
                                } else {
                                    gapY.push(0);
                                }
                            }
                            
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
                            const gapX = [], gapY = [];
                            
                            // Add a zero point at iteration 0 if the data doesn't start at 0
                            if (trainingLoss.x.length > 0 && trainingLoss.x[0] > 0) {
                                gapX.push(0);
                                gapY.push(0);
                            }
                            
                            // Add the rest of the points
                            gapX.push(...trainingLoss.x);
                            gapY.push(...trainingLoss.x.map(() => 0));
                            
                             gapTraces.push({
                                x: gapX, y: gapY, type: 'scatter', mode: 'lines',
                                name: `${sessionData.session_name} (No Val)`,
                                line: { color: sessionData.color, width: 2, dash: 'dash' } // Use assigned color
                            });
                        }
                    }
                }
            }
            
            // Calculate appropriate y-range for generalization gap
            let minGapValue = 0, maxGapValue = 0;
            if (gapTraces.length > 0) {
                const allGapValues = gapTraces.flatMap(trace => trace.y).filter(val => val !== null && val !== undefined);
                if (allGapValues.length > 0) {
                    minGapValue = Math.min(...allGapValues);
                    maxGapValue = Math.max(...allGapValues);
                    // Add 20% padding to the range
                    const padding = Math.max(0.1, (maxGapValue - minGapValue) * 0.2);
                    minGapValue = Math.min(-0.1, minGapValue - padding);
                    maxGapValue = Math.max(0.1, maxGapValue + padding);
                }
            }
            
            renderComparisonChart('generalization-comparison-chart', gapTraces, {
                title: selectedSessions.size === 1 ? 'Generalization Gap' : 'Generalization Gap Comparison',
                xaxis: { title: 'Iterations', range: [0, null] },
                yaxis: { 
                    title: 'Val Loss - Train Loss', 
                    range: [minGapValue, maxGapValue],
                    autorange: false
                },
                shapes: [
                    // Underfitting area (yellow): Val Loss - Train Loss < 0
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: minGapValue, x1: 1, y1: 0, fillcolor: 'rgba(255, 193, 7, 0.2)', line: { width: 0 }, layer: 'below' },
                    // Good fit area (green): Val Loss - Train Loss between 0 and 0.2
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0, x1: 1, y1: 0.2, fillcolor: 'rgba(40, 167, 69, 0.2)', line: { width: 0 }, layer: 'below' },
                    // Overfitting area (red): Val Loss - Train Loss > 0.2
                    { type: 'rect', xref: 'paper', yref: 'y', x0: 0, y0: 0.2, x1: 1, y1: maxGapValue, fillcolor: 'rgba(220, 53, 69, 0.2)', line: { width: 0 }, layer: 'below' }
                ],
                annotations: [
                    // Underfitting label (yellow) positioned in the negative region
                    { text: 'Underfitting', x: 0.95, y: Math.max(minGapValue + 0.05, -0.3), xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(255, 193, 7, 0.9)', size: 10 }, xanchor: 'right' },
                    // Good fit label (green) positioned in the middle of the good fit region
                    { text: 'Good Fit', x: 0.95, y: 0.1, xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(40, 167, 69, 0.9)', size: 10 }, xanchor: 'right' },
                    // Overfitting label (red) positioned in the overfitting region
                    { text: 'Overfitting', x: 0.95, y: Math.min(maxGapValue - 0.05, 0.3), xref: 'paper', yref: 'y', showarrow: false, font: { color: 'rgba(220, 53, 69, 0.9)', size: 10 }, xanchor: 'right' }
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
        const parametersModalBody = document.getElementById('parameters-modal-body');
        if (parametersModalBody) {
            parametersModalBody.innerHTML = `
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
            const modalTitle = document.querySelector('#parameters-modal .modal-title');
            if (modalTitle) {
                modalTitle.textContent = `Session Parameters: ${sessionId}`;
            }
            
            // Setup copy button with the raw data
            const copyButton = document.getElementById('copy-parameters-btn');
            if (copyButton) {
                // Remove any existing event listeners
                const newCopyButton = copyButton.cloneNode(true);
                copyButton.parentNode.replaceChild(newCopyButton, copyButton);
                
                // Add new event listener
                newCopyButton.addEventListener('click', () => {
                    const textToCopy = JSON.stringify(rawData, null, 2);
                    navigator.clipboard.writeText(textToCopy)
                        .then(() => {
                            // Show success tooltip
                            const tooltip = new bootstrap.Tooltip(newCopyButton, {
                                title: 'Copied!',
                                trigger: 'manual',
                                placement: 'top'
                            });
                            tooltip.show();
                            setTimeout(() => tooltip.hide(), 2000);
                        })
                        .catch(err => {
                            console.error('Failed to copy parameters:', err);
                            alert('Failed to copy parameters to clipboard');
                        });
                });
            }
            
            // Show the modal
            const parametersModal = new bootstrap.Modal(document.getElementById('parameters-modal'));
            parametersModal.show();
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

// Function to update session badges with actual config data
async function updateSessionBadgesWithActualData(sessions) {
    console.log('Updating session badges with actual config data...');
    
    // Process sessions in batches to avoid overwhelming the server
    const batchSize = 3;
    for (let i = 0; i < sessions.length; i += batchSize) {
        const batch = sessions.slice(i, i + batchSize);
        
        await Promise.allSettled(batch.map(async session => {
            try {
                if (!session.log_file) return;
                
                // Fetch the actual config
                const response = await fetch('/api/logs/raw', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ log_file: session.log_file })
                });
                
                if (!response.ok) return;
                
                const sessionDetails = await response.json();
                let config = null;
                
                try {
                    const rawData = JSON.parse(sessionDetails.logs);
                    config = rawData.config || {};
                } catch (e) {
                    console.log('Could not parse session config for', session.session_name);
                    return;
                }
                
                // Get the actual fine_tune_type and other parameters
                const actualFineTuneType = config.fine_tune_type;
                const actualWeightDecay = config.weight_decay;
                const actualLrDecayFactor = config.lr_decay_factor;
                const actualMaxSeqLength = config.max_seq_length;
                
                if (!actualFineTuneType && !actualWeightDecay && !actualLrDecayFactor && !actualMaxSeqLength) return;
                
                // Update the badge in the UI
                const sessionId = session.session_id || session.id || '';
                const escapedSessionId = escapeSelector(sessionId);
                const sessionCard = document.querySelector(`#session-card-${escapedSessionId}`);
                
                if (sessionCard) {
                    const trainingBadges = sessionCard.querySelector('.training-badges');
                    if (trainingBadges) {
                        // Find and update the fine-tune type badge
                        const badges = trainingBadges.querySelectorAll('.badge');
                        let fineTuneBadge = null;
                        
                        badges.forEach(badge => {
                            const text = badge.textContent.trim();
                            if (text === 'Full' || text === 'LoRA' || text === 'DoRA') {
                                fineTuneBadge = badge;
                            }
                        });
                        
                        // Update fine-tune type badge if we have that info
                        if (fineTuneBadge && actualFineTuneType) {
                            // Update the badge text and color
                            const displayType = actualFineTuneType === 'lora' ? 'LoRA' : 
                                               actualFineTuneType === 'dora' ? 'DoRA' : 'Full';
                            const bgColor = displayType === 'Full' ? 'bg-primary' : 
                                           displayType === 'LoRA' ? 'bg-success' : 'bg-info';
                            
                            fineTuneBadge.textContent = displayType;
                            fineTuneBadge.className = `badge ${bgColor}`;
                            
                            console.log(`Updated ${session.session_name}: ${actualFineTuneType} -> ${displayType}`);
                        }
                        
                        // Update learning rate display with actual parameters
                        if (actualWeightDecay || actualLrDecayFactor) {
                            const lrInfoItems = sessionCard.querySelectorAll('.info-item .info-text');
                            let lrInfoSpan = null;
                            
                            // Find the LR info span
                            lrInfoItems.forEach(span => {
                                if (span.textContent.includes('LR:')) {
                                    lrInfoSpan = span;
                                }
                            });
                            
                            if (lrInfoSpan) {
                                // Extract current LR value
                                const currentLrMatch = lrInfoSpan.textContent.match(/LR:\s*([^\s|]+)/);
                                const currentLr = currentLrMatch ? currentLrMatch[1] : '';
                                
                                // Build new display
                                let newLrDisplay = `LR: ${currentLr}`;
                                let additionalParams = [];
                                
                                if (actualLrDecayFactor) {
                                    additionalParams.push(`LDR ${actualLrDecayFactor}`);
                                }
                                
                                if (actualWeightDecay) {
                                    additionalParams.push(`WD ${actualWeightDecay}`);
                                }
                                
                                if (additionalParams.length > 0) {
                                    newLrDisplay += ` | ${additionalParams.join(' | ')}`;
                                }
                                
                                lrInfoSpan.textContent = newLrDisplay;
                                console.log(`Updated LR display for ${session.session_name}: ${newLrDisplay}`);
                            }
                        }
                        
                        // Update sequence length badge if we have actual data
                        if (actualMaxSeqLength) {
                            const trainingBadges = sessionCard.querySelector('.training-badges');
                            if (trainingBadges) {
                                // Check if sequence length badge already exists
                                const existingSeqBadges = trainingBadges.querySelectorAll('.badge.bg-secondary');
                                let hasSeqBadge = false;
                                
                                existingSeqBadges.forEach(badge => {
                                    const text = badge.textContent.trim();
                                    if (/^\d+$/.test(text)) {
                                        hasSeqBadge = true;
                                    }
                                });
                                
                                // Add sequence length badge if it doesn't exist
                                if (!hasSeqBadge) {
                                    const seqBadge = document.createElement('span');
                                    seqBadge.className = 'badge bg-secondary';
                                    seqBadge.textContent = actualMaxSeqLength;
                                    trainingBadges.appendChild(document.createTextNode(' '));
                                    trainingBadges.appendChild(seqBadge);
                                    
                                    console.log(`Added sequence length badge for ${session.session_name}: ${actualMaxSeqLength}`);
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                console.log('Error updating badge for', session.session_name, ':', error);
            }
        }));
        
        // Small delay between batches to avoid overwhelming the server
        if (i + batchSize < sessions.length) {
            await new Promise(resolve => setTimeout(resolve, 200));
        }
    }
    
    console.log('Finished updating session badges');
}

// Function to show session folder in a modal
async function showSessionFolder(sessionId) {
    try {
        // Get the session data to find the log file path
        const sessionsResponse = await fetch('/api/training/sessions');
        const sessionsData = await sessionsResponse.json();
        const sessions = sessionsData.training_sessions || [];
        const session = sessions.find(s => s.session_id === sessionId);
        
        if (!session || !session.log_file) {
            alert('Session folder not found');
            return;
        }
        
        // Extract the directory path from the log file path
        const logFilePath = session.log_file;
        const sessionDirectory = logFilePath.substring(0, logFilePath.lastIndexOf('/'));
        
        console.log('Showing folder for session:', sessionId, 'Path:', sessionDirectory);
        
        // Use the existing file browser modal
        window.currentBrowserCallback = null; // No callback needed for viewing
        window.currentBrowserType = 'view';
        
        // Set modal title
        const modal = document.getElementById('file-browser-modal');
        const modalTitle = modal.querySelector('#file-browser-title');
        modalTitle.textContent = `Session Folder: ${session.session_name || sessionId}`;
        
        // Hide the select button since we're just viewing
        const selectBtn = modal.querySelector('#browser-select-btn');
        selectBtn.style.display = 'none';
        
        // Load the directory contents
        await loadDirectoryContents(sessionDirectory);
        
        // Show the modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Reset modal when it's hidden to restore normal functionality for other uses
        modal.addEventListener('hidden.bs.modal', () => {
            const selectBtn = modal.querySelector('#browser-select-btn');
            const modalTitle = modal.querySelector('#file-browser-title');
            selectBtn.style.display = 'block';
            modalTitle.textContent = 'Select Directory';
        }, { once: true });
        
    } catch (error) {
        console.error('Error showing session folder:', error);
        alert('Error showing session folder: ' + error.message);
    }
}

// Function to delete a session with confirmation
async function deleteSession(sessionId) {
    try {
        // Get the session data for confirmation
        const sessionsResponse = await fetch('/api/training/sessions');
        const sessionsData = await sessionsResponse.json();
        const sessions = sessionsData.training_sessions || [];
        const session = sessions.find(s => s.session_id === sessionId);
        
        if (!session) {
            alert('Session not found');
            return;
        }
        
        const sessionName = session.session_name || sessionId;
        
        // Show confirmation dialog
        const confirmed = confirm(
            `Are you sure you want to delete the training session?\n\n` +
            `Session: ${sessionName}\n` +
            `Model: ${session.base_model || 'Unknown'}\n` +
            `Iterations: ${session.latest_iteration || 'N/A'}\n\n` +
            `This action cannot be undone and will delete all session files.`
        );
        
        if (!confirmed) {
            return;
        }
        
        // Show loading
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        loadingMessage.textContent = `Deleting session: ${sessionName}...`;
        loadingOverlay.classList.remove('d-none');
        
        // Call the delete API
        const response = await fetch('/api/training/sessions/delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        // Hide loading
        loadingOverlay.classList.add('d-none');
        
        if (result.success) {
            console.log(`Successfully deleted session: ${sessionName}`);
            
            // Remove from selected sessions if it was selected
            selectedSessions.delete(sessionId);
            updateSelectionSummary();
            
            // Refresh the sessions list
            await loadSessions();
            
            // Show success message
            alert(`Session "${sessionName}" has been successfully deleted.`);
        } else {
            throw new Error(result.error || 'Unknown error during deletion');
        }
        
    } catch (error) {
        // Hide loading
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.classList.add('d-none');
        
        console.error('Error deleting session:', error);
        alert(`Error deleting session: ${error.message}`);
    }
}

// Helper function to load directory contents (adapted from existing file browser)
async function loadDirectoryContents(path) {
    const browserContent = document.getElementById('browser-content');
    const browserLoading = document.getElementById('browser-loading');
    const currentPathInput = document.getElementById('browser-current-path');
    
    try {
        browserLoading.classList.remove('d-none');
        browserContent.innerHTML = '';
        
        const response = await fetch(`/api/filesystem/browse?path=${encodeURIComponent(path)}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to load directory');
        }
        
        currentPathInput.value = data.current_path;
        
        let html = '';
        
        // Add items (directories and files)
        if (data.items && data.items.length > 0) {
            data.items.forEach(item => {
                if (item.is_directory) {
                    html += `
                        <div class="border-bottom p-2 browser-item" data-type="directory" data-path="${item.path}">
                            <i class="fas fa-folder text-primary me-2"></i>
                            <span>${item.name}</span>
                            <small class="text-muted ms-auto">${item.description}</small>
                        </div>
                    `;
                } else {
                    const icon = getFileIcon(item.name);
                    html += `
                        <div class="border-bottom p-2 browser-item" data-type="file" data-path="${item.path}">
                            <i class="${icon} me-2"></i>
                            <span>${item.name}</span>
                            <small class="text-muted ms-auto">${item.description}</small>
                        </div>
                    `;
                }
            });
        }
        
        if (html === '') {
            html = '<div class="p-3 text-muted text-center">Empty directory</div>';
        }
        
        browserContent.innerHTML = html;
        
        // Add click handlers for navigation (only for viewing)
        browserContent.querySelectorAll('.browser-item[data-type="directory"]').forEach(item => {
            item.style.cursor = 'pointer';
            item.addEventListener('click', () => {
                const newPath = item.getAttribute('data-path');
                loadDirectoryContents(newPath);
            });
        });
        
    } catch (error) {
        console.error('Error loading directory:', error);
        browserContent.innerHTML = `<div class="p-3 text-danger">Error: ${error.message}</div>`;
    } finally {
        browserLoading.classList.add('d-none');
    }
}

// Helper functions for file browser
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    switch (ext) {
        case 'json': return 'fas fa-file-code text-warning';
        case 'safetensors': return 'fas fa-cube text-info';
        case 'py': return 'fab fa-python text-success';
        case 'md': return 'fab fa-markdown text-primary';
        case 'txt': return 'fas fa-file-alt text-secondary';
        case 'yaml': case 'yml': return 'fas fa-file-code text-orange';
        default: return 'fas fa-file text-muted';
    }
}

function formatFileSize(bytes) {
    if (!bytes) return '';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

console.log('Compare.js Plotly version loaded');

// Search functionality for filtering sessions
function initializeSessionSearch() {
    const searchInput = document.getElementById('session-search-input');
    const clearSearchBtn = document.getElementById('clear-search-btn');
    
    if (!searchInput || !clearSearchBtn) return;
    
    // Search input handler
    searchInput.addEventListener('input', filterSessions);
    
    // Clear search button handler
    clearSearchBtn.addEventListener('click', () => {
        searchInput.value = '';
        filterSessions();
        searchInput.focus();
    });
    
    // Enter key handler
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            searchInput.value = '';
            filterSessions();
        }
    });
}

function filterSessions() {
    const searchInput = document.getElementById('session-search-input');
    const sessionsList = document.getElementById('compare-sessions-list');
    
    if (!searchInput || !sessionsList) return;
    
    const searchTerm = searchInput.value.toLowerCase().trim();
    const sessionCards = sessionsList.querySelectorAll('.session-card');
    
    let visibleCount = 0;
    
    sessionCards.forEach(card => {
        const cardText = card.textContent.toLowerCase();
        const isVisible = searchTerm === '' || cardText.includes(searchTerm);
        
        card.style.display = isVisible ? 'block' : 'none';
        if (isVisible) visibleCount++;
    });
    
    // Show/hide "no results" message
    let noResultsMsg = sessionsList.querySelector('.no-search-results');
    if (visibleCount === 0 && searchTerm !== '') {
        if (!noResultsMsg) {
            noResultsMsg = document.createElement('div');
            noResultsMsg.className = 'no-search-results text-center text-muted py-3';
            noResultsMsg.innerHTML = `
                <i class="fas fa-search fa-2x mb-2"></i>
                <p>No sessions found matching "${searchTerm}"</p>
                <small>Try different keywords or clear the search</small>
            `;
            sessionsList.appendChild(noResultsMsg);
        }
    } else if (noResultsMsg) {
        noResultsMsg.remove();
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
    max-height: 677px;
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

/* Session Card Styling - More compact */
.session-card {
    transition: all 0.2s ease-in-out;
    border-left: 4px solid transparent;
    padding: 8px;
    border-radius: 6px;
    margin-bottom: 6px;
    cursor: pointer;
}

/* Light mode - default background for session cards */
.session-card {
    background-color: #f8f9fa;
}

/* Dark mode - using proper dark theme background */
[data-theme="dark"] .session-card {
    background-color: #2d2d2d !important;
    border-color: #404040;
}

.session-card:hover {
    background-color: rgba(0, 123, 255, 0.1);
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.2) !important;
    border-left-color: #007AFF !important;
}

/* Selected state styling - IMPORTANT: Make this very visible with blue background */
.selected-session-card {
    background-color: rgba(13, 110, 253, 0.25) !important;
    border-left-color: #0d6efd !important;
    border-left-width: 4px !important;
    border-left-style: solid !important;
    box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.3) !important;
    position: relative !important;
    z-index: 1 !important;
    border: 2px solid rgba(13, 110, 253, 0.5) !important;
    border-radius: 8px !important;
}

/* Add a blue overlay to make selection more obvious */
.selected-session-card::after {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border: 2px solid rgba(13, 110, 253, 0.6);
    border-radius: 8px;
    pointer-events: none;
    z-index: -1;
    background: rgba(13, 110, 253, 0.1);
}

/* Dark mode selected state */
[data-theme="dark"] .selected-session-card {
    background-color: rgba(13, 110, 253, 0.35) !important;
    border-left-color: #0d6efd !important;
    border-left-width: 4px !important;
    border-left-style: solid !important;
    box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.4) !important;
    border: 2px solid rgba(13, 110, 253, 0.6) !important;
}

[data-theme="dark"] .selected-session-card::after {
    background: rgba(13, 110, 253, 0.15);
    border: 2px solid rgba(13, 110, 253, 0.7);
}

/* Session Header - Redesigned for compact layout */
.session-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.session-name {
    font-weight: 600 !important;
    font-size: 13px !important;
    color: var(--text-color) !important;
    line-height: 1.2;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: 8px;
}

.loss-badges {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex-shrink: 0;
    align-items: flex-end;
}

.loss-badges .badge {
    font-size: 9px !important;
    padding: 2px 6px !important;
    border-radius: 8px !important;
    line-height: 1.2;
    white-space: nowrap;
}

.iteration-badge {
    font-size: 9px !important;
    padding: 2px 6px !important;
    border-radius: 8px !important;
    flex-shrink: 0;
}

/* Training badges row */
.training-badges {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
}

.training-badges .badge {
    font-size: 8px !important;
    padding: 1px 5px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* Compact info layout */
.session-compact-info {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-bottom: 8px;
}

.info-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px !important;
}

.info-item i {
    width: 12px;
    font-size: 9px !important;
    opacity: 0.7;
}

.info-text {
    color: var(--text-color) !important;
    line-height: 1.2;
    font-weight: 400;
}

/* Action buttons - extra small size */
.session-actions {
    display: flex;
    gap: 3px;
    margin-top: 6px;
}

.btn-xs {
    padding: 2px 4px !important;
    font-size: 9px !important;
    border-radius: 3px !important;
    min-width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
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
            background-color: rgba(0, 123, 255, 0.15) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3) !important;
            border-left-color: #007AFF !important;
        }
        
        /* Summary Table Styles */
        #sessions-summary-table {
            font-size: 0.875rem;
        }
        
        #sessions-summary-table th {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border-color);
            padding: 0.75rem 0.5rem;
        }
        
        #sessions-summary-table td {
            padding: 0.75rem 0.5rem;
            vertical-align: middle;
            border-bottom: 1px solid var(--border-color);
        }
        
        #sessions-summary-table .btn-xs {
            padding: 0.25rem 0.375rem;
            font-size: 0.75rem;
            margin: 0 0.125rem;
        }
        
        #sessions-summary-table .badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
        }
        
        #sessions-summary-table .btn-group {
            display: flex;
            flex-wrap: nowrap;
            gap: 2px;
        }
        
        /* Dark mode table styling - specific to summary table only */
        [data-theme="dark"] #sessions-summary-table {
    color: var(--text-color);
}

        [data-theme="dark"] #sessions-summary-table thead.table-light {
            background-color: var(--surface-color) !important;
            border-color: var(--border-color) !important;
        }
        
        [data-theme="dark"] #sessions-summary-table thead.table-light th {
            color: var(--text-color) !important;
            background-color: var(--surface-color) !important;
        }
        
        [data-theme="dark"] #sessions-summary-table tbody td {
            color: var(--text-color) !important;
            background-color: var(--bg-color) !important;
            border-color: var(--border-color) !important;
        }
        
        [data-theme="dark"] #sessions-summary-table tbody tr:hover td {
            background-color: var(--surface-color) !important;
        }

[data-theme="dark"] .selected-session-card {
    background-color: rgba(13, 110, 253, 0.25) !important;
    border-left-color: #0d6efd !important;
}

/* Selection Summary Styling */


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

// Listen for theme changes and regenerate comparison charts
function handleThemeChange() {
    // Only regenerate if we have active comparisons
    if (selectedSessions.size >= 1) {
        console.log('Theme changed, regenerating comparison charts...');
        generateComparison();
    }
}

// Handle session card hover events
function handleSessionHover(sessionId) {
    // Only apply hover effects for selected sessions
    if (selectedSessions.has(sessionId)) {
        hoveredSessionId = sessionId;
        regenerateChartsWithHighlighting();
    }
}

function handleSessionHoverEnd() {
    hoveredSessionId = null;
    regenerateChartsWithHighlighting();
}

// Regenerate charts with current hover highlighting
function regenerateChartsWithHighlighting() {
    if (selectedSessions.size >= 1) {
        generateComparison();
    }
}

// Summary table functions
function generateSummaryTable() {
    const tableCard = document.getElementById('summary-table-card');
    const tbody = document.getElementById('sessions-summary-tbody');
    
    if (!tableCard || !tbody || selectedSessions.size === 0) {
        hideSummaryTable();
        return;
    }
    
    // Show the table
    tableCard.style.display = 'block';
    
    // Convert selected sessions to array and sort by validation loss
    const sessionsArray = Array.from(selectedSessions.entries()).map(([sessionId, sessionData]) => {
        return { sessionId, ...sessionData };
    });
    
    // Sort by best validation loss (lowest first)
    sessionsArray.sort((a, b) => {
        const aLoss = getBestValidationLoss(a);
        const bLoss = getBestValidationLoss(b);
        return aLoss - bLoss;
    });
    
    // Generate table rows
    tbody.innerHTML = '';
    
    sessionsArray.forEach((session, index) => {
        try {
            const row = createSummaryTableRow(session, index);
            tbody.appendChild(row);
        } catch (error) {
            console.error('Error creating table row for session:', session.sessionId, error);
            // Create a fallback row with basic info
            const fallbackRow = createFallbackTableRow(session, index);
            tbody.appendChild(fallbackRow);
        }
    });
}

function hideSummaryTable() {
    const tableCard = document.getElementById('summary-table-card');
    if (tableCard) {
        tableCard.style.display = 'none';
    }
}

function getBestValidationLoss(sessionData) {
    try {
        if (sessionData.charts?.loss?.data) {
            const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
            if (validationLoss?.y && validationLoss.y.length > 0) {
                return Math.min(...validationLoss.y);
            }
        }
        return Infinity; // Return high value if no loss data available
    } catch (error) {
        console.warn('Error getting validation loss for session:', sessionData.session_name, error);
        return Infinity;
    }
}

function getBestCheckpointInfo(sessionData) {
    try {
        const bestLoss = getBestValidationLoss(sessionData);
        if (bestLoss === Infinity) return { checkpoint: 'N/A', iteration: 'N/A' };
        
        // Find the iteration with the best loss
        if (sessionData.charts?.loss?.data) {
            const validationLoss = sessionData.charts.loss.data.find(c => c.name === 'Validation Loss');
            if (validationLoss?.x && validationLoss?.y) {
                const bestIndex = validationLoss.y.indexOf(bestLoss);
                const bestIteration = validationLoss.x[bestIndex];
                return {
                    checkpoint: `checkpoint-${bestIteration.toString().padStart(6, '0')}`,
                    iteration: bestIteration
                };
            }
        }
        return { checkpoint: 'N/A', iteration: 'N/A' };
    } catch (error) {
        console.warn('Error getting best checkpoint for session:', sessionData.session_name, error);
        return { checkpoint: 'N/A', iteration: 'N/A' };
    }
}

function createSummaryTableRow(session, index) {
    const row = document.createElement('tr');
    
    // Use the SessionDataManager methods to extract all data consistently
    const params = sessionDataManager.extractTrainingParameters(session);
    const loraParams = sessionDataManager.extractLoRAParameters(session);
    const bestLoss = sessionDataManager.getBestValidationLoss(session);
    const bestCheckpoint = sessionDataManager.getBestCheckpointInfo(session);
    
    // Extract model name from session name or session_id
    const modelName = extractModelName(session.session_name || session.session_id);
    
    // Format date and time separately
    const dateTime = formatSessionDateTime(session);
    
    // Detect training type and method using existing logic
    const trainingType = detectTrainingType(session);
    const trainingMethod = detectTrainingMethod(session);
    
    console.log('Table row data for', session.session_name, ':', {
        params,
        loraParams,
        bestLoss,
        bestCheckpoint,
        config: session.config
    });
    
    row.innerHTML = `
        <td>
            <div class="fw-bold">${modelName}</div>
            <small class="text-muted">${dateTime}</small>
        </td>
        <td>
            <span class="badge bg-${getTrainingTypeBadgeColor(trainingType)}">${trainingType}</span>
        </td>
        <td>
            <span class="badge bg-warning text-dark">${trainingMethod}</span>
        </td>
        <td>
            <span class="badge bg-secondary">${params.max_seq_length}</span>
        </td>
        <td>${params.max_iterations}</td>
        <td>
            <div class="fw-bold">${bestCheckpoint.checkpoint}</div>
        </td>
        <td>
            <span class="fw-bold ${bestLoss.loss === Infinity ? 'text-muted' : 'text-success'}">
                ${bestLoss.loss === Infinity ? 'N/A' : bestLoss.loss.toFixed(4)}
            </span>
        </td>
        <td>${formatScientificNotation(params.learning_rate)}</td>
        <td>${params.lr_decay_factor}</td>
        <td>${params.weight_decay}</td>
        <td>${params.batch_size}</td>
        <td>${params.warmup_steps}</td>
        <td>${loraParams.rank}</td>
        <td>${loraParams.scale}</td>
        <td>${loraParams.dropout}</td>
        <td>${loraParams.layers}</td>
        <td>
            <small>${loraParams.target_modules}</small>
        </td>
        <td>
            <div class="btn-group" role="group">
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="showSessionParameters('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="View Parameters">
                    <i class="fas fa-file-code"></i>
                </button>
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="fuseSessionAdapter('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="Fuse Adapter">
                    <i class="fas fa-layer-group"></i>
                </button>
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="testSessionInPlayground('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="Test in Playground">
                    <i class="fas fa-flask"></i>
                </button>
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="showSessionFolder('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="View Folder">
                    <i class="fas fa-folder-open"></i>
                </button>
                <button class="btn btn-xs btn-outline-danger" 
                        onclick="deleteSession('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="Delete Session">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </td>
    `;
    
    return row;
}

function getTrainingTypeBadgeColor(trainingType) {
    switch (trainingType.toLowerCase()) {
        case 'lora': return 'success';
        case 'dora': return 'info';
        case 'full': return 'primary';
        default: return 'secondary';
    }
}

function detectTrainingType(session) {
    // Use existing detection logic from session cards
    if (session.session_name?.toLowerCase().includes('lora') || 
        session.log_file?.toLowerCase().includes('lora') ||
        session.session_id?.toLowerCase().includes('lora')) {
        return 'LoRA';
    }
    if (session.session_name?.toLowerCase().includes('dora') || 
        session.log_file?.toLowerCase().includes('dora') ||
        session.session_id?.toLowerCase().includes('dora')) {
        return 'DoRA';
    }
    return 'Full';
}

function detectTrainingMethod(session) {
    // Detect CPT vs SFT based on session patterns
    if (session.session_name?.toLowerCase().includes('sft') || 
        session.session_name?.toLowerCase().includes('instruct') ||
        session.log_file?.toLowerCase().includes('sft')) {
        return 'SFT';
    }
    return 'CPT'; // Default to CPT
}

function extractModelName(sessionName) {
    if (!sessionName) return 'Unknown';
    
    // Remove common prefixes
    let modelName = sessionName.replace(/^(mlx-community\/|microsoft\/|meta-llama\/|google\/|huggingface\/)/i, '');
    
    // Extract model name from session naming pattern
    // Pattern: model_name_training_params_date_time
    const parts = modelName.split('_');
    
    // Try to identify the model part (usually first few segments before training params)
    if (parts.length >= 3) {
        // Look for common training indicators to split
        const trainingIndicators = ['lr', 'bs', 'iter', 'seq', 'lora', 'dora', '2025'];
        let modelParts = [];
        
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            const hasTrainingIndicator = trainingIndicators.some(indicator => 
                part.toLowerCase().includes(indicator.toLowerCase())
            );
            
            if (hasTrainingIndicator) {
                break;
            }
            modelParts.push(part);
        }
        
        if (modelParts.length > 0) {
            return modelParts.join('_');
        }
    }
    
    // Fallback: just clean up the session name
    return modelName.split('_').slice(0, 3).join('_');
}

function formatSessionDateTime(session) {
    // Try to extract date/time from session name first
    const sessionName = session.session_name || session.session_id || '';
    
    // Look for date pattern in session name (YYYY-MM-DD_HH-MM)
    const dateMatch = sessionName.match(/(\d{4}-\d{2}-\d{2})[_-](\d{2})[_-](\d{2})/);
    if (dateMatch) {
        const [, date, hour, minute] = dateMatch;
        return `${date} ${hour}:${minute}`;
    }
    
    // Fallback to start_time if available
    if (session.start_time) {
        const startDate = new Date(session.start_time);
        return startDate.toLocaleString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    return 'N/A';
}

// DEPRECATED: These functions have been moved to SessionDataManager class
// function extractParametersFromConfig(config) - use sessionDataManager.extractTrainingParameters(sessionData)
// function extractLoRAParameters(config, trainingType) - use sessionDataManager.extractLoRAParameters(sessionData)

function createFallbackTableRow(session, index) {
    const row = document.createElement('tr');
    
    // Basic fallback parameters
    const params = getTrainingParameters(session, session.extraConfig);
    const modelName = extractModelName(session.session_name || session.session_id);
    const dateTime = formatSessionDateTime(session);
    const trainingType = detectTrainingType(session);
    const trainingMethod = detectTrainingMethod(session);
    
    row.innerHTML = `
        <td>
            <div class="fw-bold">${modelName}</div>
            <small class="text-muted">${dateTime}</small>
        </td>
        <td>
            <span class="badge bg-${getTrainingTypeBadgeColor(trainingType)}">${trainingType}</span>
        </td>
        <td>
            <span class="badge bg-warning text-dark">${trainingMethod}</span>
        </td>
        <td>
            <span class="badge bg-secondary">${params.max_seq_length || 'N/A'}</span>
        </td>
        <td>${params.max_iterations || 'N/A'}</td>
        <td>
            <div class="text-muted">N/A</div>
        </td>
        <td>
            <span class="text-muted">N/A</span>
        </td>
        <td>${formatScientificNotation(params.learning_rate) || 'N/A'}</td>
        <td>${params.lr_decay_factor || 'N/A'}</td>
        <td>${params.weight_decay || 'N/A'}</td>
        <td>${params.batch_size || 'N/A'}</td>
        <td>${params.warmup_steps || 'N/A'}</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>
            <div class="btn-group" role="group">
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="showSessionParameters('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="View Parameters">
                    <i class="fas fa-file-code"></i>
                </button>
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="fuseSessionAdapter('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="Fuse Adapter">
                    <i class="fas fa-layer-group"></i>
                </button>
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="testSessionInPlayground('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="Test in Playground">
                    <i class="fas fa-flask"></i>
                </button>
                <button class="btn btn-xs btn-outline-secondary" 
                        onclick="showSessionFolder('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="View Folder">
                    <i class="fas fa-folder-open"></i>
                </button>
                <button class="btn btn-xs btn-outline-danger" 
                        onclick="deleteSession('${session.sessionId.replace(/'/g, '\\\'')}'); event.stopPropagation();"
                        title="Delete Session">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </td>
    `;
    
    return row;
}

// Watch for theme changes on the body element
const themeObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
            handleThemeChange();
        }
    });
});

// Start observing theme changes
if (document.body) {
    themeObserver.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-theme']
    });
} 