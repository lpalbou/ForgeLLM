/**
 * Formatting Utilities
 * 
 * Common formatting functions for the UI.
 */

const formatUtil = {
    /**
     * Format a number with thousands separators
     * 
     * @param {number} num - Number to format
     * @returns {string} Formatted number
     */
    formatNumber(num) {
        if (num === undefined || num === null) return '—';
        return new Intl.NumberFormat().format(num);
    },
    
    /**
     * Format a decimal number with specified precision
     * 
     * @param {number} num - Number to format
     * @param {number} precision - Number of decimal places
     * @returns {string} Formatted decimal
     */
    formatDecimal(num, precision = 2) {
        if (num === undefined || num === null) return '—';
        return Number(num).toFixed(precision);
    },
    
    /**
     * Format a file size
     * 
     * @param {number} bytes - Size in bytes
     * @param {number} precision - Number of decimal places
     * @returns {string} Formatted file size
     */
    formatBytes(bytes, precision = 2) {
        if (bytes === 0) return '0 Bytes';
        if (bytes === undefined || bytes === null) return '—';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(precision)) + ' ' + sizes[i];
    },
    
    /**
     * Format a duration in seconds
     * 
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration
     */
    formatDuration(seconds) {
        if (seconds === undefined || seconds === null) return '—';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        let result = '';
        
        if (hours > 0) {
            result += `${hours}h `;
        }
        
        if (minutes > 0 || hours > 0) {
            result += `${minutes}m `;
        }
        
        result += `${secs}s`;
        
        return result.trim();
    },
    
    /**
     * Format a date
     * 
     * @param {string|number|Date} date - Date to format
     * @returns {string} Formatted date
     */
    formatDate(date) {
        if (!date) return '—';
        
        try {
            const d = new Date(date);
            return d.toLocaleDateString();
        } catch (error) {
            console.error('Error formatting date:', error);
            return '—';
        }
    },
    
    /**
     * Format a datetime
     * 
     * @param {string|number|Date} date - Date to format
     * @returns {string} Formatted datetime
     */
    formatDateTime(date) {
        if (!date) return '—';
        
        try {
            const d = new Date(date);
            return d.toLocaleString();
        } catch (error) {
            console.error('Error formatting datetime:', error);
            return '—';
        }
    },
    
    /**
     * Format a model name
     * 
     * @param {string} modelName - Model name to format
     * @returns {string} Formatted model name
     */
    formatModelName(modelName) {
        if (!modelName) return 'Unknown Model';
        
        // Remove organization prefix
        const parts = modelName.split('/');
        const name = parts.length > 1 ? parts[1] : parts[0];
        
        // Replace hyphens and underscores with spaces
        return name.replace(/[-_]/g, ' ');
    },
    
    /**
     * Format a percentage
     * 
     * @param {number} value - Value to format
     * @param {number} precision - Number of decimal places
     * @returns {string} Formatted percentage
     */
    formatPercent(value, precision = 1) {
        if (value === undefined || value === null) return '—';
        return `${Number(value).toFixed(precision)}%`;
    },
    
    /**
     * Format a learning rate
     * 
     * @param {number} lr - Learning rate to format
     * @returns {string} Formatted learning rate
     */
    formatLearningRate(lr) {
        if (lr === undefined || lr === null) return '—';
        return lr.toExponential(2);
    }
}; 