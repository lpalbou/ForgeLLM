/**
 * Charts Utility
 * 
 * Utilities for creating and updating charts for training visualization.
 */

const chartsUtil = {
    /**
     * Chart instances
     */
    charts: {},
    
    /**
     * Create a loss chart
     * 
     * @param {string} elementId - ID of the canvas element
     * @param {Object} data - Chart data
     * @returns {Chart} - Chart.js instance
     */
    createLossChart(elementId, data = {}) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.iterations || [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: data.trainLoss || [],
                        borderColor: 'rgb(46, 134, 171)',
                        backgroundColor: 'rgba(46, 134, 171, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: false
                    },
                    {
                        label: 'Validation Loss',
                        data: data.valLoss || [],
                        borderColor: 'rgb(162, 59, 114)',
                        backgroundColor: 'rgba(162, 59, 114, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training & Validation Loss'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        beginAtZero: false
                    }
                }
            }
        });
        
        this.charts[elementId] = chart;
        return chart;
    },
    
    /**
     * Create a perplexity chart
     * 
     * @param {string} elementId - ID of the canvas element
     * @param {Object} data - Chart data
     * @returns {Chart} - Chart.js instance
     */
    createPerplexityChart(elementId, data = {}) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        // Convert loss to perplexity
        const trainPerplexity = (data.trainLoss || []).map(loss => Math.exp(loss));
        const valPerplexity = (data.valLoss || []).map(loss => Math.exp(loss));
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.iterations || [],
                datasets: [
                    {
                        label: 'Training Perplexity',
                        data: trainPerplexity,
                        borderColor: 'rgb(46, 134, 171)',
                        backgroundColor: 'rgba(46, 134, 171, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: false
                    },
                    {
                        label: 'Validation Perplexity',
                        data: valPerplexity,
                        borderColor: 'rgb(162, 59, 114)',
                        backgroundColor: 'rgba(162, 59, 114, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training & Validation Perplexity'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Perplexity'
                        },
                        beginAtZero: false
                    }
                }
            }
        });
        
        this.charts[elementId] = chart;
        return chart;
    },
    
    /**
     * Create a learning rate chart
     * 
     * @param {string} elementId - ID of the canvas element
     * @param {Object} data - Chart data
     * @returns {Chart} - Chart.js instance
     */
    createLearningRateChart(elementId, data = {}) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.iterations || [],
                datasets: [
                    {
                        label: 'Learning Rate',
                        data: data.learningRate || [],
                        borderColor: 'rgb(241, 143, 1)',
                        backgroundColor: 'rgba(241, 143, 1, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Learning Rate Schedule'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Learning Rate'
                        },
                        type: 'logarithmic'
                    }
                }
            }
        });
        
        this.charts[elementId] = chart;
        return chart;
    },
    
    /**
     * Create a performance chart
     * 
     * @param {string} elementId - ID of the canvas element
     * @param {Object} data - Chart data
     * @returns {Chart} - Chart.js instance
     */
    createPerformanceChart(elementId, data = {}) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.iterations || [],
                datasets: [
                    {
                        label: 'Tokens/Second',
                        data: data.tokensPerSec || [],
                        borderColor: 'rgb(40, 167, 69)',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Speed'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Tokens/Second'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
        
        this.charts[elementId] = chart;
        return chart;
    },
    
    /**
     * Create a memory usage chart
     * 
     * @param {string} elementId - ID of the canvas element
     * @param {Object} data - Chart data
     * @returns {Chart} - Chart.js instance
     */
    createMemoryChart(elementId, data = {}) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.iterations || [],
                datasets: [
                    {
                        label: 'Memory Usage (GB)',
                        data: data.memoryUsage || [],
                        borderColor: 'rgb(220, 53, 69)',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Memory Usage'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Memory (GB)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
        
        this.charts[elementId] = chart;
        return chart;
    },
    
    /**
     * Update a chart with new data
     * 
     * @param {string} elementId - ID of the canvas element
     * @param {Object} data - Chart data
     */
    updateChart(elementId, data = {}) {
        const chart = this.charts[elementId];
        
        if (!chart) {
            console.error(`Chart not found: ${elementId}`);
            return;
        }
        
        // Update labels
        if (data.iterations) {
            chart.data.labels = data.iterations;
        }
        
        // Update datasets based on chart type
        switch (chart.config.type) {
            case 'line':
                if (chart.data.datasets[0].label === 'Training Loss') {
                    // Loss chart
                    if (data.trainLoss) {
                        chart.data.datasets[0].data = data.trainLoss;
                    }
                    if (data.valLoss) {
                        chart.data.datasets[1].data = data.valLoss;
                    }
                } else if (chart.data.datasets[0].label === 'Training Perplexity') {
                    // Perplexity chart
                    if (data.trainLoss) {
                        chart.data.datasets[0].data = data.trainLoss.map(loss => Math.exp(loss));
                    }
                    if (data.valLoss) {
                        chart.data.datasets[1].data = data.valLoss.map(loss => Math.exp(loss));
                    }
                } else if (chart.data.datasets[0].label === 'Learning Rate') {
                    // Learning rate chart
                    if (data.learningRate) {
                        chart.data.datasets[0].data = data.learningRate;
                    }
                } else if (chart.data.datasets[0].label === 'Tokens/Second') {
                    // Performance chart
                    if (data.tokensPerSec) {
                        chart.data.datasets[0].data = data.tokensPerSec;
                    }
                } else if (chart.data.datasets[0].label === 'Memory Usage (GB)') {
                    // Memory chart
                    if (data.memoryUsage) {
                        chart.data.datasets[0].data = data.memoryUsage;
                    }
                }
                break;
        }
        
        // Update chart
        chart.update();
    },
    
    /**
     * Create a Plotly chart from data
     * 
     * @param {string} elementId - ID of the div element
     * @param {Object} chartData - Plotly chart data from API
     */
    createPlotlyChart(elementId, chartData) {
        if (!chartData) {
            return;
        }
        
        Plotly.newPlot(elementId, chartData.data, chartData.layout);
    }
}; 