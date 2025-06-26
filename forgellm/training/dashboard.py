"""
Training dashboard generator with comprehensive metrics visualization
"""

import json
import logging
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


def load_training_data(json_file: str) -> Dict[str, Any]:
    """
    Load training data from a JSON file
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Dictionary containing training data
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return {"error": str(e), "metrics": []}


class DashboardGenerator:
    """Generate comprehensive training dashboards with advanced metrics"""
    
    def __init__(self):
        """Initialize the dashboard generator"""
        self.fig = None
        self.axes = None
    
    def create_dashboard(
        self, 
        json_file: str, 
        output_dir: str = "training_dashboard", 
        output_name: str = "training_dashboard.png",
        dpi: int = 200,
        figsize: Tuple[int, int] = (16, 12)
    ) -> str:
        """
        Create a comprehensive training dashboard from a metrics JSON file
        
        Args:
            json_file: Path to the metrics JSON file
            output_dir: Directory to save the dashboard
            output_name: Name of the output file
            dpi: DPI for the output image
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to the generated dashboard image
        """
        # Load training data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract metrics
        metrics = data.get('metrics', [])
        config = data.get('config', {})
        
        if not metrics:
            logger.error("No metrics found in JSON file")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with grid layout for multiple plots
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(4, 3, figure=self.fig)
        
        # Extract data for plotting
        iterations = [m.get('iteration', 0) for m in metrics]
        train_loss = [m.get('train_loss', None) for m in metrics]
        val_loss = [m.get('val_loss', None) for m in metrics]
        learning_rate = [m.get('learning_rate', None) for m in metrics]
        tokens_per_sec = [m.get('tokens_per_sec', None) for m in metrics]
        peak_memory = [m.get('peak_memory_gb', None) for m in metrics]
        
        # Filter out None values for validation loss
        val_iterations = []
        val_loss_filtered = []
        for i, loss in zip(iterations, val_loss):
            if loss is not None:
                val_iterations.append(i)
                val_loss_filtered.append(loss)
        
        # Calculate perplexity
        train_ppl = [np.exp(loss) if loss and loss < 20 else float('nan') for loss in train_loss]
        val_ppl = [np.exp(loss) if loss and loss < 20 else float('nan') for loss in val_loss_filtered]
        
        # Plot 1: Training and Validation Loss
        ax1 = self.fig.add_subplot(gs[0, :2])
        self._create_loss_visualization(ax1, iterations, train_loss, val_iterations, val_loss_filtered)
        
        # Plot 2: Perplexity
        ax2 = self.fig.add_subplot(gs[0, 2])
        self._create_perplexity_visualization(ax2, iterations, train_ppl, val_iterations, val_ppl)
        
        # Plot 3: Learning Rate
        ax3 = self.fig.add_subplot(gs[1, 0])
        self._create_learning_rate_schedule(ax3, iterations, learning_rate, config)
        
        # Plot 4: Performance (Tokens/sec)
        ax4 = self.fig.add_subplot(gs[1, 1])
        self._create_performance_metrics(ax4, iterations, tokens_per_sec, 'speed')
        
        # Plot 5: Memory Usage
        ax5 = self.fig.add_subplot(gs[1, 2])
        self._create_performance_metrics(ax5, iterations, peak_memory, 'memory')
        
        # Plot 6: Loss Stability Analysis
        ax6 = self.fig.add_subplot(gs[2, 0])
        self._create_loss_stability_analysis(ax6, iterations, train_loss)
        
        # Plot 7: Overfitting Analysis
        ax7 = self.fig.add_subplot(gs[2, 1])
        self._create_overfitting_analysis(ax7, iterations, train_loss, val_iterations, val_loss_filtered)
        
        # Plot 8: Training Progress
        ax8 = self.fig.add_subplot(gs[2, 2])
        self._create_training_progress(ax8, iterations, config)
        
        # Plot 9: Configuration Summary
        ax9 = self.fig.add_subplot(gs[3, :])
        self._add_config_summary(ax9, config, metrics)
        
        # Add title
        model_name = config.get('model', 'Unknown Model')
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        self.fig.suptitle(f'Training Dashboard: {model_name}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        output_file = output_path / output_name
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Dashboard saved to {output_file}")
        return str(output_file)
    
    def _create_loss_visualization(self, ax, iterations, train_loss, val_iterations, val_loss):
        """Create loss visualization with trend lines"""
        # Plot raw data
        ax.plot(iterations, train_loss, 'b-', alpha=0.6, label='Train Loss')
        if val_iterations:
            ax.plot(val_iterations, val_loss, 'r-', alpha=0.6, label='Validation Loss')
        
        # Add trend lines if we have enough data
        if len(iterations) > 10:
            # Simple moving average for trend
            window = min(10, len(iterations) // 5)
            if window > 0:
                train_trend = np.convolve(np.array(train_loss), np.ones(window)/window, mode='valid')
                trend_x = iterations[window-1:]
                ax.plot(trend_x, train_trend, 'b-', linewidth=2, label='Train Trend')
                
                if len(val_iterations) > window:
                    val_trend = np.convolve(np.array(val_loss), np.ones(window)/window, mode='valid')
                    val_trend_x = val_iterations[window-1:]
                    ax.plot(val_trend_x, val_trend, 'r-', linewidth=2, label='Val Trend')
        
        ax.set_title('Loss Curves', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    def _create_perplexity_visualization(self, ax, iterations, train_ppl, val_iterations, val_ppl):
        """Create perplexity visualization"""
        ax.plot(iterations, train_ppl, 'b-', label='Train PPL')
        if val_iterations:
            ax.plot(val_iterations, val_ppl, 'r-', label='Validation PPL')
        ax.set_title('Perplexity', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    def _create_learning_rate_schedule(self, ax, iterations, learning_rate, config):
        """Create learning rate schedule visualization with theoretical curve"""
        # Plot actual learning rates
        ax.plot(iterations, learning_rate, 'g-', label='Actual LR')
        
        # Plot theoretical schedule if we have config
        if config:
            lr_schedule = config.get('lr_schedule')
            if isinstance(lr_schedule, str):
                schedule_name = lr_schedule
            elif isinstance(lr_schedule, dict):
                schedule_name = lr_schedule.get('name', 'unknown')
            else:
                schedule_name = 'unknown'
                
            base_lr = config.get('learning_rate', 0.0)
            max_iterations = config.get('max_iterations', max(iterations) if iterations else 0)
            warmup_steps = config.get('warmup_steps', 0)
            lr_decay_factor = config.get('lr_decay_factor', 0.1)
            
            # Generate theoretical curve
            x = np.linspace(0, max_iterations, 100)
            y = np.zeros_like(x)
            
            # Apply warmup
            mask_warmup = x < warmup_steps
            if warmup_steps > 0:
                y[mask_warmup] = base_lr * (x[mask_warmup] / warmup_steps)
            
            # Apply schedule
            mask_schedule = x >= warmup_steps
            if schedule_name == 'cosine_decay':
                # Cosine decay from base_lr to base_lr * decay_factor
                progress = (x[mask_schedule] - warmup_steps) / (max_iterations - warmup_steps)
                y[mask_schedule] = base_lr * (lr_decay_factor + (1 - lr_decay_factor) * 
                                           (1 + np.cos(np.pi * progress)) / 2)
            elif schedule_name == 'linear_decay':
                # Linear decay from base_lr to base_lr * decay_factor
                progress = (x[mask_schedule] - warmup_steps) / (max_iterations - warmup_steps)
                y[mask_schedule] = base_lr * (1 - (1 - lr_decay_factor) * progress)
            else:
                # Constant schedule
                y[mask_schedule] = base_lr
            
            # Plot theoretical schedule
            ax.plot(x, y, 'g--', alpha=0.6, label='Theoretical')
            
            # Add schedule name to title
            ax.set_title(f'Learning Rate ({schedule_name})', fontsize=14)
        else:
            ax.set_title('Learning Rate', fontsize=14)
            
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        if config:
            ax.legend()
    
    def _create_performance_metrics(self, ax, iterations, metrics_data, metric_type='speed'):
        """Create performance metrics visualization"""
        if metric_type == 'speed':
            ax.plot(iterations, metrics_data, 'c-')
            ax.set_title('Training Speed', fontsize=14)
            ax.set_ylabel('Tokens/sec', fontsize=12)
            # Add fill below curve
            ax.fill_between(iterations, 0, metrics_data, alpha=0.2, color='c')
        else:  # memory
            ax.plot(iterations, metrics_data, 'r-')
            ax.set_title('Memory Usage', fontsize=14)
            ax.set_ylabel('Memory (GB)', fontsize=12)
            # Add fill below curve
            ax.fill_between(iterations, 0, metrics_data, alpha=0.2, color='r')
            
        ax.set_xlabel('Iterations', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _create_loss_stability_analysis(self, ax, iterations, train_loss):
        """Create loss stability analysis visualization"""
        if len(iterations) < 10:
            ax.text(0.5, 0.5, "Not enough data\nfor stability analysis", 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Loss Stability', fontsize=14)
            ax.axis('off')
            return
            
        # Calculate rolling variance with window size
        window = min(10, len(iterations) // 5)
        if window < 2:
            window = 2
            
        variances = []
        for i in range(window, len(train_loss)):
            window_data = train_loss[i-window:i]
            if all(x is not None for x in window_data):
                variances.append(np.var(window_data))
            else:
                variances.append(np.nan)
                
        # Prepend NaNs for alignment with iterations
        variances = [np.nan] * window + variances
        
        # Plot variance
        ax.plot(iterations, variances, 'm-')
        ax.set_title('Loss Stability (lower is better)', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Loss Variance', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add color bands for stability assessment
        ax_ylim = ax.get_ylim()
        if ax_ylim[1] > 0:
            # Create color bands
            ax.axhspan(0, 0.01, alpha=0.2, color='green', label='Excellent')
            ax.axhspan(0.01, 0.05, alpha=0.2, color='yellow', label='Good')
            ax.axhspan(0.05, ax_ylim[1], alpha=0.2, color='red', label='Unstable')
            ax.legend(loc='upper right')
    
    def _create_overfitting_analysis(self, ax, iterations, train_loss, val_iterations, val_loss):
        """Create overfitting analysis visualization"""
        if not val_iterations or len(val_iterations) < 2:
            ax.text(0.5, 0.5, "Not enough validation data\nfor overfitting analysis", 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Overfitting Analysis', fontsize=14)
            ax.axis('off')
            return
            
        # Calculate generalization gap (val_loss - train_loss)
        gen_gaps = []
        gap_iterations = []
        
        for i, val_iter in enumerate(val_iterations):
            # Find closest train iteration
            train_idx = np.argmin(np.abs(np.array(iterations) - val_iter))
            if train_idx < len(train_loss) and train_loss[train_idx] is not None and val_loss[i] is not None:
                gen_gaps.append(val_loss[i] - train_loss[train_idx])
                gap_iterations.append(val_iter)
        
        if not gen_gaps:
            ax.text(0.5, 0.5, "Could not calculate\ngeneralization gap", 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Overfitting Analysis', fontsize=14)
            ax.axis('off')
            return
            
        # Plot generalization gap
        ax.plot(gap_iterations, gen_gaps, 'purple')
        ax.set_title('Generalization Gap (Val - Train)', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Gap', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference line at 0
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add color bands for overfitting assessment
        ax_ylim = ax.get_ylim()
        min_y, max_y = ax_ylim
        mid_point = (min_y + max_y) / 2
        
        # Create color bands
        if min_y < 0 and max_y > 0:
            ax.axhspan(min_y, -0.1, alpha=0.2, color='green', label='Underfitting')
            ax.axhspan(-0.1, 0.1, alpha=0.2, color='blue', label='Good fit')
            ax.axhspan(0.1, max_y, alpha=0.2, color='red', label='Overfitting')
            ax.legend(loc='upper right')
    
    def _create_training_progress(self, ax, iterations, config):
        """Create training progress visualization"""
        max_iterations = config.get('max_iterations', max(iterations))
        progress = (max(iterations) / max_iterations) * 100
        
        # Create progress bar
        ax.barh(['Progress'], [progress], color='blue', height=0.5)
        ax.set_title('Training Progress', fontsize=14)
        ax.set_xlabel('Percent Complete', fontsize=12)
        ax.set_xlim([0, 100])
        
        # Add percentage text
        ax.text(progress/2, 0, f"{progress:.1f}%", ha='center', va='center', 
               fontsize=14, color='white', fontweight='bold')
        
        # Remove y-axis ticks
        ax.set_yticks([])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    def _add_config_summary(self, ax, config: Dict[str, Any], metrics: List[Dict[str, Any]]):
        """Add configuration summary to the dashboard"""
        ax.axis('off')
        
        # Prepare text
        model_name = config.get('model', 'Unknown Model')
        batch_size = config.get('batch_size', 'N/A')
        learning_rate = config.get('learning_rate', 'N/A')
        max_iterations = config.get('max_iterations', 'N/A')
        fine_tune_type = config.get('fine_tune_type', 'N/A')
        max_seq_length = config.get('max_seq_length', 'N/A')
        lr_schedule = config.get('lr_schedule', 'N/A')
        if isinstance(lr_schedule, dict):
            lr_schedule = lr_schedule.get('name', 'N/A')
        warmup_steps = config.get('warmup_steps', 'N/A')
        lr_decay_factor = config.get('lr_decay_factor', 'N/A')
        
        # Get latest metrics
        latest_metrics = metrics[-1] if metrics else {}
        current_iteration = latest_metrics.get('iteration', 0)
        latest_train_loss = latest_metrics.get('train_loss', 'N/A')
        if latest_train_loss != 'N/A':
            latest_train_ppl = math.exp(latest_train_loss) if latest_train_loss < 20 else 'N/A'
        else:
            latest_train_ppl = 'N/A'
            
        latest_val_loss = latest_metrics.get('val_loss', 'N/A')
        if latest_val_loss != 'N/A':
            latest_val_ppl = math.exp(latest_val_loss) if latest_val_loss < 20 else 'N/A'
        else:
            latest_val_ppl = 'N/A'
        
        # Find best validation loss
        best_val_loss = float('inf')
        best_iteration = 0
        for m in metrics:
            val_loss = m.get('val_loss')
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = m.get('iteration', 0)
        
        if best_val_loss == float('inf'):
            best_val_loss = 'N/A'
            best_val_ppl = 'N/A'
        else:
            best_val_ppl = math.exp(best_val_loss) if best_val_loss < 20 else 'N/A'
        
        # Calculate training time
        training_time = 'N/A'
        if len(metrics) >= 2:
            try:
                start_time = metrics[0].get('timestamp')
                end_time = metrics[-1].get('timestamp')
                if start_time and end_time:
                    from datetime import datetime
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    duration = end_dt - start_dt
                    hours = duration.total_seconds() / 3600
                    training_time = f"{hours:.1f} hours"
            except Exception as e:
                logger.warning(f"Failed to calculate training time: {e}")
        
        # Create summary text
        summary = (
            f"Model: {model_name}\n\n"
            f"Training Configuration:\n"
            f"- Batch Size: {batch_size}\n"
            f"- Learning Rate: {learning_rate}\n"
            f"- Max Iterations: {max_iterations}\n"
            f"- Fine-tune Type: {fine_tune_type}\n"
            f"- Max Sequence Length: {max_seq_length}\n"
            f"- LR Schedule: {lr_schedule}\n"
            f"- Warmup Steps: {warmup_steps}\n"
            f"- LR Decay Factor: {lr_decay_factor}\n\n"
            f"Training Progress:\n"
            f"- Current Iteration: {current_iteration} / {max_iterations}\n"
            f"- Training Time: {training_time}\n"
            f"- Latest Train Loss: {latest_train_loss} (PPL: {latest_train_ppl})\n"
            f"- Latest Val Loss: {latest_val_loss} (PPL: {latest_val_ppl})\n"
            f"- Best Val Loss: {best_val_loss} (PPL: {best_val_ppl}, iteration {best_iteration})"
        )
        
        # Add summary text with nice formatting
        ax.text(0.5, 0.5, summary, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def identify_best_checkpoints(self, data, top_k: int = 3):
        """
        Identify the best checkpoints based on validation loss and other metrics
        
        Args:
            data: Training data dictionary
            top_k: Number of top checkpoints to return
            
        Returns:
            List of best checkpoint dictionaries
        """
        metrics = data.get('metrics', [])
        if not metrics:
            return []
            
        # Filter metrics with validation loss
        val_metrics = [m for m in metrics if m.get('val_loss') is not None]
        if not val_metrics:
            return []
            
        # Sort by validation loss (ascending)
        val_metrics.sort(key=lambda x: x.get('val_loss', float('inf')))
        
        # Take top K
        best_checkpoints = []
        for i, metric in enumerate(val_metrics[:top_k]):
            checkpoint = {
                'iteration': metric.get('iteration', 0),
                'val_loss': metric.get('val_loss'),
                'train_loss': metric.get('train_loss'),
                'rank': i + 1
            }
            
            # Calculate perplexity
            if checkpoint['val_loss'] is not None:
                checkpoint['val_perplexity'] = math.exp(checkpoint['val_loss']) if checkpoint['val_loss'] < 20 else float('inf')
            
            if checkpoint['train_loss'] is not None:
                checkpoint['train_perplexity'] = math.exp(checkpoint['train_loss']) if checkpoint['train_loss'] < 20 else float('inf')
            
            # Calculate generalization gap
            if checkpoint['val_loss'] is not None and checkpoint['train_loss'] is not None:
                checkpoint['generalization_gap'] = checkpoint['val_loss'] - checkpoint['train_loss']
            
            # Generate selection reason
            if i == 0:
                checkpoint['selection_reason'] = "Lowest validation loss"
            elif i == 1:
                checkpoint['selection_reason'] = "Second lowest validation loss"
            else:
                checkpoint['selection_reason'] = f"Top {i+1} validation loss"
                
            best_checkpoints.append(checkpoint)
            
        return best_checkpoints


def create_comprehensive_dashboard(
    json_file: str, 
    output_dir: str = "training_dashboard", 
    output_name: str = "training_dashboard.png"
) -> str:
    """
    Create a comprehensive training dashboard from a metrics JSON file
    
    Args:
        json_file: Path to the metrics JSON file
        output_dir: Directory to save the dashboard
        output_name: Name of the output file
        
    Returns:
        Path to the generated dashboard image
    """
    generator = DashboardGenerator()
    return generator.create_dashboard(json_file, output_dir, output_name)


def identify_best_checkpoints(data, top_k: int = 3):
    """
    Identify the best checkpoints based on validation loss and other metrics
    
    Args:
        data: Training data dictionary
        top_k: Number of top checkpoints to return
        
    Returns:
        List of best checkpoint dictionaries
    """
    generator = DashboardGenerator()
    return generator.identify_best_checkpoints(data, top_k)


__all__ = [
    "create_comprehensive_dashboard",
    "identify_best_checkpoints",
    "load_training_data"
] 