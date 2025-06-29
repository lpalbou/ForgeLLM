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
        # Configure matplotlib to use non-GUI backend for thread safety
        import matplotlib
        matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
        
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
        
        # Filter out None values for training loss
        train_iterations = []
        train_loss_filtered = []
        for i, loss in zip(iterations, train_loss):
            if loss is not None:
                train_iterations.append(i)
                train_loss_filtered.append(loss)
        
        # Filter out None values for validation loss
        val_iterations = []
        val_loss_filtered = []
        for i, loss in zip(iterations, val_loss):
            if loss is not None:
                val_iterations.append(i)
                val_loss_filtered.append(loss)
        
        # Calculate perplexity with 3 decimal precision
        train_ppl = [round(np.exp(loss), 3) if loss and loss < 20 else float('nan') for loss in train_loss_filtered]
        val_ppl = [round(np.exp(loss), 3) if loss and loss < 20 else float('nan') for loss in val_loss_filtered]
        
        # Plot 1: Training and Validation Loss
        ax1 = self.fig.add_subplot(gs[0, :2])
        self._create_loss_visualization(ax1, train_iterations, train_loss_filtered, val_iterations, val_loss_filtered)
        
        # Plot 2: Perplexity
        ax2 = self.fig.add_subplot(gs[0, 2])
        self._create_perplexity_visualization(ax2, train_iterations, train_ppl, val_iterations, val_ppl)
        
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
        self._create_loss_stability_analysis(ax6, train_iterations, train_loss_filtered)
        
        # Plot 7: Overfitting Analysis
        ax7 = self.fig.add_subplot(gs[2, 1])
        self._create_overfitting_analysis(ax7, train_iterations, train_loss_filtered, val_iterations, val_loss_filtered)
        
        # Plot 8: Training Progress
        ax8 = self.fig.add_subplot(gs[2, 2])
        self._create_training_progress(ax8, iterations, config)
        
        # Plot 9: Configuration Summary
        ax9 = self.fig.add_subplot(gs[3, :])
        self._add_config_summary(ax9, config, metrics)
        
        # Add title
        model_name = data.get('model_name') or config.get('model', 'Unknown Model')
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        self._model_name = model_name  # Store for config summary
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
        ax.plot(iterations, train_ppl, 'b-', label='Train Perplexity')
        if val_iterations:
            ax.plot(val_iterations, val_ppl, 'r-', label='Validation Perplexity')
        ax.set_title('Perplexity', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    def _create_learning_rate_schedule(self, ax, iterations, learning_rate, config):
        """Create learning rate schedule visualization with theoretical curve"""
        # Filter out None values for learning rate
        filtered_iterations = []
        filtered_lr = []
        for i, lr in zip(iterations, learning_rate):
            if lr is not None:
                filtered_iterations.append(i)
                filtered_lr.append(lr)
        
        # Plot actual learning rates
        if filtered_lr:
            ax.plot(filtered_iterations, filtered_lr, 'g-', label='Actual LR')
        else:
            ax.text(0.5, 0.5, "No learning rate data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Learning Rate', fontsize=14)
            ax.axis('off')
            return
        
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
        # Filter out None values
        filtered_iterations = []
        filtered_data = []
        for i, data in zip(iterations, metrics_data):
            if data is not None:
                filtered_iterations.append(i)
                filtered_data.append(data)
        
        if not filtered_data:
            ax.text(0.5, 0.5, f"No {metric_type} data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'Training Speed' if metric_type == 'speed' else 'Memory Usage', fontsize=14)
            ax.axis('off')
            return
        
        if metric_type == 'speed':
            ax.plot(filtered_iterations, filtered_data, 'c-')
            ax.set_title('Training Speed', fontsize=14)
            ax.set_ylabel('Tokens/sec', fontsize=12)
            # Add fill below curve
            ax.fill_between(filtered_iterations, 0, filtered_data, alpha=0.2, color='c')
        else:  # memory
            ax.plot(filtered_iterations, filtered_data, 'r-')
            ax.set_title('Memory Usage', fontsize=14)
            ax.set_ylabel('Memory (GB)', fontsize=12)
            # Add fill below curve
            ax.fill_between(filtered_iterations, 0, filtered_data, alpha=0.2, color='r')
            
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
            variances.append(np.var(window_data))
                
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
            if train_idx < len(train_loss):
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
        
        # Prepare text - look for model_name in parent data first
        model_name = getattr(self, '_model_name', None) or config.get('model', 'Unknown Model')
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
            latest_train_ppl = round(math.exp(latest_train_loss), 3) if latest_train_loss < 20 else 'N/A'
        else:
            latest_train_ppl = 'N/A'
            
        latest_val_loss = latest_metrics.get('val_loss', 'N/A')
        if latest_val_loss != 'N/A':
            latest_val_ppl = round(math.exp(latest_val_loss), 3) if latest_val_loss < 20 else 'N/A'
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
            best_val_ppl = round(math.exp(best_val_loss), 3) if best_val_loss < 20 else 'N/A'
        
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
            f"- Latest Train Loss: {latest_train_loss} (Perplexity: {latest_train_ppl})\n"
            f"- Latest Val Loss: {latest_val_loss} (Perplexity: {latest_val_ppl})\n"
            f"- Best Val Loss: {best_val_loss} (Perplexity: {best_val_ppl}, iteration {best_iteration})"
        )
        
        # Add summary text with nice formatting
        ax.text(0.5, 0.5, summary, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def identify_best_checkpoints(self, data, top_k: int = 3):
        """
        Identify the best checkpoints using a multi-metric composite score.
        
        This uses a sophisticated approach that considers:
        1. Validation loss (lower is better) - weight 0.50
        2. Generalization gap (val_loss - train_loss) - weight 0.20
        3. Loss stability (higher is better) - weight 0.20
        4. Convergence trend - weight 0.10
        
        Args:
            data: Training data dictionary
            top_k: Number of best checkpoints to return
            
        Returns:
            List of best checkpoints with scores and selection reasons
        """
        metrics = data.get('metrics', [])
        if not metrics:
            return []
            
        max_iterations = data.get('config', {}).get('max_iterations') or metrics[-1].get('iteration', 0)
        
        # Collect candidate checkpoints (those where we logged val_loss)
        candidates = []
        for m in metrics:
            if m.get('val_loss') is None or m.get('iteration') is None:
                continue
                
            val = m.get('val_loss')
            train = m.get('train_loss')
            
            if val is None or train is None:
                continue
                
            entry = {
                'iteration': m['iteration'],
                'val_loss': m['val_loss'],
                'train_loss': train,
                'rank': 0  # Will be set later
            }
            
            # Extract checkpoint path from compound string
            checkpoint_path = m.get('checkpoint_path')
            if checkpoint_path:
                # The checkpoint_path contains both paths, extract the numbered one
                # e.g., "models/cpt/.../adapters.safetensors and models/cpt/.../0000025_adapters.safetensors."
                parts = checkpoint_path.split(' and ')
                for part in parts:
                    part = part.rstrip('.')  # Remove trailing period
                    if f"{m['iteration']:07d}_adapters.safetensors" in part:
                        entry['path'] = part
                        break
                else:
                    # Fallback: use the last part if no numbered match found
                    if parts:
                        entry['path'] = parts[-1].rstrip('.')
                    else:
                        entry['path'] = checkpoint_path.rstrip('.')
            else:
                entry['path'] = None
            
            # Calculate perplexity
            entry['val_perplexity'] = math.exp(val) if val < 20 else float('inf')
            entry['train_perplexity'] = math.exp(train) if train < 20 else float('inf')
            
            # Compute generalization gap
            entry['generalization_gap'] = entry['val_loss'] - entry['train_loss']
            
            # Calculate stability score
            entry['stability_score'] = self._calculate_loss_stability(metrics, entry['iteration'])
            
            # Calculate training progress
            entry['training_progress'] = entry['iteration'] / max_iterations if max_iterations else 0
            
            # Calculate convergence score
            entry['convergence_score'] = self._calculate_convergence_score(metrics, entry['iteration'])
            
            candidates.append(entry)
        
        if not candidates:
            return []
        
        # Compute rank for each metric
        def assign_rank(key, reverse=False):
            sorted_cand = sorted(candidates, key=lambda c: c[key], reverse=reverse)
            for rank, c in enumerate(sorted_cand, 1):
                c[f'rank_{key}'] = rank
        
        assign_rank('val_loss', reverse=False)  # lower is better
        assign_rank('generalization_gap', reverse=False)  # lower is better
        assign_rank('stability_score', reverse=True)  # higher is better
        assign_rank('convergence_score', reverse=True)  # higher is better
        
        # Composite score with weights
        weights = {
            'val_loss': 0.50,
            'generalization_gap': 0.20,
            'stability_score': 0.20,
            'convergence_score': 0.10
        }
        
        for c in candidates:
            c['composite_score'] = (
                weights['val_loss'] * c['rank_val_loss'] +
                weights['generalization_gap'] * c['rank_generalization_gap'] +
                weights['stability_score'] * c['rank_stability_score'] +
                weights['convergence_score'] * c['rank_convergence_score']
            )
        
        # Apply mild recency bias
        for c in candidates:
            # Slight preference for more recent checkpoints when scores are close
            progress_bonus = 0.05 * c['training_progress']
            c['final_score'] = c['composite_score'] - progress_bonus
        
        # Sort by final score (lower is better)
        sorted_candidates = sorted(candidates, key=lambda c: c['final_score'])
        
        # Set rank and generate selection reason for each checkpoint
        for i, c in enumerate(sorted_candidates[:top_k]):
            c['rank'] = i + 1
            c['selection_reason'] = self._generate_selection_reason(c)
        
        return sorted_candidates[:top_k]
        
    def _calculate_loss_stability(self, metrics, target_iteration, window_size=5):
        """
        Calculate loss stability around a checkpoint
        
        Args:
            metrics: List of metrics dictionaries
            target_iteration: Target iteration to calculate stability for
            window_size: Window size for stability calculation
            
        Returns:
            Stability score (higher is more stable)
        """
        # Find metrics within window
        window_metrics = []
        for m in metrics:
            if m.get('iteration') is None or m.get('train_loss') is None:
                continue
                
            iteration = m['iteration']
            if abs(iteration - target_iteration) <= window_size:
                window_metrics.append((iteration, m['train_loss']))
        
        if len(window_metrics) < 3:  # Need at least 3 points for meaningful stability
            return 0.0
            
        # Sort by iteration
        window_metrics.sort(key=lambda x: x[0])
        
        # Calculate variance of differences
        diffs = [abs(b[1] - a[1]) for a, b in zip(window_metrics[:-1], window_metrics[1:])]
        if not diffs:
            return 0.0
            
        # Lower variance means more stable (invert for higher=better)
        variance = np.var(diffs) if diffs else 0
        stability = 1.0 / (1.0 + variance)
        
        return stability
        
    def _calculate_convergence_score(self, metrics, target_iteration, window_size=10):
        """
        Calculate convergence trend around a checkpoint
        
        Args:
            metrics: List of metrics dictionaries
            target_iteration: Target iteration to calculate convergence for
            window_size: Window size for convergence calculation
            
        Returns:
            Convergence score (higher is better convergence)
        """
        # Find metrics before target within window
        before_metrics = []
        for m in metrics:
            if m.get('iteration') is None or m.get('train_loss') is None:
                continue
                
            iteration = m['iteration']
            if iteration < target_iteration and target_iteration - iteration <= window_size:
                before_metrics.append((iteration, m['train_loss']))
        
        if len(before_metrics) < 3:  # Need at least 3 points for meaningful trend
            return 0.0
            
        # Sort by iteration
        before_metrics.sort(key=lambda x: x[0])
        
        # Calculate slope of loss curve
        iterations = [x[0] for x in before_metrics]
        losses = [x[1] for x in before_metrics]
        
        # Simple linear regression
        mean_iter = np.mean(iterations)
        mean_loss = np.mean(losses)
        numerator = sum((x - mean_iter) * (y - mean_loss) for x, y in zip(iterations, losses))
        denominator = sum((x - mean_iter) ** 2 for x in iterations)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Negative slope means decreasing loss (good)
        # Convert to a score where higher is better
        convergence_score = -slope * 100  # Scale for readability
        
        # Cap at reasonable values
        return max(0.0, min(10.0, convergence_score))
        
    def _generate_selection_reason(self, checkpoint):
        """
        Generate a human-readable reason for checkpoint selection
        
        Args:
            checkpoint: Checkpoint dictionary
            
        Returns:
            Selection reason string
        """
        reasons = []
        rank = checkpoint.get('rank', 0)
        
        # Primary reason based on rank
        if rank == 1:
            reasons.append("Best overall checkpoint with optimal balance of metrics")
        elif rank == 2:
            reasons.append("Strong runner-up with excellent validation performance")
        elif rank == 3:
            reasons.append("Solid third choice with good generalization properties")
        else:
            reasons.append(f"Rank {rank} checkpoint with good overall metrics")
        
        # Add specific strengths
        if checkpoint.get('rank_val_loss') == 1:
            reasons.append("Lowest validation loss")
        elif checkpoint.get('rank_val_loss') <= 3:
            reasons.append("Very low validation loss")
            
        if checkpoint.get('rank_generalization_gap') == 1:
            reasons.append("Smallest generalization gap")
        elif checkpoint.get('rank_generalization_gap') <= 3:
            reasons.append("Small generalization gap")
            
        if checkpoint.get('rank_stability_score') == 1:
            reasons.append("Highest loss stability")
        elif checkpoint.get('rank_stability_score') <= 3:
            reasons.append("Very stable loss curve")
            
        if checkpoint.get('rank_convergence_score') == 1:
            reasons.append("Best convergence trend")
        elif checkpoint.get('rank_convergence_score') <= 3:
            reasons.append("Strong convergence trend")
        
        # Return formatted reason
        return f"{reasons[0]}. " + (f"Notable for: {', '.join(reasons[1:])}" if len(reasons) > 1 else "")


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
    Identify the best checkpoints from training data using a multi-metric composite score.
    
    This uses a sophisticated approach that considers:
    1. Validation loss (lower is better) - weight 0.50
    2. Generalization gap (val_loss - train_loss) - weight 0.20
    3. Loss stability (higher is better) - weight 0.20
    4. Convergence trend - weight 0.10
    
    Args:
        data: Training data dictionary
        top_k: Number of best checkpoints to return
        
    Returns:
        List of best checkpoints with scores and selection reasons
    """
    # Use the DashboardGenerator's implementation for consistency
    generator = DashboardGenerator()
    return generator.identify_best_checkpoints(data, top_k)


def generate_web_chart_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate chart data for web dashboard from training metrics
    
    Args:
        data: Training data containing metrics
        
    Returns:
        Dictionary containing chart data for Plotly.js
    """
    try:
        metrics = data.get('metrics', [])
        if not metrics:
            return {}
        
        # Extract data arrays
        iterations = [m.get('iteration', 0) for m in metrics]
        train_loss = [m.get('train_loss') for m in metrics if m.get('train_loss') is not None]
        val_loss = [m.get('val_loss') for m in metrics if m.get('val_loss') is not None]
        learning_rate = [m.get('learning_rate') for m in metrics if m.get('learning_rate') is not None]
        tokens_per_sec = [m.get('tokens_per_sec') for m in metrics if m.get('tokens_per_sec') is not None]
        peak_memory = [m.get('peak_memory_gb') for m in metrics if m.get('peak_memory_gb') is not None]
        
        # Get iterations for each metric (some might be sparse)
        train_iterations = [m.get('iteration', 0) for m in metrics if m.get('train_loss') is not None]
        val_iterations = [m.get('iteration', 0) for m in metrics if m.get('val_loss') is not None]
        lr_iterations = [m.get('iteration', 0) for m in metrics if m.get('learning_rate') is not None]
        speed_iterations = [m.get('iteration', 0) for m in metrics if m.get('tokens_per_sec') is not None]
        memory_iterations = [m.get('iteration', 0) for m in metrics if m.get('peak_memory_gb') is not None]
        
        charts = {}
        
        # Loss Chart
        if train_loss or val_loss:
            loss_data = []
            if train_loss:
                loss_data.append({
                    'x': train_iterations,
                    'y': train_loss,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Training Loss',
                    'line': {'color': '#2E86AB', 'width': 2},
                    'marker': {'size': 4}
                })
            if val_loss:
                loss_data.append({
                    'x': val_iterations,
                    'y': val_loss,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Validation Loss',
                    'line': {'color': '#F24236', 'width': 2},
                    'marker': {'size': 4}
                })
            
            charts['loss'] = {
                'data': loss_data,
                'layout': {
                    'title': 'Training Loss',
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Loss'},
                    'showlegend': True,
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
                }
            }
        
        # Perplexity Chart
        if train_loss or val_loss:
            ppl_data = []
            if train_loss:
                train_ppl = [math.exp(min(loss, 20)) for loss in train_loss]  # Cap at exp(20) to avoid overflow
                ppl_data.append({
                    'x': train_iterations,
                    'y': train_ppl,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Training Perplexity',
                    'line': {'color': '#2E86AB', 'width': 2},
                    'marker': {'size': 4}
                })
            if val_loss:
                val_ppl = [math.exp(min(loss, 20)) for loss in val_loss]  # Cap at exp(20) to avoid overflow
                ppl_data.append({
                    'x': val_iterations,
                    'y': val_ppl,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Validation Perplexity',
                    'line': {'color': '#F24236', 'width': 2},
                    'marker': {'size': 4}
                })
            
            charts['perplexity'] = {
                'data': ppl_data,
                'layout': {
                    'title': 'Perplexity',
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Perplexity', 'type': 'log'},
                    'showlegend': True,
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
                }
            }
        
        # Learning Rate Chart
        if learning_rate:
            charts['learning_rate'] = {
                'data': [{
                    'x': lr_iterations,
                    'y': learning_rate,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Learning Rate',
                    'line': {'color': '#A23B72', 'width': 2},
                    'marker': {'size': 4}
                }],
                'layout': {
                    'title': 'Learning Rate',
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Learning Rate', 'type': 'log'},
                    'showlegend': False,
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
                }
            }
        
        # Speed Chart
        if tokens_per_sec:
            speed_data = []
            speed_data.append({
                'x': speed_iterations,
                'y': tokens_per_sec,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Tokens/sec',
                'line': {'color': '#F18F01', 'width': 2},
                'marker': {'size': 4}
            })
            
            if peak_memory:
                # Add memory as secondary y-axis
                speed_data.append({
                    'x': memory_iterations,
                    'y': peak_memory,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Memory (GB)',
                    'line': {'color': '#C73E1D', 'width': 2},
                    'marker': {'size': 4},
                    'yaxis': 'y2'
                })
            
            charts['speed'] = {
                'data': speed_data,
                'layout': {
                    'title': 'Performance Metrics',
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Tokens/sec'},
                    'yaxis2': {
                        'title': 'Memory (GB)',
                        'overlaying': 'y',
                        'side': 'right'
                    } if peak_memory else None,
                    'showlegend': True,
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
                }
            }
        
        return charts
        
    except Exception as e:
        logger.error(f"Error generating web chart data: {e}")
        return {}


__all__ = [
    "create_comprehensive_dashboard",
    "identify_best_checkpoints",
    "load_training_data"
] 