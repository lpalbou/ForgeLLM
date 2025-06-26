"""Training monitoring and metrics tracking."""

import time
import math
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training progress with overfitting detection and early stopping."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize training monitor.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.train_loss_history = []
        self.valid_loss_history = []
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        self.start_time = time.time()
        self.iteration_times = []
        
    def log_metrics(
        self,
        iteration: int,
        train_loss: float,
        valid_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        tokens_per_sec: float = 0.0,
        memory_gb: float = 0.0,
    ) -> None:
        """Log comprehensive training metrics with time estimates.
        
        Args:
            iteration: Current iteration
            train_loss: Training loss
            valid_loss: Validation loss
            learning_rate: Current learning rate
            tokens_per_sec: Training speed in tokens per second
            memory_gb: Peak memory usage in GB
        """
        # Record metrics
        self.train_loss_history.append(train_loss)

        # Only track validation loss when provided
        if valid_loss is not None and not (isinstance(valid_loss, float) and math.isnan(valid_loss)):
            self.valid_loss_history.append(valid_loss)

        # Compute perplexity cheaply (avoid math range error on huge losses)
        ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
        
        # Calculate timing
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Use overall elapsed average for ETA for a more stable estimate
        if iteration > 0:
            avg_iter_time = elapsed / iteration
            remaining_iters = self.config.max_iterations - iteration
            eta_seconds = remaining_iters * avg_iter_time
            eta_minutes = int(eta_seconds // 60)
            # Format with thousands separator using spaces (e.g., 1 800)
            eta_formatted = format(eta_minutes, ",").replace(",", " ")
            eta_str = f"{eta_formatted}mn"
        else:
            eta_str = "calculating..."
            
        # Calculate progress
        progress = (iteration / self.config.max_iterations) * 100
        
        valid_display = f"{valid_loss:.4f}" if valid_loss is not None else "—"

        logger.info(
            f"🚀 Iter {iteration:>6}/{self.config.max_iterations} ({progress:>5.1f}%) | "
            f"📉 Train: {train_loss:.4f} (ppl {ppl:,.1f}) | 📊 Valid: {valid_display} | "
            f"📈 LR: {learning_rate:.2e} | ⚡ {tokens_per_sec:.0f} tok/s | "
            f"💾 {memory_gb:.1f}GB | ⏱️  ETA: {eta_str}"
        )
        
        # Record iteration time for ETA calculation
        if hasattr(self, '_last_log_time'):
            iter_time = current_time - self._last_log_time
            self.iteration_times.append(iter_time)
        self._last_log_time = current_time
        
    def check_overfitting(self) -> bool:
        """Check if model is overfitting using validation loss increase threshold.
        
        Returns:
            True if overfitting detected, False otherwise
        """
        if len(self.valid_loss_history) < 3:
            return False
            
        # Check if validation loss has increased by more than threshold
        recent_valid_loss = self.valid_loss_history[-1]
        best_valid_loss = min(self.valid_loss_history)
        
        if best_valid_loss > 0:  # Avoid division by zero
            loss_increase_ratio = (recent_valid_loss - best_valid_loss) / best_valid_loss
            
            if loss_increase_ratio > self.config.overfitting_threshold:
                logger.warning(f"🚨 OVERFITTING DETECTED! Validation loss increased by {loss_increase_ratio:.1%} "
                             f"(threshold: {self.config.overfitting_threshold:.1%})")
                return True
                
        return False
    
    def should_stop_early(self, current_valid_loss: float) -> bool:
        """Return True when training should stop early.

        The method always *tracks* validation loss and patience, but it only
        instructs the caller to stop when early-stopping is **enabled** and one
        of the stop criteria is met (no improvement for N checks or clear
        over-fitting).
        
        Args:
            current_valid_loss: Current validation loss
            
        Returns:
            True if training should stop early, False otherwise
        """
        # Always keep track of best validation loss & patience so that, if the
        # user enables early-stopping part-way through training, the historical
        # statistics are still meaningful.
        if current_valid_loss < self.best_valid_loss - self.config.min_loss_improvement:
            self.best_valid_loss = current_valid_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # If early-stopping is disabled, never request termination – just
        # return False after book-keeping above.
        if not self.config.enable_early_stopping:
            return False

        # --- Below this point early-stopping is enabled ---

        # Check blatant over-fitting first.
        if self.check_overfitting():
            logger.warning("🛑 Early stopping due to overfitting")
            return True

        # Check patience window.
        if self.patience_counter >= self.config.early_stopping_patience:
            logger.warning(
                "🛑 Early stopping due to no improvement for %d checks",
                self.patience_counter,
            )
            return True

        return False
    
    def log_final_summary(self) -> Dict[str, Any]:
        """Log comprehensive final training summary.
        
        Returns:
            Dictionary with summary statistics
        """
        total_time = time.time() - self.start_time
        
        if len(self.train_loss_history) >= 2:
            initial_train_loss = self.train_loss_history[0]
            final_train_loss = self.train_loss_history[-1]
            if initial_train_loss > 0:
                train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
            else:
                train_improvement = 0
        else:
            train_improvement = 0
            
        if len(self.valid_loss_history) >= 2:
            initial_valid_loss = self.valid_loss_history[0]
            final_valid_loss = self.valid_loss_history[-1]
            if initial_valid_loss > 0:
                valid_improvement = ((initial_valid_loss - final_valid_loss) / initial_valid_loss) * 100
            else:
                valid_improvement = 0
        else:
            valid_improvement = 0
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        logger.info(f"📈 Training loss improvement: {train_improvement:+.1f}%")
        logger.info(f"📊 Validation loss improvement: {valid_improvement:+.1f}%")
        logger.info(f"🎯 Best validation loss: {self.best_valid_loss:.4f}")
        logger.info(f"🔄 Total iterations completed: {len(self.train_loss_history)}")
        
        # Check final status
        is_overfitting = self.check_overfitting()
        if is_overfitting:
            logger.warning("⚠️  Final status: Model shows signs of overfitting")
        else:
            logger.info("✅ Final status: Training completed successfully")
        
        logger.info("=" * 80)
        
        # Return summary statistics
        return {
            "total_time_hours": total_time / 3600,
            "train_loss_improvement_percent": train_improvement,
            "valid_loss_improvement_percent": valid_improvement,
            "best_valid_loss": self.best_valid_loss,
            "total_iterations": len(self.train_loss_history),
            "is_overfitting": is_overfitting,
            "final_train_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "final_valid_loss": self.valid_loss_history[-1] if self.valid_loss_history else None,
        }


class AdvancedTrainingMonitor:
    """Monitor training progress with overfitting detection and early stopping"""
    
    def __init__(self, config: TrainingConfig):
        """Initialize training monitor"""
        self.config = config
        self.train_loss_history = []
        self.valid_loss_history = []
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        self.start_time = time.time()
        self.iteration_times = []
        
    def log_metrics(
        self,
        iteration: int,
        train_loss: float,
        valid_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        tokens_per_sec: float = 0.0,
        memory_gb: float = 0.0,
    ):
        """Log comprehensive training metrics with time estimates"""
        
        # Record metrics
        self.train_loss_history.append(train_loss)

        # Only track validation loss when provided
        if valid_loss is not None and not (isinstance(valid_loss, float) and math.isnan(valid_loss)):
            self.valid_loss_history.append(valid_loss)

        # Compute perplexity cheaply (avoid math range error on huge losses)
        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
        valid_ppl = math.exp(valid_loss) if valid_loss and valid_loss < 20 else float("inf")
        
        # Calculate timing
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Use overall elapsed average for ETA for a more stable estimate
        if iteration > 0:
            avg_iter_time = elapsed / iteration
            remaining_iters = self.config.max_iterations - iteration
            eta_seconds = remaining_iters * avg_iter_time
            eta_minutes = int(eta_seconds // 60)
            # Format with thousands separator using spaces (e.g., 1 800)
            eta_formatted = format(eta_minutes, ",").replace(",", " ")
            eta_str = f"{eta_formatted}mn"
        else:
            eta_str = "calculating..."
            
        # Calculate progress
        progress = (iteration / self.config.max_iterations) * 100
        
        valid_display = f"{valid_loss:.4f}" if valid_loss is not None else "—"

        logger.info(
            f"🚀 Iter {iteration:>6}/{self.config.max_iterations} ({progress:>5.1f}%) | "
            f"📉 Train: {train_loss:.4f} (ppl {train_ppl:,.1f}) | 📊 Valid: {valid_display} | "
            f"📈 LR: {learning_rate:.2e} | ⚡ {tokens_per_sec:.0f} tok/s | "
            f"💾 {memory_gb:.1f}GB | ⏱️  ETA: {eta_str}"
        )
        
        # Record iteration time for ETA calculation
        if hasattr(self, '_last_log_time'):
            iter_time = current_time - self._last_log_time
            self.iteration_times.append(iter_time)
        self._last_log_time = current_time
        
        # Store the latest metrics for status reporting
        self._latest_metrics = {
            'iteration': iteration,
            'train_loss': train_loss,
            'val_loss': valid_loss,
            'train_perplexity': train_ppl,
            'val_perplexity': valid_ppl,
            'learning_rate': learning_rate,
            'tokens_per_sec': tokens_per_sec,
            'peak_memory_gb': memory_gb,
            'elapsed_minutes': elapsed / 60,
            'eta_minutes': eta_minutes if iteration > 0 else None,
            'progress': progress
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get the current training status for web UI display
        
        Returns:
            Dictionary with current training status
        """
        if not hasattr(self, '_latest_metrics'):
            # No metrics recorded yet
            return {
                'current_iteration': 0,
                'max_iterations': self.config.max_iterations,
                'train_loss': None,
                'val_loss': None,
                'progress': 0,
                'elapsed_minutes': 0,
                'eta_minutes': None
            }
        
        # Return the latest metrics
        return self._latest_metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display
        
        Returns:
            Dictionary with dashboard data
        """
        # Calculate iterations
        iterations = list(range(1, len(self.train_loss_history) + 1))
        
        # Calculate perplexity values
        train_perplexity = [math.exp(loss) if loss < 20 else float("inf") for loss in self.train_loss_history]
        val_perplexity = [math.exp(loss) if loss < 20 else float("inf") for loss in self.valid_loss_history]
        
        # Return dashboard data
        return {
            'iterations': iterations,
            'train_loss': self.train_loss_history,
            'val_loss': self.valid_loss_history,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity,
            'best_val_loss': self.best_valid_loss,
            'config': self.config.__dict__
        }
    
    def check_overfitting(self) -> bool:
        """Check if model is overfitting using validation loss increase threshold"""
        if len(self.valid_loss_history) < 3:
            return False
            
        # Check if validation loss has increased by more than threshold
        recent_valid_loss = self.valid_loss_history[-1]
        best_valid_loss = min(self.valid_loss_history)
        
        if best_valid_loss > 0:  # Avoid division by zero
            loss_increase_ratio = (recent_valid_loss - best_valid_loss) / best_valid_loss
            
            if loss_increase_ratio > self.config.overfitting_threshold:
                logger.warning(f"🚨 OVERFITTING DETECTED! Validation loss increased by {loss_increase_ratio:.1%} "
                             f"(threshold: {self.config.overfitting_threshold:.1%})")
                return True
                
        return False
    
    def should_stop_early(self, current_valid_loss: float) -> bool:
        """Return True when training should stop early.

        The method always *tracks* validation loss and patience, but it only
        instructs the caller to stop when early-stopping is **enabled** and one
        of the stop criteria is met (no improvement for N checks or clear
        over-fitting).
        """

        # Always keep track of best validation loss & patience so that, if the
        # user enables early-stopping part-way through training, the historical
        # statistics are still meaningful.
        if current_valid_loss < self.best_valid_loss - self.config.min_loss_improvement:
            self.best_valid_loss = current_valid_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # If early-stopping is disabled, never request termination – just
        # return False after book-keeping above.
        if not self.config.enable_early_stopping:
            return False

        # --- Below this point early-stopping is enabled ---

        # Check blatant over-fitting first.
        if self.check_overfitting():
            logger.warning("🛑 Early stopping due to overfitting")
            return True

        # Check patience window.
        if self.patience_counter >= self.config.early_stopping_patience:
            logger.warning(
                "🛑 Early stopping due to no improvement for %d checks",
                self.patience_counter,
            )
            return True

        return False
    
    def log_final_summary(self):
        """Log comprehensive final training summary"""
        total_time = time.time() - self.start_time
        
        if len(self.train_loss_history) >= 2:
            initial_train_loss = self.train_loss_history[0]
            final_train_loss = self.train_loss_history[-1]
            if initial_train_loss > 0:
                train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
            else:
                train_improvement = 0
        else:
            train_improvement = 0
            
        if len(self.valid_loss_history) >= 2:
            initial_valid_loss = self.valid_loss_history[0]
            final_valid_loss = self.valid_loss_history[-1]
            if initial_valid_loss > 0:
                valid_improvement = ((initial_valid_loss - final_valid_loss) / initial_valid_loss) * 100
            else:
                valid_improvement = 0
        else:
            valid_improvement = 0
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        logger.info(f"📈 Training loss improvement: {train_improvement:+.1f}%")
        logger.info(f"📊 Validation loss improvement: {valid_improvement:+.1f}%")
        logger.info(f"🎯 Best validation loss: {self.best_valid_loss:.4f}")
        logger.info(f"🔄 Total iterations completed: {len(self.train_loss_history)}")
        
        # Check final status
        if self.check_overfitting():
            logger.warning("⚠️  Final status: Model shows signs of overfitting")
        else:
            logger.info("✅ Final status: Training completed successfully")
        
        logger.info("=" * 80) 