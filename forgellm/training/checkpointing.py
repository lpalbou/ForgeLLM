"""Checkpoint management for training."""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage training checkpoints."""
    
    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        """Initialize checkpoint manager.
        
        Args:
            output_dir: Output directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
        
    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from output directory."""
        if not self.output_dir.exists():
            return
            
        # Find all adapter checkpoint files
        checkpoint_files = list(self.output_dir.glob("*_adapters.safetensors"))
        
        # Also check for the final adapter file
        final_adapter = self.output_dir / "adapters.safetensors"
        if final_adapter.exists():
            checkpoint_files.append(final_adapter)
            
        for checkpoint_file in checkpoint_files:
            # Extract iteration from filename (e.g., 0000500_adapters.safetensors -> 500)
            if checkpoint_file.name == "adapters.safetensors":
                iteration = float('inf')  # Final adapter has highest priority
            else:
                try:
                    iteration = int(checkpoint_file.stem.split("_")[0])
                except (ValueError, IndexError):
                    logger.warning(f"Could not extract iteration from checkpoint file: {checkpoint_file}")
                    continue
                    
            self.checkpoints.append({
                "iteration": iteration,
                "path": checkpoint_file,
                "is_final": checkpoint_file.name == "adapters.safetensors"
            })
            
        # Sort checkpoints by iteration
        self.checkpoints.sort(key=lambda x: x["iteration"])
        
        logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints from {self.output_dir}")
        
    def save_checkpoint(self, iteration: int, source_path: Path) -> Path:
        """Save a new checkpoint.
        
        Args:
            iteration: Current iteration
            source_path: Path to source checkpoint file
            
        Returns:
            Path to saved checkpoint
        """
        # Format iteration with leading zeros
        formatted_iteration = f"{iteration:07d}"
        
        # Create checkpoint filename
        checkpoint_filename = f"{formatted_iteration}_adapters.safetensors"
        checkpoint_path = self.output_dir / checkpoint_filename
        
        # Copy checkpoint file
        shutil.copy2(source_path, checkpoint_path)
        
        # Add to checkpoints list
        self.checkpoints.append({
            "iteration": iteration,
            "path": checkpoint_path,
            "is_final": False
        })
        
        # Sort checkpoints by iteration
        self.checkpoints.sort(key=lambda x: x["iteration"])
        
        # Prune old checkpoints if needed
        self._prune_old_checkpoints()
        
        logger.info(f"Saved checkpoint at iteration {iteration}: {checkpoint_path}")
        
        return checkpoint_path
        
    def save_final_checkpoint(self, source_path: Path) -> Path:
        """Save final checkpoint.
        
        Args:
            source_path: Path to source checkpoint file
            
        Returns:
            Path to saved checkpoint
        """
        # Create final checkpoint path
        final_checkpoint_path = self.output_dir / "adapters.safetensors"
        
        # Copy checkpoint file
        shutil.copy2(source_path, final_checkpoint_path)
        
        # Add to checkpoints list or update existing
        for checkpoint in self.checkpoints:
            if checkpoint["is_final"]:
                checkpoint["path"] = final_checkpoint_path
                break
        else:
            self.checkpoints.append({
                "iteration": float('inf'),
                "path": final_checkpoint_path,
                "is_final": True
            })
            
        logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
        
        return final_checkpoint_path
        
    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay within max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
            
        # Sort by iteration (ascending)
        sorted_checkpoints = sorted(
            [cp for cp in self.checkpoints if not cp["is_final"]], 
            key=lambda x: x["iteration"]
        )
        
        # Keep the most recent checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        # Remove old checkpoints
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = checkpoint["path"]
                os.remove(checkpoint_path)
                self.checkpoints.remove(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint['path']}: {e}")
                
    def get_best_checkpoint(self, metrics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint based on validation loss.
        
        Args:
            metrics: List of training metrics
            
        Returns:
            Best checkpoint or None if no checkpoints with validation loss
        """
        # Create a mapping of iteration to validation loss
        iter_to_val_loss = {}
        for metric in metrics:
            iteration = metric.get("iteration")
            val_loss = metric.get("val_loss")
            if iteration is not None and val_loss is not None:
                iter_to_val_loss[iteration] = val_loss
                
        # Find checkpoints with validation loss
        checkpoints_with_val_loss = []
        for checkpoint in self.checkpoints:
            iteration = checkpoint["iteration"]
            if iteration in iter_to_val_loss:
                checkpoints_with_val_loss.append({
                    **checkpoint,
                    "val_loss": iter_to_val_loss[iteration]
                })
                
        if not checkpoints_with_val_loss:
            return None
            
        # Find checkpoint with lowest validation loss
        best_checkpoint = min(checkpoints_with_val_loss, key=lambda x: x["val_loss"])
        
        return best_checkpoint
        
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint.
        
        Returns:
            Latest checkpoint or None if no checkpoints
        """
        if not self.checkpoints:
            return None
            
        # Get the checkpoint with the highest iteration
        return max(self.checkpoints, key=lambda x: x["iteration"])
        
    def get_checkpoint_path(self, iteration: Optional[int] = None) -> Optional[Path]:
        """Get path to checkpoint at specified iteration or latest.
        
        Args:
            iteration: Iteration to get checkpoint for, or None for latest
            
        Returns:
            Path to checkpoint or None if not found
        """
        if iteration is None:
            latest_checkpoint = self.get_latest_checkpoint()
            return latest_checkpoint["path"] if latest_checkpoint else None
            
        # Find checkpoint at specified iteration
        for checkpoint in self.checkpoints:
            if checkpoint["iteration"] == iteration:
                return checkpoint["path"]
                
        return None 