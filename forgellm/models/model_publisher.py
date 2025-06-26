"""
Model Publisher - Handles publishing models to a central repository
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelPublisher:
    """Handles publishing models to a central repository"""
    
    def __init__(self):
        """Initialize model publisher"""
        self.published_dir = Path("published")
        self.published_dir.mkdir(exist_ok=True)
    
    def publish_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Publish a checkpoint to the central repository
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dict with publication info
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                return {"success": False, "error": f"Checkpoint {checkpoint_path} not found"}
            
            # Get model name from parent directory
            model_name = checkpoint_path.parent.name
            
            # Create target directory
            target_dir = self.published_dir / model_name
            target_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy checkpoint file
            target_path = target_dir / checkpoint_path.name
            shutil.copy2(checkpoint_path, target_path)
            
            # Copy model config if it exists
            config_path = checkpoint_path.parent / "config.json"
            if config_path.exists():
                shutil.copy2(config_path, target_dir / "config.json")
            
            # Copy assets directory if it exists
            assets_dir = checkpoint_path.parent / "assets"
            if assets_dir.exists() and assets_dir.is_dir():
                target_assets_dir = target_dir / "assets"
                target_assets_dir.mkdir(exist_ok=True)
                for asset_file in assets_dir.glob("*"):
                    shutil.copy2(asset_file, target_assets_dir / asset_file.name)
            
            return {
                "success": True,
                "model_name": model_name,
                "checkpoint_name": checkpoint_path.name,
                "published_path": str(target_path)
            }
        except Exception as e:
            logger.error(f"Error publishing checkpoint: {e}")
            return {"success": False, "error": str(e)}
    
    def list_published_models(self) -> Dict[str, Any]:
        """List all published models
        
        Returns:
            Dict with list of published models
        """
        try:
            models = []
            for model_dir in self.published_dir.glob("*"):
                if model_dir.is_dir():
                    model_name = model_dir.name
                    checkpoints = []
                    
                    # Get all checkpoint files
                    for checkpoint_file in model_dir.glob("*_adapters.safetensors"):
                        checkpoint_name = checkpoint_file.name
                        checkpoints.append({
                            "name": checkpoint_name,
                            "path": str(checkpoint_file),
                            "size": checkpoint_file.stat().st_size / (1024 * 1024),  # Size in MB
                            "created": checkpoint_file.stat().st_mtime
                        })
                    
                    # Sort checkpoints by creation time (newest first)
                    checkpoints.sort(key=lambda x: x["created"], reverse=True)
                    
                    models.append({
                        "name": model_name,
                        "path": str(model_dir),
                        "checkpoints": checkpoints,
                        "created": model_dir.stat().st_mtime
                    })
            
            # Sort models by creation time (newest first)
            models.sort(key=lambda x: x["created"], reverse=True)
            
            return {"success": True, "models": models}
        except Exception as e:
            logger.error(f"Error listing published models: {e}")
            return {"success": False, "error": str(e)} 