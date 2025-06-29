"""
Configuration classes for training and fine-tuning
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, ClassVar


@dataclass
class TrainingConfig:
    """Configuration for continued pre-training with SOTA best practices"""
    
    # Model configuration
    model_name: str
    input_dir: str
    output_dir: str = "models"
    data_dir: str = "data/pretraining"
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 5e-6
    max_seq_length: int = 2048
    max_tokens_per_file: int = 1000000
    save_every: int = 500
    max_checkpoints: int = 5
    max_iterations: int = 10000
    warmup_steps: int = 150
    
    # Fine-tuning configuration
    fine_tune_type: str = "full"  # "full", "lora", or "dora"
    num_layers: int = -1  # -1 for all layers
    
    # Data mixture strategy
    data_mixture_ratio: float = 0.95  # 95% domain data, 5% general data
    
    # Overfitting detection and early stopping
    overfitting_threshold: float = 0.30
    early_stopping_patience: int = 3
    min_loss_improvement: float = 0.001
    validation_split: float = 0.1
    validation_fast_pct: float = 1.0  # Percentage of validation set for quick checks
    steps_per_eval: int = 25
    steps_per_report: int = 5
    enable_early_stopping: bool = True
    use_lr_rewarming: bool = True
    
    # Learning rate scheduling
    lr_schedule: str = "cosine_decay"  # "cosine_decay", "linear_decay", "constant"
    lr_decay_factor: float = 0.1
    weight_decay: float = 0.01
    
    # Miscellaneous
    seed: int = 42
    val_batches: Optional[int] = None
    
    # Class variable for default config paths
    DEFAULT_CONFIG_PATH: ClassVar[str] = "configs/cpt_default.yaml"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in asdict(self).items() if not k.startswith("DEFAULT_")}
    
    def save(self, path: Optional[str] = None) -> str:
        """Save config to YAML file
        
        Args:
            path: Path to save config file. If None, uses output_dir/config.yaml
            
        Returns:
            Path to saved config file
        """
        if path is None:
            # Use output_dir/config.yaml as default
            path = os.path.join(self.output_dir, "config.yaml")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save config
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
        return path
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file
        
        Args:
            path: Path to config file
            
        Returns:
            TrainingConfig object
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        # Filter out keys not in TrainingConfig
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary
        
        Args:
            config_dict: Dictionary of config values
            
        Returns:
            TrainingConfig object
        """
        # Filter out keys not in TrainingConfig
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Keep input_dir as-is - let the data processor handle path resolution
        # This avoids issues with different working directories between web interface and training process
        
        return cls(**filtered_dict)


@dataclass
class InstructTuningConfig:
    """Configuration for instruction fine-tuning with SOTA best practices"""
    
    # Model configuration
    base_model_path: str
    base_model_name: str
    output_dir: str = "models/instruct_tuned"
    data_dir: str = "data/instruction_tuning"
    
    # Training parameters
    batch_size: int = 6
    learning_rate: float = 5e-6
    max_seq_length: int = 2048
    max_iterations: int = 300
    save_every: int = 25
    eval_every: int = 25
    
    # LoRA configuration
    lora_layers: int = 16
    
    # Learning rate scheduling
    warmup_steps: int = 50
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    min_lr_ratio: float = 0.1
    
    # Data configuration
    dataset_ratio: float = 0.1  # Ratio of dataset conversations in training data
    max_train_examples: int = 10000
    max_val_examples: int = 1000
    
    # Miscellaneous
    seed: int = 42
    
    # Class variable for default config paths
    DEFAULT_CONFIG_PATH: ClassVar[str] = "configs/ift_default.yaml"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in asdict(self).items() if not k.startswith("DEFAULT_")}
    
    def save(self, path: Optional[str] = None) -> str:
        """Save config to YAML file
        
        Args:
            path: Path to save config file. If None, uses output_dir/config.yaml
            
        Returns:
            Path to saved config file
        """
        if path is None:
            # Use output_dir/config.yaml as default
            path = os.path.join(self.output_dir, "config.yaml")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save config
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
        return path
    
    @classmethod
    def load(cls, path: str) -> "InstructTuningConfig":
        """Load config from YAML file
        
        Args:
            path: Path to config file
            
        Returns:
            InstructTuningConfig object
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        # Filter out keys not in InstructTuningConfig
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InstructTuningConfig":
        """Create config from dictionary
        
        Args:
            config_dict: Dictionary of config values
            
        Returns:
            InstructTuningConfig object
        """
        # Filter out keys not in InstructTuningConfig
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


def create_default_configs():
    """Create default configuration files if they don't exist"""
    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Create default CPT config
    cpt_config_path = configs_dir / "cpt_default.yaml"
    if not cpt_config_path.exists():
        cpt_config = TrainingConfig(model_name="mlx-community/gemma-3-4b-it-bf16", input_dir="dataset")
        cpt_config.save(str(cpt_config_path))
    
    # Create default IFT config
    ift_config_path = configs_dir / "ift_default.yaml"
    if not ift_config_path.exists():
        ift_config = InstructTuningConfig(
            base_model_path="models/cpt/latest",
            base_model_name="mlx-community/gemma-3-4b-it-bf16"
        )
        ift_config.save(str(ift_config_path)) 