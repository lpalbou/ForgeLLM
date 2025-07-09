"""
ForgeLLM - A toolkit for continued pre-training and fine-tuning language models with MLX-LM
"""

__version__ = "0.3.8"
__author__ = "Laurent-Philippe Albou"
__email__ = "lpalbou@gmail.com"
__license__ = "MIT"

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import submodules
from . import models
from . import training
from . import api
from . import cli

# Import key components for easier access (lazy imports to avoid circular dependencies)
def get_training_config():
    from .training.config import TrainingConfig
    return TrainingConfig

def get_continued_pretrainer():
    from .training.trainer import ContinuedPretrainer
    return ContinuedPretrainer

def get_model_manager():
    from .models.model_manager import ModelManager
    return ModelManager

__all__ = ["models", "training", "api", "cli", "__version__", 
           "get_training_config", "get_continued_pretrainer", "get_model_manager"] 