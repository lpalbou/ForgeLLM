"""
ForgeLLM - A toolkit for continued pre-training and fine-tuning language models with MLX-LM
"""

__version__ = "0.1.1"
__author__ = "Laurent-Philippe Albou"
__email__ = "lpalbou@gmail.com"
__license__ = "MIT"

# Import key components for easier access
from forgellm.training.config import TrainingConfig
from forgellm.training.trainer import ContinuedPretrainer
from forgellm.models.model_manager import ModelManager

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from . import models
from . import training
from . import api
from . import cli

__all__ = ["models", "training", "api", "cli", "__version__"] 