# ForgeLLM

A comprehensive toolkit for continued pre-training and fine-tuning of language models with MLX-LM.

## Features

- **Continued Pre-Training**: Train models on your domain-specific data
- **Instruction Fine-Tuning**: Create instruction-following models
- **Web Interface**: Monitor training progress and test models
- **Configuration Management**: Easily create and manage training configurations
- **Dashboard Generation**: Visualize training metrics
- **Model Publishing**: Convert checkpoints to shareable formats

## Installation

```bash
# Install from the current directory
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/lpalbou/forgellm.git
```

## Quick Start

### Web Interface

```bash
# Start the web interface
python -m forgellm.forgellm_web
```

Access the web interface at http://localhost:5001

### Command Line Interface

```bash
# Create default configurations
python -m forgellm.cli.commands config create-defaults

# Start continued pre-training
python -m forgellm.cli.commands train --config configs/cpt_default.yaml

# Start instruction tuning
python -m forgellm.cli.commands instruct --config configs/ift_default.yaml

# Generate text with a trained model
python -m forgellm.cli.commands generate \
    --model "mlx-community/gemma-3-4b-it-bf16" \
    --adapter-path "models/cpt/my_model/adapters.safetensors" \
    --prompt "Write a short story about AI"
```

## Documentation

For more detailed documentation, see:

- [Testing Guide](../docs/TESTING_GUIDE.md)
- [Architecture](../docs/architecture.md)
- [Usage Guide](../docs/USAGE_GUIDE.md)

## Python API

```python
from forgellm.training.config import TrainingConfig
from forgellm.training.trainer import ContinuedPretrainer
from forgellm.models.model_manager import ModelManager

# Create configuration
config = TrainingConfig(
    model_name="mlx-community/gemma-3-4b-it-bf16",
    input_dir="mnemosyne",
    output_dir="models",
    batch_size=4,
    learning_rate=5e-6,
    max_iterations=10000
)

# Initialize trainer
trainer = ContinuedPretrainer(config)

# Run training
trainer.run_training()

# Generate text
model_manager = ModelManager()
model_manager.load("mlx-community/gemma-3-4b-it-bf16", "models/cpt/my_model/adapters.safetensors")
response = model_manager.generate("Write a short story about AI", max_tokens=200)
print(response)
```

## Requirements

- Python 3.8+
- MLX 0.0.7+
- MLX-LM 0.0.3+
- Flask 2.0.0+
- Flask-SocketIO 5.0.0+

## License

[MIT](LICENSE) 