# ForgeLLM Testing Guide

This guide provides instructions for testing the refactored ForgeLLM package.

## Installation

First, install the package in development mode:

```bash
# Install the package
pip install -e ./forgellm
```

## Web Interface

Start the web interface:

```bash
# Start the web interface (default port 5001)
python forgellm/forgellm_web.py

# If port 5001 is in use, specify a different port
python forgellm/forgellm_web.py --port 5002

# Use legacy static/template folders if needed
python forgellm/forgellm_web.py --static-folder static --template-folder templates
```

The web interface will be available at http://localhost:5001 (or the port you specified).

## Configuration Management

Create and manage configurations:

```bash
# Create default configurations
python -m forgellm.cli.commands config create-defaults

# Show a configuration
python -m forgellm.cli.commands config show configs/cpt_default.yaml

# Export a configuration
python -m forgellm.cli.commands config export --type cpt --output my_config.yaml
```

## Training

Start a training job:

```bash
# Using a configuration file
python -m forgellm.cli.commands train --config configs/cpt_default.yaml

# Using command-line arguments
python -m forgellm.cli.commands train \
    --model "mlx-community/gemma-3-4b-it-bf16" \
    --input-dir "mnemosyne" \
    --batch-size 4 \
    --learning-rate 5e-6 \
    --max-iterations 1000
```

## Text Generation

Generate text with a trained model:

```bash
python -m forgellm.cli.commands generate \
    --model "mlx-community/gemma-3-4b-it-bf16" \
    --adapter-path "models/cpt/my_model/adapters.safetensors" \
    --prompt "Write a short story about AI" \
    --max-tokens 200
```

## Dashboard Generation

Generate a training dashboard:

```bash
python -m forgellm.cli.commands dashboard "models/cpt/my_model/CPT_20250624_191349.json"
```

## Model Publishing

Publish a checkpoint to a shareable format:

```bash
python -m forgellm.cli.commands publish "models/cpt/my_model/0000500_adapters.safetensors"
```

## Dataset Information

Get information about a dataset:

```bash
python -m forgellm.cli.commands dataset --input-dir "mnemosyne"
```

## Instruction Tuning

Start instruction tuning:

```bash
# Using a configuration file
python -m forgellm.cli.commands instruct --config configs/ift_default.yaml

# Using command-line arguments
python -m forgellm.cli.commands instruct \
    --base-model-path "models/cpt/my_model" \
    --base-model-name "mlx-community/gemma-3-4b-it-bf16" \
    --output-dir "models/ift/my_instruct_model"
```

## Python API Usage

You can also use ForgeLLM programmatically:

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

## Troubleshooting

If you encounter any issues:

1. **Import errors**: Make sure you've installed the package with `pip install -e ./forgellm`
2. **Port in use**: Use a different port with `--port` option
3. **Missing templates/static files**: Use the `--template-folder` and `--static-folder` options
4. **Module not found**: Use the correct Python module path (`forgellm.cli.commands` instead of just `forgellm`)

For any other issues, check the error message and traceback for details. 