# ForgeLLM

ForgeLLM is a comprehensive platform for continued pre-training and instruction fine-tuning of large language models using MLX.

## Features

- **Continued Pre-training (CPT)**: Train models on your own data to improve domain-specific knowledge
- **Instruction Fine-Tuning (IFT)**: Fine-tune models to follow instructions better
- **Real-time Monitoring**: Track training progress with comprehensive dashboards
- **Checkpoint Management**: Save and compare multiple checkpoints to select the best model
- **Model Testing**: Test models with various prompts and compare their outputs
- **Web Interface**: User-friendly interface for all operations

## Architecture

ForgeLLM uses a modular architecture with separate components:

1. **Web Server**: Flask-based web interface and API
2. **Model Server**: Separate process for model loading and inference
3. **Training Pipeline**: Components for continued pre-training and instruction fine-tuning
4. **Dashboard Generator**: Tools for visualizing training progress

For more details, see [Model Server Architecture](docs/MODEL_SERVER_ARCHITECTURE.md).

## Installation

### Prerequisites

- Python 3.9+
- MLX (for Apple Silicon Macs)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forgellm.git
   cd forgellm
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Starting the Web Interface

```bash
python forgellm_web.py
```

This will start the web interface at http://localhost:5000.

### Command Line Interface

ForgeLLM also provides a command-line interface for various operations:

```bash
# Load and test a model
python forgellm_cli.py model test --model mlx-community/gemma-3-1b-it-bf16 --prompt "Tell me about continued pre-training"

# Start continued pre-training
python forgellm_cli.py train --model-name mlx-community/gemma-3-1b-it-bf16 --input-dir data/corpus --output-dir models/cpt --batch-size 4 --learning-rate 5e-6 --max-iterations 1000
```

### Using the Model Server Directly

You can also use the model server directly for model loading and inference:

```bash
# Start the model server
python model_server.py --host localhost --port 5001

# Load a model (using curl)
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "mlx-community/gemma-3-1b-it-bf16"}' http://localhost:5001/api/model/load

# Check model status
curl http://localhost:5001/api/model/status

# Generate text
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me about continued pre-training", "max_tokens": 100}' http://localhost:5001/api/model/generate
```

## Web Interface Guide

The web interface has three main tabs:

### 1. Training

Configure and start training jobs:
- Select a model
- Configure training parameters
- Monitor training progress in real-time
- View and select checkpoints

### 2. Monitoring

Monitor active and completed training jobs:
- View training metrics (loss, perplexity, etc.)
- See learning rate schedule
- Track training speed
- Identify best checkpoints

### 3. Testing

Test and compare models:
- Load base models or fine-tuned models
- Generate text with custom prompts
- Adjust generation parameters
- Compare outputs from different models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- MLX team for the MLX framework
- Hugging Face for model weights and tokenizers 