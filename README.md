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

## Downloading Models

Before you can train or test models, you need to download them to your local cache. ForgeLLM uses the HuggingFace Hub to download and cache models locally.

### Prerequisites

Install the HuggingFace CLI:
```bash
pip install huggingface_hub
```

### Recommended Models

We recommend starting with these models for different use cases:

#### Small Models (1-2B parameters) - Good for testing and quick experiments
```bash
# Qwen3 1.7B - Excellent small model for continued pre-training
huggingface-cli download Qwen/Qwen3-1.7B

# Gemma 3 1B - Google's efficient small model
huggingface-cli download mlx-community/gemma-3-1b-pt-bf16
huggingface-cli download mlx-community/gemma-3-1b-it-bf16

# Qwen2.5 1.5B - Another excellent small option
huggingface-cli download mlx-community/Qwen23-1.7B-bf16
```

#### Medium Models (3-4B parameters) - Good balance of performance and speed
```bash
# Qwen3 4B - Excellent for continued pre-training
huggingface-cli download mlx-community/Qwen3-4B-bf16

# Gemma 3 4B - Larger Gemma model
huggingface-cli download mlx-community/gemma-3-4b-pt-bf16
huggingface-cli download mlx-community/gemma-3-4b-it-bf16

# Qwen2.5 3B - Good instruction-following model
huggingface-cli download mlx-community/Qwen2.5-3B-Instruct-4bit
```

#### Large Models (7-8B parameters) - Best performance, requires more memory
```bash
# Qwen3 8B - High-performance model
huggingface-cli download mlx-community/Qwen3-8B-bf16

# Llama 3.1 8B - Meta's flagship model
huggingface-cli download mlx-community/Meta-Llama-3.1-8B-bf16
huggingface-cli download mlx-community/Meta-Llama-3.1-8B-Instruct-bf16

# Qwen2.5 7B - Latest Qwen model
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Model Storage and Cache

All downloaded models are stored in your local HuggingFace cache directory:
- **macOS/Linux**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

This is where ForgeLLM looks for models during:
- **Training**: Base models for continued pre-training or instruction fine-tuning
- **Testing**: Loading models for text generation and evaluation
- **Published Models**: Your trained CPT models are also stored here with a `published/` prefix

### Model Types

ForgeLLM works with different model types:

- **Base Models** (e.g., `Qwen3-4B-bf16`): Pre-trained models good for continued pre-training
- **Instruct Models** (e.g., `Qwen3-4B-it-bf16`): Fine-tuned for instruction following
- **Quantized Models** (e.g., `4bit`, `8bit`): Smaller memory footprint, slightly lower quality
- **BF16 Models**: Full precision, best quality but larger memory usage

### Verifying Downloads

You can verify your downloaded models appear in ForgeLLM by:

1. Starting the web interface: `python forgellm_web.py`
2. Going to the Testing tab
3. Checking the model dropdown - your downloaded models should appear

Or using the CLI:
```bash
python forgellm_cli.py model list
```

### Storage Requirements

Typical storage requirements per model:
- **1-2B models**: 2-4 GB
- **3-4B models**: 6-8 GB  
- **7-8B models**: 14-16 GB

Make sure you have sufficient disk space before downloading large models.

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