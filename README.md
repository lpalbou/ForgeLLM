# ForgeLLM

ForgeLLM is a comprehensive platform for continued pre-training and instruction fine-tuning of large language models using MLX on Apple Silicon.

## What ForgeLLM Does

- **🚀 Train**: Continued pre-training (CPT) via web interface *(IFT coming soon - see [Development Perspectives](docs/perspectives.md))*
- **📊 Monitor**: Real-time training dashboards and checkpoint management
- **🆚 Compare**: Enable comparison of multiple training sessions with validation loss, perplexity, stability and generalization gap
- **🔗 Fuse**: Merge LoRA/DoRA adapters with base models for deployment
- **⚡ Quantize**: Convert models to 8-bit or 4-bit precision for efficient deployment
- **💬 Chat & Test**: Interactive chat with models and adapters via CLI or web
- **📦 Publish**: Convert and publish trained models with comprehensive documentation

### Screenshots
Training:
![Training](docs/assets/training-tab.png)

Monitoring:
![Monitoring](docs/assets/monitoring-tab.png)

Compare:
![Compare](docs/assets/compare-tab.png)

Testing:
![Testing](docs/assets/testing-tab.png)

## Quick Start

### 1. Installation

#### Option A: Install from PyPI (Recommended)

```bash
# Install latest version
pip install forgellm

# Install specific version
pip install forgellm==0.3.7

# Upgrade existing installation
pip install --upgrade forgellm
```

#### Option B: Install from Source (Development)

```bash
git clone https://github.com/lpalbou/forgellm.git
cd forgellm
pip install -e .
```

> **Requirements**: Python 3.9+ and Apple Silicon Mac (M1/M2/M3/M4). All dependencies including MLX are installed automatically.

### 2. Download Models

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download a model (examples)
huggingface-cli download mlx-community/gemma-3-1b-it-bf16     # Small model
huggingface-cli download mlx-community/Qwen3-4B-bf16         # Medium model
```

### 3. Start ForgeLLM

```bash
# Start both servers (recommended)
forgellm start

# Opens web interface at http://localhost:5002
# Model server runs at http://localhost:5001
```

That's it! 🎉

## Usage

### Web Interface (Recommended)

The web interface provides everything you need:

```bash
forgellm start                    # Start both servers
# or
forgellm web --port 5002         # Web interface only
forgellm server --port 5001      # Model server only (separate terminal)
```

**Web Interface Features:**
- **Training Tab**: Configure and start CPT training *(IFT support coming soon)*
- **Monitoring Tab**: View training progress and dashboards  
- **Testing Tab**: Chat with models and test different prompts

### Command Line Interface

The CLI is perfect for quick model testing and interactive chat:

```bash
# Interactive chat with a model (REPL mode)
forgellm cli generate --model mlx-community/gemma-3-1b-it-bf16

# Single prompt test
forgellm cli generate --model mlx-community/gemma-3-1b-it-bf16 --prompt "Hello, how are you?"

# Get model architecture info
forgellm cli info --model mlx-community/gemma-3-1b-it-bf16

# Test with an adapter (your trained model)
forgellm cli generate --model mlx-community/Qwen3-4B-bf16 --adapter-path models/cpt/my_trained_model
```

**REPL Mode Commands:**
- Type normally to chat
- `/help` - Show available commands
- `/q` or `/exit` - Quit
- `/stats` - Show session statistics
- `/system [prompt]` - Set/show system prompt

## Model Downloads

ForgeLLM works with MLX-compatible models from HuggingFace. All models are cached locally in `~/.cache/huggingface/hub/`.

### Recommended Models

**Small Models (1-2B) - Good for testing:**
```bash
huggingface-cli download mlx-community/gemma-3-1b-it-bf16
huggingface-cli download mlx-community/gemma-3-1b-pt-bf16
```

**Medium Models (3-4B) - Good balance:**
```bash
huggingface-cli download mlx-community/Qwen3-4B-bf16
huggingface-cli download mlx-community/gemma-3-4b-it-bf16
```

**Large Models (7-8B) - Best quality:**
```bash
huggingface-cli download mlx-community/Qwen3-8B-bf16
huggingface-cli download mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
```

### Model Types

- **Base Models** (`-bf16`, `-pt-`): Ideal for continued pre-training, clean slate for domain adaptation
- **Instruct Models** (`-it-`, `-Instruct-`): Can also be used for continued pre-training with careful data mixing
- **Quantized Models** (`-4bit`, `-8bit`): Smaller memory usage, slightly lower quality

### Continued Pre-training: Base vs Instruct Models

**Base Models (Recommended for CPT):**
- ✅ No instruction-following capabilities to preserve
- ✅ Clean foundation for domain-specific knowledge
- ✅ Higher learning rates and longer training possible

**Instruct Models (Advanced CPT):**
- ✅ Better at learning from complex documents (recent research)
- ⚠️ Requires careful data mixing (1-5% original pretraining data)
- ⚠️ Lower learning rates to prevent catastrophic forgetting
- ⚠️ Shorter training to avoid losing instruction-following abilities

Choose base models for straightforward domain adaptation, instruct models when you need better knowledge absorption from complex documents.

> **📖 For detailed CPT best practices and latest research findings, see [docs/cpt.md](docs/cpt.md)**

## Training Your Own Models

### Continued Pre-Training (CPT) - Available Now

1. **Prepare Data**: Place text files in `dataset/` directory
2. **Start Web Interface**: `forgellm start`
3. **Training Tab**: Configure model, data, and parameters
4. **Monitor**: Watch progress in real-time
5. **Publish**: Convert best checkpoints to full models

Training is currently only available through the web interface.

### Instruction Fine-Tuning (IFT) - Coming Soon

IFT capabilities are currently in development. For technical details and implementation roadmap, see **[Development Perspectives](docs/perspectives.md)**.

## Directory Structure

```
forgellm/
├── dataset/          # Your training data (text files)
├── models/           # Trained model outputs
│   ├── cpt/         # Continued pre-training models
│   └── ift/         # Instruction fine-tuning models (coming soon)
└── data/            # Processed training data
```

## Commands Reference

### Main Commands

```bash
forgellm start                    # Start both servers (recommended)
forgellm web [--port 5002]       # Web interface only
forgellm server [--port 5001]    # Model server only
forgellm cli <command>            # Command-line operations
```

### CLI Commands

```bash
# Interactive chat (REPL mode)
forgellm cli generate --model <model>

# Single prompt
forgellm cli generate --model <model> --prompt "Your question"

# Model information
forgellm cli info --model <model>

# Test with adapter
forgellm cli generate --model <model> --adapter-path <path>
```

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 5-20GB per model
- **Python**: 3.9+
- **MLX**: Automatically installed

## Architecture

ForgeLLM uses a clean separation:

- **Model Server** (`forgellm server`): Handles model loading and inference
- **Web Server** (`forgellm web`): Provides UI and training coordination
- **CLI** (`forgellm cli`): Direct model interaction and testing

This allows you to use just the CLI for testing, or the full web interface for training.

## Documentation

### 📚 Comprehensive Guides

- **[Getting Started](docs/getting_started.md)**: Complete setup and first training session
- **[Architecture](docs/architecture.md)**: System design and component overview
- **[Data Flow](docs/data_flow.md)**: How data moves through the system
- **[API Reference](docs/api_reference.md)**: Complete REST API and CLI documentation
- **[CPT Best Practices](docs/cpt.md)**: Advanced continued pre-training techniques
- **[Development Perspectives](docs/perspectives.md)**: Current capabilities and IFT roadmap

### 🔧 Technical Documentation

- **Architecture**: Multi-process design with model server separation
- **Training Pipeline**: Real-time monitoring with automatic checkpoint management
- **Model Publishing**: LoRA to full model conversion with comprehensive documentation
- **Error Recovery**: Robust error handling and automatic recovery mechanisms

## Contributing

Contributions welcome! Please submit pull requests.

## License

MIT License - see LICENSE file.

## Acknowledgments

- **ForgeLLM Team**: Continued pre-training platform
- **[MLX-LM](https://github.com/ml-explore/mlx-lm)**: Apple's MLX framework for LLMs
- **[MLX](https://github.com/ml-explore/mlx)**: Apple's machine learning framework 