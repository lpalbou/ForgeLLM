# 🚀 Quick Start Guide: Continued Pre-training with MLX-LM

## ✅ System Status: READY

Your continued pre-training system is fully set up and tested! Here's everything you need to know:

## 📊 Your Data

- **Documents Found**: 42 files in `mnemosyne/` directory
- **Supported Formats**: `.txt`, `.md`, `.rst`, `.py`, `.json`
- **Processing**: Recursive scanning of all subdirectories
- **Ready for Training**: ✅ Yes

## 🎯 Quick Commands

### Test the System
```bash
./run_training.sh test
```

### Start Training (Recommended First Run)
```bash
# Quick 500-iteration test
./run_training.sh quick

# Full training with default model (Gemma 3-4B)
./run_training.sh

# Use different models
./run_training.sh llama     # Llama 3.1-8B
./run_training.sh mistral   # Mistral 7B
```

### Custom Training
```bash
# Direct Python usage
python continued_pretraining.py --model mlx-community/gemma-3-4b-it-bf16

# Custom parameters
python continued_pretraining.py \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --batch-size 2 \
    --learning-rate 3e-6 \
    --max-iterations 5000
```

## 📋 Your Questions Answered

### ❓ Multiple Documents vs Single Document Training?
**✅ Answer: Multiple documents in batch is significantly better**

The system automatically:
1. Processes all 42 documents in `mnemosyne/`
2. Chunks them into 2048-token segments
3. Shuffles and batches multiple chunks together
4. Trains on diverse document batches for stable gradients

### ❓ When to Save the Model?
**✅ Answer: Smart checkpointing every 500 iterations**

The system automatically:
- Saves checkpoints every 500 iterations
- Keeps the last 5 checkpoints (auto-cleanup)
- Implements early stopping when loss plateaus
- Saves final model after training completion

## 🎨 Training Configurations Available

| Command | Model | Description | Time | Memory |
|---------|-------|-------------|------|--------|
| `./run_training.sh quick` | Gemma 3-4B | 500 iterations test | ~30 min | Low |
| `./run_training.sh` | Gemma 3-4B | Full training | ~4-6 hours | Medium |
| `./run_training.sh llama` | Llama 3.1-8B | Larger model | ~6-8 hours | High |
| `./run_training.sh intensive` | Gemma 3-4B | 20K iterations | ~12+ hours | Medium |
| `./run_training.sh memory-efficient` | Gemma 3-4B | Low memory usage | ~6-8 hours | Very Low |

## 📊 Expected Training Output

```
🎯 Training Configuration: Default continued pre-training
Model: mlx-community/gemma-3-4b-it-bf16
=== Preparing Training Data ===
Found 42 documents to process
Processed 156 text chunks with ~89,234 tokens
Created 140 training and 16 validation examples
=== Starting Continued Pre-training ===
Training examples: 140
Validation examples: 16
Max iterations: 10,000
Checkpoints saved every 500 iterations

Iter    500 | Loss: 3.2451 | LR: 5.00e-06 | Elapsed: 12.3m | Best: 3.2451
Iter   1000 | Loss: 2.9876 | LR: 5.00e-06 | Elapsed: 24.7m | Best: 2.9876
...
```

## 📁 Output Structure

After training, you'll find:

```
models/continued_pretrained/
├── checkpoint-500/          # Checkpoint at 500 iterations
├── checkpoint-1000/         # Checkpoint at 1000 iterations
├── checkpoint-1500/         # Checkpoint at 1500 iterations
├── checkpoint-2000/         # Checkpoint at 2000 iterations
├── checkpoint-2500/         # Checkpoint at 2500 iterations
└── final/                   # Final trained model
```

## 🔧 Troubleshooting

### Out of Memory?
```bash
./run_training.sh memory-efficient
# or
python continued_pretraining.py --batch-size 1 --max-seq-length 1024
```

### Want Faster Training?
```bash
python continued_pretraining.py --batch-size 8  # If you have enough memory
```

### Training Too Slow?
```bash
./run_training.sh quick  # Test with 500 iterations first
```

## 🎯 Best Practices Implemented

✅ **Batch Processing**: Multiple documents per batch for stable gradients  
✅ **Smart Checkpointing**: Save every 500 iterations with auto-cleanup  
✅ **Early Stopping**: Prevent overfitting with loss monitoring  
✅ **Memory Optimization**: Gradient checkpointing and efficient chunking  
✅ **Validation**: 10% of data held out for validation  
✅ **Comprehensive Logging**: Full training progress tracking  

## 🚀 Ready to Start?

1. **Quick Test**: `./run_training.sh quick` (recommended first run)
2. **Full Training**: `./run_training.sh` 
3. **Monitor Progress**: `tail -f continued_pretraining.log`
4. **Use Trained Model**: Load from `models/continued_pretrained/`

Your system is production-ready and follows all MLX-LM best practices! 🎉 

# ForgeLLM Usage Guide

## Overview
This guide covers how to use ForgeLLM for continued pretraining and fine-tuning of LLMs using MLX.

## CLI Usage

### Basic Commands

#### Model Information
Get detailed information about model architecture and formatting:
```bash
# Basic model info
python forgellm_cli.py info --model "Qwen/Qwen2.5-7B-Instruct"

# Show formatting examples
python forgellm_cli.py info --model "google/gemma-2-9b-it" --show-example
```

#### Model Testing
Test a model with a prompt:
```bash
# Test with automatic architecture detection
python forgellm_cli.py model test --model "mlx-community/Qwen2.5-0.5B-Instruct-4bit" --prompt "Hello!" --max-tokens 100

# Test with custom parameters
python forgellm_cli.py model test --model "google/gemma-2-9b-it" --prompt "Write a story" --max-tokens 200 --temperature 0.8
```

#### Interactive REPL
Start an interactive chat session:
```bash
# Start REPL mode
python forgellm_cli.py generate --model "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# REPL commands:
# /help - Show available commands
# /info - Show model architecture information
# /format - Show current formatting details
# /system [prompt] - Set system prompt
# /save <filename> - Save conversation
# /load <filename> - Load conversation
# /stats - Show session statistics
# /q - Quit
```

### Model Architecture Support

The CLI automatically detects model architectures and applies appropriate formatting:

#### Supported Architectures
- **Qwen**: Alibaba's Qwen family (Qwen 1.5, Qwen2, Qwen3)
  - Special handling: INSTRUCT by default, BASE only if "base" in name
  - Format: `<|im_start|>role\ncontent<|im_end|>`
  
- **Gemma**: Google's Gemma family
  - Special handling: No explicit system messages - uses assistant turns
  - Format: `<start_of_turn>role\ncontent<end_of_turn>`
  
- **Llama**: Meta's Llama family
  - Format: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>content<|eot_id|>`
  
- **Mistral**: Mistral AI's models
  - Format: `<s>[INST] content [/INST]`
  
- **Phi**: Microsoft's Phi family
  - Format: `<|user|>\ncontent<|end|><|assistant|>`

#### Special Cases

**Qwen Models**:
- `Qwen/Qwen2.5-7B-Instruct` → INSTRUCT (default)
- `Qwen/Qwen2.5-7B-base` → BASE (contains "base")

**Gemma Models**:
- System prompts transformed to assistant messages
- Example: `System: You are helpful` → `<start_of_turn>model\nSystem: You are helpful<end_of_turn>`
- PT models (e.g., `gemma-3-4b-pt-8bit`) correctly detected as BASE

**BASE vs INSTRUCT Detection**:
- **BASE patterns**: `base`, `pt`, `pretrained`, `foundation`, `raw`, `vanilla`, `untuned`, `completion`
- **INSTRUCT patterns**: `instruct`, `chat`, `it`, `sft`, `dpo`, `rlhf`, `assistant`, `alpaca`, `vicuna`
- BASE: Prompts used as-is (no chat template formatting)
- INSTRUCT: Proper chat template formatting applied

## Web Interface Usage

### System Prompts
The web interface supports system prompts for all model types:

1. **Identity Prompts**: "I am Mnemosyne, an AI assistant..."
2. **Personality Prompts**: "You are a helpful pirate. Always respond with 'Arrr'..."
3. **Expert Prompts**: "You are a geography expert. Provide detailed information..."

### Model Detection
The frontend automatically detects model types and shows appropriate icons:
- 🤖 INSTRUCT models (supports chat templates)
- ⚡ BASE models (raw text completion)

### Testing Tab
Use the testing tab to verify system prompt functionality:
1. Select a model
2. Enter a system prompt
3. Send test messages
4. Verify the model follows instructions

## Training

### Dataset Preparation
```bash
# Analyze dataset
python forgellm_cli.py dataset --input-dir /path/to/data

# Start training
python forgellm_cli.py train --model-name "mlx-community/Qwen2.5-0.5B-Instruct-4bit" --input-dir /path/to/data --output-dir ./checkpoints
```

### Web Training Interface
1. Navigate to the Training tab
2. Select base model and dataset
3. Configure hyperparameters
4. Monitor training progress in real-time
5. View metrics and loss curves

## Best Practices

### System Prompts
- **For Gemma**: Use behavioral prompts rather than identity changes
- **For Qwen**: Both identity and behavioral prompts work well
- **For BASE models**: Prepend instructions directly to prompts

### Model Selection
- Use INSTRUCT models for conversational tasks
- Use BASE models for completion tasks
- Check model architecture support before deployment

### Performance
- Smaller models (0.5B-1B) for testing and development
- Larger models (7B+) for production use
- Consider 4-bit quantization for memory efficiency

## Troubleshooting

### Common Issues
1. **System prompts not working**: Check if model supports system messages
2. **Wrong formatting**: Verify architecture detection with `info` command
3. **Memory issues**: Use quantized models or reduce batch size

### Debug Commands
```bash
# Check model architecture
python forgellm_cli.py info --model "your-model-name"

# Test formatting
python forgellm_cli.py info --model "your-model-name" --show-example

# Interactive debugging
python forgellm_cli.py generate --model "your-model-name"
# Then use /info and /format commands
``` 