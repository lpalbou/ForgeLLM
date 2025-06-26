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