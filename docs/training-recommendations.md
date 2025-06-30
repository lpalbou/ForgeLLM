# Training Recommendations for Large Language Models

## Overview

This guide provides comprehensive training recommendations for large language models based on empirical analysis of successful CPT (Continued Pre-Training) sessions on a MacBook Pro M4 Max with 128GB unified memory. The recommendations are calibrated using training data from:

- **Qwen3-1.7B-bf16**: Full fine-tuning with 300K tokens dataset
- **Gemma3-12B-it-bf16**: Full fine-tuning with 300K tokens dataset

## Hardware Platform Specifications

- **System**: MacBook Pro M4 Max
- **Memory**: 128GB Unified Memory
- **Framework**: MLX-LM
- **Dataset Size**: ~300K tokens
- **Precision**: bf16 (bfloat16)

## Baseline Training Analysis

### Qwen3-1.7B Performance Baseline
- **Peak Memory Usage**: 20.6 GB
- **Training Speed**: 0.10-0.15 iterations/sec
- **Token Throughput**: 300-450 tokens/sec
- **Batch Size**: 3
- **Learning Rate**: 3e-6
- **Training Duration**: ~1 hour (400 iterations)
- **Final Loss**: 1.835 (train), 1.806 (val)

### Gemma3-12B Performance Baseline
- **Peak Memory Usage**: 102.1 GB
- **Training Speed**: 0.025-0.040 iterations/sec
- **Token Throughput**: 55-80 tokens/sec
- **Batch Size**: 2
- **Learning Rate**: 6e-6
- **Training Duration**: ~4.5 hours (505 iterations, incomplete)
- **Final Loss**: 0.511 (train), 1.202 (val)

## Model-Specific Recommendations

### 1. Gemma3-27B

#### Full Fine-Tuning
**Feasibility**: ❌ **NOT RECOMMENDED**
- **Estimated Memory**: 180-220 GB (exceeds 128GB limit)
- **Alternative**: Use LoRA/DoRA only

#### LoRA Fine-Tuning
**Feasibility**: ✅ **RECOMMENDED**

```yaml
# Recommended LoRA Configuration
batch_size: 1
learning_rate: 4e-6
max_iterations: 1200
lora_layers: 16
lora_rank: 64
lora_alpha: 128
max_seq_length: 2048  # Reduced to manage memory
warmup_steps: 60
lr_schedule: "cosine_decay"
gradient_checkpointing: true
```

**Expected Performance**:
- Memory Usage: 85-95 GB
- Training Speed: 0.015-0.025 iterations/sec
- Estimated Duration: 15-20 hours
- Token Throughput: 30-50 tokens/sec

#### DoRA Fine-Tuning
**Feasibility**: ✅ **RECOMMENDED** (Better than LoRA)

```yaml
# Recommended DoRA Configuration
batch_size: 1
learning_rate: 3e-6
max_iterations: 1000
dora_layers: 12
lora_rank: 32
lora_alpha: 64
max_seq_length: 2048
warmup_steps: 50
lr_schedule: "cosine_decay"
gradient_checkpointing: true
```

### 2. Qwen3-14B

#### Full Fine-Tuning
**Feasibility**: ⚠️ **MARGINAL** (High memory pressure)

```yaml
# Conservative Full Fine-Tuning Configuration
batch_size: 1
learning_rate: 4e-6
max_iterations: 800
max_seq_length: 2048  # Reduced from 3072
warmup_steps: 40
lr_schedule: "cosine_decay"
gradient_checkpointing: true
weight_decay: 0.01
```

**Expected Performance**:
- Memory Usage: 110-125 GB (near limit)
- Training Speed: 0.020-0.030 iterations/sec
- Estimated Duration: 8-12 hours

#### LoRA Fine-Tuning
**Feasibility**: ✅ **HIGHLY RECOMMENDED**

```yaml
# Recommended LoRA Configuration
batch_size: 2
learning_rate: 5e-6
max_iterations: 600
lora_layers: 20
lora_rank: 128
lora_alpha: 256
max_seq_length: 3072
warmup_steps: 30
lr_schedule: "cosine_decay"
```

**Expected Performance**:
- Memory Usage: 65-75 GB
- Training Speed: 0.025-0.035 iterations/sec
- Estimated Duration: 5-7 hours

#### DoRA Fine-Tuning
**Feasibility**: ✅ **RECOMMENDED**

```yaml
# Recommended DoRA Configuration
batch_size: 2
learning_rate: 4e-6
max_iterations: 500
dora_layers: 16
lora_rank: 64
lora_alpha: 128
max_seq_length: 3072
warmup_steps: 25
```

### 3. Qwen3-32B

#### Full Fine-Tuning
**Feasibility**: ❌ **NOT FEASIBLE**
- **Estimated Memory**: 250-300 GB (far exceeds limit)

#### LoRA Fine-Tuning
**Feasibility**: ⚠️ **MARGINAL** (Very tight memory)

```yaml
# Conservative LoRA Configuration
batch_size: 1
learning_rate: 3e-6
max_iterations: 800
lora_layers: 8  # Reduced layers
lora_rank: 32  # Reduced rank
lora_alpha: 64
max_seq_length: 1536  # Significantly reduced
warmup_steps: 40
gradient_checkpointing: true
```

**Expected Performance**:
- Memory Usage: 115-125 GB
- Training Speed: 0.010-0.020 iterations/sec
- Estimated Duration: 12-20 hours

#### DoRA Fine-Tuning
**Feasibility**: ✅ **PREFERRED OVER LORA**

```yaml
# Recommended DoRA Configuration
batch_size: 1
learning_rate: 2e-6
max_iterations: 600
dora_layers: 6
lora_rank: 24
lora_alpha: 48
max_seq_length: 1536
warmup_steps: 30
```

### 4. Mistral-Small-24B-Instruct-2501

#### Full Fine-Tuning
**Feasibility**: ❌ **NOT RECOMMENDED**
- **Estimated Memory**: 160-200 GB

#### LoRA Fine-Tuning
**Feasibility**: ✅ **RECOMMENDED**

```yaml
# Recommended LoRA Configuration
batch_size: 1
learning_rate: 4e-6
max_iterations: 1000
lora_layers: 14
lora_rank: 96
lora_alpha: 192
max_seq_length: 2048
warmup_steps: 50
lr_schedule: "cosine_decay"
```

**Expected Performance**:
- Memory Usage: 90-105 GB
- Training Speed: 0.018-0.028 iterations/sec
- Estimated Duration: 10-15 hours

#### DoRA Fine-Tuning
**Feasibility**: ✅ **HIGHLY RECOMMENDED**

```yaml
# Recommended DoRA Configuration
batch_size: 1
learning_rate: 3e-6
max_iterations: 800
dora_layers: 10
lora_rank: 48
lora_alpha: 96
max_seq_length: 2048
warmup_steps: 40
```

## General Training Guidelines

### Memory Management Strategies

1. **Gradient Checkpointing**: Always enable for large models
2. **Sequence Length Optimization**: 
   - Start with 1536-2048 for 20B+ models
   - Use 3072 only for <15B models
3. **Batch Size Scaling**: 
   - Use batch_size=1 for models >20B
   - Use batch_size=2 for 10-20B models
   - Use batch_size=3+ for <10B models

### Learning Rate Guidelines

Based on model size scaling from baseline data:

- **1-5B models**: 3e-6 to 5e-6
- **5-15B models**: 4e-6 to 6e-6
- **15-30B models**: 2e-6 to 4e-6
- **30B+ models**: 1e-6 to 3e-6

### LoRA/DoRA Parameter Guidelines

#### LoRA Configuration by Model Size:
- **<10B**: rank=128, alpha=256, layers=20-24
- **10-20B**: rank=64-96, alpha=128-192, layers=12-16
- **20-30B**: rank=32-64, alpha=64-128, layers=8-12
- **30B+**: rank=16-32, alpha=32-64, layers=6-8

#### DoRA Advantages:
- 15-25% better convergence than LoRA
- Similar memory footprint
- Slightly slower training (5-10%)
- Better final model quality

### Training Duration Optimization

For 300K token dataset:
- **Target iterations**: 2-3 passes through dataset
- **Early stopping**: Monitor validation loss plateau
- **Checkpoint frequency**: Every 25-50 iterations
- **Validation frequency**: Every 25 iterations

## Monitoring and Troubleshooting

### Memory Pressure Indicators
- System memory >95%: Reduce batch_size or sequence_length
- Swap usage >10GB: Consider smaller model or LoRA
- Training crashes: Enable gradient_checkpointing

### Performance Optimization
- **Slow training**: Increase batch_size if memory allows
- **Poor convergence**: Increase learning_rate or warmup_steps
- **Overfitting**: Add weight_decay (0.01-0.02) or early stopping

### Quality Metrics Targets
Based on baseline analysis:
- **Initial loss**: 2.3-2.6 (normal)
- **Target final loss**: 1.2-1.8 (good convergence)
- **Validation gap**: <0.3 (avoid overfitting)

## Summary Table

| Model | Size | Full FT | LoRA | DoRA | Est. Memory | Est. Duration | Feasibility |
|-------|------|---------|------|------|-------------|---------------|-------------|
| **Gemma3-27B** | 27B | ❌ No | ✅ Yes | ✅ Yes | 85-95 GB | 15-20h | ⚠️ Tight |
| **Qwen3-14B** | 14B | ⚠️ Marginal | ✅ Yes | ✅ Yes | 65-125 GB | 5-12h | ✅ Good |
| **Qwen3-32B** | 32B | ❌ No | ⚠️ Tight | ✅ Yes | 115-125 GB | 12-20h | ⚠️ Marginal |
| **Mistral-Small-24B** | 24B | ❌ No | ✅ Yes | ✅ Yes | 90-105 GB | 10-15h | ✅ Good |

### Key Recommendations:

1. **Qwen3-14B**: Best balance of capability and feasibility
2. **Mistral-Small-24B**: Excellent choice for instruction following
3. **Gemma3-27B**: Highest capability but requires careful memory management
4. **Qwen3-32B**: Only attempt with DoRA and reduced parameters

### Success Factors:
- **DoRA > LoRA** for quality when memory allows
- **Sequence length** is the primary memory lever
- **Batch size=1** is often necessary for 20B+ models
- **Gradient checkpointing** is essential for large models
- **Monitor system memory** continuously during training

### Risk Mitigation:
- Start with conservative parameters
- Test with short runs before full training
- Have swap space available (32GB+)
- Monitor temperature and performance throttling
- Use incremental checkpoint saving 