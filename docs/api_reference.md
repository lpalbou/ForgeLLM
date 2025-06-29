# ForgeLLM API Reference

## Overview

ForgeLLM provides both REST API endpoints and command-line interfaces for interacting with the platform. This document covers all available APIs, their parameters, and response formats.

## Base URLs

- **Web API**: `http://localhost:5002/api`
- **Model Server**: `http://localhost:5001/api`

## Authentication

Currently, ForgeLLM does not require authentication for local development use.

## REST API Endpoints

### Training Management

#### Start Training

Start a new training session (CPT or IFT).

```http
POST /api/training/start
Content-Type: application/json
```

**Request Body:**
```json
{
  "model_name": "mlx-community/Qwen3-4B-bf16",
  "training_type": "cpt",
  "dataset": "my_domain",
  "max_iterations": 500,
  "batch_size": 2,
  "learning_rate": 5e-5,
  "lr_schedule": "cosine",
  "data_mixture_ratio": 0.8,
  "overfitting_threshold": 0.1,
  "save_every": 50,
  "weight_decay": 0.01,
  "warmup_steps": 10,
  "grad_checkpoint": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Training started successfully",
  "training_id": "cpt_20250115_103045",
  "output_dir": "models/cpt/Qwen3_4B_bf16_lr5e_05_bs2_iter500_seq3072_2025-01-15_10-30"
}
```

#### Get Training Status

Get the current status of training.

```http
GET /api/training/status
```

**Response:**
```json
{
  "success": true,
  "is_training": true,
  "training_type": "cpt",
  "current_iteration": 150,
  "max_iterations": 500,
  "progress_percentage": 30.0,
  "current_loss": 2.345,
  "learning_rate": 0.00004,
  "tokens_per_second": 234.5,
  "memory_usage_gb": 12.3,
  "eta_minutes": 45,
  "output_dir": "models/cpt/Qwen3_4B_bf16_lr5e_05_bs2_iter500_seq3072_2025-01-15_10-30"
}
```

#### Stop Training

Stop the current training session.

```http
POST /api/training/stop
```

**Response:**
```json
{
  "success": true,
  "message": "Training stopped successfully"
}
```

### Model Management

#### Load Model

Load a model in the model server.

```http
POST /api/model/load
Content-Type: application/json
```

**Request Body:**
```json
{
  "model_name": "mlx-community/Qwen3-4B-bf16",
  "adapter_path": "models/cpt/my_trained_model"  // Optional
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model loading started",
  "model_name": "mlx-community/Qwen3-4B-bf16",
  "adapter_path": "models/cpt/my_trained_model"
}
```

#### Get Model Status

Get the status of the currently loaded model.

```http
GET /api/model/status
```

**Response:**
```json
{
  "success": true,
  "loaded": true,
  "is_loading": false,
  "model_name": "mlx-community/Qwen3-4B-bf16",
  "adapter_path": "models/cpt/my_trained_model",
  "model_type": "instruct",
  "memory_usage_gb": 8.2
}
```

#### Generate Text

Generate text using the loaded model.

```http
POST /api/model/generate
Content-Type: application/json
```

**Request Body:**
```json
{
  "prompt": "What is machine learning?",
  "history": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "streaming": false,
  "is_base_model": false
}
```

**Response:**
```json
{
  "success": true,
  "response": "Machine learning is a subset of artificial intelligence...",
  "tokens_generated": 156,
  "generation_time_seconds": 2.3,
  "tokens_per_second": 67.8
}
```

#### Unload Model

Unload the current model to free memory.

```http
POST /api/model/unload
```

**Response:**
```json
{
  "success": true,
  "message": "Model unloaded successfully"
}
```

### Data Management

#### List Models

Get available models (cached, CPT, IFT, published).

```http
GET /api/models
```

**Response:**
```json
{
  "success": true,
  "cached_models": [
    {
      "name": "mlx-community/Qwen3-4B-bf16",
      "type": "base",
      "size_gb": 8.2,
      "cached": true
    }
  ],
  "cpt_models": [
    {
      "name": "Qwen3_4B_bf16_lr5e_05_bs2_iter500",
      "path": "cpt/Qwen3_4B_bf16_lr5e_05_bs2_iter500",
      "size_gb": 8.5,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "ift_models": [],
  "published_models": []
}
```

#### Get Checkpoints

List available checkpoints for the current training session.

```http
GET /api/checkpoints
```

**Response:**
```json
{
  "success": true,
  "checkpoints": [
    {
      "iteration": 300,
      "path": "models/cpt/my_model/0000300_adapters.safetensors",
      "validation_loss": 2.234,
      "rank": 1,
      "is_best": true,
      "size_mb": 45.2
    },
    {
      "iteration": 250,
      "path": "models/cpt/my_model/0000250_adapters.safetensors", 
      "validation_loss": 2.267,
      "rank": 2,
      "is_best": false,
      "size_mb": 45.1
    }
  ],
  "best_checkpoints": [300, 250, 200]
}
```

#### Get Dataset Info

Get information about available datasets.

```http
GET /api/dataset/info
```

**Response:**
```json
{
  "success": true,
  "datasets": [
    {
      "name": "my_domain",
      "path": "dataset/my_domain",
      "file_count": 15,
      "total_size_mb": 12.5,
      "estimated_tokens": 125000
    },
    {
      "name": "medical",
      "path": "dataset/medical", 
      "file_count": 8,
      "total_size_mb": 8.2,
      "estimated_tokens": 82000
    }
  ]
}
```

### Dashboard & Monitoring

#### Get Dashboard Data

Get training dashboard data for visualization.

```http
GET /api/dashboard/data?training_dir=models/cpt/my_model
```

**Response:**
```json
{
  "success": true,
  "training_data": [
    {
      "iteration": 50,
      "training_loss": 3.456,
      "validation_loss": 3.234,
      "learning_rate": 0.00005,
      "tokens_per_second": 234.5,
      "memory_usage_gb": 12.3,
      "timestamp": "2025-01-15T10:35:00Z"
    }
  ],
  "best_checkpoints": [300, 250, 200],
  "training_config": {
    "model_name": "mlx-community/Qwen3-4B-bf16",
    "max_iterations": 500,
    "batch_size": 2,
    "learning_rate": 5e-5
  },
  "performance_metrics": {
    "avg_tokens_per_second": 234.5,
    "peak_memory_gb": 14.2,
    "training_stability": 0.95
  }
}
```

#### Get Real-time Dashboard

Get real-time training metrics.

```http
GET /api/dashboard/realtime
```

**Response:**
```json
{
  "success": true,
  "current_metrics": {
    "iteration": 156,
    "training_loss": 2.345,
    "validation_loss": 2.456,
    "learning_rate": 0.000048,
    "tokens_per_second": 245.2,
    "memory_usage_gb": 12.8,
    "gpu_utilization": 0.87,
    "timestamp": "2025-01-15T10:40:00Z"
  },
  "is_training": true
}
```

### Model Publishing

#### Publish Checkpoint

Convert a checkpoint to a full published model.

```http
POST /api/training/publish_checkpoint
Content-Type: application/json
```

**Request Body:**
```json
{
  "checkpoint_path": "models/cpt/my_model/0000300_adapters.safetensors",
  "training_dir": "models/cpt/my_model"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Checkpoint published successfully",
  "published_model_path": "models/published/Qwen3_4B_bf16_lr5e_05_bs2_iter500_0000300_20250115_104500",
  "model_name": "published/Qwen3_4B_bf16_lr5e_05_bs2_iter500_0000300_20250115_104500",
  "dashboard_path": "models/published/Qwen3_4B_bf16_lr5e_05_bs2_iter500_0000300_20250115_104500/training_dashboard.png",
  "readme_path": "models/published/Qwen3_4B_bf16_lr5e_05_bs2_iter500_0000300_20250115_104500/README.md"
}
```

### System & Health

#### Health Check

Check system health and component status.

```http
GET /api/health
```

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "components": {
    "web_server": "running",
    "model_server": "running", 
    "training_pipeline": "idle"
  },
  "system_info": {
    "memory_total_gb": 32.0,
    "memory_available_gb": 18.5,
    "disk_free_gb": 245.8,
    "python_version": "3.11.5",
    "mlx_version": "0.15.0"
  }
}
```

#### Browse Filesystem

Browse project directories (for dataset/model selection).

```http
GET /api/filesystem/browse?path=dataset
```

**Response:**
```json
{
  "success": true,
  "path": "dataset",
  "directories": [
    {
      "name": "my_domain",
      "type": "directory",
      "file_count": 15,
      "size_mb": 12.5
    },
    {
      "name": "medical",
      "type": "directory",
      "file_count": 8,
      "size_mb": 8.2
    }
  ],
  "files": [
    {
      "name": "readme.txt",
      "type": "file",
      "size_mb": 0.1
    }
  ]
}
```

## Command Line Interface

### Main Commands

#### Start Services

```bash
# Start both model server and web interface
forgellm start [--server-port 5001] [--web-port 5002] [--host localhost]

# Start individual services
forgellm server [--host localhost] [--port 5001] [--model MODEL] [--adapter ADAPTER]
forgellm web [--host 0.0.0.0] [--port 5002] [--debug]
```

### CLI Commands

#### Generate Text

Interactive or single-shot text generation.

```bash
# Interactive chat (REPL mode)
forgellm cli generate --model mlx-community/Qwen3-4B-bf16

# Single prompt
forgellm cli generate --model mlx-community/Qwen3-4B-bf16 --prompt "Hello, world!"

# With adapter
forgellm cli generate --model mlx-community/Qwen3-4B-bf16 --adapter-path models/cpt/my_model

# Advanced options
forgellm cli generate \
  --model mlx-community/Qwen3-4B-bf16 \
  --prompt "Explain machine learning" \
  --max-tokens 200 \
  --temperature 0.7 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

**REPL Commands:**
- `/help` - Show available commands
- `/q` or `/exit` - Quit REPL
- `/stats` - Show session statistics
- `/system [prompt]` - Set/show system prompt
- `/clear` - Clear conversation history

#### Model Information

Get detailed model information.

```bash
# Basic model info
forgellm cli info --model mlx-community/Qwen3-4B-bf16

# With adapter info
forgellm cli info --model mlx-community/Qwen3-4B-bf16 --adapter-path models/cpt/my_model
```

**Output:**
```
Model Information:
  Name: mlx-community/Qwen3-4B-bf16
  Type: Instruct Model
  Architecture: Qwen3
  Parameters: ~4B
  Quantization: bf16
  Context Length: 32,768 tokens
  
Adapter Information:
  Path: models/cpt/my_model
  Type: LoRA Adapter
  Rank: 16
  Alpha: 32
  Target Modules: q_proj, v_proj, o_proj
```

## WebSocket Events

ForgeLLM uses WebSocket for real-time updates during training.

### Client Events

#### Subscribe to Training Updates

```javascript
socket.emit('subscribe_training', {
  training_id: 'cpt_20250115_103045'
});
```

### Server Events

#### Training Progress Update

```javascript
socket.on('training_update', (data) => {
  console.log('Training update:', data);
  // {
  //   iteration: 156,
  //   training_loss: 2.345,
  //   validation_loss: 2.456,
  //   learning_rate: 0.000048,
  //   tokens_per_second: 245.2,
  //   memory_usage_gb: 12.8,
  //   progress_percentage: 31.2,
  //   eta_minutes: 42
  // }
});
```

#### Training Complete

```javascript
socket.on('training_complete', (data) => {
  console.log('Training completed:', data);
  // {
  //   success: true,
  //   final_iteration: 500,
  //   final_loss: 2.123,
  //   training_time_minutes: 87,
  //   best_checkpoints: [300, 250, 200],
  //   output_dir: "models/cpt/..."
  // }
});
```

#### Training Error

```javascript
socket.on('training_error', (data) => {
  console.error('Training error:', data);
  // {
  //   error: "Out of memory",
  //   iteration: 156,
  //   suggestion: "Reduce batch size or use a smaller model"
  // }
});
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (model/file not found)
- `409` - Conflict (training already running)
- `500` - Internal Server Error
- `503` - Service Unavailable (model server not running)

### Error Response Format

```json
{
  "success": false,
  "error": "Model not found",
  "error_code": "MODEL_NOT_FOUND",
  "details": {
    "model_name": "invalid-model",
    "searched_paths": [
      "~/.cache/huggingface/hub/",
      "models/base/"
    ]
  },
  "suggestion": "Download the model using: huggingface-cli download invalid-model"
}
```

## Rate Limiting

Currently, ForgeLLM does not implement rate limiting for local development use.

## SDK Examples

### Python SDK Example

```python
import requests
import json

class ForgeLLMClient:
    def __init__(self, base_url="http://localhost:5002"):
        self.base_url = base_url
    
    def start_training(self, config):
        """Start a training session."""
        response = requests.post(
            f"{self.base_url}/api/training/start",
            json=config
        )
        return response.json()
    
    def get_training_status(self):
        """Get current training status."""
        response = requests.get(f"{self.base_url}/api/training/status")
        return response.json()
    
    def generate_text(self, prompt, model_name, **kwargs):
        """Generate text using a model."""
        # Load model first
        requests.post(f"{self.base_url}/api/model/load", json={
            "model_name": model_name
        })
        
        # Generate text
        response = requests.post(f"{self.base_url}/api/model/generate", json={
            "prompt": prompt,
            **kwargs
        })
        return response.json()

# Usage
client = ForgeLLMClient()

# Start training
training_config = {
    "model_name": "mlx-community/Qwen3-4B-bf16",
    "training_type": "cpt",
    "dataset": "my_domain",
    "max_iterations": 500
}
result = client.start_training(training_config)
print(f"Training started: {result['training_id']}")

# Generate text
response = client.generate_text(
    "What is machine learning?",
    "mlx-community/Qwen3-4B-bf16",
    max_tokens=200,
    temperature=0.7
)
print(f"Generated: {response['response']}")
```

### JavaScript SDK Example

```javascript
class ForgeLLMClient {
    constructor(baseUrl = 'http://localhost:5002') {
        this.baseUrl = baseUrl;
    }
    
    async startTraining(config) {
        const response = await fetch(`${this.baseUrl}/api/training/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        return response.json();
    }
    
    async getTrainingStatus() {
        const response = await fetch(`${this.baseUrl}/api/training/status`);
        return response.json();
    }
    
    async generateText(prompt, modelName, options = {}) {
        // Load model
        await fetch(`${this.baseUrl}/api/model/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        
        // Generate text
        const response = await fetch(`${this.baseUrl}/api/model/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, ...options })
        });
        return response.json();
    }
}

// Usage
const client = new ForgeLLMClient();

// Start training
const trainingConfig = {
    model_name: 'mlx-community/Qwen3-4B-bf16',
    training_type: 'cpt',
    dataset: 'my_domain',
    max_iterations: 500
};

client.startTraining(trainingConfig)
    .then(result => console.log('Training started:', result.training_id));

// Generate text
client.generateText(
    'What is machine learning?',
    'mlx-community/Qwen3-4B-bf16',
    { max_tokens: 200, temperature: 0.7 }
).then(response => console.log('Generated:', response.response));
```

This API reference provides comprehensive coverage of all ForgeLLM endpoints and interfaces, enabling developers to integrate with the platform programmatically. 