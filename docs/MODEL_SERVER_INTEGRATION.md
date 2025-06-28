# Model Server Integration Guide

This guide explains how to integrate with the ForgeLLM model server from other applications.

## Overview

The ForgeLLM model server provides a simple HTTP API for model loading and inference. You can use this API to integrate with the model server from any application that can make HTTP requests.

## Starting the Model Server

Before you can integrate with the model server, you need to start it:

```bash
python model_server.py --host localhost --port 5001
```

You can also preload a model when starting the server:

```bash
python model_server.py --host localhost --port 5001 --model mlx-community/gemma-3-1b-it-bf16
```

## API Endpoints

The model server provides the following API endpoints:

### Status Endpoint

**Endpoint**: `GET /api/model/status`

**Description**: Get the status of the currently loaded model.

**Response**:
```json
{
  "success": true,
  "loaded": true,
  "is_loading": false,
  "model_name": "mlx-community/gemma-3-1b-it-bf16",
  "adapter_path": null
}
```

### Load Endpoint

**Endpoint**: `POST /api/model/load`

**Description**: Load a model.

**Request**:
```json
{
  "model_name": "mlx-community/gemma-3-1b-it-bf16",
  "adapter_path": null
}
```

**Response**:
```json
{
  "success": true,
  "message": "Model mlx-community/gemma-3-1b-it-bf16 loading started",
  "model_name": "mlx-community/gemma-3-1b-it-bf16",
  "adapter_path": null
}
```

### Generate Endpoint

**Endpoint**: `POST /api/model/generate`

**Description**: Generate text using the loaded model.

**Request**:
```json
{
  "prompt": "Hello, world!",
  "max_tokens": 100
}
```

**Response**:
```json
{
  "success": true,
  "text": "Hello! How can I assist you today?",
  "generation_time": 1.5
}
```

## Integration Examples

### Python Integration

Here's an example of integrating with the model server from a Python application:

```python
import requests

class ModelClient:
    def __init__(self, server_url="http://localhost:5001"):
        self.server_url = server_url
    
    def get_status(self):
        """Get the status of the currently loaded model."""
        response = requests.get(f"{self.server_url}/api/model/status")
        return response.json()
    
    def load_model(self, model_name, adapter_path=None):
        """Load a model."""
        data = {
            "model_name": model_name,
            "adapter_path": adapter_path
        }
        response = requests.post(f"{self.server_url}/api/model/load", json=data)
        return response.json()
    
    def generate_text(self, prompt, max_tokens=100):
        """Generate text using the loaded model."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens
        }
        response = requests.post(f"{self.server_url}/api/model/generate", json=data)
        return response.json()

# Usage example
client = ModelClient()
status = client.get_status()
if not status["loaded"]:
    client.load_model("mlx-community/gemma-3-1b-it-bf16")
    # Wait for the model to load
    import time
    while True:
        status = client.get_status()
        if status["loaded"]:
            break
        if not status["is_loading"]:
            print(f"Error loading model: {status.get('error')}")
            break
        time.sleep(1)

# Generate text
result = client.generate_text("Hello, world!")
print(result["text"])
```

### JavaScript Integration

Here's an example of integrating with the model server from a JavaScript application:

```javascript
class ModelClient {
  constructor(serverUrl = "http://localhost:5001") {
    this.serverUrl = serverUrl;
  }

  async getStatus() {
    const response = await fetch(`${this.serverUrl}/api/model/status`);
    return response.json();
  }

  async loadModel(modelName, adapterPath = null) {
    const response = await fetch(`${this.serverUrl}/api/model/load`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model_name: modelName,
        adapter_path: adapterPath
      })
    });
    return response.json();
  }

  async generateText(prompt, maxTokens = 100) {
    const response = await fetch(`${this.serverUrl}/api/model/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: maxTokens
      })
    });
    return response.json();
  }
}

// Usage example
const client = new ModelClient();
(async () => {
  const status = await client.getStatus();
  if (!status.loaded) {
    await client.loadModel("mlx-community/gemma-3-1b-it-bf16");
    // Wait for the model to load
    while (true) {
      const status = await client.getStatus();
      if (status.loaded) {
        break;
      }
      if (!status.is_loading) {
        console.error(`Error loading model: ${status.error}`);
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // Generate text
  const result = await client.generateText("Hello, world!");
  console.log(result.text);
})();
```

### cURL Integration

You can also use cURL to integrate with the model server:

```bash
# Get status
curl http://localhost:5001/api/model/status

# Load model
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "mlx-community/gemma-3-1b-it-bf16"}' http://localhost:5001/api/model/load

# Generate text
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, world!", "max_tokens": 100}' http://localhost:5001/api/model/generate
```

## Error Handling

The model server returns appropriate HTTP status codes and error messages for different error conditions:

- `400 Bad Request`: Invalid request (e.g., missing required parameters)
- `404 Not Found`: Endpoint not found
- `500 Internal Server Error`: Server error (e.g., error loading model or generating text)

Error responses include a JSON object with an `error` field containing a description of the error:

```json
{
  "success": false,
  "error": "No model loaded"
}
```

## Security Considerations

The model server is designed for local use and does not include authentication. If you need to expose the model server to a network, consider:

1. Adding authentication (e.g., API keys)
2. Using HTTPS
3. Limiting access to trusted clients
4. Running the server behind a reverse proxy with security features

## Performance Considerations

When integrating with the model server, keep these performance considerations in mind:

1. **Model Loading**: Loading a model can take some time, especially for large models. Use the status endpoint to check if the model is loaded before generating text.
2. **Concurrency**: The model server processes requests sequentially. If you need to handle multiple requests concurrently, consider running multiple model servers on different ports.
3. **Memory Usage**: Large models require significant memory. Ensure your system has enough memory to load the model.
4. **Request Timeouts**: Set appropriate timeouts for requests to the model server, especially for text generation which can take some time for long outputs. 