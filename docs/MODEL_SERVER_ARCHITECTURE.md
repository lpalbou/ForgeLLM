# Model Server Architecture

## Overview

The ForgeLLM project has been refactored to use a separate model server process for handling model loading and inference. This document explains the architecture and design decisions behind this approach.

## Problem Statement

The original architecture had several issues:

1. **Blocking API Calls**: Model loading would block the web server, causing API calls to stall indefinitely.
2. **Memory Management**: Loading large models in the same process as the web server could lead to out-of-memory errors.
3. **Process Isolation**: Errors in model loading or inference could crash the entire web server.

## Solution: Separate Model Server

The new architecture uses a separate HTTP server (`model_server.py`) that runs in its own process and handles all model-related operations. The web server communicates with this model server via HTTP requests.

```
┌─────────────────┐       HTTP       ┌─────────────────┐
│                 │    Requests      │                 │
│   Web Server    │◄─────────────────►   Model Server  │
│  (Flask/API)    │                  │ (model_server.py)│
│                 │                  │                 │
└─────────────────┘                  └─────────────────┘
```

## Key Components

### 1. Model Server (`model_server.py`)

A standalone HTTP server that handles:
- Model loading
- Text generation
- Status reporting

The model server loads models in a separate thread to keep its API responsive.

### 2. ModelManager

The `ModelManager` class has been refactored to act as a client to the model server:
- It ensures the model server is running
- It sends HTTP requests to the model server for model operations
- It maintains the state of the currently loaded model

### 3. API Routes

The API routes have been updated to use the new `ModelManager` implementation:
- `/api/model/load` - Starts loading a model in the model server
- `/api/model/status` - Gets the status of the currently loaded model
- `/api/model/generate` - Generates text using the loaded model
- `/api/model/unload` - Unloads the current model

## Benefits

1. **Non-blocking API**: The web server remains responsive even during model loading.
2. **Better Resource Management**: The model server can be restarted independently of the web server.
3. **Improved Stability**: Errors in model loading or inference don't affect the web server.
4. **Scalability**: The model server could be deployed on a separate machine if needed.

## Implementation Details

### Model Loading Process

1. The client sends a request to `/api/model/load`
2. The web server forwards the request to the model server
3. The model server starts loading the model in a background thread
4. The web server returns immediately with a success message
5. The client can check the loading status by polling `/api/model/status`

### Text Generation Process

1. The client sends a request to `/api/model/generate`
2. The web server forwards the request to the model server
3. The model server generates text using the loaded model
4. The web server returns the generated text to the client

## Error Handling

The new architecture includes improved error handling:

1. **Loading Errors**: If model loading fails, the error is reported in the status response.
2. **Generation Errors**: If text generation fails, the error is returned in the response.
3. **Connection Errors**: If the model server is not running, the `ModelManager` will try to start it.

## Testing

The new architecture includes comprehensive tests:

1. **Unit Tests**: Tests for the `ModelManager` class and model server.
2. **Integration Tests**: Tests for the API routes and model server communication.
3. **End-to-End Tests**: Tests for the complete workflow from the client to the model server.

## Future Improvements

1. **Authentication**: Add authentication between the web server and model server.
2. **Multiple Models**: Support loading multiple models in separate processes.
3. **Distributed Deployment**: Support deploying the model server on a separate machine.
4. **Load Balancing**: Add support for multiple model servers with load balancing. 