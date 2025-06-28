# Testing Guide

This document provides instructions for testing the ForgeLLM system.

## Test Structure

The test suite is organized into several categories:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test the interaction between components
3. **End-to-End Tests**: Test the complete system workflow

## Running Tests

### Running All Tests

To run all tests:

```bash
cd forgellm
python -m unittest discover -s tests
```

### Running Specific Test Categories

To run specific test categories:

```bash
# Run unit tests
python -m unittest discover -s tests -p "test_*.py" -k "not integration and not e2e"

# Run integration tests
python -m unittest discover -s tests -p "test_integration.py"

# Run end-to-end tests
python -m unittest discover -s tests -p "test_e2e.py"
```

### Running Individual Tests

To run a specific test file:

```bash
python -m unittest tests/test_model_server.py
```

To run a specific test case:

```bash
python -m unittest tests.test_model_server.TestModelServer.test_server_status
```

## Model Server Tests

The model server tests verify the functionality of the standalone model server component. These tests include:

1. **Server Status**: Tests that the server status endpoint works correctly
2. **Model Loading**: Tests loading a model into the server
3. **Text Generation**: Tests generating text with a loaded model
4. **Error Handling**: Tests error handling in the model server

To run the model server tests:

```bash
python -m unittest tests/test_model_server.py
```

### Environment Variables

The model server tests use the following environment variables:

- `TEST_MODEL`: The model to use for testing (default: "mlx-community/gemma-3-1b-it-bf16")
  - Set to "skip" to skip tests that require a model

Example:

```bash
# Use a specific model for testing
TEST_MODEL="mlx-community/gemma-3-1b-it-bf16" python -m unittest tests/test_model_server.py

# Skip tests that require a model
TEST_MODEL="skip" python -m unittest tests/test_model_server.py
```

## Model Manager Tests

The model manager tests verify the functionality of the ModelManager class, which acts as a client to the model server. These tests include:

1. **Initialization**: Tests that the ModelManager initializes correctly
2. **Server Connection**: Tests that the ModelManager connects to the model server
3. **Model Loading**: Tests loading a model through the ModelManager
4. **Text Generation**: Tests generating text through the ModelManager
5. **Error Handling**: Tests error handling in the ModelManager

To run the model manager tests:

```bash
python -m unittest tests/test_model_manager.py
```

## Integration Tests

The integration tests verify the interaction between the web server and model server. These tests include:

1. **Health Endpoint**: Tests that the health endpoint works correctly
2. **Model Status**: Tests that the model status endpoint works correctly
3. **Model Loading and Generation**: Tests the complete flow of loading a model and generating text
4. **Model List**: Tests that the model list endpoint works correctly
5. **Training Status**: Tests that the training status endpoint works correctly
6. **Error Handling**: Tests error handling in the web server

To run the integration tests:

```bash
python -m unittest tests/test_integration.py
```

### Environment Variables

The integration tests use the following environment variables:

- `TEST_MODEL`: The model to use for testing (default: "mlx-community/gemma-3-1b-it-bf16")
  - Set to "skip" to skip tests that require a model

Example:

```bash
# Use a specific model for testing
TEST_MODEL="mlx-community/gemma-3-1b-it-bf16" python -m unittest tests/test_integration.py

# Skip tests that require a model
TEST_MODEL="skip" python -m unittest tests/test_integration.py
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Test File Naming**: Name test files with the prefix `test_`
2. **Test Class Naming**: Name test classes with the prefix `Test`
3. **Test Method Naming**: Name test methods with the prefix `test_`
4. **Documentation**: Add docstrings to test classes and methods
5. **Isolation**: Ensure tests are isolated and don't depend on each other
6. **Cleanup**: Clean up any resources created during tests

Example:

```python
class TestNewFeature(unittest.TestCase):
    """Tests for the new feature."""
    
    def setUp(self):
        """Set up test environment."""
        # Set up code here
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up code here
    
    def test_new_feature_works(self):
        """Test that the new feature works correctly."""
        # Test code here
```

## Continuous Integration

The test suite is run automatically on every push to the repository using GitHub Actions. The workflow is defined in `.github/workflows/tests.yml`.

To run the tests locally before pushing:

```bash
# Run all tests
python -m unittest discover -s tests

# Run linting
flake8 forgellm

# Run type checking
mypy forgellm
``` 