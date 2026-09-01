# Tests for embeddings-service

This directory contains Python tests for the embeddings-service.

## Test Structure

- `test_integration.py` - Integration tests for the deployed service at emb.confuse.site
- `test_config_validation.py` - Unit tests for configuration validation
- `test_embedding_generation.py` - Tests for embedding generation logic and event schemas

## Running Tests

### Prerequisites

Install Python dependencies:
```bash
pip install -r ../requirements.txt
```

### Run All Tests

```bash
# From the embeddings-service directory
pytest

# With coverage
pytest --cov=.

# With verbose output
pytest -v
```

### Run Specific Test Files

```bash
# Run integration tests only
pytest tests/test_integration.py

# Run config validation tests only
pytest tests/test_config_validation.py

# Run embedding generation tests only
pytest tests/test_embedding_generation.py
```

### Run Specific Test Classes

```bash
# Run health check tests only
pytest tests/test_integration.py::TestEmbeddingsServiceHealth

# Run configuration tests only
pytest tests/test_config_validation.py::TestEnvironmentVariables
```

## Test Categories

### Integration Tests (`test_integration.py`)

These tests verify the deployed service at https://emb.confuse.site:

- **Health Check Tests**: Verify the `/health` endpoint responds correctly
- **Connectivity Tests**: Test DNS resolution, HTTPS, and multiple requests
- **Header Tests**: Validate response headers and metadata
- **Error Handling Tests**: Test error handling for invalid requests
- **Performance Tests**: Measure response times and concurrent request handling

### Configuration Tests (`test_config_validation.py`)

These tests validate environment variable configuration:

- **Environment Variables**: Test required and optional variables
- **Configuration Scenarios**: Test minimal, full, and alternative configurations
- **URL Validation**: Validate API endpoint URL formats
- **Type Validation**: Test numeric validation for ports, timeouts, etc.

### Embedding Generation Tests (`test_embedding_generation.py`)

These tests validate the core embedding generation logic and Kafka event flow:

- **Event Schemas**: Validate input/output event structures match Rust code
- **Embedding API Logic**: Test NVIDIA NIM API request/response formats
- **Kafka Event Flow**: Simulate Kafka consumption and publication
- **Error Handling**: Test API failures, empty content, truncation
- **Model Configuration**: Validate model selection and dimension mapping

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Python tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v
```

## Notes

- Integration tests require network access to emb.confuse.site
- Tests use pytest framework with asyncio support
- Tests are designed to be fast and reliable
- All tests are independent and can be run in any order
- Embedding generation tests validate the core business logic without requiring actual Kafka infrastructure
- Test coverage includes: HTTP endpoints, configuration validation, event schemas, API logic, and error handling
