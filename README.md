# Embeddings Service

**Port**: 3011

A high-performance Kafka-based embedding generation service for the ConFuse platform. This service consumes chunk events from Kafka, generates embeddings using NVIDIA NIM/Gemini API, and publishes the results back to Kafka for the unified-processor to store in FalkorDB.

## Features

- **Kafka-Based Architecture**: Event-driven processing via Apache Kafka
- **NVIDIA NIM/Gemini API Embeddings**: Uses NVIDIA NIM or Gemini API models for embedding generation
- **Batch Processing**: Efficient batch embedding generation for multiple chunks
- **Health Check Endpoint**: Simple HTTP health check for deployment monitoring
- **Auto-Retry**: Resilient Kafka publishing with retry logic and DLQ fallback
- **Keep-Alive**: Self-ping mechanism to prevent cloud provider spin-down
- **Production Ready**: Optimized Rust implementation with proper error handling

## How to run the microservice

### Prerequisites

- Rust 1.70+
- Python 3.8+ (for testing)
- Apache Kafka cluster (Aiven or self-hosted)
- NVIDIA NIM API key or Gemini API key

### Installation

```bash
# Install Rust dependencies
cargo build --release

# Install Python dependencies (for testing)
pip install -r requirements.txt
```

### Configuration

Set environment variables:

```bash
# Server Configuration
export HOST="0.0.0.0"
export PORT="3011"
export WORKERS="1"

# Model Configuration
export DEFAULT_EMBEDDING_MODEL="nvidia/embeddings/003"
export MAX_BATCH_SIZE="32"
export MODEL_TIMEOUT_SECS="30"
export NVIDIA_NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_NIM_API_KEY="your-api-key"

# Kafka Configuration
export KAFKA_BOOTSTRAP_SERVERS="your-kafka-bootstrap-servers"
export EMBEDDINGS_SERVICE_KAFKA_GROUP_ID="embeddings-service-group"
export KAFKA_INPUT_TOPIC="chunk-raw-events"
export KAFKA_OUTPUT_TOPIC="embedding-generated-events"
export KAFKA_ENABLED="true"
```

### How to run the microservice

```bash
# Development
cargo run

# Production
cargo run --release
```

## Architecture

```
embeddings-service/
├── src/
│   ├── models/          # Model management and generation
│   ├── generators/      # Batch and streaming embedding generation
│   ├── infra/           # Kafka infrastructure (events, worker)
│   ├── config.rs        # Configuration management
│   ├── error.rs         # Error types
│   ├── lib.rs           # Library exports
│   └── main.rs          # Application entry with health check server
└── main.rs              # Application entry
```

## Integration

The embeddings service integrates with:
- **Kafka**: Event-driven communication with other services
- **unified-processor**: Consumes embedding-generated events to store in FalkorDB
- **Data processing pipeline**: Receives chunk-raw events from document processors

## Event Flow

1. **Input**: Service consumes `SimplifiedChunkRawEvent` from Kafka input topic
2. **Processing**: Generates embeddings for each chunk using configured model
3. **Output**: Publishes `SimplifiedEmbeddingGeneratedEvent` to Kafka output topic
4. **Storage**: unified-processor consumes output events and stores in FalkorDB

## Development

### Adding New Models

1. Update the model configuration in environment variables
2. Modify `ModelManager` in `src/models/` if custom model logic is needed
3. Test with the new model before deploying

### Testing

#### Rust Tests
```bash
# Run unit tests
cargo test

# Run with specific test output
cargo test -- --nocapture
```

#### Python Tests
Python integration, configuration, and embedding generation tests are provided in the `tests/` directory.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run all Python tests (47 tests total)
pytest

# Run with coverage
pytest --cov=.

# Run specific test files
pytest tests/test_integration.py           # HTTP endpoint tests
pytest tests/test_config_validation.py    # Configuration validation
pytest tests/test_embedding_generation.py # Core business logic tests
```

See [tests/README.md](tests/README.md) for detailed testing information.

### Performance

The service is optimized for:
- High-throughput batch processing
- Efficient memory usage with Vec pre-allocation
- Resilient Kafka operations with retry logic
- Minimal overhead for health checks

## Deployment

The service includes:
- **Health check server**: Responds to `GET /health` with "OK"
- **Kafka health check**: Verifies Kafka connectivity before starting worker
- **Keep-alive mechanism**: Self-pings every 10 minutes to prevent cloud spin-down
- **Graceful error handling**: Automatic retry on Kafka disconnection
