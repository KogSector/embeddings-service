# Embeddings Service Documentation

## Overview

The Embeddings Service provides high-performance text embedding generation for the ConFuse platform. It supports multiple embedding models and provides both single and batch processing capabilities.

## Features

- **Multiple Model Support**: Local models via Ollama and OpenAI embeddings
- **Batch Processing**: Efficient batch embedding generation
- **High Performance**: Rust-based architecture with Python ML integration
- **Flexible API**: RESTful API with comprehensive endpoints
- **Model Management**: Dynamic model loading and caching

## Architecture

```
embeddings-service/
├── src/
│   ├── api/              # REST API endpoints
│   ├── core/             # Core embedding logic
│   ├── generators/       # Embedding model interfaces
│   └── main.rs           # Service entry point
├── docs/                 # Documentation
├── Cargo.toml            # Rust dependencies
├── pyproject.toml        # Python dependencies
└── README.md             # Service documentation
```

## Quick Start

```bash
# Build the Rust service
cargo build --release

# Install Python dependencies
pip install numpy sentence-transformers

# Set environment variables
export OLLAMA_URL=http://localhost:11434
export OPENAI_API_KEY=your_api_key

# Run the service
cargo run
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/embed` | POST | Generate single embedding |
| `/api/v1/embed/batch` | POST | Generate batch embeddings |
| `/api/v1/models` | GET | List available models |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | 3010 | Server port |
| `OLLAMA_URL` | No | http://localhost:11434 | Ollama server URL |
| `OPENAI_API_KEY` | No | - | OpenAI API key |
| `DEFAULT_MODEL` | No | nomic-embed-text | Default embedding model |

## Ports

- **3010**: Primary service port

## Documentation Files

- [API Reference](api-reference.md) - Complete API documentation
- [Architecture](architecture.md) - Detailed architecture and design decisions
- [Deployment](deployment.md) - Deployment instructions and configuration
