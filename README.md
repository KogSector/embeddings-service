# Embeddings Service

**Port**: 3001

A high-performance embedding generation and vector search service for the ConFuse platform.

## Features

- **Open Source Models**: SentenceTransformers embedding models
- **Batch Processing**: Efficient batch embedding generation
- **Vector Search**: Similarity search with pgvector
- **Caching**: Redis-based caching for performance
- **REST API**: Clean HTTP API for integration
- **Hybrid Architecture**: Rust performance with Python ML models

## Quick Start

### Prerequisites

- Rust 1.70+
- Python 3.11+
- PostgreSQL with pgvector extension
- Redis

### Installation

```bash
# Install Rust dependencies
cargo build --release

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration

Set environment variables:

```bash
export POSTGRES_URL="postgresql://localhost/embeddings"
export REDIS_URL="redis://localhost"
```

### Running

```bash
# Development
cargo run

# Production
cargo run --release
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Generate Embedding
```bash
POST /api/v1/generate
{
  "text": "Your text here",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### Batch Generate
```bash
POST /api/v1/generate/batch
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "batch_size": 32
}
```

### Vector Search
```bash
GET /api/v1/search?q=query text&model=sentence-transformers/all-MiniLM-L6-v2&limit=10
```

### List Models
```bash
GET /api/v1/models
```

## Architecture

```
embeddings-service/
├── src/
│   ├── models/          # Model management
│   ├── generators/      # Embedding generation
│   ├── storage/         # Database & cache
│   ├── api/            # HTTP endpoints
│   └── core/           # Configuration & errors
└── main.rs             # Application entry
```

## Integration

The embeddings service integrates with:
- **unified-processor**: For document and code processing
- **relation-graph**: For knowledge graph operations
- **api-backend**: For API gateway routing

## Development

### Adding New Models

1. Implement the `EmbeddingModel` trait in `src/models/`
2. Register the model in `ModelManager`
3. Update the model list in API responses

### Testing

```bash
cargo test
```

### Performance

The service is optimized for:
- High-throughput batch processing
- Concurrent model loading
- Efficient memory usage
- Fast vector operations
