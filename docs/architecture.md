# Embeddings Service Architecture

## Overview

The Embeddings Service is a high-performance text embedding generation service built with Rust and Python integration. It provides a unified API for multiple embedding models and providers.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Embeddings Service                        │
├─────────────────────────────────────────────────────────────┤
│  Rust API Layer (Port 3010)                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Health    │ │   Embed     │ │     Models API          │ │
│  │   Endpoint  │ │   Endpoint  │ │     Endpoint            │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Rust Core Logic                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Model     │ │   Batch     │ │     Cache               │ │
│  │  Manager    │ │ Processor   │ │   Manager               │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Python Integration Layer (PyO3)                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Ollama    │ │   OpenAI    │ │   Sentence              │ │
│  │   Bridge    │ │   Bridge    │ │   Transformers          │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  External Services                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Ollama    │ │   OpenAI    │ │   Model Cache           │ │
│  │   Server    │ │     API     │ │   (Memory/Redis)        │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Rust API Layer
- **Framework**: Axum web framework
- **Purpose**: High-performance HTTP server
- **Features**: Async request handling, middleware, rate limiting
- **Port**: 3010

### 2. Model Manager
- **Language**: Rust
- **Purpose**: Manages embedding model lifecycle
- **Responsibilities**:
  - Model registration and discovery
  - Dynamic model loading/unloading
  - Model health monitoring
  - Provider abstraction

### 3. Batch Processor
- **Language**: Rust
- **Purpose**: Efficient batch processing
- **Features**:
  - Concurrent embedding generation
  - Memory-efficient batching
  - Error handling and retry logic
  - Progress tracking

### 4. Cache Manager
- **Language**: Rust
- **Purpose**: Embedding caching and retrieval
- **Strategies**:
  - In-memory LRU cache
  - Optional Redis cache
  - Persistent cache for frequently used embeddings
  - Cache invalidation policies

### 5. Model Providers

#### Ollama Bridge (Python)
- **Technology**: Ollama Python client
- **Purpose**: Local model inference
- **Features**:
  - Local model hosting
  - Custom model support
  - Zero network latency
  - Privacy-preserving

#### OpenAI Bridge (Python)
- **Technology**: OpenAI Python client
- **Purpose**: Cloud-based embeddings
- **Features**:
  - High-quality embeddings
  - Multiple model sizes
  - Pay-per-token pricing
  - Global availability

#### Sentence Transformers (Python)
- **Technology**: Hugging Face Transformers
- **Purpose**: Local model inference
- **Features**:
  - Wide model selection
  - Custom model loading
  - GPU acceleration support
  - Offline capability

## Data Flow

### Single Embedding Request
```
1. HTTP Request → Rust API
2. Validation → Model Manager
3. Model Selection → Provider Bridge
4. Text Processing → Python Model
5. Embedding Generation → Cache Manager
6. Cache Storage → Response
```

### Batch Embedding Request
```
1. HTTP Request → Rust API
2. Batch Validation → Batch Processor
3. Text Chunking → Model Manager
4. Concurrent Processing → Provider Bridges
5. Parallel Generation → Cache Manager
6. Batch Assembly → Response
```

### Model Discovery
```
1. Service Start → Model Manager
2. Provider Scan → Available Models
3. Model Registration → Model Registry
4. Health Checks → Model Status
5. API Exposure → Models Endpoint
```

## Technology Stack

### Rust Dependencies
- `axum`: Web framework
- `tokio`: Async runtime
- `serde`: Serialization
- `pyo3`: Python integration
- `reqwest`: HTTP client
- `lru`: LRU cache implementation

### Python Dependencies
- `ollama`: Ollama client
- `openai`: OpenAI client
- `sentence-transformers`: Local models
- `numpy`: Numerical operations
- `torch`: PyTorch for model inference

### Optional Dependencies
- `redis-py`: Redis cache client
- `transformers`: Hugging Face models
- `accelerate`: GPU acceleration

## Model Management

### Model Registry
```rust
pub struct Model {
    pub id: String,
    pub name: String,
    pub provider: Provider,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub capabilities: Vec<String>,
}

pub enum Provider {
    Ollama,
    OpenAI,
    Local,
}
```

### Model Loading Strategy
- **Lazy Loading**: Models loaded on first request
- **Preloading**: Critical models loaded at startup
- **Unloading**: Unused models unloaded after timeout
- **Memory Management**: GPU memory monitoring and cleanup

### Model Configuration
```yaml
models:
  ollama:
    - id: "nomic-embed-text"
      name: "Nomic Embed Text"
      dimensions: 768
      preload: true
    - id: "qwen2.5:7b"
      name: "Qwen 2.5 7B"
      dimensions: 3584
      preload: false
  
  openai:
    - id: "text-embedding-3-small"
      name: "OpenAI Text Embedding 3 Small"
      dimensions: 1536
    - id: "text-embedding-3-large"
      name: "OpenAI Text Embedding 3 Large"
      dimensions: 3072
```

## Performance Optimization

### Concurrency
- **Async Processing**: Non-blocking I/O operations
- **Thread Pools**: Dedicated threads for model inference
- **Batch Optimization**: Intelligent batching for throughput
- **Resource Pooling**: Connection and GPU resource pooling

### Memory Management
- **Model Sharing**: Multiple requests share loaded models
- **Memory Monitoring**: Track GPU and RAM usage
- **Garbage Collection**: Proactive model unloading
- **Cache Eviction**: LRU eviction for embedding cache

### Caching Strategy
- **Multi-level Cache**: Memory → Redis → Database
- **Cache Keys**: Content-based hashing
- **TTL Management**: Expiration policies
- **Cache Warming**: Preload common embeddings

## Security

### Input Validation
- Text length limits
- Token count validation
- Content sanitization
- Rate limiting per client

### API Security
- Request authentication (future)
- API key management
- Request signing
- CORS configuration

### Model Security
- Model access controls
- Input/output filtering
- Privacy-preserving options
- Audit logging

## Monitoring and Observability

### Metrics
- Request rate and latency
- Model usage statistics
- Error rates by model
- Cache hit/miss ratios
- Resource utilization

### Logging
- Structured JSON logs
- Request/response logging
- Error stack traces
- Performance metrics

### Health Checks
- Model availability
- Provider connectivity
- Resource utilization
- Cache status

## Deployment Architecture

### Container Strategy
- Multi-stage Docker builds
- Model volume mounting
- Environment-based configuration
- Health check endpoints

### Scaling Considerations
- **Horizontal Scaling**: Multiple service instances
- **Model Distribution**: Different models on different instances
- **Load Balancing**: Request distribution strategies
- **Resource Allocation**: CPU/GPU resource management

### High Availability
- Model redundancy
- Failover mechanisms
- Graceful degradation
- Circuit breakers

## Integration Points

### ConFuse Platform Integration
- **Unified Processor**: Primary consumer
- **Relation Graph**: Semantic search integration
- **API Gateway**: Service discovery and routing
- **Monitoring**: Centralized logging and metrics

### External Integrations
- **Ollama**: Local model hosting
- **OpenAI**: Cloud-based embeddings
- **Hugging Face**: Model repository
- **Vector Databases**: Embedding storage

## Future Enhancements

### Planned Features
- **Custom Models**: User-defined model support
- **Model Fine-tuning**: On-device fine-tuning
- **Multi-modal**: Image and text embeddings
- **Edge Deployment**: Lightweight edge models

### Performance Improvements
- **GPU Optimization**: CUDA and Metal support
- **Quantization**: Model size reduction
- **Streaming**: Real-time embedding generation
- **Distributed Inference**: Multi-GPU support
