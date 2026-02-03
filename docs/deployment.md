# Embeddings Service Deployment Guide

## Overview

This guide covers deployment strategies for the Embeddings Service in various environments.

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM (16GB+ for GPU models)
- **Storage**: 20GB+ available space
- **GPU**: Optional NVIDIA GPU with CUDA support
- **Network**: Access to Ollama and/or OpenAI API

### Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)
- Rust 1.70+ (for local development)
- Ollama (for local models)

## Environment Configuration

### Required Environment Variables

```bash
# Service Configuration
PORT=3010
HOST=0.0.0.0
LOG_LEVEL=info

# Model Providers
OLLAMA_URL=http://localhost:11434
OPENAI_API_KEY=your_openai_api_key

# Default Model
DEFAULT_MODEL=nomic-embed-text
```

### Optional Environment Variables

```bash
# Performance Tuning
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT_SECONDS=60
BATCH_SIZE=32
CACHE_SIZE=1000

# Model Configuration
MODEL_CACHE_DIR=/app/models
GPU_ENABLED=true
GPU_MEMORY_FRACTION=0.8

# Cache Configuration
REDIS_URL=redis://localhost:6379
CACHE_TTL_SECONDS=3600

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
RATE_LIMIT_BURST=100
```

## Docker Deployment

### Standard Dockerfile
```bash
# Build the image
docker build -t embeddings-service:latest .

# Run with environment variables
docker run -d \
  --name embeddings-service \
  -p 3010:3010 \
  -e OLLAMA_URL=http://localhost:11434 \
  -e OPENAI_API_KEY=your_api_key \
  embeddings-service:latest
```

### Docker Compose with Ollama

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  embeddings-service:
    build: .
    ports:
      - "3010:3010"
    environment:
      - OLLAMA_URL=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEFAULT_MODEL=nomic-embed-text
      - CACHE_SIZE=1000
    depends_on:
      - ollama
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3010/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - model_cache:/app/models

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  model_cache:
  ollama_data:
  redis_data:
```

Deploy with:
```bash
docker-compose up -d

# Pull models
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull qwen2.5:7b
```

### GPU-enabled Docker Compose

```yaml
version: '3.8'

services:
  embeddings-service:
    build: .
    ports:
      - "3010:3010"
    environment:
      - OLLAMA_URL=http://ollama:11434
      - GPU_ENABLED=true
      - GPU_MEMORY_FRACTION=0.8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - ollama-gpu
    volumes:
      - model_cache:/app/models

  ollama-gpu:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  model_cache:
  ollama_data:
```

## Kubernetes Deployment

### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: embeddings-service
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: embeddings-service-config
  namespace: embeddings-service
data:
  PORT: "3010"
  HOST: "0.0.0.0"
  LOG_LEVEL: "info"
  OLLAMA_URL: "http://ollama:11434"
  DEFAULT_MODEL: "nomic-embed-text"
  CACHE_SIZE: "1000"
```

### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: embeddings-service-secrets
  namespace: embeddings-service
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embeddings-service
  namespace: embeddings-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embeddings-service
  template:
    metadata:
      labels:
        app: embeddings-service
    spec:
      containers:
      - name: embeddings-service
        image: embeddings-service:latest
        ports:
        - containerPort: 3010
        envFrom:
        - configMapRef:
            name: embeddings-service-config
        - secretRef:
            name: embeddings-service-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3010
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3010
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 10Gi
```

### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: embeddings-service
  namespace: embeddings-service
spec:
  selector:
    app: embeddings-service
  ports:
  - name: http
    port: 3010
    targetPort: 3010
  type: ClusterIP
```

### Ollama Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: embeddings-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        volumeMounts:
        - name: ollama-data
          mountPath: /root/.ollama
      volumes:
      - name: ollama-data
        persistentVolumeClaim:
          claimName: ollama-pvc
```

### Persistent Volume Claim
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-pvc
  namespace: embeddings-service
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

## Local Development

### Prerequisites Setup
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install ollama openai sentence-transformers numpy torch

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### Running the Service

#### Development Server
```bash
# Set environment variables
export OLLAMA_URL=http://localhost:11434
export OPENAI_API_KEY=your_api_key
export DEFAULT_MODEL=nomic-embed-text

# Build and run
cargo run

# Or for development with hot reload
cargo watch -x run
```

#### Testing
```bash
# Run tests
cargo test

# Run integration tests
cargo test --test integration

# Run with coverage
cargo tarpaulin --out Html
```

### Model Setup
```bash
# Pull models
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# List available models
ollama list

# Test model
ollama run nomic-embed-text "Hello, world!"
```

## Monitoring and Logging

### Health Checks
- **Endpoint**: `/health`
- **Method**: GET
- **Response**: Service status, model availability, cache statistics

### Metrics
Monitor these key metrics:
- Request rate and response times
- Model usage statistics
- Cache hit/miss ratios
- Error rates by model and provider
- GPU utilization (if applicable)
- Memory usage patterns

### Logging
The service uses structured logging:
```rust
use tracing::{info, warn, error};

info!("Embedding generated", model = "nomic-embed-text", tokens = 100);
warn!("Model overload detected", model = "qwen2.5:7b", queue_size = 50);
error!("Embedding failed", error = %error, model = "nomic-embed-text");
```

### Prometheus Metrics
```rust
use prometheus::{Counter, Histogram, Gauge};

let embedding_requests_total = Counter::new("embedding_requests_total", "Total embedding requests");
let embedding_duration = Histogram::new("embedding_duration_seconds", "Embedding generation duration");
let cache_size = Gauge::new("cache_size", "Current cache size");
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker logs embeddings-service

# Common causes:
# - Missing Ollama connection
# - Invalid OpenAI API key
# - Port conflicts
```

#### Model Loading Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull missing models
ollama pull nomic-embed-text

# Check GPU availability
nvidia-smi
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats embeddings-service

# Check cache performance
curl http://localhost:3010/health | jq '.cache'

# Profile with perf
perf record -g cargo run
perf report
```

#### Memory Issues
```bash
# Check memory usage
docker stats --no-stream embeddings-service

# Clear model cache
curl -X DELETE http://localhost:3010/api/v1/cache

# Restart service
docker restart embeddings-service
```

### Debug Mode
Enable debug logging:
```bash
export RUST_LOG=debug
export LOG_LEVEL=debug
cargo run
```

### Model Debugging
```bash
# Test model directly
ollama run nomic-embed-text "test text"

# Check model info
ollama show nomic-embed-text

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Scaling Considerations

### Horizontal Scaling
- **Stateless Design**: Multiple instances can run behind a load balancer
- **Model Distribution**: Different models on different instances
- **Cache Coordination**: Redis for shared cache
- **Load Balancing**: Round-robin or least-connections

### Vertical Scaling
- **Memory Scaling**: More memory for larger models
- **GPU Scaling**: Multiple GPUs for parallel inference
- **CPU Scaling**: More cores for concurrent requests
- **Storage Scaling**: Faster SSD for model loading

### Model Optimization
- **Quantization**: Reduce model size with minimal quality loss
- **Pruning**: Remove unnecessary model parameters
- **Distillation**: Use smaller, faster models
- **Caching**: Cache frequently used embeddings

## Security

### Production Security
- **API Authentication**: Implement API key or token-based auth
- **Rate Limiting**: Prevent abuse and ensure fair usage
- **Input Validation**: Sanitize all inputs
- **HTTPS**: Use TLS in production

### Network Security
- **Firewall Rules**: Restrict access to necessary ports
- **VPC/Private Networks**: Isolate service communication
- **Ingress/Egress**: Control network traffic
- **DDoS Protection**: Implement rate limiting and filtering

### Data Security
- **Encryption**: Encrypt data at rest and in transit
- **Access Controls**: Implement role-based access
- **Audit Logging**: Log all access and modifications
- **Privacy**: Ensure user data privacy

## Cost Optimization

### OpenAI API Costs
- **Model Selection**: Use smaller models when possible
- **Batch Processing**: Reduce API calls with batching
- **Caching**: Cache results to avoid repeated calls
- **Monitoring**: Track token usage and costs

### Infrastructure Costs
- **Right-sizing**: Choose appropriate instance sizes
- **Auto-scaling**: Scale based on demand
- **Spot Instances**: Use spot instances for non-critical workloads
- **Resource Sharing**: Share resources across services

### Model Costs
- **Local Models**: Free but require hardware
- **Cloud Models**: Pay-per-token but no hardware
- **Hybrid Approach**: Use local for common, cloud for specialized
- **Model Optimization**: Reduce model size and inference cost
