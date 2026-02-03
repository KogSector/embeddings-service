# Embeddings Service API Reference

## Base URL
```
http://localhost:3010
```

## Authentication
Currently no authentication is required. This will be updated as the platform security evolves.

## Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "embeddings-service",
  "version": "0.1.0",
  "models": {
    "available": ["nomic-embed-text", "qwen2.5:7b"],
    "default": "nomic-embed-text"
  }
}
```

### Generate Single Embedding
```http
POST /api/v1/embed
```

Generate embedding for a single text input.

**Request Body:**
```json
{
  "text": "string",
  "model": "string (optional)",
  "dimensions": "number (optional)"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "model": "nomic-embed-text",
  "dimensions": 768,
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### Generate Batch Embeddings
```http
POST /api/v1/embed/batch
```

Generate embeddings for multiple texts in a single request.

**Request Body:**
```json
{
  "texts": ["string1", "string2", "string3"],
  "model": "string (optional)",
  "dimensions": "number (optional)"
}
```

**Response:**
```json
{
  "embeddings": [
    {
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    },
    {
      "embedding": [0.4, 0.5, 0.6, ...],
      "index": 1
    },
    {
      "embedding": [0.7, 0.8, 0.9, ...],
      "index": 2
    }
  ],
  "model": "nomic-embed-text",
  "dimensions": 768,
  "usage": {
    "prompt_tokens": 30,
    "total_tokens": 30
  }
}
```

### List Available Models
```http
GET /api/v1/models
```

Get list of available embedding models.

**Response:**
```json
{
  "models": [
    {
      "id": "nomic-embed-text",
      "name": "Nomic Embed Text",
      "provider": "ollama",
      "dimensions": 768,
      "max_tokens": 8192
    },
    {
      "id": "text-embedding-3-small",
      "name": "OpenAI Text Embedding 3 Small",
      "provider": "openai",
      "dimensions": 1536,
      "max_tokens": 8192
    }
  ]
}
```

### Model Information
```http
GET /api/v1/models/{model_id}
```

Get detailed information about a specific model.

**Response:**
```json
{
  "id": "nomic-embed-text",
  "name": "Nomic Embed Text",
  "provider": "ollama",
  "dimensions": 768,
  "max_tokens": 8192,
  "description": "Open source embedding model from Nomic AI",
  "pricing": {
    "input_tokens": "free"
  },
  "capabilities": [
    "text-embedding",
    "semantic-search",
    "clustering"
  ]
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

### Common Error Codes

- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Model not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Service error
- `503 Service Unavailable` - Model service unavailable

### Model-Specific Errors

- `model_not_found` - Requested model is not available
- `model_overloaded` - Model is currently overloaded
- `invalid_model` - Invalid model name or format
- `embedding_failed` - Embedding generation failed

## Request Limits

### Single Embedding
- **Max text length**: 8192 tokens (varies by model)
- **Rate limit**: 100 requests/minute

### Batch Embedding
- **Max batch size**: 100 texts per request
- **Max text length**: 8192 tokens per text
- **Rate limit**: 10 requests/minute

## Model Providers

### Ollama (Local Models)
- **Models**: nomic-embed-text, qwen2.5:7b, custom models
- **Pricing**: Free (local hosting)
- **Latency**: Low (local processing)
- **Setup**: Requires Ollama server

### OpenAI (Cloud Models)
- **Models**: text-embedding-3-small, text-embedding-3-large
- **Pricing**: Pay-per-token
- **Latency**: Medium (network dependent)
- **Setup**: Requires OpenAI API key

## Usage Examples

### cURL Example
```bash
# Single embedding
curl -X POST http://localhost:3010/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# Batch embedding
curl -X POST http://localhost:3010/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World", "Embeddings"]}'
```

### Python Example
```python
import requests

# Single embedding
response = requests.post("http://localhost:3010/api/v1/embed", 
                        json={"text": "Hello, world!"})
embedding = response.json()["embedding"]

# Batch embedding
response = requests.post("http://localhost:3010/api/v1/embed/batch",
                        json={"texts": ["Hello", "World", "Embeddings"]})
embeddings = [item["embedding"] for item in response.json()["embeddings"]]
```

## Performance Considerations

### Optimization Tips
- Use batch processing for multiple texts
- Choose appropriate model size for your use case
- Consider caching frequently used embeddings
- Monitor token usage for cost management

### Benchmarking
- **Single embedding**: ~50ms (local), ~200ms (OpenAI)
- **Batch embedding (10 texts)**: ~200ms (local), ~500ms (OpenAI)
- **Throughput**: ~20 embeddings/second (local), ~5 embeddings/second (OpenAI)

## Monitoring

### Metrics to Monitor
- Request rate and response times
- Error rates by model and endpoint
- Token usage and costs
- Model loading and caching performance

### Health Monitoring
- Model availability checks
- Ollama server connectivity
- OpenAI API rate limits
- Memory and CPU usage
