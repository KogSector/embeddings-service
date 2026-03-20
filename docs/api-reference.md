# Embeddings Service — API Reference

## Overview

The embeddings service generates vector embeddings for text content and stores chunk nodes in FalkorDB. It serves as the embedding + storage layer in the ConFuse pipeline.

**Architecture:** gRPC-primary (called by unified-processor), HTTP for direct access and health checks.

## HTTP Endpoints

### Health Check
```
GET /health
```

### Generate Single Embedding
```
POST /api/v1/generate
```
Generate an embedding for a single text input.

**Request Body:**
```json
{
  "text": "function hello() { return 'world'; }",
  "model": "all-MiniLM-L6-v2"
}
```

**Response:**
```json
{
  "embeddings": [0.023, -0.041, ...],
  "dimension": 384,
  "model_used": "all-MiniLM-L6-v2"
}
```

### Generate Batch Embeddings
```
POST /api/v1/generate/batch
```
Generate embeddings for multiple texts.

### Graphiti Endpoints
```
GET  /api/v1/graphiti/health
POST /api/v1/graphiti/generate
POST /api/v1/graphiti/chunks
GET  /api/v1/graphiti/models
```

## gRPC Service

**Proto:** `proto/embeddings.proto`  
**Package:** `confuse.embeddings.v1`  
**Port:** `50054` (default)

### RPCs

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `Embed` | `EmbedRequest` | `EmbedResponse` | Single embedding generation |
| `BatchEmbed` | `BatchEmbedRequest` | `BatchEmbedResponse` | Batch embedding generation |
| `GetModelInfo` | `ModelInfoRequest` | `ModelInfo` | Model metadata |
| `ProcessAndStoreChunks` | `ProcessChunksRequest` | `ProcessChunksResponse` | **Primary pipeline RPC** — generates embeddings AND stores chunk nodes in FalkorDB |

### ProcessAndStoreChunks (Primary Pipeline RPC)

This is the main RPC called by `unified-processor`. It:
1. Receives `ChunkData` objects with content, metadata, and entity hints
2. Generates vector embeddings for each chunk
3. Stores chunk nodes with embeddings in FalkorDB
4. Returns success/failure counts

**Request:**
```protobuf
message ProcessChunksRequest {
  repeated ChunkData chunks = 1;
  optional string model = 2;
  map<string, string> options = 3;
}
```

**Response:**
```protobuf
message ProcessChunksResponse {
  int32 chunks_stored = 1;
  int32 chunks_failed = 2;
  string model_used = 3;
  repeated string errors = 4;
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8060` | HTTP server port |
| `GRPC_PORT` | `50054` | gRPC server port |
| `FALKORDB_HOST` | `localhost` | FalkorDB host |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `FALKORDB_GRAPH_NAME` | `confuse_knowledge` | FalkorDB graph name |
| `DEFAULT_MODEL` | `all-MiniLM-L6-v2` | Default embedding model |
| `AUTH_MIDDLEWARE_URL` | `http://localhost:3010` | Auth middleware URL |
