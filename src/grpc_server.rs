//! Embeddings Service gRPC Server
//! Handles gRPC requests from unified-processor for batch embedding generation
//! and storing Chunk nodes in FalkorDB.

use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};

use crate::AppState;
use crate::proto::confuse::embeddings::v1::{
    embeddings_server::{Embeddings, EmbeddingsServer},
    EmbedRequest, EmbedResponse,
    BatchEmbedRequest, BatchEmbedResponse, EmbeddingResult,
    ModelInfoRequest, ModelInfo,
    ProcessChunksRequest, ProcessChunksResponse, ChunkResult,
};

pub struct EmbeddingsGrpcService {
    state: AppState,
    default_model: String,
}

impl EmbeddingsGrpcService {
    pub fn new(state: AppState, default_model: String) -> Self {
        Self { state, default_model }
    }
}

#[tonic::async_trait]
impl Embeddings for EmbeddingsGrpcService {
    /// Generate a single embedding for one text input.
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        let model_name = req.model.unwrap_or_else(|| self.default_model.clone());

        tracing::info!("gRPC Embed: text_len={}, model={}", req.text.len(), model_name);

        let model = self.state.model_manager
            .ensure_model_loaded(&model_name)
            .await
            .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        let embeddings = model
            .generate(vec![req.text])
            .await
            .map_err(|e| Status::internal(format!("Embedding generation failed: {e}")))?;

        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| Status::internal("No embedding returned"))?;

        let dimension = embedding.len() as i32;
        let chunk_id = req.chunk_id.unwrap_or_default();

        Ok(Response::new(EmbedResponse {
            embeddings: embedding,
            dimension,
            model_used: model_name,
            chunk_id,
        }))
    }

    /// Generate embeddings for a batch of texts.
    async fn batch_embed(
        &self,
        request: Request<BatchEmbedRequest>,
    ) -> Result<Response<BatchEmbedResponse>, Status> {
        let req = request.into_inner();
        let model_name = req.model.unwrap_or_else(|| self.default_model.clone());
        let texts_count = req.texts.len();

        tracing::info!("gRPC BatchEmbed: texts={}, model={}", texts_count, model_name);

        if req.texts.is_empty() {
            return Ok(Response::new(BatchEmbedResponse {
                results: vec![],
                model_used: model_name,
            }));
        }

        let model = self.state.model_manager
            .ensure_model_loaded(&model_name)
            .await
            .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        let all_embeddings = model
            .generate(req.texts)
            .await
            .map_err(|e| Status::internal(format!("Batch embedding failed: {e}")))?;

        let results: Vec<EmbeddingResult> = all_embeddings
            .into_iter()
            .enumerate()
            .map(|(idx, emb)| EmbeddingResult {
                embeddings: emb,
                index: idx as i32,
                chunk_id: req.chunk_ids.get(idx).cloned().unwrap_or_default(),
            })
            .collect();

        Ok(Response::new(BatchEmbedResponse {
            results,
            model_used: model_name,
        }))
    }

    /// Return model metadata.
    async fn get_model_info(
        &self,
        request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfo>, Status> {
        let req = request.into_inner();
        let model_name = req.model_name.unwrap_or_else(|| self.default_model.clone());

        tracing::info!("gRPC GetModelInfo: model={}", model_name);

        let model = self.state.model_manager
            .ensure_model_loaded(&model_name)
            .await
            .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        Ok(Response::new(ModelInfo {
            model_name: model.name().to_string(),
            dimension: model.dimension() as i32,
            max_sequence_length: 512, // Standard for sentence-transformers
            provider: if model_name.contains("ollama") { "ollama".to_string() } else { "sentence-transformers".to_string() },
        }))
    }

    /// Process chunks from unified-processor: generate embeddings and store as
    /// Chunk nodes in FalkorDB.
    async fn process_and_store_chunks(
        &self,
        request: Request<ProcessChunksRequest>,
    ) -> Result<Response<ProcessChunksResponse>, Status> {
        let req = request.into_inner();
        let model_name = req.model.unwrap_or_else(|| self.default_model.clone());
        let total_chunks = req.chunks.len() as i32;

        tracing::info!(
            "gRPC ProcessAndStoreChunks: chunks={}, model={}",
            total_chunks,
            model_name
        );

        if req.chunks.is_empty() {
            return Ok(Response::new(ProcessChunksResponse {
                results: vec![],
                model_used: model_name,
                total_chunks: 0,
                chunks_stored: 0,
                chunks_failed: 0,
                errors: vec![],
            }));
        }

        // Ensure model is loaded
        let model = self.state.model_manager
            .ensure_model_loaded(&model_name)
            .await
            .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        // Extract texts for batch embedding
        let texts: Vec<String> = req.chunks.iter().map(|c| c.content.clone()).collect();

        // Generate embeddings for all chunks in one batch call
        let embeddings = model
            .generate(texts)
            .await
            .map_err(|e| Status::internal(format!("Batch embedding failed: {e}")))?;

        let mut results = Vec::with_capacity(req.chunks.len());
        let mut chunks_stored = 0i32;
        let mut chunks_failed = 0i32;
        let mut errors = Vec::new();

        for (i, chunk) in req.chunks.iter().enumerate() {
            let embedding = match embeddings.get(i) {
                Some(e) => e.clone(),
                None => {
                    let msg = format!("Missing embedding for chunk {}", chunk.chunk_id);
                    errors.push(msg.clone());
                    results.push(ChunkResult {
                        chunk_id: chunk.chunk_id.clone(),
                        stored_node_id: String::new(),
                        stored: false,
                        error: Some(msg),
                    });
                    chunks_failed += 1;
                    continue;
                }
            };

            // Store in FalkorDB if client is available
            if let Some(falcordb_client) = &self.state.falcordb_client {
                let source_uuid = uuid::Uuid::parse_str(&chunk.source_id)
                    .unwrap_or_else(|_| uuid::Uuid::new_v4());

                let vector_chunk = crate::storage::falcordb_client::VectorChunk {
                    id: uuid::Uuid::new_v4(),
                    embedding,
                    chunk_text: chunk.content.clone(),
                    chunk_index: i,
                    document_id: source_uuid,
                    source_id: chunk.source_id.clone(),
                    created_at: chrono::DateTime::from_timestamp(chunk.created_at, 0)
                        .unwrap_or_else(|| chrono::Utc::now()),
                    updated_at: chrono::Utc::now(),
                    metadata: serde_json::json!({
                        "file_path": chunk.file_path,
                        "chunk_type": chunk.chunk_type,
                        "level": chunk.level,
                        "tier": chunk.tier,
                        "confidence": chunk.confidence,
                    }),
                };

                match falcordb_client.store_vector_chunk(&vector_chunk).await {
                    Ok(node_id) => {
                        results.push(ChunkResult {
                            chunk_id: chunk.chunk_id.clone(),
                            stored_node_id: node_id,
                            stored: true,
                            error: None,
                        });
                        chunks_stored += 1;
                    }
                    Err(e) => {
                        let msg = format!("Store failed for chunk {}: {e}", chunk.chunk_id);
                        errors.push(msg.clone());
                        results.push(ChunkResult {
                            chunk_id: chunk.chunk_id.clone(),
                            stored_node_id: String::new(),
                            stored: false,
                            error: Some(msg),
                        });
                        chunks_failed += 1;
                    }
                }
            } else {
                // No FalkorDB — log warning, still count as stored so pipeline continues
                tracing::warn!(
                    "FalkorDB not configured; chunk {} will not be persisted",
                    chunk.chunk_id
                );
                results.push(ChunkResult {
                    chunk_id: chunk.chunk_id.clone(),
                    stored_node_id: String::new(),
                    stored: false,
                    error: Some("FalkorDB client not initialized".to_string()),
                });
                chunks_failed += 1;
            }
        }

        tracing::info!(
            "gRPC ProcessAndStoreChunks complete: stored={}, failed={}",
            chunks_stored,
            chunks_failed
        );

        Ok(Response::new(ProcessChunksResponse {
            results,
            model_used: model_name,
            total_chunks,
            chunks_stored,
            chunks_failed,
            errors,
        }))
    }
}

/// Start the embeddings gRPC server on `GRPC_HOST:GRPC_PORT`.
pub async fn start_grpc_server(
    state: AppState,
    default_model: String,
    grpc_host: String,
    grpc_port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{grpc_host}:{grpc_port}").parse()?;

    tracing::info!("Starting embeddings gRPC server on {}", addr);

    let svc = EmbeddingsGrpcService::new(state, default_model);

    Server::builder()
        .add_service(EmbeddingsServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
