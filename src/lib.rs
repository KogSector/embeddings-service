//! Embeddings Service Library
//! 
//! A high-performance embedding generation service
//! for the ConFuse platform with FalcorDB vector storage.

pub mod api;
pub mod core;
pub mod generators;
pub mod storage;
pub mod models;
pub mod events;
pub mod grpc_server;

// Include generated protobuf code
pub mod proto {
    pub mod confuse {
        pub mod embeddings {
            pub mod v1 {
                include!(concat!(env!("OUT_DIR"), "/confuse.embeddings.v1.rs"));
            }
        }
    }
}

use std::sync::Arc;
use std::collections::HashMap;
use crate::core::{Config, Result};
use crate::models::ModelManager;
use crate::storage::falcordb_client::FalcorDBClient;
use crate::proto::confuse::embeddings::v1::{ChunkData, ChunkResult, ProcessChunksResponse};
use crate::events::EmbeddingEventPublisher;

// Application state for Axum
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
    pub falcordb_client: Option<Arc<FalcorDBClient>>,
}

pub struct EmbeddingsService {
    pub config: Config,
    pub model_manager: Arc<ModelManager>,
    pub falcordb_client: Option<Arc<FalcorDBClient>>,
    pub event_publisher: EmbeddingEventPublisher,
}

impl EmbeddingsService {
    pub fn new(
        config: Config,
        model_manager: Arc<ModelManager>,
        falcordb_client: Option<Arc<FalcorDBClient>>,
        event_publisher: EmbeddingEventPublisher,
    ) -> Self {
        Self {
            config,
            model_manager,
            falcordb_client,
            event_publisher,
        }
    }

    pub async fn _falcordb_client() -> Option<Arc<FalcorDBClient>> {
        use crate::storage::falcordb_client::FalcorDBConfig;
        
        match FalcorDBConfig::from_env() {
            Ok(config) => {
                match FalcorDBClient::new(config).await {
                    Ok(client) => {
                        tracing::info!("FalcorDB client initialized successfully");
                        Some(Arc::new(client))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize FalcorDB client: {}. Vector storage will be disabled.", e);
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!("FalcorDB configuration not found: {}. Vector storage will be disabled.", e);
                None
            }
        }
    }

    pub async fn initialize_default_model(&self) -> Result<()> {
        self.model_manager.load_model(&self.config.models.default_model).await
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    
    pub fn app_state(self) -> AppState {
        AppState {
            model_manager: self.model_manager,
            falcordb_client: self.falcordb_client,
        }
    }

    /// Process chunks from unified-processor and store them as Chunk nodes in FalkorDB
    pub async fn process_and_store_chunks(
        &self,
        chunks: Vec<ChunkData>,
        model: &str,
        _options: HashMap<String, String>,
    ) -> Result<ProcessChunksResponse> {
        if chunks.is_empty() {
            return Ok(ProcessChunksResponse {
                results: Vec::new(),
                model_used: model.to_string(),
                total_chunks: 0,
                chunks_stored: 0,
                chunks_failed: 0,
                errors: Vec::new(),
            });
        }

        let total_chunks = chunks.len();
        let mut results = Vec::with_capacity(total_chunks);
        let mut chunks_stored = 0;
        let mut chunks_failed = 0;
        let mut errors = Vec::new();

        tracing::info!("Processing {} chunks with model: {}", total_chunks, model);

        // Ensure model is loaded
        let embedding_model = self.model_manager.ensure_model_loaded(model).await?;

        // Extract texts for batch embedding generation
        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
        
        // Generate embeddings for all texts
        let embeddings = embedding_model.generate(texts).await
            .map_err(|e| {
                tracing::error!("Failed to generate embeddings: {}", e);
                crate::core::EmbeddingError::GenerationError(format!("Batch embedding generation failed: {}", e))
            })?;

        // Store each chunk with its embedding in FalcorDB
        let mut stored_chunks = Vec::new();
        if let Some(falcordb_client) = &self.falcordb_client {
            for (i, chunk) in chunks.iter().enumerate() {
                let embedding: &Vec<f32> = embeddings.get(i).ok_or_else(|| {
                    crate::core::EmbeddingError::GenerationError("Missing embedding for chunk".to_string())
                })?;

                // Convert ChunkData to VectorChunk
                let vector_chunk = crate::storage::falcordb_client::VectorChunk {
                    id: uuid::Uuid::parse_str(&chunk.chunk_id)
                        .map_err(|e| crate::core::EmbeddingError::InvalidInput(format!("Invalid chunk_id: {}", e)))?,
                    embedding: embedding.clone(),
                    chunk_text: chunk.content.clone(),
                    chunk_index: i,
                    document_id: uuid::Uuid::parse_str(&chunk.source_id)
                        .map_err(|e| crate::core::EmbeddingError::InvalidInput(format!("Invalid source_id: {}", e)))?,
                    source_id: chunk.source_id.clone(),
                    created_at: chrono::DateTime::from_timestamp(chunk.created_at, 0)
                        .unwrap_or_else(|| chrono::Utc::now()),
                    updated_at: chrono::Utc::now(),
                    metadata: self.convert_chunk_metadata(chunk)?,
                };

                match falcordb_client.store_vector_chunk(&vector_chunk).await {
                    Ok(stored_id) => {
                        results.push(ChunkResult {
                            chunk_id: chunk.chunk_id.clone(),
                            stored_node_id: stored_id,
                            stored: true,
                            error: None,
                        });
                        chunks_stored += 1;
                        stored_chunks.push(vector_chunk);
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to store chunk {}: {}", chunk.chunk_id, e);
                        tracing::error!("{}", error_msg);
                        results.push(ChunkResult {
                            chunk_id: chunk.chunk_id.clone(),
                            stored_node_id: String::new(),
                            stored: false,
                            error: Some(error_msg.clone()),
                        });
                        chunks_failed += 1;
                        errors.push(error_msg.clone());
                        
                        // Publish processing failed event for this chunk
                        if let Err(pub_err) = self.event_publisher.publish_processing_failed(
                            &chunk.source_id,
                            &chunk.chunk_id,
                            &error_msg,
                            "STORAGE_ERROR"
                        ).await {
                            tracing::error!("Failed to publish processing failed event: {}", pub_err);
                        }
                    }
                }
            }
        } else {
            let error_msg = "FalcorDB client not available".to_string();
            tracing::error!("{}", error_msg);
            return Err(crate::core::EmbeddingError::ConfigError(error_msg));
        }

        // Publish embedding generated event for successfully stored chunks
        if !stored_chunks.is_empty() {
            let start_time = std::time::Instant::now();
            if let Err(pub_err) = self.event_publisher.publish_embedding_generated(
                &stored_chunks.first().map(|c| c.source_id.as_str()).unwrap_or("unknown"),
                &stored_chunks,
                model,
                start_time.elapsed().as_millis() as u64
            ).await {
                tracing::error!("Failed to publish embedding generated event: {}", pub_err);
            }
        }

        tracing::info!("Processed {} chunks: {} stored, {} failed", total_chunks, chunks_stored, chunks_failed);

        Ok(ProcessChunksResponse {
            results,
            model_used: model.to_string(),
            total_chunks: total_chunks as i32,
            chunks_stored: chunks_stored as i32,
            chunks_failed: chunks_failed as i32,
            errors,
        })
    }

    /// Convert ChunkData metadata to VectorChunk metadata format
    fn convert_chunk_metadata(&self, chunk: &ChunkData) -> Result<serde_json::Value> {
        let mut metadata = serde_json::Map::new();
        
        metadata.insert("file_path".to_string(), serde_json::Value::String(chunk.file_path.clone()));
        metadata.insert("chunk_type".to_string(), serde_json::Value::String(chunk.chunk_type.clone()));
        metadata.insert("level".to_string(), serde_json::Value::String(chunk.level.clone()));
        metadata.insert("tier".to_string(), serde_json::Value::String(chunk.tier.clone()));
        metadata.insert("confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(chunk.confidence as f64).unwrap()));
        
        // Add line range if present
        if let Some(ref chunk_metadata) = chunk.metadata {
            if !chunk_metadata.line_range.is_empty() {
                metadata.insert("line_range".to_string(), serde_json::Value::Array(
                    chunk_metadata.line_range.iter().map(|&v| serde_json::Value::Number(serde_json::Number::from(v))).collect()
                ));
            }
            
            // Add complexity score
            metadata.insert("complexity_score".to_string(), serde_json::Value::Number(serde_json::Number::from(chunk_metadata.complexity_score)));
            metadata.insert("token_count".to_string(), serde_json::Value::Number(serde_json::Number::from(chunk_metadata.token_count)));
            
            // Add custom metadata
            for (key, value) in &chunk_metadata.custom {
                metadata.insert(key.clone(), serde_json::Value::String(value.clone()));
            }
        }

        Ok(serde_json::Value::Object(metadata))
    }

    /// Generate embeddings for a single chunk (internal method for Kafka processing)
    pub async fn generate_embeddings_internal(
        &self,
        content: &str,
        chunk_id: Option<&str>,
        source_id: &str,
    ) -> Result<Vec<f32>> {
        let model = &self.config.models.default_model;
        let embedding_model = self.model_manager.ensure_model_loaded(model).await?;
        
        let embedding = embedding_model
            .generate(vec![content.to_string()])
            .await
            .map_err(|e| crate::core::EmbeddingError::GenerationError(format!("Failed to generate embedding: {}", e)))?;
        
        let embedding = embedding.into_iter().next()
            .ok_or_else(|| crate::core::EmbeddingError::GenerationError("No embedding generated".to_string()))?;
        
        tracing::debug!(
            "Generated embedding for chunk {} from source {}: {} dimensions",
            chunk_id.unwrap_or("unknown"),
            source_id,
            embedding.len()
        );
        
        Ok(embedding)
    }
}
