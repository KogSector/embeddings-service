//! Embeddings Service Library
//! 
//! A high-performance embedding generation service
//! for the ConFuse platform with FalcorDB vector storage.

pub mod api;
pub mod core;
pub mod generators;
pub mod models;
// Events and storage removed for stateless embedding logic
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


// Application state for Axum
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
}

pub struct EmbeddingsService {
    pub config: Config,
    pub model_manager: Arc<ModelManager>,
}

impl EmbeddingsService {
    pub fn new(
        config: Config,
        model_manager: Arc<ModelManager>,
    ) -> Self {
        Self {
            config,
            model_manager,
        }
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
