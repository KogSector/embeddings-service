//! Embeddings Service Library
//! 
//! A high-performance embedding generation service
//! for the ConFuse platform (generate-only, Kafka-based).

pub mod api;
pub mod config;
pub mod error;
pub mod generators;
pub mod models;
pub mod infra;
// Events and storage removed for stateless embedding logic

// Include generated protobuf code (removed - no longer needed for Kafka-only architecture)
/*
pub mod proto {
    pub mod confuse {
        pub mod embeddings {
            pub mod v1 {
                include!(concat!(env!("OUT_DIR"), "/confuse.embeddings.v1.rs"));
            }
        }
    }
}
*/

use std::sync::Arc;
pub use crate::config::Config;
pub use crate::error::{EmbeddingError, Result};
pub use crate::models::{ModelManager, EmbeddingModel};


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
            .map_err(|e| crate::EmbeddingError::GenerationError(format!("Failed to generate embedding: {}", e)))?;
        
        let embedding = embedding.into_iter().next()
            .ok_or_else(|| crate::EmbeddingError::GenerationError("No embedding generated".to_string()))?;
        
        tracing::debug!(
            "Generated embedding for chunk {} from source {}: {} dimensions",
            chunk_id.unwrap_or("unknown"),
            source_id,
            embedding.len()
        );
        
        Ok(embedding)
    }
}
