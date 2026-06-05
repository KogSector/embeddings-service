use crate::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel as FastEmbedModelEnum};
use std::sync::{Arc, Mutex};

use super::models::EmbeddingModel;

/// FastEmbed-based embedding model (local in-process inference via ONNX)
pub struct FastEmbedModel {
    name: String,
    dimension: usize,
    model: Arc<Mutex<TextEmbedding>>,
}

impl FastEmbedModel {
    pub fn new(model_name: &str, _config: &Config) -> Result<Self> {
        // Normalize model name: strip `:latest` or similar tag suffixes
        // so "nomic-embed-text:latest" resolves the same as "nomic-embed-text".
        let normalized_name = model_name.split(':').next().unwrap_or(model_name);

        // Map model name to fastembed's EmbeddingModel enum.
        // FastEmbed supports many models out of the box.
        let (fastembed_model, dimension) = match normalized_name {
            "nomic-embed-text" => (FastEmbedModelEnum::NomicEmbedTextV15, 768),
            "all-minilm" => (FastEmbedModelEnum::AllMiniLML6V2, 384),
            "bge-small-en-v1.5" => (FastEmbedModelEnum::BGESmallENV15, 384),
            "mxbai-embed-large" => (FastEmbedModelEnum::MxbaiEmbedLargeV1, 1024),
            _ => (FastEmbedModelEnum::NomicEmbedTextV15, 768), // Default
        };

        let model = TextEmbedding::try_new(InitOptions::new(fastembed_model).with_show_download_progress(true))
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to initialize FastEmbed model: {}", e)))?;

        Ok(Self {
            name: model_name.to_string(),
            dimension,
            model: Arc::new(Mutex::new(model)),
        })
    }
}

#[async_trait]
impl EmbeddingModel for FastEmbedModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // FastEmbed processes embeddings synchronously using Rayon (parallel CPU threads)
        // We use spawn_blocking to avoid blocking the async executor
        let model = Arc::clone(&self.model);
        
        let result = tokio::task::spawn_blocking(move || {
            let mut locked_model = model.lock().unwrap();
            locked_model.embed(texts, None)
        })
        .await
        .map_err(|e| EmbeddingError::GenerationError(format!("Task execution failed: {}", e)))?
        .map_err(|e| EmbeddingError::GenerationError(format!("FastEmbed inference failed: {}", e)))?;

        Ok(result)
    }
}
