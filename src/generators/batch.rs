//! Batch embedding generation

use crate::core::Result;
use crate::models::{ModelManager, EmbeddingModel};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchEmbeddingRequest {
    pub texts: Vec<String>,
    pub model: String,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimension: usize,
    pub processing_time_ms: u64,
    pub total_texts: usize,
}

pub struct BatchGenerator {
    model_manager: ModelManager,
}

impl BatchGenerator {
    pub fn new(model_manager: ModelManager) -> Self {
        Self { model_manager }
    }

    pub async fn generate(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let start_time = Instant::now();
        
        let model = self.model_manager.ensure_model_loaded(&request.model).await?;
        let batch_size = request.batch_size.unwrap_or(32);
        
        let embeddings = model.generate_batch(request.texts, batch_size).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(BatchEmbeddingResponse {
            embeddings,
            model: request.model,
            dimension: model.dimension(),
            processing_time_ms: processing_time,
            total_texts: embeddings.len(),
        })
    }

    pub async fn generate_single(&self, text: String, model_name: &str) -> Result<Vec<f32>> {
        let request = BatchEmbeddingRequest {
            texts: vec![text],
            model: model_name.to_string(),
            batch_size: Some(1),
        };
        
        let response = self.generate(request).await?;
        response.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::core::EmbeddingError::GenerationError("No embedding generated".to_string()))
    }
}
