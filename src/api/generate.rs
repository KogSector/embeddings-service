//! Embedding generation endpoints

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::generators::BatchGenerator;
use crate::models::ModelManager;

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub text: String,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BatchGenerateRequest {
    pub texts: Vec<String>,
    pub model: Option<String>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub embedding: Vec<f32>,
    pub model: String,
    pub dimension: usize,
}

#[derive(Debug, Serialize)]
pub struct BatchGenerateResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimension: usize,
    pub processing_time_ms: u64,
    pub total_texts: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

pub async fn generate_embeddings(
    State(model_manager): State<Arc<ModelManager>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, StatusCode> {
    let model_name = request.model.unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    
    let generator = BatchGenerator::new((*model_manager).clone());
    
    match generator.generate_single(request.text, &model_name).await {
        Ok(embedding) => {
            let model = model_manager.get_model(&model_name).await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            Ok(Json(GenerateResponse {
                embedding,
                model: model_name,
                dimension: model.dimension(),
            }))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn generate_batch_embeddings(
    State(model_manager): State<Arc<ModelManager>>,
    Json(request): Json<BatchGenerateRequest>,
) -> Result<Json<BatchGenerateResponse>, StatusCode> {
    let model_name = request.model.unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    
    let generator = BatchGenerator::new((*model_manager).clone());
    
    let batch_request = crate::generators::BatchEmbeddingRequest {
        texts: request.texts,
        model: model_name.clone(),
        batch_size: request.batch_size,
    };
    
    match generator.generate(batch_request).await {
        Ok(response) => Ok(Json(BatchGenerateResponse {
            embeddings: response.embeddings,
            model: response.model,
            dimension: response.dimension,
            processing_time_ms: response.processing_time_ms,
            total_texts: response.total_texts,
        })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}
