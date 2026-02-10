//! Graphiti-compatible API endpoints for the embeddings service

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::models::ModelManager;

#[derive(Debug, Deserialize)]
pub struct GraphitiEmbeddingRequest {
    pub texts: Vec<String>,
    pub model: Option<String>,
    pub embedding_dim: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct GraphitiEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimension: usize,
    pub usage: Option<GraphitiUsage>,
}

#[derive(Debug, Serialize)]
pub struct GraphitiUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct ChunkEmbeddingRequest {
    pub chunks: Vec<DocumentChunk>,
    pub model: Option<String>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct ChunkEmbeddingResponse {
    pub embeddings: Vec<ChunkEmbedding>,
    pub model: String,
    pub total_processed: usize,
    pub failed: usize,
}

#[derive(Debug, Serialize)]
pub struct ChunkEmbedding {
    pub chunk_id: String,
    pub embedding: Vec<f32>,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfoQuery {
    pub format: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct GraphitiModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub dimension: usize,
    pub max_tokens: Option<u32>,
    pub supports_graphiti: bool,
}

/// Generate embeddings in Graphiti-compatible format
pub async fn generate_graphiti_embeddings(
    State(app_state): State<crate::AppState>,
    Json(request): Json<GraphitiEmbeddingRequest>,
) -> Result<Json<GraphitiEmbeddingResponse>, StatusCode> {
    let model_manager = &app_state.model_manager;
    let model_name = request.model.as_deref().unwrap_or("nomic-embed-text");
    
    // Ensure model is loaded
    let model = match model_manager.ensure_model_loaded(model_name).await {
        Ok(model) => model,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    // Generate embeddings
    let embeddings = match model.generate(request.texts).await {
        Ok(embeddings) => embeddings,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    let dimension = model.dimension();
    
    let response = GraphitiEmbeddingResponse {
        embeddings,
        model: model_name.to_string(),
        dimension,
        usage: Some(GraphitiUsage {
            prompt_tokens: 0,
            total_tokens: 0,
        }),
    };

    Ok(Json(response))
}

/// Process document chunks with embeddings
pub async fn process_chunks(
    State(app_state): State<crate::AppState>,
    Json(request): Json<ChunkEmbeddingRequest>,
) -> Result<Json<ChunkEmbeddingResponse>, StatusCode> {
    let model_manager = &app_state.model_manager;
    let model_name = request.model.as_deref().unwrap_or("nomic-embed-text");
    let batch_size = request.batch_size.unwrap_or(32);
    
    // Ensure model is loaded
    let model = match model_manager.ensure_model_loaded(model_name).await {
        Ok(model) => model,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    let mut chunk_embeddings = Vec::new();
    let mut failed = 0;

    // Process chunks in batches
    for chunk_batch in request.chunks.chunks(batch_size) {
        let texts: Vec<String> = chunk_batch.iter().map(|chunk| chunk.content.clone()).collect();
        
        match model.generate_batch(texts, batch_size).await {
            Ok(batch_embeddings) => {
                for (chunk, embedding) in chunk_batch.iter().zip(batch_embeddings) {
                    chunk_embeddings.push(ChunkEmbedding {
                        chunk_id: chunk.id.clone(),
                        embedding,
                        success: true,
                        error: None,
                    });
                }
            }
            Err(e) => {
                for chunk in chunk_batch {
                    chunk_embeddings.push(ChunkEmbedding {
                        chunk_id: chunk.id.clone(),
                        embedding: vec![],
                        success: false,
                        error: Some(format!("Batch processing failed: {}", e)),
                    });
                    failed += 1;
                }
            }
        }
    }

    let response = ChunkEmbeddingResponse {
        embeddings: chunk_embeddings,
        model: model_name.to_string(),
        total_processed: request.chunks.len(),
        failed,
    };

    Ok(Json(response))
}

/// List models compatible with Graphiti
pub async fn list_graphiti_models(
    State(app_state): State<crate::AppState>,
    Query(_query): Query<ModelInfoQuery>,
) -> Json<Vec<GraphitiModelInfo>> {
    let model_manager = &app_state.model_manager;
    let loaded_models = model_manager.list_models().await;
    let mut models = Vec::new();

    for model_name in loaded_models {
        if let Ok(model) = model_manager.get_model(&model_name).await {
            let model_info = GraphitiModelInfo {
                id: model_name.clone(),
                name: model_name.clone(),
                provider: "local".to_string(),
                dimension: model.dimension(),
                max_tokens: Some(8192),
                supports_graphiti: true,
            };
            models.push(model_info);
        }
    }

    models.extend_from_slice(&[
        GraphitiModelInfo {
            id: "nomic-embed-text".to_string(),
            name: "Nomic Embed Text".to_string(),
            provider: "local".to_string(),
            dimension: 768,
            max_tokens: Some(8192),
            supports_graphiti: true,
        },
    ]);

    Json(models)
}

/// Health check for Graphiti integration
pub async fn graphiti_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "embeddings-service",
        "graphiti_support": true,
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}
