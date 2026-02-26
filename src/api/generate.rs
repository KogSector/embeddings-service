//! Embedding generation endpoints

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::generators::BatchGenerator;
use crate::storage::falcordb_client::VectorChunk;
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub text: String,
    pub model: Option<String>,
    #[serde(default)]
    pub document_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub chunk_index: Option<usize>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct BatchGenerateRequest {
    pub texts: Vec<String>,
    pub model: Option<String>,
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub document_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub embedding: Vec<f32>,
    pub model: String,
    pub dimension: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct BatchGenerateResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimension: usize,
    pub processing_time_ms: u64,
    pub total_texts: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

pub async fn generate_embeddings(
    State(app_state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model_name = request.model.unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    let model_manager = &app_state.model_manager;
    
    let generator = BatchGenerator::new((**model_manager).clone());
    
    let embedding = generator.generate_single(request.text.clone(), &model_name).await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "generation_failed".to_string(),
                    message: "Failed to generate embedding".to_string(),
                    details: Some(e.to_string()),
                })
            )
        })?;
    
    let model = model_manager.get_model(&model_name).await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "model_error".to_string(),
                    message: "Failed to retrieve model information".to_string(),
                    details: Some(e.to_string()),
                })
            )
        })?;
    
    // Store in FalcorDB if client is available and metadata is provided
    let (embedding_id, stored) = if let Some(falcordb_client) = &app_state.falcordb_client {
        if let (Some(doc_id_str), Some(src_id)) = (&request.document_id, &request.source_id) {
            // Parse document_id
            let document_id = Uuid::parse_str(doc_id_str)
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "invalid_document_id".to_string(),
                            message: "Invalid document_id format".to_string(),
                            details: Some(e.to_string()),
                        })
                    )
                })?;
            
            let chunk = VectorChunk {
                id: Uuid::new_v4(),
                embedding: embedding.clone(),
                chunk_text: request.text,
                chunk_index: request.chunk_index.unwrap_or(0),
                document_id,
                source_id: src_id.clone(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                metadata: request.metadata.unwrap_or_else(|| serde_json::json!({})),
            };
            
            match falcordb_client.store_vector_chunk(&chunk).await {
                Ok(id) => (Some(id), Some(true)),
                Err(e) => {
                    tracing::error!("Failed to store vector in FalcorDB: {}", e);
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "storage_failed".to_string(),
                            message: "Failed to store vector in FalcorDB".to_string(),
                            details: Some(e.to_string()),
                        })
                    ));
                }
            }
        } else {
            (None, Some(false))
        }
    } else {
        (None, None)
    };
    
    Ok(Json(GenerateResponse {
        embedding,
        model: model_name,
        dimension: model.dimension(),
        embedding_id,
        stored,
    }))
}

pub async fn generate_batch_embeddings(
    State(app_state): State<AppState>,
    Json(request): Json<BatchGenerateRequest>,
) -> Result<Json<BatchGenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model_name = request.model.unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    let model_manager = &app_state.model_manager;
    
    let generator = BatchGenerator::new((**model_manager).clone());
    
    let batch_request = crate::generators::BatchEmbeddingRequest {
        texts: request.texts.clone(),
        model: model_name.clone(),
        batch_size: request.batch_size,
    };
    
    let response = generator.generate(batch_request).await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "generation_failed".to_string(),
                    message: "Failed to generate batch embeddings".to_string(),
                    details: Some(e.to_string()),
                })
            )
        })?;
    
    // Store in FalcorDB if client is available and metadata is provided
    let (embedding_ids, stored) = if let Some(falcordb_client) = &app_state.falcordb_client {
        if let (Some(doc_id_str), Some(src_id)) = (&request.document_id, &request.source_id) {
            // Parse document_id
            let document_id = Uuid::parse_str(doc_id_str)
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "invalid_document_id".to_string(),
                            message: "Invalid document_id format".to_string(),
                            details: Some(e.to_string()),
                        })
                    )
                })?;
            
            // Create vector chunks for batch storage
            let chunks: Vec<VectorChunk> = response.embeddings.iter()
                .zip(request.texts.iter())
                .enumerate()
                .map(|(idx, (embedding, text))| VectorChunk {
                    id: Uuid::new_v4(),
                    embedding: embedding.clone(),
                    chunk_text: text.clone(),
                    chunk_index: idx,
                    document_id,
                    source_id: src_id.clone(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    metadata: request.metadata.clone().unwrap_or_else(|| serde_json::json!({})),
                })
                .collect();
            
            match falcordb_client.batch_store_chunks(chunks).await {
                Ok(ids) => (Some(ids), Some(true)),
                Err(e) => {
                    tracing::error!("Failed to batch store vectors in FalcorDB: {}", e);
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "storage_failed".to_string(),
                            message: "Failed to store vectors in FalcorDB".to_string(),
                            details: Some(e.to_string()),
                        })
                    ));
                }
            }
        } else {
            (None, Some(false))
        }
    } else {
        (None, None)
    };
    
    Ok(Json(BatchGenerateResponse {
        embeddings: response.embeddings,
        model: response.model,
        dimension: response.dimension,
        processing_time_ms: response.processing_time_ms,
        total_texts: response.total_texts,
        embedding_ids,
        stored,
    }))
}
