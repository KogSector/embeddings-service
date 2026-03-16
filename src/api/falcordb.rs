//! FalcorDB integration for storing embeddings directly in graph nodes

use crate::models::ModelManager;
use crate::storage::falcordb_client::{FalcorDBClient, VectorChunk};
use crate::core::{EmbeddingError, Result};
use axum::{
    extract::{Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct FalcorDBEmbeddingRequest {
    pub texts: Vec<String>,
    pub chunk_ids: Vec<String>,  // Corresponding chunk IDs
    pub document_id: String,      // Document ID for all chunks
    pub source_id: String,        // Source identifier
    pub models: Option<Vec<String>>, // Multiple models for ensemble
    pub store_in_falcordb: bool,
    pub chunk_index_start: Option<usize>, // Starting index for chunks
}

#[derive(Debug, Serialize)]
pub struct FalcorDBEmbeddingResponse {
    pub embeddings: Vec<FalcorDBEmbeddingResult>,
    pub models_used: Vec<String>,
    pub fusion_method: String,
    pub falcordb_chunks_created: usize,
    pub postgres_stored: usize,
}

#[derive(Debug, Serialize)]
pub struct FalcorDBEmbeddingResult {
    pub chunk_id: String,
    pub text: String,
    pub embeddings: HashMap<String, Vec<f32>>, // Multiple model embeddings
    pub fused_embedding: Vec<f32>, // Fused/ensemble embedding
    pub quality_score: f32,
    pub falcordb_stored: bool,
    pub postgres_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingFusionConfig {
    pub method: FusionMethod,
    pub weights: Option<HashMap<String, f32>>,
    pub normalize: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionMethod {
    Average,
    WeightedAverage,
    Concatenation,
    MaxPooling,
    Attention,
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::Average
    }
}

/// Store embeddings in FalcorDB with optional PostgreSQL backup
pub async fn store_embeddings_falcordb(
    Query(params): Query<FalcorDBEmbeddingRequest>,
    State(app_state): State<crate::AppState>,
) -> Result<Json<FalcorDBEmbeddingResponse>> {
    let start_time = std::time::Instant::now();

    // Validate request
    if params.texts.is_empty() {
        return Err(core::error::EmbeddingError::InvalidInput("No texts provided for embedding generation".to_string()));
    }

    if params.texts.len() != params.chunk_ids.len() {
        return Err(core::error::EmbeddingError::InvalidInput("Texts and chunk IDs must have the same length".to_string()));
    }

    if params.store_in_falcordb && app_state.falcordb_client.is_none() {
        return Err(core::error::EmbeddingError::ConfigError("FalcorDB storage requested but client not available".to_string()));
    }

    // Generate embeddings for all texts
    let mut embeddings_map = HashMap::new();
    let mut models_used = Vec::new();

    // Use default model or specified models
    let models = if let Some(ref models) = params.models {
        models.clone()
    } else {
        vec!["sentence-transformers/all-MiniLM-L6-v2".to_string()]
    };

    for model in &models {
        match app_state.model_manager.generate(&params.texts, model).await {
            Ok(embeddings) => {
                embeddings_map.insert(model.clone(), embeddings);
                models_used.push(model.clone());
            }
            Err(e) => {
                tracing::error!(
                    model = %model,
                    error = %e,
                    "Failed to generate embeddings with model"
                );
                return Err(core::error::EmbeddingError::GenerationError(format!("Failed to generate embeddings with model {}: {}", model, e)));
            }
        }
    }

    // Fuse embeddings if multiple models
    let fused_embeddings = if embeddings_map.len() > 1 {
        fuse_embeddings(&embeddings_map, &FusionMethod::Average).await?
    } else {
        embeddings_map.values().next().unwrap().clone()
    };

    // Store in FalcorDB if requested
    let mut falcordb_stored_count = 0;
    if params.store_in_falcordb {
        if let Some(ref falcordb_client) = app_state.falcordb_client {
            let chunks: Vec<VectorChunk> = params
                .texts
                .iter()
                .zip(&params.chunk_ids)
                .zip(&fused_embeddings)
                .enumerate()
                .map(|(idx, ((text, chunk_id), embedding))| {
                    let chunk_index = params.chunk_index_start.unwrap_or(0) + idx;
                    let document_uuid = uuid::Uuid::parse_str(&params.document_id)
                        .unwrap_or_else(|_| uuid::Uuid::new_v4());
                    
                    VectorChunk::new(
                        text.clone(),
                        embedding.clone(),
                        document_uuid,
                        params.source_id.clone(),
                        chunk_index,
                    )
                })
                .collect();

            match falcordb_client.batch_store_chunks(chunks).await {
                Ok(stored_ids) => {
                    falcordb_stored_count = stored_ids.len();
                    tracing::info!(
                        stored_count = falcordb_stored_count,
                        "Successfully stored embeddings in FalcorDB"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Failed to store embeddings in FalcorDB"
                    );
                    return Err(core::error::EmbeddingError::ConnectionError(format!("Failed to store embeddings in FalcorDB: {}", e)));
                }
            }
        }
    }

    // Prepare response
    let results: Vec<FalcorDBEmbeddingResult> = params
        .texts
        .iter()
        .zip(&params.chunk_ids)
        .zip(&fused_embeddings)
        .enumerate()
        .map(|(idx, ((text, chunk_id), embedding))| {
            let quality_score = calculate_embedding_quality(&embedding);
            
            FalcorDBEmbeddingResult {
                chunk_id: chunk_id.clone(),
                text: text.clone(),
                embeddings: embeddings_map
                    .iter()
                    .map(|(model, embeddings)| {
                        (model.clone(), embeddings[idx].clone())
                    })
                    .collect(),
                fused_embedding: fused_embeddings[idx].clone(),
                quality_score,
                falcordb_stored: params.store_in_falcordb && idx < falcordb_stored_count,
                postgres_id: None, // PostgreSQL storage not implemented in this version
            }
        })
        .collect();

    let response = FalcorDBEmbeddingResponse {
        embeddings: results,
        models_used,
        fusion_method: "average".to_string(),
        falcordb_chunks_created: falcordb_stored_count,
        postgres_stored: 0,
    };

    let duration = start_time.elapsed();
    tracing::info!(
        duration_ms = duration.as_millis(),
        chunks_processed = response.embeddings.len(),
        falcordb_stored = falcordb_stored_count,
        "Embeddings generated and stored successfully"
    );

    Ok(Json(response))
}

/// Fuse embeddings from multiple models using specified method
async fn fuse_embeddings(
    embeddings_map: &HashMap<String, Vec<Vec<f32>>>,
    method: &FusionMethod,
) -> Result<Vec<Vec<f32>>> {
    let embedding_dim = embeddings_map.values().next().map(|v| v[0].len()).unwrap_or(0);
    let mut fused_embeddings = Vec::with_capacity(embeddings_map.values().next().map(|v| v.len()).unwrap_or(0));

    for i in 0..embeddings_map.values().next().map(|v| v.len()).unwrap_or(0) {
        let mut fused_embedding = vec![0.0; embedding_dim];

        match method {
            FusionMethod::Average => {
                let mut sum = 0.0;
                for embeddings in embeddings_map.values() {
                    for (j, &val) in embeddings[i].iter().enumerate() {
                        fused_embedding[j] += val;
                    }
                    sum += 1.0;
                }
                for val in &mut fused_embedding {
                    *val /= sum;
                }
            }
            FusionMethod::WeightedAverage => {
                // Simple equal weights for now - can be extended
                let weight = 1.0 / embeddings_map.len() as f32;
                for embeddings in embeddings_map.values() {
                    for (j, &val) in embeddings[i].iter().enumerate() {
                        fused_embedding[j] += val * weight;
                    }
                }
            }
            FusionMethod::MaxPooling => {
                for embeddings in embeddings_map.values() {
                    for (j, &val) in embeddings[i].iter().enumerate() {
                        if val > fused_embedding[j] {
                            fused_embedding[j] = val;
                        }
                    }
                }
            }
            FusionMethod::Concatenation => {
                // Not implemented for this example
                return Err(EmbeddingError::InvalidInput(
                    "Concatenation fusion not implemented".to_string(),
                ));
            }
            FusionMethod::Attention => {
                // Not implemented for this example
                return Err(EmbeddingError::InvalidInput(
                    "Attention fusion not implemented".to_string(),
                ));
            }
        }

        fused_embeddings.push(fused_embedding);
    }

    Ok(fused_embeddings)
}

/// Calculate quality score for an embedding based on various metrics
fn calculate_embedding_quality(embedding: &[f32]) -> f32 {
    if embedding.is_empty() {
        return 0.0;
    }

    // Simple quality metrics
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let variance: f32 = embedding.iter().map(|x| (x - embedding.iter().sum::<f32>() / embedding.len() as f32).powi(2)).sum::<f32>() / embedding.len() as f32;
    
    // Quality score based on normalization and variance
    let normalization_score = if norm > 0.0 && norm < 10.0 { 1.0 } else { 0.5 };
    let variance_score = if variance > 0.01 && variance < 1.0 { 1.0 } else { 0.5 };
    
    (normalization_score + variance_score) / 2.0
}

/// Get FalcorDB statistics
pub async fn get_falcordb_stats(
    State(app_state): State<crate::AppState>,
) -> Result<Json<serde_json::Value>> {
    if let Some(ref falcordb_client) = app_state.falcordb_client {
        match falcordb_client.health_check().await {
            Ok(_) => {
                let stats = serde_json::json!({
                    "status": "healthy",
                    "database": "falkordb",
                    "vector_storage": "enabled",
                    "connection": {
                        "host": falcordb_client.config().host,
                        "port": falcordb_client.config().port,
                        "database": falcordb_client.config().database
                    },
                    "configuration": {
                        "vector_dimension": falcordb_client.config().vector_dimension,
                        "similarity_threshold": falcordb_client.config().similarity_threshold,
                        "max_results": falcordb_client.config().max_results
                    }
                });
                Ok(Json(stats))
            }
            Err(e) => {
                tracing::error!(error = %e, "FalcorDB health check failed");
                return Err(core::error::EmbeddingError::ConnectionError(format!("FalcorDB health check failed: {}", e)));
            }
        }
    } else {
        let stats = serde_json::json!({
            "status": "disabled",
            "database": "falkordb",
            "vector_storage": "disabled",
            "reason": "FalcorDB client not initialized"
        });
        Ok(Json(stats))
    }
}

/// Test FalcorDB connectivity
pub async fn test_falcordb_connection(
    State(app_state): State<crate::AppState>,
) -> Result<Json<serde_json::Value>> {
    if let Some(ref falcordb_client) = app_state.falcordb_client {
        match falcordb_client.health_check().await {
            Ok(_) => {
                let response = serde_json::json!({
                    "status": "success",
                    "message": "FalcorDB connection successful",
                    "database": {
                        "host": falcordb_client.config().host,
                        "port": falcordb_client.config().port,
                        "database": falcordb_client.config().database
                    }
                });
                Ok(Json(response))
            }
            Err(e) => {
                tracing::error!(error = %e, "FalcorDB connection test failed");
                return Err(core::error::EmbeddingError::ConnectionError(format!("FalcorDB connection test failed: {}", e)));
                return Err(core::error::EmbeddingError::ConfigError("FalcorDB client not available".to_string()));
            }
        }
    } else {
        return Err(core::error::EmbeddingError::ConfigError("FalcorDB client not available".to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_method_default() {
        let method = FusionMethod::default();
        assert!(matches!(method, FusionMethod::Average));
    }

    #[test]
    fn test_calculate_embedding_quality() {
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let quality = calculate_embedding_quality(&embedding);
        assert!(quality > 0.0);
        assert!(quality <= 1.0);
    }

    #[test]
    fn test_calculate_embedding_quality_empty() {
        let embedding = vec![];
        let quality = calculate_embedding_quality(&embedding);
        assert_eq!(quality, 0.0);
    }

    #[tokio::test]
    async fn test_fuse_embeddings_average() {
        let mut embeddings_map = HashMap::new();
        embeddings_map.insert("model1".to_string(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        embeddings_map.insert("model2".to_string(), vec![vec![2.0, 3.0], vec![4.0, 5.0]]);

        let result = fuse_embeddings(&embeddings_map, &FusionMethod::Average).await;
        assert!(result.is_ok());

        let fused = result.unwrap();
        assert_eq!(fused.len(), 2);
        assert!((fused[0][0] - 1.5).abs() < 0.001); // (1.0 + 2.0) / 2
        assert!((fused[0][1] - 2.5).abs() < 0.001); // (2.0 + 3.0) / 2
    }

    #[tokio::test]
    async fn test_fuse_embeddings_max_pooling() {
        let mut embeddings_map = HashMap::new();
        embeddings_map.insert("model1".to_string(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        embeddings_map.insert("model2".to_string(), vec![vec![2.0, 3.0], vec![4.0, 5.0]]);

        let result = fuse_embeddings(&embeddings_map, &FusionMethod::MaxPooling).await;
        assert!(result.is_ok());

        let fused = result.unwrap();
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0][0], 2.0); // max(1.0, 2.0)
        assert_eq!(fused[0][1], 3.0); // max(2.0, 3.0)
    }

    #[tokio::test]
    async fn test_fuse_embeddings_concatenation_not_implemented() {
        let mut embeddings_map = HashMap::new();
        embeddings_map.insert("model1".to_string(), vec![vec![1.0, 2.0]]);

        let result = fuse_embeddings(&embeddings_map, &FusionMethod::Concatenation).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not implemented"));
    }
}
