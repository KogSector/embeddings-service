//! Vector search endpoints

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::ModelManager;
use crate::storage::PostgresStorage;

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
    pub model: Option<String>,
    pub limit: Option<usize>,
    pub threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ModelsQuery {
    pub loaded: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub model: String,
    pub results: Vec<SearchResult>,
    pub total_found: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub similarity: f32,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub available: Vec<String>,
    pub loaded: Vec<String>,
    pub default: String,
}

pub async fn search_similar(
    Query(query): Query<SearchQuery>,
    State(app_state): State<crate::AppState>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let model_manager = &app_state.model_manager;
    let storage = &app_state.postgres_storage;
    let model_name = query.model.unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    let limit = query.limit.unwrap_or(10);
    let threshold = query.threshold.unwrap_or(0.7);
    
    // Generate embedding for query
    let generator = crate::generators::BatchGenerator::new((**model_manager).clone());
    
    let query_embedding = generator.generate_single(query.q.clone(), &model_name).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Search for similar vectors
    match storage.search_similar(&query_embedding, &model_name, limit, threshold).await {
        Ok(records) => {
            let results: Vec<SearchResult> = records
                .into_iter()
                .map(|record| SearchResult {
                    id: record.id.to_string(),
                    text: record.text,
                    similarity: 0.0, // TODO: Calculate actual similarity
                    metadata: record.metadata,
                })
                .collect();
            
            let total_found = results.len();
            Ok(Json(SearchResponse {
                query: query.q,
                model: model_name,
                results,
                total_found,
            }))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn list_models(
    Query(_query): Query<ModelsQuery>,
    State(model_manager): State<Arc<ModelManager>>,
) -> Result<Json<ModelsResponse>, StatusCode> {
    let loaded = model_manager.list_models().await;
    let available = vec![
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        "sentence-transformers/all-mpnet-base-v2".to_string(),
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
    ];
    
    Ok(Json(ModelsResponse {
        available,
        loaded,
        default: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
    }))
}
