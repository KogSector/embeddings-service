//! Health check endpoints

use axum::{
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub timestamp: String,
}

pub async fn health_check() -> Result<Json<HealthResponse>, StatusCode> {
    let response = HealthResponse {
        status: "healthy".to_string(),
        service: "embeddings-service".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    Ok(Json(response))
}
