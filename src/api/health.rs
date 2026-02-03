//! Health check endpoints

use axum::{
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use crate::core::Config;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub timestamp: String,
    pub components: ComponentHealth,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub database: String,
    pub cache: String,
    pub models_loaded: usize,
}

pub async fn health_check() -> Result<Json<HealthResponse>, StatusCode> {
    let response = HealthResponse {
        status: "healthy".to_string(),
        service: "embeddings-service".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        components: ComponentHealth {
            database: "connected".to_string(), // TODO: Actually check
            cache: "connected".to_string(),   // TODO: Actually check
            models_loaded: 0,                 // TODO: Actually count
        },
    };

    Ok(Json(response))
}
