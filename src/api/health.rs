//! Health check endpoints

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub timestamp: String,
    pub falcordb_connected: bool,
    pub model_loaded: bool,
}

pub async fn health_check(
    State(app_state): State<AppState>,
) -> Result<Json<HealthResponse>, StatusCode> {
    // Check FalcorDB connectivity
    let falcordb_connected = if let Some(falcordb_client) = &app_state.falcordb_client {
        falcordb_client.health_check().await.is_ok()
    } else {
        false
    };
    
    // Check if default model is loaded
    let model_loaded = app_state.model_manager
        .get_model("sentence-transformers/all-MiniLM-L6-v2")
        .await
        .is_ok();
    
    // Determine overall health status
    let status = if model_loaded {
        "healthy"
    } else {
        "unhealthy"
    };
    
    let response = HealthResponse {
        status: status.to_string(),
        service: "embeddings-service".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        falcordb_connected,
        model_loaded,
    };

    if status == "healthy" {
        Ok(Json(response))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}
