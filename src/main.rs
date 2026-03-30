//! Main entry point for the embeddings service
//! gRPC pipeline: unified-processor calls ProcessAndStoreChunks → generates embeddings → stores in FalkorDB

use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use embeddings_service::{
    api::{health_check, generate_embeddings, generate_batch_embeddings,
          generate_graphiti_embeddings, process_chunks, list_graphiti_models, graphiti_health},
    core::Config,
    models::ModelManager,
    EmbeddingsService,
};


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env.map (non-sensitive) and .env.secret (sensitive overrides)
    dotenvy::from_filename(".env.map").ok();
    dotenvy::from_filename(".env.secret").ok();
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Starting embeddings service on {}:{}", config.server.host, config.server.port);

    // Initialize service components
    let model_manager = Arc::new(ModelManager::new(config.clone()));
    

    // Ensure default model is loaded at startup
    model_manager.ensure_model_loaded(&config.models.default_model)
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let app_state = embeddings_service::AppState {
        model_manager: model_manager.clone(),
    };

    // -- Shared Middleware (from confuse-common) --
    let auth_service_url = std::env::var("AUTH_MIDDLEWARE_URL")
        .unwrap_or_else(|_| "http://auth-middleware:3010".to_string());
    let auth_bypass = std::env::var("AUTH_BYPASS_ENABLED")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
    let auth_layer = confuse_common::middleware::AxumAuthLayer::new(auth_service_url, auth_bypass);

    let rate_limit = confuse_common::middleware::AxumRateLimitConfig::default_for_service(20);

    // Build router — generate-only endpoints with FalcorDB storage
    let app = Router::new()
        // Health
        .route("/health", get(health_check))
        // Generation endpoints
        .route("/api/v1/generate", post(generate_embeddings))
        .route("/api/v1/generate/batch", post(generate_batch_embeddings))
        // Graphiti endpoints
        .route("/api/v1/graphiti/health", get(graphiti_health))
        .route("/api/v1/graphiti/generate", post(generate_graphiti_embeddings))
        .route("/api/v1/graphiti/chunks", post(process_chunks))
        .route("/api/v1/graphiti/models", get(list_graphiti_models))
        // FalcorDB endpoints (commented out for now)
        // .route("/api/v1/falcordb/store", post(store_embeddings_falcordb))\
        // .route("/api/v1/falcordb/stats", get(get_falcordb_stats))
        // .route("/api/v1/falcordb/test", get(test_falcordb_connection))
        .with_state(app_state.clone())
        .layer(axum::middleware::from_fn(confuse_common::middleware::security_headers_middleware))
        .layer(axum::middleware::from_fn(confuse_common::middleware::zero_trust_middleware))
        .layer(axum::middleware::from_fn_with_state(rate_limit.clone(), confuse_common::middleware::axum_rate_limit_middleware))
        .layer(axum::middleware::from_fn_with_state(auth_layer.clone(), confuse_common::middleware::axum_auth_middleware))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        );


    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Starting HTTP server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

