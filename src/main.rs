//! Main entry point for the embeddings service.
//! Provides REST endpoints for embedding generation via Kafka communication with unified-processor.

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
    api::{generate_embeddings, generate_batch_embeddings},
    Config,
    models::ModelManager,
};


#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    // Check Kafka health before starting (Kafka is required)
    tracing::info!("Checking Kafka connectivity...");
    use confuse_common::events::EventProducer;
    EventProducer::new(&config.kafka.bootstrap_servers)
        .map_err(|e| anyhow::anyhow!("Kafka is required but not available: {}", e))?;
    tracing::info!("Kafka health check passed");

    // Initialize service components
    let model_manager = Arc::new(ModelManager::new(config.clone()));
    

    // Ensure default model is loaded at startup
    model_manager.ensure_model_loaded(&config.models.default_model)
        .await?;

    let app_state = embeddings_service::AppState {
        model_manager: model_manager.clone(),
    };

    // Initialize and start Kafka worker (required for service to function)
    let kafka_worker = embeddings_service::infra::kafka_worker::KafkaWorker::new(
        config.clone(),
        model_manager.clone(),
    )?;

    tokio::spawn(async move {
        if let Err(e) = kafka_worker.start().await {
            tracing::error!("Kafka worker failed: {}", e);
        }
    });

    // -- Shared Middleware (from confuse-common) --
    let auth_service_url = std::env::var("AUTH_MIDDLEWARE_URL")
        .unwrap_or_else(|_| "http://auth-middleware:3010".to_string());
    let auth_grpc_url = std::env::var("AUTH_GRPC_URL")
        .unwrap_or_else(|_| "http://localhost:50058".to_string());
    let auth_bypass = std::env::var("AUTH_BYPASS_ENABLED")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
    let auth_layer = confuse_common::middleware::AxumAuthLayer::with_grpc(
        auth_service_url,
        auth_grpc_url,
        auth_bypass,
    ).await;

    let rate_limit = confuse_common::middleware::AxumRateLimitConfig::default_for_service(20);

    // Build router — generate-only endpoints (Kafka-based communication with unified-processor)
    let app = Router::new()
        // Generation endpoints
        .route("/api/v1/generate", post(generate_embeddings))
        .route("/api/v1/generate/batch", post(generate_batch_embeddings))
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

