//! Main entry point for the embeddings service (generate-only)

use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
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
    EmbeddingsService,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Starting embeddings service on {}:{}", config.server.host, config.server.port);

    // Initialize service (no storage initialization needed)
    let service = EmbeddingsService::new(config.clone()).await?;
    service.initialize_default_model().await?;

    // Start Kafka consumer
    if let Err(e) = crate::core::kafka::start_kafka_consumer(service.clone()).await {
        tracing::warn!("Failed to start Kafka consumer: {}", e);
    }

    let app_state = service.app_state();

    // -- Shared Middleware (from confuse-common) --
    let auth_service_url = std::env::var("AUTH_MIDDLEWARE_URL")
        .unwrap_or_else(|_| "http://auth-middleware:3010".to_string());
    let auth_bypass = std::env::var("AUTH_BYPASS_ENABLED")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
    let auth_layer = confuse_common::middleware::AxumAuthLayer::new(auth_service_url, auth_bypass);

    let rate_limit = confuse_common::middleware::AxumRateLimitConfig::default_for_service(20);

    // Build router — generate-only endpoints
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
        .with_state(app_state)
        .layer(axum::middleware::from_fn(confuse_common::middleware::security_headers_middleware))
        .layer(axum::middleware::from_fn(confuse_common::middleware::zero_trust_middleware))
        .layer(axum::middleware::from_fn_with_state(rate_limit.clone(), confuse_common::middleware::axum_rate_limit_middleware))
        .layer(axum::middleware::from_fn_with_state(auth_layer.clone(), confuse_common::middleware::axum_auth_middleware))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        );

    // Start gRPC server in background
    let grpc_service = service.clone();
    let grpc_config = config.clone();
    let grpc_handle = tokio::spawn(async move {
        if let Err(e) = crate::grpc_server::start_grpc_server(grpc_service, grpc_config).await {
            tracing::error!("gRPC server failed: {}", e);
        }
    });

    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    // Wait for gRPC server to finish
    let _ = grpc_handle.await;

    Ok(())
}
