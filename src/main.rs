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

    let app_state = service.app_state();

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
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        );

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
