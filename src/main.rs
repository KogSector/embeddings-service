//! Main entry point for the embeddings service

use axum::{
    routing::{get, post, delete},
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
    api::{health_check, generate_embeddings, generate_batch_embeddings, search_similar, list_models, 
          generate_graphiti_embeddings, process_chunks, list_graphiti_models, graphiti_health,
          generate_neo4j_embeddings, get_neo4j_embeddings, 
          search_neo4j_embeddings, batch_store_neo4j_embeddings, delete_neo4j_embedding},
    core::Config,
    EmbeddingsService,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Starting embeddings service on {}:{}", config.server.host, config.server.port);

    // Initialize service
    let service = EmbeddingsService::new(config.clone()).await?;
    service.initialize_default_model().await?;

    // Build router
    let app = Router::new()
        // Standard endpoints
        .route("/health", get(health_check))
        .route("/api/v1/generate", post(generate_embeddings))
        .route("/api/v1/generate/batch", post(generate_batch_embeddings))
        .route("/api/v1/search", get(search_similar))
        .route("/api/v1/models", get(list_models))
        // Graphiti-compatible endpoints
        .route("/api/v1/graphiti/health", get(graphiti_health))
        .route("/api/v1/graphiti/generate", post(generate_graphiti_embeddings))
        .route("/api/v1/graphiti/chunks", post(process_chunks))
        .route("/api/v1/graphiti/models", get(list_graphiti_models))
        // Neo4j integration endpoints
        .route("/api/v1/neo4j/generate", post(generate_neo4j_embeddings))
        .route("/api/v1/neo4j/get", get(get_neo4j_embeddings))
        .route("/api/v1/neo4j/search", post(search_neo4j_embeddings))
        .route("/api/v1/neo4j/batch", post(batch_store_neo4j_embeddings))
        .route("/api/v1/neo4j/delete", delete(delete_neo4j_embedding))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        )
        .with_state(service.model_manager.clone())
        .with_state(service.postgres_storage.clone());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

