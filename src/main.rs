//! Main entry point for the embeddings service
//! Kafka-streamed pipeline: consumes chunks.raw → generates embeddings → produces chunks.embedded

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
use confuse_common::events::{EventConsumer, EventProducer, Topics, ChunkRawEvent, ChunkEnrichedEvent, EmbeddingGeneratedEvent};

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
    tracing::info!("gRPC server on {}:{}", config.grpc_host, config.grpc_port);

    // Initialize service (no storage initialization needed)
    let service = EmbeddingsService::new(config.clone()).await?;
    service.initialize_default_model().await?;

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

    // Start Kafka consumer for chunks.raw
    let kafka_service = service.clone();
    let kafka_handle = tokio::spawn(async move {
        if let Err(e) = start_kafka_consumer(kafka_service).await {
            tracing::error!("Kafka consumer failed: {}", e);
        }
    });

    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    // Wait for background tasks
    let _ = grpc_handle.await;
    let _ = kafka_handle.await;

    Ok(())
}

/// Start Kafka consumer to process chunks.raw events
async fn start_kafka_consumer(service: EmbeddingsService) -> anyhow::Result<()> {
    let bootstrap_servers = std::env::var("KAFKA_BOOTSTRAP_SERVERS")
        .unwrap_or_else(|_| "localhost:9092".to_string());
    let group_id = std::env::var("KAFKA_GROUP_ID")
        .unwrap_or_else(|_| "embeddings-service".to_string());
    
    let consumer = confuse_common::events::EventConsumer::new(&bootstrap_servers, &group_id)?;
    
    tracing::info!("Starting Kafka consumer for chunks.raw");
    
    consumer.subscribe(&[Topics::CHUNKS_RAW]).await?;
    
    let mut stream = consumer.stream();
    
    while let Some(result) = stream.next().await {
        match result {
            Ok(msg) => {
                if let Some(payload) = msg.payload() {
                    match serde_json::from_slice::<ChunkRawEvent>(payload) {
                        Ok(chunk_event) => {
                            // Process chunk and generate embeddings
                            if let Err(e) = process_chunk_with_embeddings(&service, &chunk_event).await {
                                tracing::error!("Failed to process chunk {}: {}", chunk_event.chunk_id, e);
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to deserialize chunk event: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Kafka consumer error: {}", e);
            }
        }
    }
    
    tracing::info!("Kafka consumer stopped");
    Ok(())
}

/// Process a single chunk and generate embeddings
async fn process_chunk_with_embeddings(
    service: &EmbeddingsService, 
    chunk_event: &ChunkRawEvent
) -> anyhow::Result<()> {
    // Generate embeddings for the chunk content
    let embedding_result = service.generate_embeddings_internal(
        &chunk_event.content,
        Some(&chunk_event.chunk_id),
        &chunk_event.source_id,
        &chunk_event.file_id,
    ).await?;
    
    // Publish chunks.embedded event back to unified-processor
    let bootstrap_servers = std::env::var("KAFKA_BOOTSTRAP_SERVERS")
        .unwrap_or_else(|_| "localhost:9092".to_string());
    
    let producer = confuse_common::events::EventProducer::new(&bootstrap_servers)?;
    
    let enriched_event = ChunkEnrichedEvent {
        headers: confuse_common::events::EventHeaders::new(
            "embeddings-service",
            "chunk.enriched"
        )
        .with_correlation_id(&chunk_event.chunk_id),
        metadata: confuse_common::events::EventMetadata::default(),
        source_id: chunk_event.source_id.clone(),
        file_id: chunk_event.file_id.clone(),
        chunk_id: chunk_event.chunk_id.clone(),
        content: chunk_event.content.clone(),
        chunk_type: chunk_event.chunk_type.clone(),
        entity_hints: chunk_event.entity_hints.clone(),
        relationship_context: chunk_event.relationship_context.clone(),
        embedding: Some(embedding_result.embeddings),
        quality_score: Some(embedding_result.quality_score),
        created_at: chrono::Utc::now(),
    };
    
    producer.publish(Topics::CHUNKS_EMBEDDED, &enriched_event).await?;
    
    tracing::info!("Processed chunk {} and published embeddings", chunk_event.chunk_id);
    Ok(())
}
