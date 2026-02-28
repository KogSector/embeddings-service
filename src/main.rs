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

    // Start Kafka consumer pipeline (chunks.raw → embed → chunks.embedded)
    let kafka_service = service.clone();
    let kafka_handle = tokio::spawn(async move {
        if let Err(e) = run_kafka_pipeline(kafka_service).await {
            tracing::error!("Kafka pipeline failed: {}", e);
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

/// Run the Kafka streaming pipeline:
/// Consume from chunks.raw → generate embeddings → produce to chunks.embedded
async fn run_kafka_pipeline(service: EmbeddingsService) -> Result<(), Box<dyn std::error::Error>> {
    use confuse_common::events::{EventProducer, Topics};
    use rdkafka::consumer::{Consumer, StreamConsumer};
    use rdkafka::ClientConfig;
    use rdkafka::message::Message;
    use futures::StreamExt;
    
    let bootstrap_servers = std::env::var("KAFKA_BOOTSTRAP_SERVERS")
        .unwrap_or_else(|_| "localhost:9092".to_string());
    let group_id = std::env::var("KAFKA_GROUP_ID")
        .unwrap_or_else(|_| "embeddings-service".to_string());
    
    // Create consumer
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap_servers)
        .set("group.id", &group_id)
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", "earliest")
        .set("session.timeout.ms", "45000")
        .create()
        .map_err(|e| format!("Kafka consumer creation failed: {}", e))?;
    
    consumer.subscribe(&[Topics::CHUNKS_RAW])
        .map_err(|e| format!("Kafka subscribe failed: {}", e))?;
    
    // Create producer for chunks.embedded
    let producer = EventProducer::from_env()
        .map_err(|e| format!("Kafka producer creation failed: {}", e))?;
    
    tracing::info!("Kafka pipeline started: consuming from {} → producing to {}", 
        Topics::CHUNKS_RAW, Topics::CHUNKS_EMBEDDED);
    
    let mut stream = consumer.stream();
    while let Some(result) = stream.next().await {
        match result {
            Ok(msg) => {
                if let Some(payload) = msg.payload() {
                    match serde_json::from_slice::<serde_json::Value>(payload) {
                        Ok(chunk_msg) => {
                            let content = chunk_msg.get("content")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            
                            // Generate embedding
                            match service.generate_embedding(content).await {
                                Ok(embedding) => {
                                    // Build embedded chunk message (original + embedding)
                                    let mut embedded = chunk_msg.clone();
                                    if let Some(obj) = embedded.as_object_mut() {
                                        obj.insert("embedding".to_string(), 
                                            serde_json::json!(embedding));
                                        obj.insert("model_used".to_string(),
                                            serde_json::json!("sentence-transformers/all-MiniLM-L6-v2"));
                                    }
                                    
                                    if let Err(e) = producer.publish_to_topic(&embedded, Topics::CHUNKS_EMBEDDED).await {
                                        tracing::error!("Failed to publish embedded chunk: {}", e);
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Embedding generation failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to deserialize chunk message: {}", e);
                        }
                    }
                }
                
                // Commit offset
                if let Err(e) = rdkafka::consumer::Consumer::commit_message(
                    &consumer, &msg, rdkafka::consumer::CommitMode::Async
                ) {
                    tracing::warn!("Failed to commit offset: {}", e);
                }
            }
            Err(e) => {
                tracing::error!("Kafka consumer error: {}", e);
            }
        }
    }
    
    Ok(())
}
