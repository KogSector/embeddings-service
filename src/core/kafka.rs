use confuse_common::events::{
    consumer::{EventConsumer, EventHandler, ConsumerError, HandlerResult},
    events::ChunkCreatedEvent,
    topics::Topics,
};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{error, info, warn};
use crate::EmbeddingsService;

pub struct ChunkIngestionHandler {
    service: Arc<EmbeddingsService>,
}

impl ChunkIngestionHandler {
    pub fn new(service: Arc<EmbeddingsService>) -> Self {
        Self { service }
    }

    async fn handle_chunk_created(&self, event: ChunkCreatedEvent) -> anyhow::Result<()> {
        info!("Processing chunk created event: {}", event.chunk_id);
        // Here we would typically trigger embedding generation for this chunk
        // For now, we just log it as the service is generate-only via API
        // But in a real pipeline, we would call self.service.generate_embedding(event.content...)
        // Since content is in Blob Storage, we might need to fetch it or rely on the event metadata.
        // The event has `blob_storage_url`.
        
        info!("Chunk {} from {} is ready for embedding generation", event.chunk_id, event.source_id);
        Ok(())
    }
}

#[async_trait]
impl EventHandler for ChunkIngestionHandler {
    async fn handle(&self, topic: &str, payload: &[u8]) -> HandlerResult {
        if topic == Topics::CHUNKS_CREATED {
             let event: ChunkCreatedEvent = serde_json::from_slice(payload)
                .map_err(ConsumerError::from)?;
             
             self.handle_chunk_created(event).await
                .map_err(|e| ConsumerError::Handler(e.to_string()))?;
        }
        Ok(())
    }

    async fn handle_error(&self, topic: &str, error: &ConsumerError, _payload: Option<&[u8]>) {
        error!("Kafka error on topic {}: {}", topic, error);
    }
}

pub async fn start_kafka_consumer(service: Arc<EmbeddingsService>) -> anyhow::Result<()> {
    // Check if Kafka enabled
    if std::env::var("KAFKA_ENABLED").unwrap_or_else(|_| "false".to_string()) != "true" {
        info!("Kafka disabled for embeddings-service");
        return Ok(());
    }

    let consumer = EventConsumer::from_env()?;
    consumer.subscribe(&[Topics::CHUNKS_CREATED])?;
    
    let handler = Arc::new(ChunkIngestionHandler::new(service));
    
    tokio::spawn(async move {
        info!("Starting Kafka consumer for embeddings-service");
        if let Err(e) = consumer.run(handler).await {
            error!("Kafka consumer loop failed: {}", e);
        }
    });

    Ok(())
}
