//! Event publishing for embeddings-service

use confuse_common::events::{EventProducer, Topics, SimplifiedEmbeddingGeneratedEvent, EventHeaders, SimplifiedEmbedding};
use crate::core::Result;

/// Event publisher for embeddings-service
pub struct EmbeddingEventPublisher {
    producer: Option<EventProducer>,
}

impl EmbeddingEventPublisher {
    /// Create a new event publisher
    pub fn new() -> Self {
        Self {
            producer: Self::create_producer(),
        }
    }

    /// Create Kafka producer if configured
    #[cfg(feature = "kafka")]
    fn create_producer() -> Option<EventProducer> {
        match std::env::var("KAFKA_BOOTSTRAP_SERVERS") {
            Ok(bootstrap_servers) => {
                match EventProducer::new(&bootstrap_servers) {
                    Ok(producer) => {
                        tracing::info!("Kafka event producer initialized");
                        Some(producer)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize Kafka producer: {}. Event publishing disabled.", e);
                        None
                    }
                }
            }
            Err(_) => {
                tracing::info!("KAFKA_BOOTSTRAP_SERVERS not configured. Event publishing disabled.");
                None
            }
        }
    }

    /// Create Kafka producer if configured (no-op when kafka feature disabled)
    #[cfg(not(feature = "kafka"))]
    fn create_producer() -> Option<EventProducer> {
        tracing::info!("Kafka feature not enabled. Event publishing disabled.");
        None
    }

    /// Publish embedding generated event
    pub async fn publish_embedding_generated(
        &self,
        source_id: &str,
        chunks: &[crate::storage::falcordb_client::VectorChunk],
        model: &str,
        processing_time_ms: u64,
    ) -> Result<()> {
        if let Some(ref producer) = self.producer {
            let simplified_embeddings: Vec<SimplifiedEmbedding> = chunks
                .iter()
                .map(|chunk| {
                    // Extract language from metadata if available
                    let language = chunk.metadata
                        .as_object()
                        .and_then(|obj| obj.get("language"))
                        .and_then(|val| val.as_str())
                        .map(|s| s.to_string());

                    SimplifiedEmbedding {
                        chunk_id: chunk.id.to_string(),
                        file_id: chunk.document_id.to_string(),
                        chunk_type: "chunk".to_string(), // Default type
                        content: chunk.chunk_text.clone(),
                        language,
                        embedding: chunk.embedding.clone(),
                        model: model.to_string(),
                        dimension: chunk.embedding.len() as u32,
                    }
                })
                .collect();

            let event = SimplifiedEmbeddingGeneratedEvent {
                headers: EventHeaders::new("embeddings-service", "EMBEDDING_GENERATED")
                    .with_correlation_id(source_id),
                metadata: Default::default(),
                source_id: source_id.to_string(),
                chunks: simplified_embeddings,
                model: model.to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            };

            match producer.publish(Topics::EMBEDDING_GENERATED, &event).await {
                Ok(()) => {
                    tracing::info!("Published embedding generated event for source: {}", source_id);
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("Failed to publish embedding generated event: {}", e);
                    Err(crate::core::EmbeddingError::ConnectionError(format!("Failed to publish event: {}", e)))
                }
            }
        } else {
            tracing::debug!("Event publisher not available, skipping event publication");
            Ok(())
        }
    }

    /// Publish chunk processing failed event
    pub async fn publish_processing_failed(
        &self,
        source_id: &str,
        chunk_id: &str,
        error_message: &str,
        error_type: &str,
    ) -> Result<()> {
        if let Some(ref producer) = self.producer {
            use confuse_common::events::{SimplifiedProcessingFailedEvent, EventHeaders};

            let event = SimplifiedProcessingFailedEvent {
                headers: EventHeaders::new("embeddings-service", "PROCESSING_FAILED")
                    .with_correlation_id(source_id),
                metadata: Default::default(),
                original_topic: Topics::CHUNKS_RAW.to_string(),
                original_event_id: chunk_id.to_string(),
                error_message: error_message.to_string(),
                error_type: error_type.to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            };

            match producer.publish(Topics::DLQ_PROCESSING_FAILED, &event).await {
                Ok(()) => {
                    tracing::info!("Published processing failed event for chunk: {}", chunk_id);
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("Failed to publish processing failed event: {}", e);
                    Err(crate::core::EmbeddingError::ConnectionError(format!("Failed to publish event: {}", e)))
                }
            }
        } else {
            tracing::debug!("Event publisher not available, skipping event publication");
            Ok(())
        }
    }
}

impl Default for EmbeddingEventPublisher {
    fn default() -> Self {
        Self::new()
    }
}
