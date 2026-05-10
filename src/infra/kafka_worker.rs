use anyhow::Result;
use confuse_common::events::{
    EventConsumer, EventProducer, SimplifiedChunkRawEvent,
    SimplifiedEmbeddingGeneratedEvent, SimplifiedEmbedding,
};
use std::sync::Arc;
use tracing::{info, error};
use crate::models::ModelManager;
use crate::Config;

pub struct KafkaWorker {
    consumer: EventConsumer,
    producer: EventProducer,
    model_manager: Arc<ModelManager>,
    config: Config,
}

impl KafkaWorker {
    pub fn new(config: Config, model_manager: Arc<ModelManager>) -> Result<Self> {
        let consumer = EventConsumer::new(
            &config.kafka.bootstrap_servers,
            &config.kafka.group_id,
        )?;
        let producer = EventProducer::new(&config.kafka.bootstrap_servers)?;

        Ok(Self {
            consumer,
            producer,
            model_manager,
            config,
        })
    }

    pub async fn start(self) -> Result<()> {
        if !self.config.kafka.enabled {
            info!("Kafka worker is disabled in configuration");
            return Ok(());
        }

        info!("Starting Kafka worker for embeddings-service with retry on disconnection");

        loop {
            match self.start_worker().await {
                Ok(_) => {
                    info!("Kafka worker stopped gracefully");
                    break;
                }
                Err(e) => {
                    error!("Kafka worker error: {}. Retrying in 5 seconds...", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }

        Ok(())
    }

    async fn start_worker(&self) -> Result<()> {
        let consumer = EventConsumer::new(
            &self.config.kafka.bootstrap_servers,
            &self.config.kafka.group_id,
        )?;
        consumer.subscribe(&[&self.config.kafka.input_topic]).await?;

        let model_manager = self.model_manager.clone();
        let producer = Arc::new(EventProducer::new(&self.config.kafka.bootstrap_servers)?);
        let config = self.config.clone();

        let handler = Arc::new(move |event: SimplifiedChunkRawEvent| {
            let model_manager = model_manager.clone();
            let producer = producer.clone();
            let output_topic = config.kafka.output_topic.clone();

            Box::pin(async move {
                let chunk_count = event.chunks.len();
                info!("Processing {} chunks for source: {}", chunk_count, event.source_id);

                // Pre-allocate with known capacity — avoids reallocations.
                let mut embeddings = Vec::with_capacity(chunk_count);

                for chunk in event.chunks {
                    match model_manager.generate_embeddings(&chunk.content, None).await {
                        Ok(vector) => {
                            let dimension = vector.len() as u32;
                            let model_name = model_manager.get_default_model_name().to_string();

                            embeddings.push(SimplifiedEmbedding {
                                chunk_id: chunk.chunk_id.clone(),
                                file_id: chunk.file_id.clone(),
                                chunk_type: chunk.chunk_type.clone(),
                                language: chunk.language.clone(),
                                embedding: vector.clone(),
                                model: model_name,
                                dimension,
                            });
                        }
                        Err(e) => {
                            error!("Failed to generate embedding for chunk {}: {}", chunk.chunk_id, e);
                        }
                    }
                }

                // Publish embeddings event to unified-processor via Kafka
                if !embeddings.is_empty() {
                    let output_event = SimplifiedEmbeddingGeneratedEvent {
                        headers: event.headers.clone(),
                        metadata: event.metadata.clone(),
                        source_id: event.source_id.clone(),
                        chunks: embeddings,
                        model: model_manager.get_default_model_name().to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                    };

                    // Use resilient publish with retry and DLQ fallback
                    if let Err(e) = producer.publish_with_retry(&output_topic, &output_event, 3, None).await {
                        error!("Failed to publish embeddings event after retries for source {}: {}", event.source_id, e);
                    } else {
                        info!("Published {} embeddings for source: {}", output_event.chunks.len(), event.source_id);
                    }
                }

                Ok(())
            }) as futures::future::BoxFuture<'static, Result<()>>
        });

        consumer.consume(handler).await?;
        Ok(())
    }
}
