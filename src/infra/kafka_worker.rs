use anyhow::Result;
use confuse_common::events::{
    EventConsumer, EventProducer, SimplifiedChunkRawEvent, 
    SimplifiedEmbeddingGeneratedEvent, SimplifiedEmbedding,
};
use std::sync::Arc;
use tracing::{info, error};
use crate::models::ModelManager;
use crate::core::Config;

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

        info!("Starting Kafka worker for embeddings-service");
        self.consumer.subscribe(&[&self.config.kafka.input_topic]).await?;

        let model_manager = self.model_manager.clone();
        let producer = Arc::new(self.producer);
        let config = self.config.clone();

        let handler = Arc::new(move |event: SimplifiedChunkRawEvent| {
            let model_manager = model_manager.clone();
            let producer = producer.clone();
            let output_topic = config.kafka.output_topic.clone();

            Box::pin(async move {
                info!("Processing {} chunks for source: {}", event.chunks.len(), event.source_id);
                
                let mut embeddings = Vec::new();
                for chunk in event.chunks {
                    // Generate embedding for chunk
                    match model_manager.generate_embeddings(&chunk.content, None).await {
                        Ok(vector) => {
                            let dimension = vector.len() as u32;
                            embeddings.push(SimplifiedEmbedding {
                                chunk_id: chunk.chunk_id,
                                file_id: chunk.file_id,
                                chunk_type: chunk.chunk_type,
                                content: chunk.content,
                                language: chunk.language,
                                embedding: vector,
                                model: model_manager.get_default_model_name().to_string(),
                                dimension,
                            });
                        }
                        Err(e) => {
                            error!("Failed to generate embedding for chunk {}: {}", chunk.chunk_id, e);
                        }
                    }
                }

                if !embeddings.is_empty() {
                    let output_event = SimplifiedEmbeddingGeneratedEvent {
                        headers: event.headers.clone(),
                        metadata: event.metadata.clone(),
                        source_id: event.source_id.clone(),
                        chunks: embeddings,
                        model: model_manager.get_default_model_name().to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                    };

                    producer.publish(&output_topic, &output_event).await?;
                    info!("Published {} embeddings for source: {}", output_event.chunks.len(), event.source_id);
                }

                Ok(())
            }) as futures::future::BoxFuture<'static, Result<()>>
        });

        self.consumer.consume(handler).await?;
        Ok(())
    }
}
