use anyhow::Result;
use confuse_common::events::{
    EventConsumer, EventProducer, SimplifiedChunkRawEvent,
    SimplifiedEmbeddingGeneratedEvent, SimplifiedEmbedding,
};
use confuse_common::events::episode::{
    GraphifyEpisode, EpisodeSourceType, EpisodeChunkType,
};
use confuse_common::events::Topics;
use std::sync::Arc;
use tracing::{info, warn, error};
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

        let graphify_enabled = std::env::var("GRAPHIFY_EPISODE_EMISSION_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        if graphify_enabled {
            info!("Graphify episode emission is ENABLED in embeddings-service");
        }

        let handler = Arc::new(move |event: SimplifiedChunkRawEvent| {
            let model_manager = model_manager.clone();
            let producer = producer.clone();
            let output_topic = config.kafka.output_topic.clone();
            let graphify_on = graphify_enabled;

            Box::pin(async move {
                let chunk_count = event.chunks.len();
                info!("Processing {} chunks for source: {}", chunk_count, event.source_id);

                // Pre-allocate with known capacity — avoids reallocations.
                let mut embeddings = Vec::with_capacity(chunk_count);
                let mut graphify_episodes: Vec<GraphifyEpisode> = if graphify_on {
                    Vec::with_capacity(chunk_count)
                } else {
                    Vec::new()
                };

                for chunk in event.chunks {
                    match model_manager.generate_embeddings(&chunk.content, None).await {
                        Ok(vector) => {
                            let dimension = vector.len() as u32;
                            let model_name = model_manager.get_default_model_name().to_string();

                            // Build Graphify episode with embedding attached.
                            if graphify_on {
                                let episode = GraphifyEpisode::new(
                                    EpisodeSourceType::Codebase,
                                    event.source_id.clone(),
                                    chunk.content.clone(),
                                    "embeddings-service",
                                )
                                .with_chunk_type(EpisodeChunkType::CodeBlock)
                                .with_embedding(vector.clone(), &model_name);

                                graphify_episodes.push(episode);
                            }

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

                // Publish the legacy embedding event (existing pipeline).
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
                    if let Err(e) = producer.publish_with_retry(&output_topic, &output_event).await {
                        error!("Failed to publish embeddings event after retries for source {}: {}", event.source_id, e);
                    } else {
                        info!("Published {} embeddings for source: {}", output_event.chunks.len(), event.source_id);
                    }
                }

                // Publish Graphify episodes (new pipeline — feature-flagged).
                if graphify_on && !graphify_episodes.is_empty() {
                    let topic = Topics::GRAPHIFY_EPISODES;
                    let mut published = 0usize;
                    for ep in &graphify_episodes {
                        match ep.to_kafka_payload() {
                            Ok(payload) => {
                                // Publish Graphify episodes using resilient publish
                                if let Err(e) = producer.publish_with_retry(topic, &payload).await {
                                    warn!("Graphify episode publish failed after retries: {}", e);
                                } else {
                                    published += 1;
                                }
                            }
                            Err(e) => {
                                warn!("Graphify episode serialization failed: {}", e);
                            }
                        }
                    }
                    info!(
                        "Published {}/{} Graphify episodes for source: {}",
                        published,
                        graphify_episodes.len(),
                        event.source_id
                    );
                }

                Ok(())
            }) as futures::future::BoxFuture<'static, Result<()>>
        });

        self.consumer.consume(handler).await?;
        Ok(())
    }
}
