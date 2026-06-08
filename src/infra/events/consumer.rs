//! Event Consumer for ConFuse Platform

use rdkafka::config::ClientConfig;
use rdkafka::consumer::{StreamConsumer, Consumer};
use rdkafka::message::{Message, BorrowedMessage};
use serde::de::DeserializeOwned;
use futures::StreamExt;
use anyhow::Result;
use std::sync::Arc;
use tracing::{info, error, warn};

/// Kafka event consumer for background message processing
pub struct EventConsumer {
    consumer: StreamConsumer,
    group_id: String,
}

impl EventConsumer {
    /// Create a new event consumer with specified consumer group
    pub fn new(bootstrap_servers: &str, group_id: &str) -> Result<Self> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", bootstrap_servers)
            .set("group.id", group_id)
            .set("client.id", &format!("{}-consumer", group_id))
            .set("enable.auto.commit", "true")
            .set("auto.offset.reset", "earliest")
            .set("enable.auto.offset.store", "false")
            .set("session.timeout.ms", "10000")
            .set("heartbeat.interval.ms", "3000")
            .set("max.poll.interval.ms", "300000")
            .set("auto.commit.interval.ms", "5000")
            .create()?;

        info!("Kafka event consumer initialized with group: {}", group_id);

        Ok(Self {
            consumer,
            group_id: group_id.to_string(),
        })
    }

    /// Subscribe to topics and start consuming messages
    pub async fn subscribe(&self, topics: &[&str]) -> Result<()> {
        self.consumer.subscribe(topics)?;
        info!("Subscribed to topics: {:?}", topics);
        Ok(())
    }

    /// Start consuming messages with async handler
    pub async fn consume<F, T>(&self, handler: Arc<F>) -> Result<()>
    where
        F: Fn(T) -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync + 'static,
        T: DeserializeOwned + Send + 'static,
    {
        info!("Starting message consumption for group: {}", self.group_id);

        let mut message_stream = self.consumer.stream();
        
        while let Some(message_result) = message_stream.next().await {
            match message_result {
                Ok(message) => {
                    let payload = match message.payload() {
                        Some(p) => p,
                        None => {
                            warn!("Received message with empty payload");
                            continue;
                        }
                    };

                    // Deserialize message
                    match serde_json::from_slice::<T>(payload) {
                        Ok(event) => {
                            let handler = handler.clone();
                            tokio::spawn(async move {
                                if let Err(e) = handler(event).await {
                                    error!("Error processing message: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to deserialize message: {}", e);
                            // Continue processing other messages
                        }
                    }
                }
                Err(e) => {
                    error!("Kafka consumer error: {}", e);
                    // Continue processing
                }
            }
        }

        Ok(())
    }

    /// Get consumer group ID
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Commit offsets manually (if auto-commit is disabled)
    pub async fn commit_message(&self, message: &BorrowedMessage<'_>) -> Result<()> {
        self.consumer.commit_message(message, rdkafka::consumer::CommitMode::Async)?;
        Ok(())
    }
}

/// Trait for event handlers
pub trait EventHandler<T>: Send + Sync {
    fn handle(&self, event: T) -> futures::future::BoxFuture<'static, Result<()>>;
}

/// Convenience function to create a boxed future handler
pub fn boxed_handler<F, Fut, T>(f: F) -> impl Fn(T) -> futures::future::BoxFuture<'static, Result<()>>
where
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: futures::Future<Output = Result<()>> + Send + 'static,
    T: Send + 'static,
{
    move |event| Box::pin(f(event))
}
