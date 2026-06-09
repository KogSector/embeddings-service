//! Configuration for the embeddings service (generate-only)

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub models: ModelConfig,
    pub kafka: KafkaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    pub bootstrap_servers: String,
    pub group_id: String,
    pub input_topic: String,
    pub output_topic: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub default_model: String,
    pub max_batch_size: usize,
    pub timeout: Duration,
    pub ollama_url: Option<String>,
}

impl Config {
    pub fn from_env() -> crate::Result<Self> {
        Ok(Self {
            server: ServerConfig {
                host: std::env::var("HOST")
                    .map_err(|_| "Missing required env var: HOST")?,
                port: std::env::var("EMBEDDINGS_SERVICE_PORT")
                    .map_err(|_| "Missing required env var: EMBEDDINGS_SERVICE_PORT")?
                    .parse()
                    .map_err(|_| "Invalid EMBEDDINGS_SERVICE_PORT")?,
                workers: std::env::var("WORKERS")
                    .map_err(|_| "Missing required env var: WORKERS")?
                    .parse()
                    .map_err(|_| "Invalid WORKERS")?,
            },
            models: ModelConfig {
                default_model: std::env::var("DEFAULT_EMBEDDING_MODEL")
                    .map_err(|_| "Missing required env var: DEFAULT_EMBEDDING_MODEL")?,
                max_batch_size: std::env::var("MAX_BATCH_SIZE")
                    .map_err(|_| "Missing required env var: MAX_BATCH_SIZE")?
                    .parse()
                    .map_err(|_| "Invalid MAX_BATCH_SIZE")?,
                timeout: Duration::from_secs(
                    std::env::var("MODEL_TIMEOUT_SECS")
                        .map_err(|_| "Missing required env var: MODEL_TIMEOUT_SECS")?
                        .parse()
                        .map_err(|_| "Invalid MODEL_TIMEOUT_SECS")?
                ),
                ollama_url: std::env::var("OLLAMA_URL").ok(),
            },
            kafka: KafkaConfig {
                bootstrap_servers: std::env::var("KAFKA_BOOTSTRAP_SERVERS")
                    .map_err(|_| "Missing required env var: KAFKA_BOOTSTRAP_SERVERS")?,
                group_id: std::env::var("EMBEDDINGS_SERVICE_KAFKA_GROUP_ID")
                    .map_err(|_| "Missing required env var: EMBEDDINGS_SERVICE_KAFKA_GROUP_ID")?,
                input_topic: std::env::var("KAFKA_INPUT_TOPIC")
                    .map_err(|_| "Missing required env var: KAFKA_INPUT_TOPIC")?,
                output_topic: std::env::var("KAFKA_OUTPUT_TOPIC")
                    .map_err(|_| "Missing required env var: KAFKA_OUTPUT_TOPIC")?,
                enabled: std::env::var("KAFKA_ENABLED")
                    .map_err(|_| "Missing required env var: KAFKA_ENABLED")?
                    .parse()
                    .map_err(|_| "Invalid KAFKA_ENABLED")?,
            },
        })
    }
}
