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
}

impl Config {
    pub fn from_env() -> crate::Result<Self> {
        Ok(Self {
            server: ServerConfig {
                host: std::env::var("HOST")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: HOST".to_string()))?,
                port: std::env::var("PORT")
                    .or_else(|_| std::env::var("EMBEDDINGS_SERVICE_PORT"))
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: PORT or EMBEDDINGS_SERVICE_PORT".to_string()))?
                    .parse()
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Invalid PORT or EMBEDDINGS_SERVICE_PORT".to_string()))?,
                workers: std::env::var("WORKERS")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: WORKERS".to_string()))?
                    .parse()
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Invalid WORKERS".to_string()))?,
            },
            models: ModelConfig {
                default_model: std::env::var("DEFAULT_EMBEDDING_MODEL")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: DEFAULT_EMBEDDING_MODEL".to_string()))?,
                max_batch_size: std::env::var("MAX_BATCH_SIZE")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: MAX_BATCH_SIZE".to_string()))?
                    .parse()
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Invalid MAX_BATCH_SIZE".to_string()))?,
                timeout: Duration::from_secs(
                    std::env::var("MODEL_TIMEOUT_SECS")
                        .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: MODEL_TIMEOUT_SECS".to_string()))?
                        .parse()
                        .map_err(|_| crate::error::EmbeddingError::ConfigError("Invalid MODEL_TIMEOUT_SECS".to_string()))?
                ),
            },
            kafka: KafkaConfig {
                bootstrap_servers: std::env::var("KAFKA_BOOTSTRAP_SERVERS")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: KAFKA_BOOTSTRAP_SERVERS".to_string()))?,
                group_id: std::env::var("EMBEDDINGS_SERVICE_KAFKA_GROUP_ID")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: EMBEDDINGS_SERVICE_KAFKA_GROUP_ID".to_string()))?,
                input_topic: std::env::var("KAFKA_INPUT_TOPIC")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: KAFKA_INPUT_TOPIC".to_string()))?,
                output_topic: std::env::var("KAFKA_OUTPUT_TOPIC")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: KAFKA_OUTPUT_TOPIC".to_string()))?,
                enabled: std::env::var("KAFKA_ENABLED")
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Missing required env var: KAFKA_ENABLED".to_string()))?
                    .parse()
                    .map_err(|_| crate::error::EmbeddingError::ConfigError("Invalid KAFKA_ENABLED".to_string()))?,
            },
        })
    }
}
