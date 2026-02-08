//! Configuration for the embeddings service

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub models: ModelConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub postgres_url: String,
    pub redis_url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub default_model: String,
    pub models_dir: String,
    pub max_batch_size: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub ttl: Duration,
    pub max_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 3001,
                workers: 4,
            },
            database: DatabaseConfig {
                postgres_url: String::new(),
                redis_url: String::new(),
                max_connections: 10,
            },
            models: ModelConfig {
                default_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                models_dir: "./models".to_string(),
                max_batch_size: 32,
                timeout: Duration::from_secs(30),
            },
            cache: CacheConfig {
                ttl: Duration::from_hours(1),
                max_size: 10000,
            },
        }
    }
}

impl Config {
    pub fn from_env() -> crate::core::Result<Self> {
        let mut config = Self::default();
        
        // Override with environment variables
        config.server.host = std::env::var("EMBEDDINGS_HOST")
            .expect("EMBEDDINGS_HOST must be set");
        config.server.port = std::env::var("EMBEDDINGS_PORT")
            .expect("EMBEDDINGS_PORT must be set")
            .parse()
            .unwrap_or(3001);
        config.database.postgres_url = std::env::var("POSTGRES_URL")
            .expect("POSTGRES_URL must be set");
        config.database.redis_url = std::env::var("REDIS_URL")
            .expect("REDIS_URL must be set");
        if let Ok(default_model) = std::env::var("DEFAULT_EMBEDDING_MODEL") {
            config.models.default_model = default_model;
        }
        
        Ok(config)
    }
}
