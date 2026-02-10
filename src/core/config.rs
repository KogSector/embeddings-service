//! Configuration for the embeddings service (generate-only)

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub models: ModelConfig,
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
    pub models_dir: String,
    pub max_batch_size: usize,
    pub timeout: Duration,
    pub ollama_url: Option<String>,
    pub use_ollama: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 3011,
                workers: 4,
            },
            models: ModelConfig {
                default_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                models_dir: "./models".to_string(),
                max_batch_size: 32,
                timeout: Duration::from_secs(30),
                ollama_url: None,
                use_ollama: false,
            },
        }
    }
}

impl Config {
    pub fn from_env() -> crate::core::Result<Self> {
        let mut config = Self::default();

        // Override with environment variables
        if let Ok(host) = std::env::var("HOST") {
            config.server.host = host;
        }
        config.server.port = std::env::var("PORT")
            .unwrap_or_else(|_| "3011".to_string())
            .parse()
            .unwrap_or(3011);
        if let Ok(default_model) = std::env::var("DEFAULT_EMBEDDING_MODEL") {
            config.models.default_model = default_model;
        }
        if let Ok(batch_size) = std::env::var("MAX_BATCH_SIZE") {
            config.models.max_batch_size = batch_size.parse().unwrap_or(32);
        }
        config.models.ollama_url = std::env::var("OLLAMA_URL").ok();
        config.models.use_ollama = std::env::var("USE_OLLAMA")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        Ok(config)
    }
}
