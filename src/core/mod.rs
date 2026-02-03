//! Core functionality for the embeddings service

pub mod config;
pub mod error;

pub use config::{Config, DatabaseConfig, ModelConfig, ServerConfig, CacheConfig};
pub use error::{EmbeddingError, Result};
