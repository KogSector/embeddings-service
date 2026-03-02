//! Core functionality for the embeddings service

pub mod config;
pub mod error;

pub use config::{Config, ModelConfig, ServerConfig};
pub use error::{EmbeddingError, Result};
