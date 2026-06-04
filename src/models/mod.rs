//! Model management for the embeddings service

pub mod manager;
pub mod models;
pub mod fastembed;

pub use manager::{ModelManager};
pub use models::{EmbeddingModel, OllamaModel, ModelType};
pub use fastembed::FastEmbedModel;
