//! Model management for the embeddings service

pub mod manager;
pub mod models;

pub use manager::{ModelManager};
pub use models::{EmbeddingModel, OllamaModel, SentenceTransformersModel, ModelType};
