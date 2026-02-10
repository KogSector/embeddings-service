//! Model management for the embeddings service

pub mod manager;
pub mod sentence_transformers;
pub mod ollama;

pub use manager::{ModelManager, EmbeddingModel};
pub use sentence_transformers::SentenceTransformersModel;
pub use ollama::OllamaModel;
