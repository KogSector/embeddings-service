//! Model management for the embeddings service

pub mod manager;
pub mod sentence_transformers;

pub use manager::{ModelManager, EmbeddingModel};
pub use sentence_transformers::SentenceTransformersModel;
