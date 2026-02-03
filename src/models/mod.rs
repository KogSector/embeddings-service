//! Model management for the embeddings service

pub mod manager;
pub mod sentence_transformers;
pub mod openai;

pub use manager::ModelManager;
pub use sentence_transformers::SentenceTransformersModel;
pub use openai::OpenAIModel;
