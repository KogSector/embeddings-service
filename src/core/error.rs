//! Error types for the embeddings service

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),
    
    #[error("Failed to generate embedding: {0}")]
    GenerationError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),
    
    #[error("Neo4j error: {0}")]
    Neo4jError(#[from] neo4rs::Error),
    
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    #[error("Python error: {0}")]
    PythonError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;
