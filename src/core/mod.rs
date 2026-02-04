//! Core functionality for the embeddings service

pub mod config;
pub mod error;
pub mod vector_search;

pub use config::{Config, DatabaseConfig, ModelConfig, ServerConfig, CacheConfig};
pub use error::{EmbeddingError, Result};
pub use vector_search::{VectorSearchService, VectorSearchRequest, VectorSearchResponse, SearchResult, SearchCache};

