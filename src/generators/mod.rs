//! Embedding generation functionality

pub mod batch;
pub mod streaming;

pub use batch::{BatchGenerator, BatchEmbeddingRequest, BatchEmbeddingResponse};
pub use streaming::StreamingGenerator;
