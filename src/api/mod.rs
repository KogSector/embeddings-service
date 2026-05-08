//! API endpoints for the embeddings service (generate-only)

pub mod generate;
pub use generate::{generate_embeddings, generate_batch_embeddings};
