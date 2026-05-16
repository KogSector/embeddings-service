//! API endpoints for the embeddings service (generate-only)

pub mod generate;
pub mod health;
pub use generate::{generate_embeddings, generate_batch_embeddings};
pub use health::health_check;
