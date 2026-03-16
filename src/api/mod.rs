//! API endpoints for the embeddings service with FalcorDB storage

pub mod health;
pub mod generate;
pub mod graphiti;
pub mod falcordb;

pub use health::health_check;
pub use generate::{generate_embeddings, generate_batch_embeddings};
pub use graphiti::{generate_graphiti_embeddings, process_chunks, list_graphiti_models, graphiti_health};
pub use falcordb::{store_embeddings_falcordb, get_falcordb_stats, test_falcordb_connection};
