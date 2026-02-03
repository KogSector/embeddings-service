//! API endpoints for the embeddings service

pub mod health;
pub mod generate;
pub mod search;
pub mod graphiti;
pub mod neo4j;

pub use health::health_check;
pub use generate::{generate_embeddings, generate_batch_embeddings};
pub use search::{search_similar, list_models};
pub use graphiti::{generate_graphiti_embeddings, process_chunks, list_graphiti_models, graphiti_health};
pub use neo4j::{
    generate_neo4j_embeddings, get_neo4j_embeddings,
    search_neo4j_embeddings, batch_store_neo4j_embeddings, delete_neo4j_embedding,
};

