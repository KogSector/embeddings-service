//! Storage functionality for embeddings and vectors

pub mod neo4j;
pub mod postgres;
pub mod redis;

pub use neo4j::{Neo4jConfig, Neo4jEmbeddingRecord, Neo4jStorage};
pub use postgres::{PostgresStorage, VectorRecord};
pub use redis::RedisCache;
