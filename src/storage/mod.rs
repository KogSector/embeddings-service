//! Storage functionality for embeddings and vectors

pub mod postgres;
pub mod redis;

pub use postgres::PostgresStorage;
pub use redis::RedisCache;
