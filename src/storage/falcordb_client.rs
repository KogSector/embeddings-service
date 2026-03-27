//! FalcorDB Client Module
//!
//! Provides connection management and operations for FalcorDB (Redis wire protocol)
//! with native vector storage capabilities.

use crate::core::{EmbeddingError, Result};
use chrono::{DateTime, Utc};
use prometheus::{Counter, Histogram, HistogramOpts, Opts, Registry};
use redis::{Client, RedisResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

lazy_static::lazy_static! {
    /// Global Prometheus registry for vector storage metrics
    pub static ref VECTOR_METRICS: VectorMetrics = VectorMetrics::new();
}

/// Prometheus metrics for vector storage operations
pub struct VectorMetrics {
    /// Histogram tracking vector storage operation duration
    pub storage_duration: Histogram,
    /// Counter tracking successful vector storage operations
    pub storage_success: Counter,
    /// Counter tracking failed vector storage operations
    pub storage_failure: Counter,
}

impl VectorMetrics {
    /// Create a new VectorMetrics instance with Prometheus metrics
    pub fn new() -> Self {
        let storage_duration = Histogram::with_opts(
            HistogramOpts::new(
                "vector_storage_duration_seconds",
                "Duration of vector storage operations in seconds"
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
        )
        .expect("Failed to create storage_duration histogram");

        let storage_success = Counter::with_opts(
            Opts::new(
                "vector_storage_success_total",
                "Total number of successful vector storage operations"
            )
        )
        .expect("Failed to create storage_success counter");

        let storage_failure = Counter::with_opts(
            Opts::new(
                "vector_storage_failure_total",
                "Total number of failed vector storage operations"
            )
        )
        .expect("Failed to create storage_failure counter");

        Self {
            storage_duration,
            storage_success,
            storage_failure,
        }
    }

    /// Register metrics with a Prometheus registry
    pub fn register(&self, registry: &Registry) -> Result<()> {
        registry
            .register(Box::new(self.storage_duration.clone()))
            .map_err(|e| EmbeddingError::ConfigError(format!("Failed to register storage_duration: {}", e)))?;
        
        registry
            .register(Box::new(self.storage_success.clone()))
            .map_err(|e| EmbeddingError::ConfigError(format!("Failed to register storage_success: {}", e)))?;
        
        registry
            .register(Box::new(self.storage_failure.clone()))
            .map_err(|e| EmbeddingError::ConfigError(format!("Failed to register storage_failure: {}", e)))?;

        Ok(())
    }

    /// Record a successful storage operation
    pub fn record_success(&self, duration: Duration) {
        self.storage_duration.observe(duration.as_secs_f64());
        self.storage_success.inc();
    }

    /// Record a failed storage operation
    pub fn record_failure(&self, duration: Duration) {
        self.storage_duration.observe(duration.as_secs_f64());
        self.storage_failure.inc();
    }
}

impl Default for VectorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Vector chunk node in FalcorDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorChunk {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub chunk_text: String,
    pub chunk_index: usize,
    pub document_id: Uuid,
    pub source_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

impl VectorChunk {
    /// Create a new VectorChunk
    pub fn new(
        text: String,
        embedding: Vec<f32>,
        document_id: Uuid,
        source_id: String,
        chunk_index: usize,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            embedding,
            chunk_text: text,
            chunk_index,
            document_id,
            source_id,
            created_at: now,
            updated_at: now,
            metadata: serde_json::json!({}),
        }
    }

    /// Validate vector chunk data
    pub fn validate(&self) -> Result<()> {
        if self.embedding.len() != 384 {
            return Err(EmbeddingError::InvalidInput(format!(
                "Invalid embedding dimension: expected 384, got {}",
                self.embedding.len()
            )));
        }
        if self.chunk_text.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "Chunk text cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration for FalcorDB connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalcorDBConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub database: String,
    pub vector_dimension: usize,
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub connection_pool_size: u32,
    pub connection_timeout_ms: u64,
    pub query_timeout_ms: u64,
}

impl Default for FalcorDBConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8765,
            username: "falkor".to_string(),
            password: String::new(),
            database: "falkordb".to_string(),
            vector_dimension: 384,
            similarity_threshold: 0.75,
            max_results: 100,
            connection_pool_size: 10,
            connection_timeout_ms: 5000,
            query_timeout_ms: 30000,
        }
    }
}

impl FalcorDBConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            host: std::env::var("FALKORDB_HOST")
                .unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("FALKORDB_PORT")
                .unwrap_or_else(|_| "8765".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_PORT: {}", e)))?,
            username: std::env::var("FALKORDB_USERNAME")
                .unwrap_or_else(|_| "falkor".to_string()),
            password: std::env::var("FALKORDB_PASSWORD").unwrap_or_else(|_| "".to_string()),
            database: std::env::var("FALKORDB_DATABASE")
                .unwrap_or_else(|_| "falkordb".to_string()),
            vector_dimension: std::env::var("FALKORDB_VECTOR_DIMENSION")
                .unwrap_or_else(|_| "384".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_VECTOR_DIMENSION: {}", e)))?,
            similarity_threshold: std::env::var("FALKORDB_SIMILARITY_THRESHOLD")
                .unwrap_or_else(|_| "0.75".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_SIMILARITY_THRESHOLD: {}", e)))?,
            max_results: std::env::var("FALKORDB_MAX_RESULTS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_MAX_RESULTS: {}", e)))?,
            connection_pool_size: std::env::var("FALKORDB_CONNECTION_POOL_SIZE")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_CONNECTION_POOL_SIZE: {}", e)))?,
            connection_timeout_ms: std::env::var("FALKORDB_CONNECTION_TIMEOUT_MS")
                .unwrap_or_else(|_| "5000".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_CONNECTION_TIMEOUT_MS: {}", e)))?,
            query_timeout_ms: std::env::var("FALKORDB_QUERY_TIMEOUT_MS")
                .unwrap_or_else(|_| "30000".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALKORDB_QUERY_TIMEOUT_MS: {}", e)))?,
        })
    }
}

/// FalcorDB client with connection pooling and retry logic
#[derive(Clone)]
pub struct FalcorDBClient {
    client: Arc<Client>,
    config: FalcorDBConfig,
}

impl FalcorDBClient {
    /// Create a new FalcorDB client with connection pooling and retry logic
    pub async fn new(config: FalcorDBConfig) -> Result<Self> {
        info!(
            "Initializing FalcorDB client: {}:{} (pool_size: {})",
            config.host, config.port, config.connection_pool_size
        );

        let connection_string = if config.password.is_empty() {
            format!("redis://{}:{}", config.host, config.port)
        } else {
            format!("redis://{}:{}@{}:{}", config.username, config.password, config.host, config.port)
        };

        let client = Client::open(connection_string.as_str())
            .map_err(|e| EmbeddingError::ConfigError(format!("Failed to create FalcorDB client: {}", e)))?;

        // Test connection with retry logic (up to 3 attempts)
        let max_retries = 3;
        let mut last_error = None;
        
        for attempt in 1..=max_retries {
            match client.get_multiplexed_async_connection().await {
                Ok(mut conn) => {
                    match redis::cmd("PING").query_async::<String>(&mut conn).await {
                        Ok(_) => {
                            info!("FalcorDB client initialized successfully on attempt {}", attempt);
                            return Ok(Self {
                                client: Arc::new(client),
                                config,
                            });
                        }
                        Err(e) => {
                            warn!("FalcorDB ping failed on attempt {}/{}: {}", attempt, max_retries, e);
                            last_error = Some(e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to get FalcorDB connection on attempt {}/{}: {}", attempt, max_retries, e);
                    last_error = Some(e);
                }
            }
            
            if attempt < max_retries {
                let backoff_ms = attempt * 1000; // Linear backoff: 1s, 2s, 3s
                info!("Retrying FalcorDB connection in {}ms...", backoff_ms);
                tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms as u64)).await;
            }
        }

        error!("Failed to initialize FalcorDB client after {} attempts", max_retries);
        Err(EmbeddingError::ConfigError(format!(
            "FalcorDB connection failed after {} retries: {}",
            max_retries,
            last_error.map(|e| e.to_string()).unwrap_or_else(|| "Unknown error".to_string())
        )))
    }

    /// Health check to verify FalcorDB connectivity
    pub async fn health_check(&self) -> Result<()> {
        debug!("Performing FalcorDB health check");

        let mut conn = self.client.get_multiplexed_async_connection().await
            .map_err(|e| {
                error!("FalcorDB health check failed: {}", e);
                EmbeddingError::ConfigError(format!("Health check failed: {}", e))
            })?;

        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| {
                error!("FalcorDB health check ping failed: {}", e);
                EmbeddingError::ConfigError(format!("Health check ping failed: {}", e))
            })?;

        debug!("FalcorDB health check passed");
        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &FalcorDBConfig {
        &self.config
    }

    /// Store a single vector chunk in FalcorDB
    pub async fn store_vector_chunk(&self, chunk: &VectorChunk) -> Result<String> {
        let start = Instant::now();
        
        // Validate chunk before storage
        if let Err(e) = chunk.validate() {
            let duration = start.elapsed();
            VECTOR_METRICS.record_failure(duration);
            error!(
                chunk_id = %chunk.id,
                document_id = %chunk.document_id,
                chunk_index = chunk.chunk_index,
                error = %e,
                duration_ms = duration.as_millis(),
                "Vector chunk validation failed"
            );
            return Err(e);
        }

        debug!(
            chunk_id = %chunk.id,
            document_id = %chunk.document_id,
            source_id = %chunk.source_id,
            chunk_index = chunk.chunk_index,
            embedding_dim = chunk.embedding.len(),
            text_length = chunk.chunk_text.len(),
            "Storing vector chunk"
        );

        let mut conn = self.client.get_multiplexed_async_connection().await
            .map_err(|e| {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    chunk_index = chunk.chunk_index,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Failed to get FalcorDB connection"
                );
                EmbeddingError::ConnectionError(format!("Failed to get connection: {}", e))
            })?;

        // Create Cypher query for storing chunk
        let cypher_query = format!(
            r#"
            CREATE (vc:Chunk {{
                id: '{}',
                content: '{}',
                embedding: '{}',
                chunk_index: {},
                document_id: '{}',
                source_id: '{}',
                created_at: {},
                updated_at: {},
                metadata: '{}'
            }})
            RETURN vc.id as id
            "#,
            chunk.id,
            escape_cypher_string(&chunk.chunk_text),
            serde_json::to_string(&chunk.embedding).unwrap_or_else(|_| "[]".to_string()),
            chunk.chunk_index,
            chunk.document_id,
            escape_cypher_string(&chunk.source_id),
            chunk.created_at.timestamp(),
            chunk.updated_at.timestamp(),
            serde_json::to_string(&chunk.metadata).unwrap_or_else(|_| "{}".to_string())
        );

        let result: redis::RedisResult<Vec<String>> = redis::cmd("GRAPH.QUERY")
            .arg(&self.config.database)
            .arg(&cypher_query)
            .query_async(&mut conn)
            .await;

        match result {
            Ok(response) => {
                let duration = start.elapsed();
                VECTOR_METRICS.record_success(duration);
                info!(
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    chunk_index = chunk.chunk_index,
                    duration_ms = duration.as_millis(),
                    "Successfully stored vector chunk"
                );
                Ok(chunk.id.to_string())
            }
            Err(e) => {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    chunk_index = chunk.chunk_index,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Failed to store vector chunk"
                );
                Err(EmbeddingError::ConnectionError(format!("FalcorDB query failed: {}", e)))
            }
        }
    }

    /// Store multiple vector chunks in a single transaction
    pub async fn batch_store_chunks(&self, chunks: Vec<VectorChunk>) -> Result<Vec<String>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let chunk_count = chunks.len();

        // Validate all chunks first
        for (idx, chunk) in chunks.iter().enumerate() {
            if let Err(e) = chunk.validate() {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_index = idx,
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    total_chunks = chunk_count,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Batch validation failed for chunk"
                );
                return Err(e);
            }
        }

        info!(
            chunk_count = chunk_count,
            "Starting batch storage of vector chunks"
        );

        let mut conn = self.client.get_multiplexed_async_connection().await
            .map_err(|e| {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_count = chunk_count,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Failed to get FalcorDB connection for batch"
                );
                EmbeddingError::ConnectionError(format!("Failed to get connection: {}", e))
            })?;

        let mut ids = Vec::with_capacity(chunks.len());

        for (idx, chunk) in chunks.iter().enumerate() {
            debug!(
                progress = format!("{}/{}", idx + 1, chunk_count),
                chunk_id = %chunk.id,
                document_id = %chunk.document_id,
                chunk_index = chunk.chunk_index,
                "Storing chunk in batch"
            );

            let cypher_query = format!(
                r#"
                CREATE (vc:Chunk {{
                    id: '{}',
                    content: '{}',
                    embedding: '{}',
                    chunk_index: {},
                    document_id: '{}',
                    source_id: '{}',
                    created_at: {},
                    updated_at: {},
                    metadata: '{}'
                }})
                RETURN vc.id as id
                "#,
                chunk.id,
                escape_cypher_string(&chunk.chunk_text),
                serde_json::to_string(&chunk.embedding).unwrap_or_else(|_| "[]".to_string()),
                chunk.chunk_index,
                chunk.document_id,
                escape_cypher_string(&chunk.source_id),
                chunk.created_at.timestamp(),
                chunk.updated_at.timestamp(),
                serde_json::to_string(&chunk.metadata).unwrap_or_else(|_| "{}".to_string())
            );

            let result: redis::RedisResult<Vec<String>> = redis::cmd("GRAPH.QUERY")
                .arg(&self.config.database)
                .arg(&cypher_query)
                .query_async(&mut conn)
                .await;

            match result {
                Ok(_) => {
                    ids.push(chunk.id.to_string());
                }
                Err(e) => {
                    let duration = start.elapsed();
                    VECTOR_METRICS.record_failure(duration);
                    error!(
                        chunk_index = idx,
                        chunk_id = %chunk.id,
                        document_id = %chunk.document_id,
                        total_chunks = chunk_count,
                        error = %e,
                        duration_ms = duration.as_millis(),
                        "Failed to store chunk in batch"
                    );
                    return Err(EmbeddingError::ConnectionError(format!("Batch storage failed: {}", e)));
                }
            }
        }

        let duration = start.elapsed();
        VECTOR_METRICS.record_success(duration);
        info!(
            chunk_count = ids.len(),
            duration_ms = duration.as_millis(),
            "Successfully stored vector chunks in batch"
        );

        Ok(ids)
    }

    /// Convenience wrapper for storing a vector chunk with comprehensive logging
    /// 
    /// This method wraps `store_vector_chunk` and provides a same functionality
    /// with explicit emphasis on observability. It's useful when you want to make
    /// it clear that logging and metrics are being captured.
    pub async fn store_vector_with_logging(&self, chunk: &VectorChunk) -> Result<String> {
        self.store_vector_chunk(chunk).await
    }
}

/// Escape single quotes in Cypher strings
fn escape_cypher_string(s: &str) -> String {
    s.replace('\'', "\\'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FalcorDBConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 8765);
        assert_eq!(config.vector_dimension, 384);
        assert_eq!(config.similarity_threshold, 0.75);
        assert_eq!(config.connection_pool_size, 10);
    }

    #[test]
    fn test_config_from_env_missing_password() {
        // Clear password env var if set
        std::env::remove_var("FALKORDB_PASSWORD");
        
        let result = FalcorDBConfig::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("FALKORDB_PASSWORD"));
    }

    #[test]
    fn test_vector_chunk_new() {
        let embedding = vec![0.1; 384];
        let document_id = Uuid::new_v4();
        let source_id = "test-source".to_string();
        
        let chunk = VectorChunk::new(
            "Test chunk text".to_string(),
            embedding.clone(),
            document_id,
            source_id.clone(),
            0,
        );

        assert_eq!(chunk.chunk_text, "Test chunk text");
        assert_eq!(chunk.embedding.len(), 384);
        assert_eq!(chunk.document_id, document_id);
        assert_eq!(chunk.source_id, source_id);
        assert_eq!(chunk.chunk_index, 0);
    }

    #[test]
    fn test_vector_chunk_validate_success() {
        let embedding = vec![0.1; 384];
        let chunk = VectorChunk::new(
            "Valid chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );

        assert!(chunk.validate().is_ok());
    }

    #[test]
    fn test_vector_chunk_validate_invalid_dimension() {
        let embedding = vec![0.1; 256]; // Wrong dimension
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid embedding dimension"));
    }

    #[test]
    fn test_vector_chunk_validate_empty_text() {
        let embedding = vec![0.1; 384];
        let chunk = VectorChunk::new(
            "".to_string(), // Empty text
            embedding,
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Chunk text cannot be empty"));
    }

    #[test]
    fn test_vector_metrics_creation() {
        let metrics = VectorMetrics::new();
        
        // Verify metrics can be created without panicking
        assert_eq!(metrics.storage_success.get(), 0.0);
        assert_eq!(metrics.storage_failure.get(), 0.0);
    }

    #[test]
    fn test_vector_metrics_record_success() {
        let metrics = VectorMetrics::new();
        let duration = Duration::from_millis(100);
        
        metrics.record_success(duration);
        
        assert_eq!(metrics.storage_success.get(), 1.0);
        assert_eq!(metrics.storage_failure.get(), 0.0);
    }

    #[test]
    fn test_vector_metrics_record_failure() {
        let metrics = VectorMetrics::new();
        let duration = Duration::from_millis(50);
        
        metrics.record_failure(duration);
        
        assert_eq!(metrics.storage_success.get(), 0.0);
        assert_eq!(metrics.storage_failure.get(), 1.0);
    }

    #[test]
    fn test_vector_metrics_multiple_operations() {
        let metrics = VectorMetrics::new();
        
        metrics.record_success(Duration::from_millis(100));
        metrics.record_success(Duration::from_millis(150));
        metrics.record_failure(Duration::from_millis(200));
        
        assert_eq!(metrics.storage_success.get(), 2.0);
        assert_eq!(metrics.storage_failure.get(), 1.0);
    }

    #[test]
    fn test_escape_cypher_string() {
        assert_eq!(escape_cypher_string("test"), "test");
        assert_eq!(escape_cypher_string("test's quote"), "test\\'s quote");
        assert_eq!(escape_cypher_string("multiple '' quotes"), "multiple \\'\\' quotes");
    }
}
