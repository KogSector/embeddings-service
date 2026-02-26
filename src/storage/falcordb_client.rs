//! FalcorDB Client Module
//!
//! Provides connection management and operations for FalcorDB (Neo4j-based graph database)
//! with native vector storage capabilities.

use crate::core::{EmbeddingError, Result};
use chrono::{DateTime, Utc};
use neo4rs::{Graph, Query};
use prometheus::{Counter, Histogram, HistogramOpts, Opts, Registry};
use serde::{Deserialize, Serialize};
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
            port: 6379,
            username: "neo4j".to_string(),
            password: String::new(),
            database: "neo4j".to_string(),
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
            host: std::env::var("FALCORDB_HOST")
                .unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("FALCORDB_PORT")
                .unwrap_or_else(|_| "6379".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_PORT: {}", e)))?,
            username: std::env::var("FALCORDB_USERNAME")
                .unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("FALCORDB_PASSWORD")
                .map_err(|_| EmbeddingError::ConfigError("FALCORDB_PASSWORD not set".to_string()))?,
            database: std::env::var("FALCORDB_DATABASE")
                .unwrap_or_else(|_| "neo4j".to_string()),
            vector_dimension: std::env::var("FALCORDB_VECTOR_DIMENSION")
                .unwrap_or_else(|_| "384".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_VECTOR_DIMENSION: {}", e)))?,
            similarity_threshold: std::env::var("FALCORDB_SIMILARITY_THRESHOLD")
                .unwrap_or_else(|_| "0.75".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_SIMILARITY_THRESHOLD: {}", e)))?,
            max_results: std::env::var("FALCORDB_MAX_RESULTS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_MAX_RESULTS: {}", e)))?,
            connection_pool_size: std::env::var("FALCORDB_CONNECTION_POOL_SIZE")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_CONNECTION_POOL_SIZE: {}", e)))?,
            connection_timeout_ms: std::env::var("FALCORDB_CONNECTION_TIMEOUT_MS")
                .unwrap_or_else(|_| "5000".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_CONNECTION_TIMEOUT_MS: {}", e)))?,
            query_timeout_ms: std::env::var("FALCORDB_QUERY_TIMEOUT_MS")
                .unwrap_or_else(|_| "30000".to_string())
                .parse()
                .map_err(|e| EmbeddingError::ConfigError(format!("Invalid FALCORDB_QUERY_TIMEOUT_MS: {}", e)))?,
        })
    }
}

/// FalcorDB client with connection pooling and retry logic
#[derive(Clone)]
pub struct FalcorDBClient {
    graph: Arc<Graph>,
    config: FalcorDBConfig,
}

impl FalcorDBClient {
    /// Create a new FalcorDB client with connection pooling
    pub async fn new(config: FalcorDBConfig) -> Result<Self> {
        info!(
            "Initializing FalcorDB client: {}:{} (pool_size: {})",
            config.host, config.port, config.connection_pool_size
        );

        let uri = format!("bolt://{}:{}", config.host, config.port);
        
        // Attempt connection with retry logic
        let graph = Self::connect_with_retry(&uri, &config).await?;

        info!("FalcorDB client initialized successfully");

        Ok(Self {
            graph: Arc::new(graph),
            config,
        })
    }

    /// Connect to FalcorDB with exponential backoff retry logic
    async fn connect_with_retry(uri: &str, config: &FalcorDBConfig) -> Result<Graph> {
        const MAX_RETRIES: u32 = 5;
        const INITIAL_BACKOFF_MS: u64 = 100;

        let mut retry_count = 0;
        let mut backoff_ms = INITIAL_BACKOFF_MS;

        loop {
            match Self::attempt_connection(uri, config).await {
                Ok(graph) => {
                    if retry_count > 0 {
                        info!("Successfully connected to FalcorDB after {} retries", retry_count);
                    }
                    return Ok(graph);
                }
                Err(e) => {
                    retry_count += 1;
                    
                    if retry_count >= MAX_RETRIES {
                        error!(
                            "Failed to connect to FalcorDB after {} attempts: {}",
                            MAX_RETRIES, e
                        );
                        return Err(EmbeddingError::ConfigError(format!(
                            "Failed to connect to FalcorDB after {} retries: {}",
                            MAX_RETRIES, e
                        )));
                    }

                    warn!(
                        "Failed to connect to FalcorDB (attempt {}/{}): {}. Retrying in {}ms...",
                        retry_count, MAX_RETRIES, e, backoff_ms
                    );

                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    
                    // Exponential backoff with jitter
                    backoff_ms = (backoff_ms * 2).min(5000);
                }
            }
        }
    }

    /// Attempt a single connection to FalcorDB
    async fn attempt_connection(uri: &str, config: &FalcorDBConfig) -> Result<Graph> {
        debug!("Attempting connection to FalcorDB at {}", uri);

        let graph = Graph::new(uri, &config.username, &config.password)
            .await
            .map_err(|e| {
                EmbeddingError::ConfigError(format!("Failed to create Graph connection: {}", e))
            })?;

        Ok(graph)
    }

    /// Health check to verify FalcorDB connectivity
    pub async fn health_check(&self) -> Result<()> {
        debug!("Performing FalcorDB health check");

        let query = Query::new("RETURN 1 as health".to_string());
        
        let mut result = self.graph.execute(query).await.map_err(|e| {
            error!("FalcorDB health check failed: {}", e);
            EmbeddingError::ConfigError(format!("Health check failed: {}", e))
        })?;

        if result.next().await.map_err(|e| {
            EmbeddingError::ConfigError(format!("Health check result error: {}", e))
        })?.is_some() {
            debug!("FalcorDB health check passed");
            Ok(())
        } else {
            error!("FalcorDB health check returned no results");
            Err(EmbeddingError::ConfigError(
                "Health check failed: no results".to_string(),
            ))
        }
    }

    /// Get the underlying Graph connection
    pub fn graph(&self) -> &Graph {
        &self.graph
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

        let query = Query::new(
            r#"
            CREATE (vc:Vector_Chunk {
                id: $id,
                embedding: $embedding,
                chunk_text: $chunk_text,
                chunk_index: $chunk_index,
                document_id: $document_id,
                source_id: $source_id,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at),
                metadata: $metadata
            })
            RETURN vc.id as id
            "#
            .to_string(),
        )
        .param("id", chunk.id.to_string())
        .param("embedding", chunk.embedding.clone())
        .param("chunk_text", chunk.chunk_text.clone())
        .param("chunk_index", chunk.chunk_index as i64)
        .param("document_id", chunk.document_id.to_string())
        .param("source_id", chunk.source_id.clone())
        .param("created_at", chunk.created_at.to_rfc3339())
        .param("updated_at", chunk.updated_at.to_rfc3339())
        .param("metadata", serde_json::to_string(&chunk.metadata)?);

        let mut result = self.graph.execute(query).await.map_err(|e| {
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
            EmbeddingError::Neo4jError(e)
        })?;

        if let Some(row) = result.next().await.map_err(|e| {
            let duration = start.elapsed();
            VECTOR_METRICS.record_failure(duration);
            error!(
                chunk_id = %chunk.id,
                document_id = %chunk.document_id,
                error = %e,
                duration_ms = duration.as_millis(),
                "Failed to retrieve stored chunk ID"
            );
            EmbeddingError::Neo4jError(e)
        })? {
            let id: String = row.get("id").map_err(|e| {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Failed to extract ID from result"
                );
                EmbeddingError::Neo4jError(e)
            })?;
            
            let duration = start.elapsed();
            VECTOR_METRICS.record_success(duration);
            info!(
                chunk_id = %id,
                document_id = %chunk.document_id,
                chunk_index = chunk.chunk_index,
                duration_ms = duration.as_millis(),
                "Successfully stored vector chunk"
            );
            Ok(id)
        } else {
            let duration = start.elapsed();
            VECTOR_METRICS.record_failure(duration);
            error!(
                chunk_id = %chunk.id,
                document_id = %chunk.document_id,
                duration_ms = duration.as_millis(),
                "No ID returned after storing vector chunk"
            );
            Err(EmbeddingError::ConnectionError(
                "No ID returned from storage operation".to_string(),
            ))
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

        let mut txn = self.graph.start_txn().await.map_err(|e| {
            let duration = start.elapsed();
            VECTOR_METRICS.record_failure(duration);
            error!(
                chunk_count = chunk_count,
                error = %e,
                duration_ms = duration.as_millis(),
                "Failed to start transaction for batch storage"
            );
            EmbeddingError::Neo4jError(e)
        })?;

        let mut ids = Vec::with_capacity(chunks.len());

        for (idx, chunk) in chunks.iter().enumerate() {
            debug!(
                progress = format!("{}/{}", idx + 1, chunk_count),
                chunk_id = %chunk.id,
                document_id = %chunk.document_id,
                chunk_index = chunk.chunk_index,
                "Storing chunk in batch transaction"
            );

            let query = Query::new(
                r#"
                CREATE (vc:Vector_Chunk {
                    id: $id,
                    embedding: $embedding,
                    chunk_text: $chunk_text,
                    chunk_index: $chunk_index,
                    document_id: $document_id,
                    source_id: $source_id,
                    created_at: datetime($created_at),
                    updated_at: datetime($updated_at),
                    metadata: $metadata
                })
                RETURN vc.id as id
                "#
                .to_string(),
            )
            .param("id", chunk.id.to_string())
            .param("embedding", chunk.embedding.clone())
            .param("chunk_text", chunk.chunk_text.clone())
            .param("chunk_index", chunk.chunk_index as i64)
            .param("document_id", chunk.document_id.to_string())
            .param("source_id", chunk.source_id.clone())
            .param("created_at", chunk.created_at.to_rfc3339())
            .param("updated_at", chunk.updated_at.to_rfc3339())
            .param("metadata", serde_json::to_string(&chunk.metadata)?);

            let mut result = txn.execute(query).await.map_err(|e| {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_index = idx,
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    total_chunks = chunk_count,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Failed to store chunk in batch transaction"
                );
                EmbeddingError::Neo4jError(e)
            })?;

            if let Some(row) = result.next().await.map_err(|e| {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_index = idx,
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "Failed to retrieve chunk ID in batch transaction"
                );
                EmbeddingError::Neo4jError(e)
            })? {
                let id: String = row.get("id").map_err(|e| {
                    let duration = start.elapsed();
                    VECTOR_METRICS.record_failure(duration);
                    error!(
                        chunk_index = idx,
                        chunk_id = %chunk.id,
                        document_id = %chunk.document_id,
                        error = %e,
                        duration_ms = duration.as_millis(),
                        "Failed to extract ID from batch transaction result"
                    );
                    EmbeddingError::Neo4jError(e)
                })?;
                ids.push(id);
            } else {
                let duration = start.elapsed();
                VECTOR_METRICS.record_failure(duration);
                error!(
                    chunk_index = idx,
                    chunk_id = %chunk.id,
                    document_id = %chunk.document_id,
                    duration_ms = duration.as_millis(),
                    "No ID returned for chunk in batch transaction"
                );
                return Err(EmbeddingError::ConnectionError(
                    "No ID returned from batch storage operation".to_string(),
                ));
            }
        }

        txn.commit().await.map_err(|e| {
            let duration = start.elapsed();
            VECTOR_METRICS.record_failure(duration);
            error!(
                chunk_count = chunk_count,
                error = %e,
                duration_ms = duration.as_millis(),
                "Failed to commit batch transaction"
            );
            EmbeddingError::Neo4jError(e)
        })?;

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
    /// This method wraps `store_vector_chunk` and provides the same functionality
    /// with explicit emphasis on observability. It's useful when you want to make
    /// it clear that logging and metrics are being captured.
    pub async fn store_vector_with_logging(&self, chunk: &VectorChunk) -> Result<String> {
        self.store_vector_chunk(chunk).await
    }

    /// Create HAS_CHUNK relationship from Document to Vector_Chunk
    pub async fn create_has_chunk_relationship(
        &self,
        document_id: Uuid,
        chunk_id: Uuid,
        chunk_index: usize,
    ) -> Result<()> {
        debug!(
            "Creating HAS_CHUNK relationship: document={}, chunk={}, index={}",
            document_id, chunk_id, chunk_index
        );

        let query = Query::new(
            r#"
            MATCH (d:Document {id: $document_id})
            MATCH (vc:Vector_Chunk {id: $chunk_id})
            CREATE (d)-[r:HAS_CHUNK {
                chunk_index: $chunk_index,
                created_at: datetime()
            }]->(vc)
            RETURN r
            "#
            .to_string(),
        )
        .param("document_id", document_id.to_string())
        .param("chunk_id", chunk_id.to_string())
        .param("chunk_index", chunk_index as i64);

        self.graph.run(query).await.map_err(|e| {
            error!(
                "Failed to create HAS_CHUNK relationship: document={}, chunk={}, error={}",
                document_id, chunk_id, e
            );
            EmbeddingError::Neo4jError(e)
        })?;

        info!(
            "Created HAS_CHUNK relationship: document={}, chunk={}",
            document_id, chunk_id
        );

        Ok(())
    }

    /// Create NEXT_CHUNK relationship between sequential Vector_Chunk nodes
    pub async fn create_next_chunk_relationship(
        &self,
        from_chunk_id: Uuid,
        to_chunk_id: Uuid,
        sequence_number: usize,
    ) -> Result<()> {
        debug!(
            "Creating NEXT_CHUNK relationship: from={}, to={}, sequence={}",
            from_chunk_id, to_chunk_id, sequence_number
        );

        let query = Query::new(
            r#"
            MATCH (vc1:Vector_Chunk {id: $from_chunk_id})
            MATCH (vc2:Vector_Chunk {id: $to_chunk_id})
            CREATE (vc1)-[r:NEXT_CHUNK {
                sequence_number: $sequence_number
            }]->(vc2)
            RETURN r
            "#
            .to_string(),
        )
        .param("from_chunk_id", from_chunk_id.to_string())
        .param("to_chunk_id", to_chunk_id.to_string())
        .param("sequence_number", sequence_number as i64);

        self.graph.run(query).await.map_err(|e| {
            error!(
                "Failed to create NEXT_CHUNK relationship: from={}, to={}, error={}",
                from_chunk_id, to_chunk_id, e
            );
            EmbeddingError::Neo4jError(e)
        })?;

        info!(
            "Created NEXT_CHUNK relationship: from={}, to={}",
            from_chunk_id, to_chunk_id
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FalcorDBConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 6379);
        assert_eq!(config.vector_dimension, 384);
        assert_eq!(config.similarity_threshold, 0.75);
        assert_eq!(config.connection_pool_size, 10);
    }

    #[test]
    fn test_config_from_env_missing_password() {
        // Clear password env var if set
        std::env::remove_var("FALCORDB_PASSWORD");
        
        let result = FalcorDBConfig::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("FALCORDB_PASSWORD"));
    }

    #[tokio::test]
    async fn test_health_check_requires_connection() {
        // This test documents that health_check requires a valid connection
        // In a real test environment, you would set up a test FalcorDB instance
        
        // Skip if no test database available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return;
        }

        let config = FalcorDBConfig {
            password: std::env::var("FALCORDB_PASSWORD").unwrap_or_default(),
            ..Default::default()
        };

        let client = FalcorDBClient::new(config).await;
        
        if let Ok(client) = client {
            let health = client.health_check().await;
            assert!(health.is_ok());
        }
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
}
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
