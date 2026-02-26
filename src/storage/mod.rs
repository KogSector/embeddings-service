//! FalcorDB Vector Storage Module
//!
//! Stores and retrieves embedding vectors in FalcorDB graph nodes.
//! Each chunk becomes a node with an `embedding` property containing the vector.

pub mod falcordb_client;

#[cfg(test)]
mod falcordb_client_connection_test;

#[cfg(test)]
mod vector_chunk_schema_test;

#[cfg(test)]
mod embedding_dimension_test;

#[cfg(test)]
mod vector_storage_roundtrip_test;

#[cfg(test)]
mod batch_insert_atomicity_test;

#[cfg(test)]
mod vector_storage_metrics_test;

pub use falcordb_client::{FalcorDBClient, FalcorDBConfig, VectorChunk};

use anyhow::{anyhow, Result};
use redis::aio::MultiplexedConnection;
use redis::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info};

// =============================================================================
// Data Types
// =============================================================================

/// A vector record for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: uuid::Uuid,
    pub text: String,
    pub embedding: Vec<f32>,
    pub model: String,
    pub metadata: serde_json::Value,
}

/// An embedding record for graph storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub node_id: String,
    pub node_label: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub fused_embedding: Vec<f32>,
    pub models_used: Vec<String>,
    pub quality_score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result from a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub node_id: String,
    pub node_label: String,
    pub text: String,
    pub similarity: f32,
    pub quality_score: f32,
}

// =============================================================================
// FalcorDB Vector Storage
// =============================================================================

/// FalcorDB-backed vector storage using the Redis wire protocol.
#[derive(Clone)]
pub struct FalkorVectorStorage {
    connection: MultiplexedConnection,
    graph_name: String,
}

impl FalkorVectorStorage {
    /// Create from environment variables.
    pub async fn from_env() -> Result<Self> {
        let host = std::env::var("FALCORDB_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = std::env::var("FALCORDB_PORT")
            .unwrap_or_else(|_| "6379".to_string())
            .parse()
            .unwrap_or(6379);
        let password = std::env::var("FALCORDB_PASSWORD").ok();
        let graph_name = std::env::var("FALCORDB_GRAPH_NAME")
            .unwrap_or_else(|_| "confuse_knowledge".to_string());

        let url = match password {
            Some(ref pw) if !pw.is_empty() => format!("redis://:{}@{}:{}", pw, host, port),
            _ => format!("redis://{}:{}", host, port),
        };

        info!("Connecting FalkorVectorStorage to {}:{} graph={}", host, port, graph_name);

        let client = redis::Client::open(url.as_str())
            .map_err(|e| anyhow!("Failed to create FalcorDB client: {}", e))?;

        let connection = client
            .get_multiplexed_tokio_connection()
            .await
            .map_err(|e| anyhow!("Failed to connect to FalcorDB: {}", e))?;

        info!("FalkorVectorStorage connected successfully");

        Ok(Self {
            connection,
            graph_name,
        })
    }

    /// Execute a Cypher query via GRAPH.QUERY.
    async fn query(&mut self, cypher: &str) -> Result<Value> {
        debug!("GRAPH.QUERY {} \"{}\"", self.graph_name, cypher);

        let result: Value = redis::cmd("GRAPH.QUERY")
            .arg(&self.graph_name)
            .arg(cypher)
            .query_async(&mut self.connection)
            .await
            .map_err(|e| anyhow!("FalcorDB query failed: {}", e))?;

        Ok(result)
    }

    /// Execute a read-only Cypher query.
    async fn ro_query(&mut self, cypher: &str) -> Result<Value> {
        let result: Value = redis::cmd("GRAPH.RO_QUERY")
            .arg(&self.graph_name)
            .arg(cypher)
            .query_async(&mut self.connection)
            .await
            .map_err(|e| anyhow!("FalcorDB read-only query failed: {}", e))?;

        Ok(result)
    }

    // -------------------------------------------------------------------------
    //  Store operations
    // -------------------------------------------------------------------------

    /// Store a single embedding in a FalcorDB node.
    pub async fn store_embedding(&mut self, record: EmbeddingRecord) -> Result<String> {
        let embedding_json = serde_json::to_string(&record.fused_embedding)?;
        let models_json = serde_json::to_string(&record.models_used)?;
        let metadata_json = serde_json::to_string(&record.metadata)?;

        let cypher = format!(
            "MERGE (n:{label} {{id: '{id}'}}) \
             SET n.text_content = '{text}', \
                 n.embedding = '{embedding}', \
                 n.models_used = '{models}', \
                 n.quality_score = {quality}, \
                 n.metadata = '{metadata}', \
                 n.embedding_dim = {dim}, \
                 n.updated_at = {ts} \
             RETURN n.id",
            label = Self::escape(&record.node_label),
            id = Self::escape(&record.node_id),
            text = Self::escape(&record.text),
            embedding = Self::escape(&embedding_json),
            models = Self::escape(&models_json),
            quality = record.quality_score,
            metadata = Self::escape(&metadata_json),
            dim = record.fused_embedding.len(),
            ts = chrono::Utc::now().timestamp(),
        );

        self.query(&cypher).await?;

        info!(
            "Stored embedding in FalcorDB: node_id={}, label={}, dim={}",
            record.node_id, record.node_label, record.fused_embedding.len()
        );

        Ok(record.node_id)
    }

    /// Batch store multiple embeddings.
    pub async fn store_batch(&mut self, records: Vec<EmbeddingRecord>) -> Result<Vec<String>> {
        let mut stored_ids = Vec::new();

        for record in records {
            match self.store_embedding(record).await {
                Ok(id) => stored_ids.push(id),
                Err(e) => error!("Failed to store embedding: {}", e),
            }
        }

        Ok(stored_ids)
    }

    // -------------------------------------------------------------------------
    //  Retrieve operations
    // -------------------------------------------------------------------------

    /// Get an embedding by node ID.
    pub async fn get_embedding(&mut self, node_id: &str) -> Result<Option<EmbeddingRecord>> {
        let cypher = format!(
            "MATCH (n) WHERE n.id = '{}' RETURN n.id, n.text_content, n.embedding, n.models_used, n.quality_score, labels(n)",
            Self::escape(node_id)
        );

        let result = self.ro_query(&cypher).await?;

        // Parse the first row if present
        if let Value::Array(ref arr) = result {
            if arr.len() >= 2 {
                if let Value::Array(ref rows) = arr[1] {
                    if let Some(Value::Array(ref row)) = rows.first() {
                        return Ok(Some(self.parse_embedding_row(row)?));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Vector similarity search using cosine similarity in Cypher.
    ///
    /// This loads the query vector and computes cosine similarity against
    /// all nodes with an `embedding` property. For large graphs, consider
    /// limiting the search to specific labels.
    pub async fn search_similar(
        &mut self,
        query_embedding: &[f32],
        limit: usize,
        min_similarity: f32,
    ) -> Result<Vec<SimilarityResult>> {
        let query_json = serde_json::to_string(query_embedding)?;

        // Cypher-level cosine similarity:
        // We store embeddings as JSON strings. The comparison is done
        // by computing dot product / (norm_a * norm_b) in application code
        // after retrieval since FalcorDB doesn't have native vector functions yet.
        let cypher = format!(
            "MATCH (n:Chunk) WHERE n.embedding IS NOT NULL RETURN n.id, n.text_content, n.embedding, n.quality_score, labels(n) LIMIT {}",
            limit * 5 // Fetch extra to filter by similarity
        );

        let result = self.ro_query(&cypher).await?;
        let mut results = Vec::new();

        if let Value::Array(ref arr) = result {
            if arr.len() >= 2 {
                if let Value::Array(ref rows) = arr[1] {
                    for row_val in rows {
                        if let Value::Array(ref cells) = row_val {
                            if cells.len() >= 5 {
                                let node_id = Self::extract_string(&cells[0]);
                                let text = Self::extract_string(&cells[1]);
                                let embedding_str = Self::extract_string(&cells[2]);
                                let quality = Self::extract_float(&cells[3]);
                                let label = Self::extract_string(&cells[4]);

                                // Parse stored embedding
                                if let Ok(stored_embedding) =
                                    serde_json::from_str::<Vec<f32>>(&embedding_str)
                                {
                                    let similarity =
                                        Self::cosine_similarity(query_embedding, &stored_embedding);

                                    if similarity >= min_similarity {
                                        results.push(SimilarityResult {
                                            node_id,
                                            node_label: label,
                                            text,
                                            similarity,
                                            quality_score: quality,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by similarity descending and take top `limit`
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Delete an embedding node.
    pub async fn delete_embedding(&mut self, node_id: &str) -> Result<bool> {
        let cypher = format!(
            "MATCH (n) WHERE n.id = '{}' DELETE n RETURN count(n)",
            Self::escape(node_id)
        );
        self.query(&cypher).await?;
        Ok(true)
    }

    /// Health check.
    pub async fn ping(&mut self) -> bool {
        let result: redis::RedisResult<String> = redis::cmd("PING")
            .query_async(&mut self.connection)
            .await;
        result.is_ok()
    }

    // -------------------------------------------------------------------------
    //  Helpers
    // -------------------------------------------------------------------------

    /// Cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    fn escape(s: &str) -> String {
        s.replace('\'', "\\'").replace('\\', "\\\\")
    }

    fn extract_string(value: &Value) -> String {
        match value {
            Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
            Value::SimpleString(s) => s.clone(),
            Value::Int(n) => n.to_string(),
            _ => String::new(),
        }
    }

    fn extract_float(value: &Value) -> f32 {
        match value {
            Value::Double(f) => *f as f32,
            Value::Int(n) => *n as f32,
            Value::BulkString(s) => String::from_utf8_lossy(s).parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    fn parse_embedding_row(&self, cells: &[Value]) -> Result<EmbeddingRecord> {
        let node_id = Self::extract_string(cells.first().unwrap_or(&Value::Nil));
        let text = Self::extract_string(cells.get(1).unwrap_or(&Value::Nil));
        let embedding_str = Self::extract_string(cells.get(2).unwrap_or(&Value::Nil));
        let models_str = Self::extract_string(cells.get(3).unwrap_or(&Value::Nil));
        let quality = Self::extract_float(cells.get(4).unwrap_or(&Value::Nil));
        let label = Self::extract_string(cells.get(5).unwrap_or(&Value::Nil));

        let fused_embedding: Vec<f32> =
            serde_json::from_str(&embedding_str).unwrap_or_default();
        let models_used: Vec<String> =
            serde_json::from_str(&models_str).unwrap_or_default();

        Ok(EmbeddingRecord {
            node_id,
            node_label: label,
            text,
            embedding: fused_embedding.clone(),
            fused_embedding,
            models_used,
            quality_score: quality,
            metadata: HashMap::new(),
        })
    }
}
