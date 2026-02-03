//! Neo4j storage for embeddings using neo4rs driver
//!
//! Provides real Neo4j integration for storing and querying embeddings
//! directly in graph nodes.

use crate::core::Result;
use neo4rs::{Graph, Query};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for Neo4j connection
#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    pub max_connections: usize,
}

impl Neo4jConfig {
    pub fn from_env() -> Self {
        Self {
            uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string()),
            max_connections: std::env::var("NEO4J_MAX_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        }
    }
}

/// Record stored in Neo4j with embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jEmbeddingRecord {
    pub node_id: String,
    pub node_label: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub fused_embedding: Vec<f32>,
    pub models_used: Vec<String>,
    pub quality_score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Neo4j storage implementation
pub struct Neo4jStorage {
    graph: Arc<Graph>,
    initialized: Arc<RwLock<bool>>,
}

impl Neo4jStorage {
    /// Create new Neo4j storage connection
    pub async fn new(config: &Neo4jConfig) -> Result<Self> {
        tracing::info!(
            "Connecting to Neo4j: {} as user {}",
            config.uri,
            config.user
        );

        let graph = Graph::new(&config.uri, &config.user, &config.password).await?;

        let storage = Self {
            graph: Arc::new(graph),
            initialized: Arc::new(RwLock::new(false)),
        };

        // Ensure vector index exists
        storage.ensure_indices().await?;

        Ok(storage)
    }

    /// Ensure required indices exist
    async fn ensure_indices(&self) -> Result<()> {
        let mut initialized = self.initialized.write().await;

        if *initialized {
            return Ok(());
        }

        tracing::info!("Creating Neo4j embedding indices if they don't exist");

        // Create constraint on node_id for Embedding nodes
        let constraint_query = r#"
            CREATE CONSTRAINT embedding_node_id IF NOT EXISTS
            FOR (e:Embedding) REQUIRE e.node_id IS UNIQUE
        "#;

        if let Err(e) = self.graph.run(Query::new(constraint_query.to_string())).await {
            tracing::warn!("Could not create constraint (may already exist): {}", e);
        }

        // Create index on embedding vector (if Neo4j version supports it)
        let index_query = r#"
            CREATE INDEX embedding_text_idx IF NOT EXISTS
            FOR (e:Embedding) ON (e.text)
        "#;

        if let Err(e) = self.graph.run(Query::new(index_query.to_string())).await {
            tracing::warn!("Could not create text index (may already exist): {}", e);
        }

        *initialized = true;
        tracing::info!("Neo4j indices verified/created");

        Ok(())
    }

    /// Store a single embedding in Neo4j
    pub async fn store_embedding(&self, record: Neo4jEmbeddingRecord) -> Result<String> {
        let query = r#"
            MERGE (e:Embedding {node_id: $node_id})
            SET e.node_label = $node_label,
                e.text = $text,
                e.embedding = $embedding,
                e.fused_embedding = $fused_embedding,
                e.models_used = $models_used,
                e.quality_score = $quality_score,
                e.metadata = $metadata,
                e.updated_at = datetime()
            ON CREATE SET e.created_at = datetime()
            RETURN e.node_id as node_id
        "#;

        let metadata_json = serde_json::to_string(&record.metadata)?;

        let mut result = self
            .graph
            .execute(
                Query::new(query.to_string())
                    .param("node_id", record.node_id.clone())
                    .param("node_label", record.node_label)
                    .param("text", record.text)
                    .param("embedding", record.embedding)
                    .param("fused_embedding", record.fused_embedding)
                    .param("models_used", record.models_used)
                    .param("quality_score", record.quality_score as f64)
                    .param("metadata", metadata_json),
            )
            .await?;

        if let Some(row) = result.next().await? {
            let node_id: String = row.get("node_id")?;
            tracing::debug!("Stored embedding for node: {}", node_id);
            return Ok(node_id);
        }

        Ok(record.node_id)
    }

    /// Store multiple embeddings in batch
    pub async fn store_batch(&self, records: Vec<Neo4jEmbeddingRecord>) -> Result<Vec<String>> {
        let mut stored_ids = Vec::with_capacity(records.len());

        // Use UNWIND for batch operation
        let batch_query = r#"
            UNWIND $records as record
            MERGE (e:Embedding {node_id: record.node_id})
            SET e.node_label = record.node_label,
                e.text = record.text,
                e.embedding = record.embedding,
                e.fused_embedding = record.fused_embedding,
                e.models_used = record.models_used,
                e.quality_score = record.quality_score,
                e.metadata = record.metadata,
                e.updated_at = datetime()
            ON CREATE SET e.created_at = datetime()
            RETURN e.node_id as node_id
        "#;

        // Convert records to Neo4j-compatible format
        let records_param: Vec<HashMap<String, serde_json::Value>> = records
            .iter()
            .map(|r| {
                let mut map = HashMap::new();
                map.insert("node_id".to_string(), serde_json::json!(r.node_id));
                map.insert("node_label".to_string(), serde_json::json!(r.node_label));
                map.insert("text".to_string(), serde_json::json!(r.text));
                map.insert("embedding".to_string(), serde_json::json!(r.embedding));
                map.insert("fused_embedding".to_string(), serde_json::json!(r.fused_embedding));
                map.insert("models_used".to_string(), serde_json::json!(r.models_used));
                map.insert("quality_score".to_string(), serde_json::json!(r.quality_score));
                map.insert("metadata".to_string(), serde_json::json!(r.metadata));
                map
            })
            .collect();

        let mut result = self
            .graph
            .execute(
                Query::new(batch_query.to_string())
                    .param("records", serde_json::json!(records_param)),
            )
            .await?;

        while let Some(row) = result.next().await? {
            if let Ok(node_id) = row.get::<String>("node_id") {
                stored_ids.push(node_id);
            }
        }

        tracing::info!("Batch stored {} embeddings in Neo4j", stored_ids.len());
        Ok(stored_ids)
    }

    /// Get an embedding by node ID
    pub async fn get_embedding(&self, node_id: &str) -> Result<Option<Neo4jEmbeddingRecord>> {
        let query = r#"
            MATCH (e:Embedding {node_id: $node_id})
            RETURN e.node_id as node_id,
                   e.node_label as node_label,
                   e.text as text,
                   e.embedding as embedding,
                   e.fused_embedding as fused_embedding,
                   e.models_used as models_used,
                   e.quality_score as quality_score,
                   e.metadata as metadata
        "#;

        let mut result = self
            .graph
            .execute(Query::new(query.to_string()).param("node_id", node_id))
            .await?;

        if let Some(row) = result.next().await? {
            let metadata_str: String = row.get("metadata").unwrap_or_default();
            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(&metadata_str).unwrap_or_default();

            let record = Neo4jEmbeddingRecord {
                node_id: row.get("node_id")?,
                node_label: row.get("node_label").unwrap_or_default(),
                text: row.get("text").unwrap_or_default(),
                embedding: row.get("embedding").unwrap_or_default(),
                fused_embedding: row.get("fused_embedding").unwrap_or_default(),
                models_used: row.get("models_used").unwrap_or_default(),
                quality_score: row.get::<f64>("quality_score").unwrap_or_default() as f32,
                metadata,
            };
            return Ok(Some(record));
        }

        Ok(None)
    }

    /// Search for similar embeddings using cosine similarity
    /// Note: This is a basic implementation. For production, consider Neo4j's
    /// vector index or a dedicated vector database integration.
    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_similarity: f32,
    ) -> Result<Vec<(Neo4jEmbeddingRecord, f32)>> {
        // For now, fetch all and compute similarity in Rust
        // This should be replaced with Neo4j's native vector similarity when available
        let query = r#"
            MATCH (e:Embedding)
            WHERE e.fused_embedding IS NOT NULL
            RETURN e.node_id as node_id,
                   e.node_label as node_label,
                   e.text as text,
                   e.embedding as embedding,
                   e.fused_embedding as fused_embedding,
                   e.models_used as models_used,
                   e.quality_score as quality_score,
                   e.metadata as metadata
            LIMIT 1000
        "#;

        let mut result = self.graph.execute(Query::new(query.to_string())).await?;
        let mut candidates: Vec<(Neo4jEmbeddingRecord, f32)> = Vec::new();

        while let Some(row) = result.next().await? {
            let fused_embedding: Vec<f64> = row.get("fused_embedding").unwrap_or_default();
            let fused_f32: Vec<f32> = fused_embedding.iter().map(|&v| v as f32).collect();

            let similarity = cosine_similarity(query_embedding, &fused_f32);

            if similarity >= min_similarity {
                let metadata_str: String = row.get("metadata").unwrap_or_default();
                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&metadata_str).unwrap_or_default();

                let record = Neo4jEmbeddingRecord {
                    node_id: row.get("node_id")?,
                    node_label: row.get("node_label").unwrap_or_default(),
                    text: row.get("text").unwrap_or_default(),
                    embedding: row.get("embedding").unwrap_or_default(),
                    fused_embedding: fused_f32,
                    models_used: row.get("models_used").unwrap_or_default(),
                    quality_score: row.get::<f64>("quality_score").unwrap_or_default() as f32,
                    metadata,
                };
                candidates.push((record, similarity));
            }
        }

        // Sort by similarity descending and take top N
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);

        Ok(candidates)
    }

    /// Delete an embedding by node ID
    pub async fn delete_embedding(&self, node_id: &str) -> Result<bool> {
        let query = r#"
            MATCH (e:Embedding {node_id: $node_id})
            DETACH DELETE e
            RETURN count(e) as deleted
        "#;

        let mut result = self
            .graph
            .execute(Query::new(query.to_string()).param("node_id", node_id))
            .await?;

        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted").unwrap_or(0);
            return Ok(deleted > 0);
        }

        Ok(false)
    }

    /// Update an existing embedding
    pub async fn update_embedding(
        &self,
        node_id: &str,
        embedding: Vec<f32>,
        fused_embedding: Vec<f32>,
        models_used: Vec<String>,
        quality_score: f32,
    ) -> Result<bool> {
        let query = r#"
            MATCH (e:Embedding {node_id: $node_id})
            SET e.embedding = $embedding,
                e.fused_embedding = $fused_embedding,
                e.models_used = $models_used,
                e.quality_score = $quality_score,
                e.updated_at = datetime()
            RETURN e.node_id as node_id
        "#;

        let mut result = self
            .graph
            .execute(
                Query::new(query.to_string())
                    .param("node_id", node_id)
                    .param("embedding", embedding)
                    .param("fused_embedding", fused_embedding)
                    .param("models_used", models_used)
                    .param("quality_score", quality_score as f64),
            )
            .await?;

        Ok(result.next().await?.is_some())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        let query = r#"
            MATCH (e:Embedding)
            RETURN count(e) as total_embeddings,
                   collect(DISTINCT e.node_label)[0..10] as labels
        "#;

        let mut result = self.graph.execute(Query::new(query.to_string())).await?;
        let mut stats = HashMap::new();

        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total_embeddings").unwrap_or(0);
            let labels: Vec<String> = row.get("labels").unwrap_or_default();

            stats.insert("total_embeddings".to_string(), serde_json::json!(total));
            stats.insert("labels".to_string(), serde_json::json!(labels));
        }

        Ok(stats)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&c, &d).abs() < 0.001);
    }
}
