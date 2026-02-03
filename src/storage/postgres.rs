//! PostgreSQL storage for vectors using pgvector

use crate::core::Result;
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: Uuid,
    pub text: String,
    pub embedding: Vec<f32>,
    pub model: String,
    pub metadata: serde_json::Value,
}

pub struct PostgresStorage {
    pool: PgPool,
}

impl PostgresStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;

        // Ensure pgvector extension is enabled
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&pool)
            .await?;

        // Create embeddings table if it doesn't exist
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                text TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                model VARCHAR(255) NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            "#
        )
        .execute(&pool)
        .await?;

        // Create index for similarity search
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings USING ivfflat (embedding vector_cosine_ops)"
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    pub async fn store_embedding(&self, record: VectorRecord) -> Result<Uuid> {
        let id = record.id;
        
        sqlx::query(
            r#"
            INSERT INTO embeddings (id, text, embedding, model, metadata)
            VALUES ($1, $2, $3, $4, $5)
            "#
        )
        .bind(id)
        .bind(&record.text)
        .bind(&record.embedding)
        .bind(&record.model)
        .bind(&record.metadata)
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    pub async fn store_batch(&self, records: Vec<VectorRecord>) -> Result<Vec<Uuid>> {
        let mut ids = Vec::new();
        
        for record in records {
            let id = self.store_embedding(record).await?;
            ids.push(id);
        }
        
        Ok(ids)
    }

    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        model: &str,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<VectorRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, text, embedding, model, metadata
            FROM embeddings
            WHERE model = $1
            AND 1 - (embedding <=> $2) > $3
            ORDER BY embedding <=> $2
            LIMIT $4
            "#
        )
        .bind(model)
        .bind(query_embedding)
        .bind(threshold)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(VectorRecord {
                id: row.get("id"),
                text: row.get("text"),
                embedding: row.get("embedding"),
                model: row.get("model"),
                metadata: row.get("metadata"),
            });
        }

        Ok(results)
    }

    pub async fn get_by_id(&self, id: Uuid) -> Result<Option<VectorRecord>> {
        let row = sqlx::query(
            "SELECT id, text, embedding, model, metadata FROM embeddings WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(VectorRecord {
                id: row.get("id"),
                text: row.get("text"),
                embedding: row.get("embedding"),
                model: row.get("model"),
                metadata: row.get("metadata"),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn delete_by_id(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query("DELETE FROM embeddings WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }
}
