//! Redis cache for embeddings and metadata

use crate::core::Result;
use redis::{AsyncCommands, Client};
use serde_json;
use std::time::Duration;

pub struct RedisCache {
    client: Client,
}

impl RedisCache {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)?;
        
        // Test connection
        let mut conn = client.get_async_connection().await?;
        redis::cmd("PING").query_async::<_, String>(&mut conn).await?;
        
        Ok(Self { client })
    }

    pub async fn cache_embedding(
        &self,
        key: &str,
        embedding: &[f32],
        ttl: Duration,
    ) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        
        let value = serde_json::to_string(embedding)?;
        
        conn.set_ex::<_, _, ()>(key, value, ttl.as_secs()).await?;
        
        Ok(())
    }

    pub async fn get_cached_embedding(&self, key: &str) -> Result<Option<Vec<f32>>> {
        let mut conn = self.client.get_async_connection().await?;
        
        let result: Option<String> = conn.get(key).await?;
        
        if let Some(value) = result {
            let embedding: Vec<f32> = serde_json::from_str(&value)?;
            Ok(Some(embedding))
        } else {
            Ok(None)
        }
    }

    pub async fn cache_text_hash(
        &self,
        text: &str,
        embedding_key: &str,
        ttl: Duration,
    ) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        
        let hash_key = format!("text_hash:{}", RedisCache::compute_text_hash(text));
        
        conn.set_ex::<_, _, ()>(&hash_key, embedding_key, ttl.as_secs()).await?;
        
        Ok(())
    }

    pub async fn get_embedding_by_text(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let hash_key = format!("text_hash:{}", RedisCache::compute_text_hash(text));
        
        let mut conn = self.client.get_async_connection().await?;
        
        let embedding_key: Option<String> = conn.get(&hash_key).await?;
        
        if let Some(key) = embedding_key {
            self.get_cached_embedding(&key).await
        } else {
            Ok(None)
        }
    }

    pub async fn invalidate_cache(&self, pattern: &str) -> Result<usize> {
        let mut conn = self.client.get_async_connection().await?;
        
        let keys: Vec<String> = conn.keys(pattern).await?;
        
        if !keys.is_empty() {
            let _: () = conn.del(&keys).await?;
        }
        
        Ok(keys.len())
    }

    pub async fn get_cache_stats(&self) -> Result<serde_json::Value> {
        let mut conn = self.client.get_async_connection().await?;
        let info: String = redis::cmd("INFO").query_async(&mut conn).await?;
        
        // Parse Redis INFO output into key-value pairs
        let mut stats = serde_json::Map::new();
        for line in info.lines() {
            if let Some((key, value)) = line.split_once(':') {
                stats.insert(key.to_string(), serde_json::Value::String(value.to_string()));
            }
        }
        
        Ok(serde_json::Value::Object(stats))
    }

    fn compute_text_hash(text: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
