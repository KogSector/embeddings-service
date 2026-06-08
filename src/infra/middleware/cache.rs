//! Response caching middleware
//!
//! In-memory LRU-style cache for API responses with TTL.
//! Path-aware TTL: auth endpoints get longer TTL, search gets shorter.

use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    status: u16,
    content_type: String,
    expires_at: u64,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Default TTL for cached responses
    pub default_ttl: Duration,
    /// TTL for auth verification responses
    pub auth_ttl: Duration,
    /// TTL for search responses
    pub search_ttl: Duration,
    /// Maximum entries before eviction
    pub max_entries: usize,
    /// Whether caching is enabled
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(60),
            auth_ttl: Duration::from_secs(300),
            search_ttl: Duration::from_secs(30),
            max_entries: 10_000,
            enabled: true,
        }
    }
}

/// In-memory response cache
#[derive(Clone)]
pub struct ResponseCache {
    entries: Arc<DashMap<String, CacheEntry>>,
    config: CacheConfig,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl ResponseCache {
    pub fn new(config: CacheConfig) -> Self {
        let cache = Self {
            entries: Arc::new(DashMap::new()),
            config,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        };

        // Periodic cleanup every 60s
        let entries = cache.entries.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let now = Self::now_epoch();
                entries.retain(|_, entry| entry.expires_at > now);
            }
        });

        cache
    }

    fn now_epoch() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Build a cache key from method + path + query + user_id
    pub fn build_key(method: &str, path: &str, query: Option<&str>, user_id: Option<&str>) -> String {
        let q = query.unwrap_or("");
        let u = user_id.unwrap_or("anon");
        format!("cache:{}:{}:{}:{}", method, path, q, u)
    }

    /// Get TTL based on path
    pub fn ttl_for_path(&self, path: &str) -> Duration {
        if path.contains("/verify") || path.contains("/auth") {
            self.config.auth_ttl
        } else if path.contains("/search") {
            self.config.search_ttl
        } else {
            self.config.default_ttl
        }
    }

    /// Try to get a cached response
    pub fn get(&self, key: &str) -> Option<(Vec<u8>, u16, String)> {
        if !self.config.enabled {
            return None;
        }

        let entry = self.entries.get(key)?;
        if entry.expires_at < Self::now_epoch() {
            drop(entry);
            self.entries.remove(key);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some((entry.data.clone(), entry.status, entry.content_type.clone()))
    }

    /// Store a response in cache
    pub fn set(&self, key: &str, data: Vec<u8>, status: u16, content_type: &str, ttl: Duration) {
        if !self.config.enabled {
            return;
        }

        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries {
            let to_remove = self.config.max_entries / 10;
            let mut keys_to_remove = Vec::with_capacity(to_remove);
            let now = Self::now_epoch();
            for entry in self.entries.iter() {
                if entry.expires_at < now || keys_to_remove.len() < to_remove {
                    keys_to_remove.push(entry.key().clone());
                }
                if keys_to_remove.len() >= to_remove {
                    break;
                }
            }
            for k in keys_to_remove {
                self.entries.remove(&k);
            }
        }

        self.entries.insert(
            key.to_string(),
            CacheEntry {
                data,
                status,
                content_type: content_type.to_string(),
                expires_at: Self::now_epoch() + ttl.as_secs(),
            },
        );
    }

    /// Invalidate cache entries matching a prefix
    pub fn invalidate_prefix(&self, prefix: &str) {
        self.entries.retain(|k, _| !k.starts_with(prefix));
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.entries.len(),
        )
    }
}
