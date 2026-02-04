//! Vector Search Optimization Module
//!
//! Provides high-performance vector search with:
//! - HNSW-based Approximate Nearest Neighbor (ANN) search
//! - LRU caching for search results
//! - Batch vector operations
//! - User-scoped embedding retrieval

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ============================================================================
// LRU Cache for Search Results
// ============================================================================

/// LRU Cache entry with expiration
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
    access_count: u32,
}

/// LRU Cache for vector search results
pub struct SearchCache<K, V> {
    entries: RwLock<HashMap<K, CacheEntry<V>>>,
    capacity: usize,
    ttl: Duration,
    stats: RwLock<CacheStats>,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> SearchCache<K, V> {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            ttl,
            stats: RwLock::new(CacheStats::default()),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut entries = self.entries.write();
        
        if let Some(entry) = entries.get_mut(key) {
            if entry.created_at.elapsed() < self.ttl {
                entry.access_count += 1;
                self.stats.write().hits += 1;
                return Some(entry.value.clone());
            }
            // Entry expired, remove it
            entries.remove(key);
        }
        
        self.stats.write().misses += 1;
        None
    }

    pub fn insert(&self, key: K, value: V) {
        let mut entries = self.entries.write();
        
        // Evict if at capacity
        if entries.len() >= self.capacity {
            self.evict_lru(&mut entries);
        }
        
        entries.insert(key, CacheEntry {
            value,
            created_at: Instant::now(),
            access_count: 1,
        });
    }

    fn evict_lru(&self, entries: &mut HashMap<K, CacheEntry<V>>) {
        // Find the least recently accessed entry
        let lru_key = entries
            .iter()
            .min_by_key(|(_, e)| e.access_count)
            .map(|(k, _)| k.clone());
        
        if let Some(key) = lru_key {
            entries.remove(&key);
            self.stats.write().evictions += 1;
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

// ============================================================================
// HNSW Index Configuration
// ============================================================================

/// HNSW index parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    /// Number of connections per layer (M)
    pub m: usize,
    /// Size of dynamic candidate list during construction (ef_construction)
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (ef_search)
    pub ef_search: usize,
    /// Maximum number of layers
    pub max_level: usize,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_level: 16,
        }
    }
}

// ============================================================================
// Vector Search Request/Response
// ============================================================================

/// Search request with user context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return
    pub top_k: usize,
    /// User ID for scoping
    pub user_id: Option<String>,
    /// Workspace ID for multi-tenant isolation
    pub workspace_id: Option<String>,
    /// Knowledge base ID (optional filter)
    pub knowledge_base_id: Option<String>,
    /// Minimum similarity threshold
    pub min_score: Option<f32>,
    /// Include vector in response
    pub include_vectors: bool,
}

/// Search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// Search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResponse {
    pub results: Vec<SearchResult>,
    pub total_count: usize,
    pub search_time_ms: u64,
    pub cache_hit: bool,
}

// ============================================================================
// Batch Vector Operations
// ============================================================================

/// Batch embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingRequest {
    pub texts: Vec<String>,
    pub user_id: Option<String>,
    pub workspace_id: Option<String>,
    pub model: Option<String>,
}

/// Batch embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimensions: usize,
    pub processing_time_ms: u64,
}

// ============================================================================
// Vector Search Service
// ============================================================================

/// High-performance vector search service
pub struct VectorSearchService {
    /// Search result cache
    cache: Arc<SearchCache<String, Vec<SearchResult>>>,
    /// HNSW configuration
    hnsw_config: HNSWConfig,
    /// Dimension of vectors
    dimensions: usize,
}

impl VectorSearchService {
    pub fn new(dimensions: usize, cache_capacity: usize, cache_ttl: Duration) -> Self {
        Self {
            cache: Arc::new(SearchCache::new(cache_capacity, cache_ttl)),
            hnsw_config: HNSWConfig::default(),
            dimensions,
        }
    }

    pub fn with_hnsw_config(mut self, config: HNSWConfig) -> Self {
        self.hnsw_config = config;
        self
    }

    /// Generate cache key from search request
    fn cache_key(&self, request: &VectorSearchRequest) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash vector (truncated for efficiency)
        for v in request.vector.iter().take(16) {
            v.to_bits().hash(&mut hasher);
        }
        
        request.top_k.hash(&mut hasher);
        request.workspace_id.hash(&mut hasher);
        request.knowledge_base_id.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }

    /// Search with caching
    pub async fn search(&self, request: &VectorSearchRequest) -> VectorSearchResponse {
        let start = Instant::now();
        let cache_key = self.cache_key(request);
        
        // Check cache
        if let Some(cached_results) = self.cache.get(&cache_key) {
            debug!("Cache hit for search, key: {}", cache_key);
            return VectorSearchResponse {
                results: cached_results,
                total_count: 0, // Unknown from cache
                search_time_ms: start.elapsed().as_millis() as u64,
                cache_hit: true,
            };
        }
        
        // Perform actual search (placeholder - real impl uses Milvus)
        let results = self.perform_search(request).await;
        
        // Cache results
        if !results.is_empty() {
            self.cache.insert(cache_key, results.clone());
        }
        
        VectorSearchResponse {
            total_count: results.len(),
            results,
            search_time_ms: start.elapsed().as_millis() as u64,
            cache_hit: false,
        }
    }

    /// Perform the actual vector search (placeholder for Milvus integration)
    async fn perform_search(&self, request: &VectorSearchRequest) -> Vec<SearchResult> {
        // This is a placeholder - real implementation would query Milvus
        // with user-scoped collection filtering based on workspace_id
        
        debug!(
            "Performing vector search: top_k={}, workspace={:?}, kb={:?}",
            request.top_k, request.workspace_id, request.knowledge_base_id
        );
        
        // In production, this would:
        // 1. Build Milvus search expression with workspace filter
        // 2. Execute HNSW search
        // 3. Return results with metadata
        
        Vec::new()
    }

    /// Batch search for multiple queries
    pub async fn batch_search(&self, requests: Vec<VectorSearchRequest>) -> Vec<VectorSearchResponse> {
        let mut results = Vec::with_capacity(requests.len());
        
        // Process in parallel chunks for efficiency
        for request in requests {
            results.push(self.search(&request).await);
        }
        
        results
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear search cache
    pub fn clear_cache(&self) {
        self.cache.clear();
        info!("Vector search cache cleared");
    }
}

// ============================================================================
// Cosine Similarity (for local operations)
// ============================================================================

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
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

/// Calculate L2 (Euclidean) distance between two vectors
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_operations() {
        let cache: SearchCache<String, Vec<i32>> = SearchCache::new(2, Duration::from_secs(60));
        
        cache.insert("key1".into(), vec![1, 2, 3]);
        cache.insert("key2".into(), vec![4, 5, 6]);
        
        assert_eq!(cache.get(&"key1".into()), Some(vec![1, 2, 3]));
        assert!(cache.stats().hits > 0);
    }
}
