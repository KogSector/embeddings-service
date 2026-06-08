use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, HashSet};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use std::hash::{Hash, Hasher};

pub mod auth;
pub mod mcp;

// Original embeddings-service modules
pub mod fastembed;
pub mod manager;
pub mod models;

pub use manager::ModelManager;
pub use models::EmbeddingModel;

// Advanced data structures for performance optimization

/// High-performance cache-friendly vector for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedVector {
    pub data: Arc<Vec<f32>>,
    pub dimension: usize,
    pub norm: Option<f32>, // Cached L2 norm for faster similarity calculations
    pub hash: u64, // Cached hash for deduplication
}

impl OptimizedVector {
    pub fn new(data: Vec<f32>) -> Self {
        let dimension = data.len();
        let norm = Self::calculate_norm(&data);
        let hash = Self::calculate_hash(&data);
        
        Self {
            data: Arc::new(data),
            dimension,
            norm: Some(norm),
            hash,
        }
    }
    
    fn calculate_norm(data: &[f32]) -> f32 {
        data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    
    fn calculate_hash(data: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for &val in data {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
    
    /// Fast cosine similarity using cached norms
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }
        
        let dot_product: f32 = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
            
        match (self.norm, other.norm) {
            (Some(norm_a), Some(norm_b)) => {
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot_product / (norm_a * norm_b)
                }
            }
            _ => 0.0,
        }
    }
}

/// Spatial index for fast nearest neighbor search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialIndex {
    pub vectors: Vec<OptimizedVector>,
    pub metadata: Vec<VectorMetadata>,
    pub dimension: usize,
    pub index_type: IndexType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    LSH, // Locality Sensitive Hashing
    HNSW, // Hierarchical Navigable Small World
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub id: String,
    pub source_id: String,
    pub chunk_index: Option<usize>,
    pub timestamp: DateTime<Utc>,
    pub tags: HashSet<String>,
    pub quality_score: Option<f32>,
}

/// Bloom filter for fast membership testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    pub bits: Vec<bool>,
    pub hash_functions: usize,
    pub size: usize,
    pub items_count: usize,
}

impl BloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let size = Self::optimal_size(expected_items, false_positive_rate);
        let hash_functions = Self::optimal_hash_functions(size, expected_items);
        
        Self {
            bits: vec![false; size],
            hash_functions,
            size,
            items_count: 0,
        }
    }
    
    fn optimal_size(n: usize, p: f64) -> usize {
        (-(n as f64) * p.ln() / (2.0_f64.ln().powi(2))).ceil() as usize
    }
    
    fn optimal_hash_functions(m: usize, n: usize) -> usize {
        ((m as f64 / n as f64) * 2.0_f64.ln()).round() as usize
    }
    
    pub fn insert(&mut self, item: &str) {
        for i in 0..self.hash_functions {
            let hash = self.hash(item, i);
            self.bits[hash % self.size] = true;
        }
        self.items_count += 1;
    }
    
    pub fn contains(&self, item: &str) -> bool {
        for i in 0..self.hash_functions {
            let hash = self.hash(item, i);
            if !self.bits[hash % self.size] {
                return false;
            }
        }
        true
    }
    
    fn hash(&self, item: &str, seed: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish() as usize
    }
}

/// LRU Cache for frequently accessed data
#[derive(Debug)]
pub struct LRUCache<K, V> 
where 
    K: Hash + Eq + Clone,
    V: Clone,
{
    capacity: usize,
    map: HashMap<K, (V, usize)>,
    access_order: BTreeMap<usize, K>,
    counter: usize,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            access_order: BTreeMap::new(),
            counter: 0,
        }
    }
    
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, old_counter)) = self.map.get(key) {
            let value = value.clone();
            let old_counter = *old_counter;
            
            // Update access order
            self.access_order.remove(&old_counter);
            self.counter += 1;
            self.access_order.insert(self.counter, key.clone());
            self.map.insert(key.clone(), (value.clone(), self.counter));
            
            Some(value)
        } else {
            None
        }
    }
    
    pub fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            // Update existing
            if let Some((_, old_counter)) = self.map.get(&key) {
                self.access_order.remove(old_counter);
            }
        } else if self.map.len() >= self.capacity {
            // Evict least recently used
            if let Some((&oldest_counter, oldest_key)) = self.access_order.iter().next() {
                let oldest_key = oldest_key.clone();
                self.access_order.remove(&oldest_counter);
                self.map.remove(&oldest_key);
            }
        }
        
        self.counter += 1;
        self.access_order.insert(self.counter, key.clone());
        self.map.insert(key, (value, self.counter));
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum VcsType {
    Git,
    Subversion,
    Mercurial,
    Bazaar,
    Perforce,
    Unknown,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum VcsProvider {
    GitHub,
    GitLab,
    Bitbucket,
    Azure,
    Gitea,
    SourceForge,
    CodeCommit,
    SelfHosted,
    Local,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum CredentialType {
    PersonalAccessToken { token: String },
    UsernamePassword { username: String, password: String },
    SshKey { private_key: String, passphrase: Option<String> },
    AppPassword { username: String, app_password: String },
    None,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepositoryCredentials {
    pub credential_type: CredentialType,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepositoryConfig {
    pub branch: String,
    pub auto_sync: bool,
    pub webhook_enabled: bool,
    pub webhook_secret: Option<String>,
    pub include_branches: Vec<String>,
    pub exclude_paths: Vec<String>,
    pub include_file_extensions: Vec<String>,
    pub max_file_size_mb: u32,
    pub sync_frequency_minutes: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepositoryInfo {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub url: String,
    pub vcs_type: VcsType,
    pub provider: VcsProvider,
    pub owner: String,
    pub is_private: bool,
    pub default_branch: String,
    pub clone_url: String,
    pub ssh_url: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_synced: Option<DateTime<Utc>>,
    pub sync_status: RepositorySyncStatus,
    pub credentials: RepositoryCredentials,
    pub config: RepositoryConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum RepositorySyncStatus {
    Connected,
    Syncing,
    SyncCompleted,
    SyncFailed,
    Disconnected,
    PendingAuth,
}

#[derive(Deserialize, Debug)]
pub struct ConnectRepositoryRequest {
    pub url: String,
    pub vcs_type: Option<VcsType>,
    pub provider: Option<VcsProvider>,
    pub credentials: RepositoryCredentials,
    pub config: Option<RepositoryConfig>,
}

#[derive(Deserialize)]
pub struct ConnectRepoRequest {
    pub repo_url: String,
    pub access_token: Option<String>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct ServiceStatus {
    pub name: String,
    pub url: String,
    pub status: String,
    pub response_time_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub message: String,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            message: "Success".to_string(),
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            message: message.clone(),
            data: None,
            error: Some(message),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct UserSettings {
    pub user_id: String,
    pub profile: ProfileSettings,
    pub notifications: NotificationSettings,
    pub security: SecuritySettings,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProfileSettings {
    pub first_name: String,
    pub last_name: String,
    pub email: String,
    pub bio: Option<String>,
    pub location: Option<String>,
    pub website: Option<String>,
    pub social_links: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NotificationSettings {
    pub email_notifications: bool,
    pub push_notifications: bool,
    pub security_alerts: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SecuritySettings {
    pub two_factor_enabled: bool,
    pub session_timeout: u32,
}

#[derive(Deserialize)]
pub struct UpdateSettingsRequest {
    pub profile: Option<ProfileSettings>,
    pub notifications: Option<NotificationSettings>,
    pub security: Option<SecuritySettings>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AgentRecord {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub agent_type: String, 
    pub endpoint: Option<String>, 
    pub api_key: String, 
    pub permissions: Vec<String>, 
    pub status: AgentStatus,
    pub config: AgentConfig,
    pub created_at: String,
    pub updated_at: String,
    pub last_used: Option<String>,
    pub usage_stats: AgentUsageStats,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum AgentStatus {
    Connected,
    Pending,
    Error,
    Inactive,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AgentConfig {
    pub model: Option<String>, 
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout: Option<u32>, 
    pub custom_instructions: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AgentUsageStats {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub avg_response_time: Option<f32>,
    pub last_error: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct CreateAgentRequest {
    pub name: String,
    pub agent_type: String,
    pub endpoint: Option<String>,
    pub api_key: String,
    pub permissions: Vec<String>,
    pub config: AgentConfig,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct UpdateAgentRequest {
    pub name: Option<String>,
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
    pub permissions: Option<Vec<String>>,
    pub config: Option<AgentConfig>,
    pub status: Option<AgentStatus>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct AgentInvokeRequest {
    pub message: String,
    pub context_type: Option<String>, 
    pub include_history: Option<bool>,
}

#[derive(Serialize)]
pub struct AgentInvokeResponse {
    pub response: String,
    pub usage: AgentInvokeUsage,
    pub context_used: Vec<String>,
}

#[derive(Serialize)]
pub struct AgentInvokeUsage {
    pub tokens_used: u32,
    pub response_time_ms: u64,
}

#[derive(Serialize, Debug, Clone)]
pub struct AgentContext {
    pub repositories: Vec<RepositoryContext>,
    pub documents: Vec<DocumentContext>,
    pub urls: Vec<UrlContext>,
}

#[derive(Serialize, Debug, Clone)]
pub struct RepositoryContext {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub language: String,
    pub recent_files: Vec<String>,
    pub recent_commits: Vec<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct DocumentContext {
    pub id: String,
    pub name: String,
    pub doc_type: String,
    pub summary: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct UrlContext {
    pub id: String,
    pub url: String,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum DataSourceType {
    Repository,
    Document,
    Url,
    Dropbox,
    GoogleDrive,
    OneDrive,
    LocalFile,
    Notion,
}
