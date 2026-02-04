//! Embeddings Service Library
//! 
//! A high-performance embedding generation and vector search service
//! for the ConFuse platform.

pub mod api;
pub mod core;
pub mod generators;
pub mod models;
pub mod storage;

use std::sync::Arc;
use crate::core::{Config, Result};
use crate::models::ModelManager;
use crate::storage::{PostgresStorage, RedisCache};

// Application state for Axum
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
    pub postgres_storage: Arc<PostgresStorage>,
}

pub struct EmbeddingsService {
    pub config: Config,
    pub model_manager: Arc<ModelManager>,
    pub postgres_storage: Arc<PostgresStorage>,
    pub redis_cache: Arc<RedisCache>,
}

impl EmbeddingsService {
    pub async fn new(config: Config) -> Result<Self> {
        let model_manager = Arc::new(ModelManager::new(config.clone()));
        
        let postgres_storage = Arc::new(
            PostgresStorage::new(&config.database.postgres_url).await?
        );
        
        let redis_cache = Arc::new(
            RedisCache::new(&config.database.redis_url).await?
        );
        
        Ok(Self {
            config,
            model_manager,
            postgres_storage,
            redis_cache,
        })
    }

    pub async fn initialize_default_model(&self) -> Result<()> {
        self.model_manager.load_model(&self.config.models.default_model).await
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    
    pub fn app_state(self) -> AppState {
        AppState {
            model_manager: self.model_manager,
            postgres_storage: self.postgres_storage,
        }
    }
}
