//! Embeddings Service Library
//! 
//! A high-performance embedding generation service
//! for the ConFuse platform with FalcorDB vector storage.

pub mod api;
pub mod core;
pub mod generators;
pub mod storage;
pub mod models;

use std::sync::Arc;
use crate::core::{Config, Result};
use crate::models::ModelManager;
use crate::storage::falcordb_client::FalcorDBClient;

// Application state for Axum
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
    pub falcordb_client: Option<Arc<FalcorDBClient>>,
}

pub struct EmbeddingsService {
    pub config: Config,
    pub model_manager: Arc<ModelManager>,
    pub falcordb_client: Option<Arc<FalcorDBClient>>,
}

impl EmbeddingsService {
    pub async fn new(config: Config) -> Result<Self> {
        let model_manager = Arc::new(ModelManager::new(config.clone()));
        
        // Initialize FalcorDB client if configuration is available
        let falcordb_client = Self::initialize_falcordb_client().await;
        
        Ok(Self {
            config,
            model_manager,
            falcordb_client,
        })
    }

    async fn initialize_falcordb_client() -> Option<Arc<FalcorDBClient>> {
        use crate::storage::falcordb_client::FalcorDBConfig;
        
        match FalcorDBConfig::from_env() {
            Ok(config) => {
                match FalcorDBClient::new(config).await {
                    Ok(client) => {
                        tracing::info!("FalcorDB client initialized successfully");
                        Some(Arc::new(client))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize FalcorDB client: {}. Vector storage will be disabled.", e);
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!("FalcorDB configuration not found: {}. Vector storage will be disabled.", e);
                None
            }
        }
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
            falcordb_client: self.falcordb_client,
        }
    }
}
