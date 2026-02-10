//! Embeddings Service Library
//! 
//! A high-performance embedding generation service
//! for the ConFuse platform. Generate-only — no storage.

pub mod api;
pub mod core;
pub mod generators;
pub mod storage;
pub mod models;

use std::sync::Arc;
use crate::core::{Config, Result};
use crate::models::ModelManager;

// Application state for Axum
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
}

pub struct EmbeddingsService {
    pub config: Config,
    pub model_manager: Arc<ModelManager>,
}

impl EmbeddingsService {
    pub async fn new(config: Config) -> Result<Self> {
        let model_manager = Arc::new(ModelManager::new(config.clone()));
        
        Ok(Self {
            config,
            model_manager,
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
        }
    }
}
