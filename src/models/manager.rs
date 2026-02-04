//! Model manager for loading and managing embedding models

use crate::core::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    fn name(&self) -> &str;
    fn dimension(&self) -> usize;
    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
    async fn generate_batch(&self, texts: Vec<String>, batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        for chunk in texts.chunks(batch_size) {
            let chunk_embeddings = self.generate(chunk.to_vec()).await?;
            embeddings.extend(chunk_embeddings);
        }
        
        Ok(embeddings)
    }
}

#[derive(Clone)]
pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingModel>>>>,
    config: Config,
}

impl ModelManager {
    pub fn new(config: Config) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        let mut models = self.models.write().await;
        
        if models.contains_key(model_name) {
            return Ok(());
        }

        let model: Arc<dyn EmbeddingModel> = Arc::new(crate::models::SentenceTransformersModel::new(model_name, &self.config)?);

        models.insert(model_name.to_string(), model);
        tracing::info!("Loaded model: {}", model_name);
        
        Ok(())
    }

    pub async fn get_model(&self, model_name: &str) -> Result<Arc<dyn EmbeddingModel>> {
        let models = self.models.read().await;
        
        models
            .get(model_name)
            .cloned()
            .ok_or_else(|| EmbeddingError::ModelNotFound(model_name.to_string()))
    }

    pub async fn ensure_model_loaded(&self, model_name: &str) -> Result<Arc<dyn EmbeddingModel>> {
        // Try to get the model first
        if let Ok(model) = self.get_model(model_name).await {
            return Ok(model);
        }

        // If not loaded, load it
        self.load_model(model_name).await?;
        self.get_model(model_name).await
    }

    pub async fn list_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    pub async fn unload_model(&self, model_name: &str) -> Result<()> {
        let mut models = self.models.write().await;
        models.remove(model_name);
        tracing::info!("Unloaded model: {}", model_name);
        Ok(())
    }
}
