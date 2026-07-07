//! Embedding model implementations
//! Supports local models via SentenceTransformers (Python) and Ollama

use crate::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Embedding model trait
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

/// NVIDIA NIM based embedding model
pub struct NvidiaNimModel {
    name: String,
    api_key: String,
    base_url: String,
    dimension: usize,
    client: Client,
}

impl NvidiaNimModel {
    pub fn new(model_name: &str, config: &Config) -> Result<Self> {
        // We'll map gemini config fields conceptually to NIM or assume env variables
        let api_key = config.models.gemini_api_key.clone();
        if api_key.is_empty() {
            tracing::warn!("API_KEY is not set. Embedding generation will fail.");
        }
        
        let base_url = {
            let raw = config.models.gemini_base_url.clone();
            if raw.is_empty() || raw.contains("google") {
                "https://integrate.api.nvidia.com/v1/embeddings".to_string()
            } else if raw.ends_with("/v1/embeddings") {
                raw
            } else {
                // Ensure the path suffix is present
                let trimmed = raw.trim_end_matches('/');
                format!("{}/v1/embeddings", trimmed)
            }
        };
        
        let dimension = Self::get_model_dimension(model_name);
        let client = Client::builder()
            .timeout(config.models.timeout)
            .build()
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            name: model_name.to_string(),
            api_key,
            base_url,
            dimension,
            client,
        })
    }

    fn get_model_dimension(_model_name: &str) -> usize {
        1024 // nv-embed-v1 uses 1024 dimensions
    }

    fn get_max_input_chars(_model_name: &str) -> usize {
        7500 // NV models handle 4k tokens, ~7.5k chars is safely within bounds
    }
}

#[async_trait]
impl EmbeddingModel for NvidiaNimModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let max_chars = Self::get_max_input_chars(&self.name);
        let mut inputs = Vec::new();
        
        for text in &texts {
            if text.len() > max_chars {
                tracing::debug!("Truncating input from {} to {} chars", text.len(), max_chars);
                let mut end = max_chars;
                while !text.is_char_boundary(end) && end > 0 {
                    end -= 1;
                }
                inputs.push(text[..end].to_string());
            } else {
                inputs.push(text.clone());
            }
        }

        #[derive(Debug, Serialize)]
        struct NimRequest {
            model: String,
            input: Vec<String>,
            input_type: String,
        }

        #[derive(Debug, Deserialize)]
        struct NimEmbedding {
            embedding: Vec<f32>,
            index: usize,
        }

        #[derive(Debug, Deserialize)]
        struct NimResponse {
            data: Vec<NimEmbedding>,
        }

        let request = NimRequest {
            model: self.name.clone(),
            input: inputs,
            input_type: "query".to_string(),
        };

        let response = self.client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::GenerationError(format!("NIM request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::GenerationError(format!("NIM returned {}: {}", status, body)));
        }

        let mut result: NimResponse = response.json().await
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to parse NIM response: {}", e)))?;

        result.data.sort_by_key(|e| e.index);
        let embeddings: Vec<Vec<f32>> = result.data.into_iter().map(|e| e.embedding).collect();

        Ok(embeddings)
    }
}

/// Model type enum for configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    NvidiaNim,
}

impl ModelType {
    pub fn from_config() -> Self {
        ModelType::NvidiaNim
    }
}
