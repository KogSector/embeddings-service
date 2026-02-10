//! Ollama model integration via HTTP API
//!
//! Connects to a local or remote Ollama server for embedding generation.
//! Uses the `/api/embeddings` endpoint.

use crate::core::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Request body for Ollama embeddings API
#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    prompt: String,
}

/// Response from Ollama embeddings API
#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

pub struct OllamaModel {
    name: String,
    ollama_url: String,
    dimension: usize,
    client: reqwest::Client,
}

impl OllamaModel {
    pub fn new(model_name: &str, config: &Config) -> Result<Self> {
        let ollama_url = config.models.ollama_url
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let dimension = Self::get_model_dimension(model_name);

        let client = reqwest::Client::builder()
            .timeout(config.models.timeout)
            .build()
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            name: model_name.to_string(),
            ollama_url,
            dimension,
            client,
        })
    }

    fn get_model_dimension(model_name: &str) -> usize {
        match model_name {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            "all-minilm" => 384,
            "snowflake-arctic-embed" => 1024,
            _ => 768, // Default for most Ollama embedding models
        }
    }

    async fn generate_single(&self, text: &str) -> Result<Vec<f32>> {
        let request = OllamaEmbeddingRequest {
            model: self.name.clone(),
            prompt: text.to_string(),
        };

        let response = self.client
            .post(format!("{}/api/embeddings", self.ollama_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::GenerationError(format!(
                "Ollama request failed: {}", e
            )))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::GenerationError(format!(
                "Ollama returned {}: {}", status, body
            )));
        }

        let result: OllamaEmbeddingResponse = response.json().await
            .map_err(|e| EmbeddingError::GenerationError(format!(
                "Failed to parse Ollama response: {}", e
            )))?;

        Ok(result.embedding)
    }
}

#[async_trait]
impl crate::models::EmbeddingModel for OllamaModel {
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

        let mut embeddings = Vec::with_capacity(texts.len());
        
        // Ollama processes one text at a time via /api/embeddings
        for text in &texts {
            let embedding = self.generate_single(text).await?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}
