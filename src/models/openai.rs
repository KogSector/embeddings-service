//! OpenAI embeddings model integration

use crate::core::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct OpenAIModel {
    name: String,
    dimension: usize,
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIModel {
    pub fn new(name: &str, config: &Config) -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| EmbeddingError::ConfigError("OPENAI_API_KEY not set".to_string()))?;

        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        let dimension = match name {
            "openai/text-embedding-3-small" => 1536,
            "openai/text-embedding-3-large" => 3072,
            "openai/text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,  // Graphiti compatibility
            "text-embedding-3-large" => 3072,  // Graphiti compatibility
            _ => 1536,
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| EmbeddingError::ConfigError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            name: name.to_string(),
            dimension,
            client,
            api_key,
            base_url,
        })
    }
}

#[async_trait]
impl crate::models::EmbeddingModel for OpenAIModel {
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

        let model_name = self.name.strip_prefix("openai/").unwrap_or(&self.name);
        
        let request = OpenAIEmbeddingRequest {
            model: model_name.to_string(),
            input: texts,
        };

        let response = self.client
            .post(&format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::GenerationError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::GenerationError(format!(
                "OpenAI API error: {} - {}",
                response.status(),
                error_text
            )));
        }

        let embedding_response: OpenAIEmbeddingResponse = response.json().await
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to parse response: {}", e)))?;

        let embeddings: Vec<Vec<f32>> = embedding_response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect();

        Ok(embeddings)
    }
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}
