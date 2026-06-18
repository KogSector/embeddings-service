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

/// Ollama-based embedding model (local HTTP API)
pub struct OllamaModel {
    name: String,
    ollama_url: String,
    dimension: usize,
    client: Client,
}

impl OllamaModel {
    pub fn new(model_name: &str, config: &Config) -> Result<Self> {
        let ollama_url = config.models.ollama_url
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        let dimension = Self::get_model_dimension(model_name);
        let client = Client::builder()
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
            _ => 768,
        }
    }

    /// Returns the maximum character length we allow for the given model.
    /// Ollama embedding models typically have a 2048-token default context.
    /// Code tokenizes densely (~2-3 chars/token), so we use conservative limits.
    fn get_max_input_chars(model_name: &str) -> usize {
        match model_name {
            "nomic-embed-text" => 6000,      // 2048 tokens × ~3 chars/token
            "mxbai-embed-large" => 1500,     // 512 tokens × ~3 chars/token
            "all-minilm" => 750,             // 256 tokens × ~3 chars/token
            "snowflake-arctic-embed" => 1500, // 512 tokens × ~3 chars/token
            _ => 4000,
        }
    }

    async fn generate_single(&self, text: &str) -> Result<Vec<f32>> {
        // Truncate text that exceeds the model's context window
        let max_chars = Self::get_max_input_chars(&self.name);
        let input = if text.len() > max_chars {
            tracing::warn!(
                "Truncating input from {} to {} chars for model {}",
                text.len(),
                max_chars,
                self.name
            );
            // Find a safe char boundary to truncate at
            let mut end = max_chars;
            while !text.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            &text[..end]
        } else {
            text
        };

        #[derive(Debug, Serialize)]
        struct Options {
            num_ctx: usize,
        }

        #[derive(Debug, Serialize)]
        struct Request {
            model: String,
            prompt: String,
            options: Options,
        }

        #[derive(Debug, Deserialize)]
        struct Response {
            embedding: Vec<f32>,
        }

        let request = Request {
            model: self.name.clone(),
            prompt: input.to_string(),
            options: Options {
                num_ctx: 8192,
            },
        };

        let response = self.client
            .post(format!("{}/api/embeddings", self.ollama_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::GenerationError(format!("Ollama request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::GenerationError(format!("Ollama returned {}: {}", status, body)));
        }

        let result: Response = response.json().await
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to parse Ollama response: {}", e)))?;

        Ok(result.embedding)
    }
}

#[async_trait]
impl EmbeddingModel for OllamaModel {
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
        for text in &texts {
            let embedding = self.generate_single(text).await?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// Model type enum for configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Ollama,
}

impl ModelType {
    pub fn from_config() -> Self {
        ModelType::Ollama
    }
}
