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

/// Gemini-based embedding model (Google API)
pub struct GeminiModel {
    name: String,
    api_key: String,
    base_url: String,
    dimension: usize,
    client: Client,
}

impl GeminiModel {
    pub fn new(model_name: &str, config: &Config) -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .unwrap_or_else(|_| "".to_string());
        if api_key.is_empty() {
            tracing::warn!("GEMINI_API_KEY is not set. Embedding generation will fail.");
        }
        
        let base_url = std::env::var("GEMINI_BASE_URL")
            .unwrap_or_else(|_| "https://generativelanguage.googleapis.com".to_string());
        
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

    fn get_model_dimension(model_name: &str) -> usize {
        let normalized_name = model_name.trim_start_matches("models/");
        match normalized_name {
            "embedding-001" | "text-embedding-004" | "embedding-003" => 768,
            _ => 768,
        }
    }

    /// Truncates text conservatively if it's too long
    fn get_max_input_chars(_model_name: &str) -> usize {
        10000 // Gemini embedding handles up to 2048 tokens or more (text-embedding-004 handles 2048)
    }

    async fn generate_single(&self, text: &str) -> Result<Vec<f32>> {
        let max_chars = Self::get_max_input_chars(&self.name);
        let input = if text.len() > max_chars {
            tracing::warn!(
                "Truncating input from {} to {} chars for model {}",
                text.len(),
                max_chars,
                self.name
            );
            let mut end = max_chars;
            while !text.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            &text[..end]
        } else {
            text
        };

        #[derive(Debug, Serialize)]
        struct GeminiPart { text: String }
        #[derive(Debug, Serialize)]
        struct GeminiContent { parts: Vec<GeminiPart> }
        #[derive(Debug, Serialize)]
        struct Request {
            model: String,
            content: GeminiContent,
        }

        #[derive(Debug, Deserialize)]
        struct GeminiEmbedding { values: Vec<f32> }
        #[derive(Debug, Deserialize)]
        struct Response {
            embedding: GeminiEmbedding,
        }

        let normalized_name = self.name.trim_start_matches("models/");
        let request = Request {
            model: format!("models/{}", normalized_name),
            content: GeminiContent {
                parts: vec![GeminiPart { text: input.to_string() }],
            }
        };

        let url = format!(
            "{}/v1/models/{}:embedContent?key={}",
            self.base_url, normalized_name, self.api_key
        );

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::GenerationError(format!("Gemini request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::GenerationError(format!("Gemini returned {}: {}", status, body)));
        }

        let result: Response = response.json().await
            .map_err(|e| EmbeddingError::GenerationError(format!("Failed to parse Gemini response: {}", e)))?;

        Ok(result.embedding.values)
    }
}

#[async_trait]
impl EmbeddingModel for GeminiModel {
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
    Gemini,
}

impl ModelType {
    pub fn from_config() -> Self {
        ModelType::Gemini
    }
}
