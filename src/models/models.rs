//! Embedding model implementations
//! Supports local models via SentenceTransformers (Python) and Ollama

use crate::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use pyo3::prelude::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::task;

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

    async fn generate_single(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Debug, Serialize)]
        struct Request {
            model: String,
            prompt: String,
        }

        #[derive(Debug, Deserialize)]
        struct Response {
            embedding: Vec<f32>,
        }

        let request = Request {
            model: self.name.clone(),
            prompt: text.to_string(),
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

/// SentenceTransformers-based embedding model (Python via PyO3)
pub struct SentenceTransformersModel {
    name: String,
    dimension: usize,
    max_batch_size: usize,
}

impl SentenceTransformersModel {
    pub fn new(name: &str, _config: &Config) -> Result<Self> {
        let dimension = Self::get_model_dimension(name);
        Ok(Self {
            name: name.to_string(),
            dimension,
            max_batch_size: 32,
        })
    }

    fn get_model_dimension(model_name: &str) -> usize {
        match model_name {
            "sentence-transformers/all-MiniLM-L6-v2" => 384,
            "sentence-transformers/all-mpnet-base-v2" => 768,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" => 384,
            _ => 384,
        }
    }

    fn generate_with_python(
        py: Python<'_>,
        model_name: &str,
        texts: Vec<String>,
        _max_batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let st_module = py.import("sentence_transformers")
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to import sentence_transformers: {}", e)))?;

        let model_class = st_module.getattr("SentenceTransformer")
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to get SentenceTransformer class: {}", e)))?;

        let model = model_class.call1((model_name,))
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to create model {}: {}", model_name, e)))?;

        let py_texts = pyo3::types::PyList::new(py, &texts)
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to create Python list: {}", e)))?;

        let embeddings = model.call_method1("encode", (py_texts,))
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to encode texts: {}", e)))?;

        let embeddings = embeddings.call_method0("tolist")
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to convert embeddings: {}", e)))?;

        embeddings.extract::<Vec<Vec<f32>>>()
            .map_err(|e| EmbeddingError::PythonError(format!("Failed to extract embeddings: {}", e)))
    }
}

#[async_trait]
impl EmbeddingModel for SentenceTransformersModel {
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

        let model_name = self.name.clone();
        let max_batch_size = self.max_batch_size;

        let result = task::spawn_blocking(move || {
            Python::with_gil(|py| {
                Self::generate_with_python(py, &model_name, texts, max_batch_size)
            })
        })
        .await
        .map_err(|e| EmbeddingError::GenerationError(e.to_string()))??;

        Ok(result)
    }
}

/// Model type enum for configuration
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Ollama,
    SentenceTransformers,
}

impl ModelType {
    pub fn from_config(use_ollama: bool) -> Self {
        if use_ollama {
            ModelType::Ollama
        } else {
            ModelType::SentenceTransformers
        }
    }
}
