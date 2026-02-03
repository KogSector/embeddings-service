//! SentenceTransformers model integration

use crate::core::{Config, Result, EmbeddingError};
use async_trait::async_trait;
use pyo3::prelude::*;
use std::collections::HashMap;
use tokio::task;

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
            _ => 384, // Default
        }
    }
}

#[async_trait]
impl crate::models::EmbeddingModel for SentenceTransformersModel {
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

        // Run Python in blocking task
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

impl SentenceTransformersModel {
    fn generate_with_python(
        py: Python<'_>,
        model_name: &str,
        texts: Vec<String>,
        _max_batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // Import SentenceTransformers
        let st_module = py.import("sentence_transformers")
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to import sentence_transformers: {}", e
            )))?;

        // Get SentenceTransformer class
        let model_class = st_module.getattr("SentenceTransformer")
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to get SentenceTransformer class: {}", e
            )))?;

        // Create model instance
        let model = model_class.call1((model_name,))
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to create model {}: {}", model_name, e
            )))?;

        // Convert texts to Python list
        let py_texts = pyo3::types::PyList::new(py, &texts)
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to create Python list: {}", e
            )))?;

        // Generate embeddings
        let embeddings = model.call_method1("encode", (py_texts,))
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to encode texts: {}", e
            )))?;

        // Convert to Rust Vec<Vec<f32>>
        let embeddings = embeddings.call_method0("tolist")
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to convert embeddings: {}", e
            )))?;

        embeddings.extract::<Vec<Vec<f32>>>()
            .map_err(|e| EmbeddingError::PythonError(format!(
                "Failed to extract embeddings: {}", e
            )))
    }
}
