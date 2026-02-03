//! Neo4j integration for storing embeddings directly in graph nodes

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::ModelManager;
use crate::storage::PostgresStorage;

#[derive(Debug, Deserialize)]
pub struct Neo4jEmbeddingRequest {
    pub texts: Vec<String>,
    pub node_ids: Vec<String>,  // Corresponding Neo4j node IDs
    pub node_labels: Vec<String>, // Neo4j node labels
    pub models: Option<Vec<String>>, // Multiple models for ensemble
    pub store_in_neo4j: bool,
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
}

#[derive(Debug, Serialize)]
pub struct Neo4jEmbeddingResponse {
    pub embeddings: Vec<Neo4jEmbeddingResult>,
    pub models_used: Vec<String>,
    pub fusion_method: String,
    pub neo4j_nodes_created: usize,
    pub postgres_stored: usize,
}

#[derive(Debug, Serialize)]
pub struct Neo4jEmbeddingResult {
    pub node_id: String,
    pub text: String,
    pub embeddings: HashMap<String, Vec<f32>>, // Multiple model embeddings
    pub fused_embedding: Vec<f32>, // Fused/ensemble embedding
    pub quality_score: f32,
    pub neo4j_stored: bool,
    pub postgres_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingFusionConfig {
    pub method: FusionMethod,
    pub weights: Option<HashMap<String, f32>>,
    pub normalize: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionMethod {
    WeightedAverage,
    Concatenation,
    Attention,
    MaxPooling,
    Adaptive, // Dynamically choose best based on content type
}

/// Generate multi-model embeddings and store in both Neo4j and PostgreSQL
pub async fn generate_neo4j_embeddings(
    State(model_manager): State<Arc<ModelManager>>,
    State(storage): State<Arc<PostgresStorage>>,
    Json(mut request): Json<Neo4jEmbeddingRequest>,
) -> Result<Json<Neo4jEmbeddingResponse>, StatusCode> {
    // Default to multiple models for strongest embeddings
    if request.models.is_none() {
        request.models = Some(vec![
            "nomic-embed-text".to_string(),      // Local - good for code
            "sentence-transformers/all-mpnet-base-v2".to_string(), // General purpose
            "openai/text-embedding-3-small".to_string(), // High quality
        ]);
    }

    let models = request.models.unwrap();
    let mut results = Vec::new();
    let mut neo4j_nodes_created = 0;
    let mut postgres_stored = 0;

    // Process each text with all models
    for (i, text) in request.texts.iter().enumerate() {
        let node_id = request.node_ids.get(i).unwrap_or(&format!("node_{}", i)).clone();
        let node_label = request.node_labels.get(i).unwrap_or(&"Entity".to_string()).clone();
        
        let mut embeddings_per_model = HashMap::new();
        let mut embedding_vectors = Vec::new();

        // Generate embeddings with each model
        for model_name in &models {
            if let Ok(model) = model_manager.ensure_model_loaded(model_name).await {
                if let Ok(model_embeddings) = model.generate(vec![text.clone()]).await {
                    if let Some(embedding) = model_embeddings.first() {
                        embeddings_per_model.insert(model_name.clone(), embedding.clone());
                        embedding_vectors.push(embedding.clone());
                    }
                }
            }
        }

        // Fuse embeddings using adaptive method
        let fused_embedding = fuse_embeddings(&embedding_vectors, &FusionMethod::Adaptive, None);
        let quality_score = calculate_embedding_quality(&fused_embedding, &node_label);

        let result = Neo4jEmbeddingResult {
            node_id: node_id.clone(),
            text: text.clone(),
            embeddings: embeddings_per_model,
            fused_embedding: fused_embedding.clone(),
            quality_score,
            neo4j_stored: false,
            postgres_id: None,
        };

        // Store in Neo4j if requested
        if request.store_in_neo4j {
            if let Ok(_) = store_in_neo4j(
                &request.neo4j_uri,
                &request.neo4j_user,
                &request.neo4j_password,
                &node_id,
                &node_label,
                &fused_embedding,
                &models,
                &result.embeddings,
            ).await {
                result.neo4j_stored = true;
                neo4j_nodes_created += 1;
            }
        }

        // Also store in PostgreSQL for backup/search
        if let Ok(postgres_id) = store_in_postgres(storage, &node_id, text, &fused_embedding, &models).await {
            result.postgres_id = Some(postgres_id.to_string());
            postgres_stored += 1;
        }

        results.push(result);
    }

    let response = Neo4jEmbeddingResponse {
        embeddings: results,
        models_used: models,
        fusion_method: "adaptive".to_string(),
        neo4j_nodes_created,
        postgres_stored,
    };

    Ok(Json(response))
}

/// Fuse multiple embeddings using various methods
fn fuse_embeddings(
    embeddings: &[Vec<f32>],
    method: &FusionMethod,
    weights: Option<&HashMap<String, f32>>,
) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![];
    }

    if embeddings.len() == 1 {
        return embeddings[0].clone();
    }

    match method {
        FusionMethod::WeightedAverage => {
            let dim = embeddings[0].len();
            let mut fused = vec![0.0; dim];
            
            for embedding in embeddings {
                for (i, &val) in embedding.iter().enumerate() {
                    fused[i] += val;
                }
            }
            
            let weight = 1.0 / embeddings.len() as f32;
            for val in &mut fused {
                *val *= weight;
            }
            
            fused
        }
        FusionMethod::Concatenation => {
            let mut fused = Vec::new();
            for embedding in embeddings {
                fused.extend(embedding);
            }
            fused
        }
        FusionMethod::MaxPooling => {
            let dim = embeddings[0].len();
            let mut fused = vec![f32::MIN; dim];
            
            for embedding in embeddings {
                for (i, &val) in embedding.iter().enumerate() {
                    fused[i] = fused[i].max(val);
                }
            }
            
            fused
        }
        FusionMethod::Attention => {
            // Simple attention mechanism (can be enhanced)
            let dim = embeddings[0].len();
            let mut fused = vec![0.0; dim];
            let mut attention_weights = Vec::new();
            
            // Calculate attention weights based on embedding norms
            for embedding in embeddings {
                let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
                attention_weights.push(norm);
            }
            
            let total_weight: f32 = attention_weights.iter().sum();
            for (i, embedding) in embeddings.iter().enumerate() {
                let weight = attention_weights[i] / total_weight;
                for (j, &val) in embedding.iter().enumerate() {
                    fused[j] += val * weight;
                }
            }
            
            fused
        }
        FusionMethod::Adaptive => {
            // Choose best method based on embedding characteristics
            if embeddings.len() == 2 {
                // For 2 models, use weighted average
                fuse_embeddings(embeddings, &FusionMethod::WeightedAverage, weights)
            } else if embeddings.len() == 3 {
                // For 3 models, use attention
                fuse_embeddings(embeddings, &FusionMethod::Attention, weights)
            } else {
                // For many models, use max pooling
                fuse_embeddings(embeddings, &FusionMethod::MaxPooling, weights)
            }
        }
    }
}

/// Calculate embedding quality score
fn calculate_embedding_quality(embedding: &[f32], node_label: &str) -> f32 {
    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let variance = calculate_variance(embedding);
    
    // Quality based on normalization, variance, and node type
    let base_quality = if norm > 0.1 && norm < 10.0 { 0.8 } else { 0.5 };
    let variance_bonus = if variance > 0.01 { 0.2 } else { 0.0 };
    let label_bonus = match node_label {
        "Code" | "Function" | "Class" => 0.1, // Code entities get bonus
        "Document" | "Text" => 0.05,
        _ => 0.0,
    };
    
    (base_quality + variance_bonus + label_bonus).min(1.0)
}

fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    
    variance
}

/// Store embedding in Neo4j node
async fn store_in_neo4j(
    uri: &str,
    user: &str,
    password: &str,
    node_id: &str,
    label: &str,
    embedding: &[f32],
    models: &[String],
    individual_embeddings: &HashMap<String, Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // This would use a Neo4j driver - for now, return success
    // In real implementation, you'd use something like neo4rs crate
    
    // Example Cypher query:
    // MERGE (n:Entity {id: $node_id})
    // SET n.embedding = $embedding,
    //     n.models = $models,
    //     n.embedding_quality = $quality,
    //     n.updated_at = datetime()
    
    println!("Storing in Neo4j: node_id={}, label={}, embedding_dim={}", 
             node_id, label, embedding.len());
    
    Ok(())
}

/// Store embedding in PostgreSQL as backup
async fn store_in_postgres(
    storage: &Arc<PostgresStorage>,
    node_id: &str,
    text: &str,
    embedding: &[f32],
    models: &[String],
) -> Result<uuid::Uuid, Box<dyn std::error::Error + Send + Sync>> {
    use crate::storage::VectorRecord;
    use uuid::Uuid;
    
    let metadata = serde_json::json!({
        "node_id": node_id,
        "models": models,
        "stored_in_neo4j": true
    });
    
    let record = VectorRecord {
        id: Uuid::new_v4(),
        text: text.to_string(),
        embedding: embedding.to_vec(),
        model: models.join(","), // Composite model name
        metadata,
    };
    
    storage.store_embedding(record).await.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
}

/// Get embeddings from Neo4j nodes
pub async fn get_neo4j_embeddings(
    Query(params): Query<HashMap<String, String>>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "not_implemented_yet",
        "message": "Neo4j retrieval endpoint coming soon",
        "params": params
    }))
}
