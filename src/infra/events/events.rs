//! Event Definitions for ConFuse Platform

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Event headers included in all events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventHeaders {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: String,
    pub source_service: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

impl EventHeaders {
    pub fn new(source_service: impl Into<String>, event_type: impl Into<String>) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            event_type: event_type.into(),
            timestamp: Utc::now().to_rfc3339(),
            source_service: source_service.into(),
            correlation_id: None,
            trace_id: None,
        }
    }
    
    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }
    
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }
}

/// Event metadata for processing context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    #[serde(default)]
    pub retry_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

/// Simplified chunk structure for raw chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedChunk {
    pub id: String,
    pub file_id: String,
    pub chunk_type: String, // function, class, etc.
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f32>,
}

/// Event published when raw chunks are created (simplified flow)
/// Emitted by unified-processor; consumed by embeddings-service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedChunkRawEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub repo_name: Option<String>,
    pub chunks: Vec<SimplifiedChunk>,
    pub timestamp: String,
}

impl SimplifiedChunkRawEvent {
    pub fn topic() -> &'static str {
        "chunks.raw"
    }
}

/// Simplified embedding structure (without content - unified-processor already has it)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedEmbedding {
    pub id: String,
    pub file_id: String,
    pub chunk_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub embedding: Vec<f32>,
    pub model: String,
    pub dimension: u32,
}

/// Event published when embeddings are generated (simplified flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedEmbeddingGeneratedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub repo_name: Option<String>,
    pub chunks: Vec<SimplifiedEmbedding>,
    pub model: String,
    pub timestamp: String,
}

impl SimplifiedEmbeddingGeneratedEvent {
    pub fn topic() -> &'static str {
        "embedding.generated"
    }
}
