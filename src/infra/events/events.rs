//! Event Definitions for ConFuse Platform

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Common Types
// =============================================================================

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

/// File type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FileType {
    Unknown,
    Code,
    Document,
}

/// Source types for ingestion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    #[serde(rename = "unknown")]
    Unknown,
    #[serde(rename = "github")]
    Github,
    #[serde(rename = "gitlab")]
    Gitlab,
    #[serde(rename = "bitbucket")]
    Bitbucket,
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "gdrive")]
    Gdrive,
    #[serde(rename = "notion")]
    Notion,
    #[serde(rename = "file_upload")]
    FileUpload,
    #[serde(rename = "dropbox")]
    Dropbox,
    #[serde(rename = "onedrive")]
    Onedrive,
    #[serde(rename = "web")]
    Web,
    #[serde(rename = "url")]
    Url,
    #[serde(rename = "salesforce")]
    Salesforce,
    #[serde(rename = "hubspot")]
    Hubspot,
    #[serde(rename = "drata")]
    Drata,
    #[serde(rename = "vanta")]
    Vanta,
    #[serde(rename = "confluence")]
    Confluence,
    #[serde(rename = "sql_database")]
    SqlDatabase,
    #[serde(rename = "nosql_database")]
    NosqlDatabase,
}

// =============================================================================
// Code Events
// =============================================================================

/// Event published when a code file is ingested
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIngestedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub file_id: String,
    pub source_id: String,
    pub file_path: String,
    pub file_name: String,
    pub file_extension: String,
    pub file_size_bytes: u64,
    pub content_hash: String,
    pub storage_location: String,
    pub language: String,
    #[serde(default)]
    pub is_config: bool,
}

impl CodeIngestedEvent {
    pub fn topic() -> &'static str {
        "code.ingested"
    }
}

/// Processed code chunk with AST information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub chunk_id: String,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub chunk_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Event published when a code file has been processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProcessedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub file_id: String,
    pub source_id: String,
    pub language: String,
    pub chunks: Vec<CodeChunk>,
    #[serde(default)]
    pub functions_count: u32,
    #[serde(default)]
    pub classes_count: u32,
    #[serde(default)]
    pub imports: Vec<String>,
    pub processing_time_ms: u64,
}

impl CodeProcessedEvent {
    pub fn topic() -> &'static str {
        "code.processed"
    }
}

// =============================================================================
// Document Events
// =============================================================================

/// Event published when a document is ingested
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsIngestedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub file_id: String,
    pub source_id: String,
    pub file_path: String,
    pub file_name: String,
    pub file_extension: String,
    pub file_size_bytes: u64,
    pub content_hash: String,
    pub storage_location: String,
    pub document_type: String,
}

impl DocsIngestedEvent {
    pub fn topic() -> &'static str {
        "docs.ingested"
    }
}

/// Processed document chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocChunk {
    pub chunk_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,
    pub chunk_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_heading: Option<String>,
}

/// Table extracted from document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTable {
    pub table_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,
    pub row_count: u32,
    pub column_count: u32,
    pub content_markdown: String,
}

/// Event published when a document has been processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsProcessedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub file_id: String,
    pub source_id: String,
    pub document_type: String,
    pub chunks: Vec<DocChunk>,
    #[serde(default)]
    pub page_count: u32,
    #[serde(default)]
    pub word_count: u32,
    #[serde(default)]
    pub has_tables: bool,
    #[serde(default)]
    pub tables: Vec<ExtractedTable>,
    #[serde(default)]
    pub has_images: bool,
    pub processing_time_ms: u64,
}

impl DocsProcessedEvent {
    pub fn topic() -> &'static str {
        "docs.processed"
    }
}

// =============================================================================
// Chunk Events
// =============================================================================

/// Event published when a chunk is created and stored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkCreatedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub file_id: String,
    pub chunk_id: String,
    pub content_hash: String,
    pub blob_storage_url: String, // URL to raw content in Blob Storage
    pub chunk_type: String,       // code, text, table
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_heading: Option<String>,
}

impl ChunkCreatedEvent {
    pub fn topic() -> &'static str {
        "chunks.created"
    }
}

/// Entity hint pre-identified in a chunk by the agentic chunker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHint {
    /// The entity surface form as it appears in the text
    pub text: String,
    /// Entity type (organization, person, technology, concept, location, etc.)
    pub entity_type: String,
    /// Confidence that this is a real entity (0.0–1.0)
    pub confidence: f32,
    /// Byte offset where entity starts in the chunk content
    pub start_offset: usize,
    /// Byte offset where entity ends in the chunk content
    pub end_offset: usize,
}

/// Simplified chunk metadata for event serialization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    /// Line range in source file
    pub line_range: Option<(usize, usize)>,
    /// Byte range in source file
    pub byte_range: Option<(usize, usize)>,
    /// Complexity score (1-10)
    pub complexity_score: u8,
    /// Token count (approximate)
    pub token_count: usize,
    /// Pre-identified entity hints from agentic chunker
    #[serde(default)]
    pub entity_hints: Vec<EntityHint>,
    /// Relationship hints in "subject -> predicate -> object" notation
    #[serde(default)]
    pub relationship_context: Vec<String>,
    /// Custom key-value metadata
    #[serde(default)]
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

/// Event published when raw chunks are created with entity hints
/// Emitted by unified-processor after intelligent chunking; consumed by embeddings-service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRawEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub file_id: String,
    pub chunk_id: String,
    /// Raw text content of chunk
    pub content: String,
    /// Chunk type (code, text, table, etc.)
    pub chunk_type: String,
    /// Granularity level
    pub level: String,
    /// Processing tier applied
    pub tier: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Quality score (0.0-1.0, populated after enhancement)
    #[serde(default)]
    pub quality_score: Option<f32>,
    /// Chunk metadata
    pub chunk_metadata: ChunkMetadata,
    /// Pre-identified entity hints from agentic chunker
    #[serde(default)]
    pub entity_hints: Vec<EntityHint>,
    /// Relationship hints in "subject -> predicate -> object" notation
    #[serde(default)]
    pub relationship_context: Vec<String>,
    /// Creation timestamp
    #[serde(default = "chrono::Utc::now")]
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ChunkRawEvent {
    pub fn topic() -> &'static str {
        "chunks.raw"
    }
}

/// Event published when a chunk has been enriched with entity hints and optional embeddings.
/// to seed entity extraction with pre-identified entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEnrichedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub file_id: String,
    pub chunk_id: String,
    /// Actual text content of the chunk (stored directly to avoid blob round-trips)
    pub content: String,
    /// Chunk type (code, text, table, etc.)
    pub chunk_type: String,
    /// Pre-identified entity hints from the agentic chunker
    #[serde(default)]
    pub entity_hints: Vec<EntityHint>,
    /// Relationship hints in "subject -> predicate -> object" notation
    #[serde(default)]
    pub relationship_context: Vec<String>,
    /// Embedding vector (serialized; None if not yet generated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Quality score from the agentic chunker (0.0–1.0)
    pub quality_score: f32,
}

impl ChunkEnrichedEvent {
    pub fn topic() -> &'static str {
        "chunks.enriched"
    }
}

// =============================================================================
// Embedding Events
// =============================================================================

/// Event published when embeddings have been generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingGeneratedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub file_id: String,
    pub source_id: String,
    pub chunk_ids: Vec<String>,
    pub embedding_model: String,
    pub embedding_dimension: u32,
    pub total_chunks: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_storage_location: Option<String>,
    pub processing_time_ms: u64,
}

impl EmbeddingGeneratedEvent {
    pub fn topic() -> &'static str {
        "embedding.generated"
    }
}

// =============================================================================
// Graph Events
// =============================================================================

/// Relationship in knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelationship {
    pub source_node_id: String,
    pub target_node_id: String,
    pub relationship_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<String>,
}

/// Event published when the knowledge graph has been updated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphUpdatedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default)]
    pub nodes_added: u32,
    #[serde(default)]
    pub nodes_updated: u32,
    #[serde(default)]
    pub edges_added: u32,
    #[serde(default)]
    pub edges_updated: u32,
    #[serde(default)]
    pub sample_relationships: Vec<GraphRelationship>,
    pub processing_time_ms: u64,
}

impl GraphUpdatedEvent {
    pub fn topic() -> &'static str {
        "graph.updated"
    }
}

/// Event to request a graph build
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBuildRequestedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    #[serde(default)]
    pub force_rebuild: bool,
    pub build_type: String,
}

impl GraphBuildRequestedEvent {
    pub fn topic() -> &'static str {
        "graph.build.requested"
    }
}

/// Event published when graph build completes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBuildCompletedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub relationships_created: u32,
    pub entities_created: u32,
    pub build_time_ms: u64,
    pub build_type: String,
}

impl GraphBuildCompletedEvent {
    pub fn topic() -> &'static str {
        "graph.build.completed"
    }
}

// =============================================================================
// Source Sync Events
// =============================================================================

/// Event triggered when source sync is requested
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSyncRequestedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub source_type: SourceType,
    pub source_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    #[serde(default)]
    pub full_sync: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_by: Option<String>,
}

impl SourceSyncRequestedEvent {
    pub fn topic() -> &'static str {
        "source.sync.requested"
    }

    pub fn new(source_id: impl Into<String>, source_type: SourceType, source_url: impl Into<String>) -> Self {
        Self {
            headers: EventHeaders::new("api-backend", "SOURCE_SYNC_REQUESTED"),
            metadata: EventMetadata::default(),
            source_id: source_id.into(),
            source_type,
            source_url: source_url.into(),
            branch: None,
            access_token: None,
            full_sync: false,
            requested_by: None,
        }
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        let uid = user_id.into();
        self.metadata.user_id = Some(uid.clone());
        self.requested_by = Some(uid);
        self
    }
}

/// Event triggered when source sync completes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSyncCompletedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub files_processed: u32,
    #[serde(default)]
    pub code_files_count: u32,
    #[serde(default)]
    pub doc_files_count: u32,
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

impl SourceSyncCompletedEvent {
    pub fn topic() -> &'static str {
        "source.sync.completed"
    }
}

/// Event triggered when source sync fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSyncFailedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub error: String,
    pub error_code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

impl SourceSyncFailedEvent {
    pub fn topic() -> &'static str {
        "source.sync.failed"
    }
}

// =============================================================================
// Auth Events
// =============================================================================

/// Authentication event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub user_id: String,
    pub event_type: String, // login, logout, token_refresh, login_failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    #[serde(default)]
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

impl AuthEvent {
    pub fn topic() -> &'static str {
        "auth.events"
    }
}

/// Session event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub session_id: String,
    pub user_id: String,
    pub event_type: String, // created, destroyed, expired, refreshed
    pub expires_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,
}

impl SessionEvent {
    pub fn topic() -> &'static str {
        "session.events"
    }
}

// =============================================================================
// DLQ Events
// =============================================================================

/// Event published when processing fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingFailedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub original_topic: String,
    pub original_event_id: String,
    pub error_message: String,
    pub error_type: String,
    pub error_code: String,
    pub retry_count: u32,
    #[serde(default)]
    pub is_retryable: bool,
    pub original_payload: String,
    pub error_timestamp: String,
    pub failed_service: String,
}

impl ProcessingFailedEvent {
    pub fn topic() -> &'static str {
        "dlq.processing.failed"
    }
}

// =============================================================================
// =============================================================================

/// Simplified chunk metadata for raw chunks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimplifiedChunkMetadata {
    /// Line range in source file
    pub line_range: Option<(usize, usize)>,
    /// Byte range in source file
    pub byte_range: Option<(usize, usize)>,
    /// Complexity score (1-10)
    pub complexity_score: u8,
    /// Token count (approximate)
    pub token_count: usize,
    /// Quality score (0.0-1.0)
    #[serde(default)]
    pub quality_score: Option<f32>,
    /// Language for code chunks
    #[serde(default)]
    pub language: Option<String>,
    /// Start line number
    #[serde(default)]
    pub start_line: Option<u32>,
    /// End line number
    #[serde(default)]
    pub end_line: Option<u32>,
    /// Confidence score
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// Simplified chunk structure for raw chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedChunk {
    pub chunk_id: String,
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
    pub chunk_id: String,
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
    pub chunks: Vec<SimplifiedEmbedding>,
    pub model: String,
    pub timestamp: String,
}

impl SimplifiedEmbeddingGeneratedEvent {
    pub fn topic() -> &'static str {
        "embedding.generated"
    }
}

/// Event published when graph is updated (simplified flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedGraphUpdatedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub nodes_created: u32,
    pub edges_created: u32,
    pub timestamp: String,
}

impl SimplifiedGraphUpdatedEvent {
    pub fn topic() -> &'static str {
        "graph.updated"
    }
}

/// Event published when processing fails (simplified flow)
/// Emitted by any service; consumed by DLQ handlers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedProcessingFailedEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub original_topic: String,
    pub original_event_id: String,
    pub error_message: String,
    pub error_type: String,
    pub timestamp: String,
}

impl SimplifiedProcessingFailedEvent {
    pub fn topic() -> &'static str {
        "dlq.processing.failed"
    }
}

// =============================================================================
// Cross-Repository Relationship Events
// =============================================================================

/// Dependency detected in a repository (package imports, external libraries)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryDependency {
    /// Name of the dependency (e.g., "numpy", "express", "sqlx")
    pub name: String,
    /// Version constraint (e.g., "^1.0", ">=2.0.0")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Type of dependency: "runtime", "dev", "build", "peer"
    pub dependency_type: String,
    /// Source of detection: "manifest", "import", "api_call"
    pub detection_source: String,
    /// File where the dependency was detected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_in_file: Option<String>,
}

/// Event carrying repository-level metadata for cross-repo analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryMetadataEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_id: String,
    pub repository_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_branch: Option<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<RepositoryDependency>,
    pub timestamp: String,
}

impl RepositoryMetadataEvent {
    pub fn topic() -> &'static str {
        "repository.metadata"
    }
}

/// Event published when cross-repository relationships are detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRepoRelationshipEvent {
    pub headers: EventHeaders,
    #[serde(default)]
    pub metadata: EventMetadata,
    pub source_repository_id: String,
    pub target_repository_id: String,
    pub relationship_type: String,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence_snippet: Option<String>,
    pub timestamp: String,
}

impl CrossRepoRelationshipEvent {
    pub fn topic() -> &'static str {
        "cross.repo.relationships"
    }
}
