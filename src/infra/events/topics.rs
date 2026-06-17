//! Kafka Topic Definitions for ConFuse Platform

/// Kafka topic names used across the ConFuse platform
pub struct Topics;

impl Topics {
    // Chunk Processing Topics (unified-processor → embeddings-service)
    pub const CHUNKS_RAW: &'static str = "chunks.raw";

    // Embedding Topics (embeddings-service → unified-processor)
    pub const EMBEDDING_GENERATED: &'static str = "embedding.generated";
}

/// Get all active topic names for configuration
pub fn get_all_topics() -> Vec<&'static str> {
    vec![
        Topics::CHUNKS_RAW,
        Topics::EMBEDDING_GENERATED,
    ]
}
