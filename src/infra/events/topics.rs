//! Kafka Topic Definitions for ConFuse Platform

/// Kafka topic names used across the ConFuse platform
pub struct Topics;

impl Topics {
    // Source Sync Topics (data-connector → unified-processor)
    pub const SOURCE_SYNC_REQUESTED: &'static str = "source.sync.requested";

    // Chunk Processing Topics (unified-processor → embeddings-service)
    pub const CHUNKS_RAW: &'static str = "chunks.raw";

    // Embedding Topics (embeddings-service → unified-processor)
    pub const EMBEDDING_GENERATED: &'static str = "embedding.generated";

    // Graphify Topics (data-connector → graphify-pipeline)
    pub const GRAPHIFY_EPISODES: &'static str = "graphify.episodes.v1";
}

/// Get all active topic names for configuration
pub fn get_all_topics() -> Vec<&'static str> {
    vec![
        Topics::SOURCE_SYNC_REQUESTED,
        Topics::CHUNKS_RAW,
        Topics::EMBEDDING_GENERATED,
        Topics::GRAPHIFY_EPISODES,
    ]
}
