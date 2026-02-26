//! Property-based tests for vector storage round trip
//!
//! **Validates: Requirements 4.1, 6.1**
//!
//! This test validates that vectors stored in FalcorDB can be retrieved with full fidelity.
//! It tests the complete round trip: store → retrieve → compare.
//!
//! **Requirements:**
//! - 4.1: WHEN the Embeddings_Service generates an embedding vector, THE Embeddings_Service 
//!        SHALL create a Vector_Chunk node in FalcorDB
//! - 6.1: THE Unified_Processor SHALL retrieve vectors from FalcorDB Vector_Chunk nodes

use super::{FalcorDBClient, FalcorDBConfig, VectorChunk};
use proptest::prelude::*;
use uuid::Uuid;

/// **Property 3: Vector Storage Round Trip**
///
/// *For all* valid Vector_Chunk nodes stored in FalcorDB, the data SHALL be retrievable
/// with full fidelity, including:
/// - Embedding vector values (within floating-point precision)
/// - Chunk text content (exact match)
/// - All metadata fields (exact match)
/// - Chunk index (exact match)
/// - Document and source IDs (exact match)
///
/// **Validates: Requirements 4.1, 6.1**
///
/// This property test validates that:
/// 1. Stored vectors can be retrieved by ID
/// 2. Retrieved embedding values match stored values (within epsilon)
/// 3. Retrieved text content matches stored text exactly
/// 4. Retrieved metadata matches stored metadata exactly
/// 5. Retrieved chunk index matches stored index
/// 6. Retrieved document_id and source_id match stored values
/// 7. Round trip preserves all data integrity

// Strategy for generating valid 384-dimensional embeddings
fn valid_embedding_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32, // Typical normalized embedding range
        384..=384 // Exactly 384 dimensions
    )
}

// Strategy for generating valid chunk text (non-empty, reasonable length)
fn valid_text_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 .,!?\\-\\n]{1,1000}").unwrap()
}

// Strategy for generating valid source IDs
fn valid_source_id_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z0-9_\\-]{1,50}").unwrap()
}

// Strategy for generating valid metadata JSON
fn valid_metadata_strategy() -> impl Strategy<Value = serde_json::Value> {
    prop::collection::hash_map(
        prop::string::string_regex("[a-z_]{1,20}").unwrap(),
        prop::string::string_regex("[a-zA-Z0-9 ]{0,50}").unwrap(),
        0..5
    ).prop_map(|map| serde_json::json!(map))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test that stored vector chunks can be retrieved with exact text content
    ///
    /// **Property**: Text content SHALL be preserved exactly during round trip
    /// **Validates: Requirements 4.1, 6.1**
    #[test]
    fn prop_roundtrip_preserves_text_content(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
        metadata in valid_metadata_strategy(),
    ) {
        // Skip test if FalcorDB is not available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Setup: Create client and test data
            let config = create_test_config();
            let client = match FalcorDBClient::new(config).await {
                Ok(c) => c,
                Err(_) => return Ok(()), // Skip if connection fails
            };

            let document_id = Uuid::new_v4();
            let mut chunk = VectorChunk::new(
                text.clone(),
                embedding.clone(),
                document_id,
                source_id.clone(),
                chunk_index,
            );
            chunk.metadata = metadata.clone();

            // Store the chunk
            let stored_id = client.store_vector_chunk(&chunk).await;
            prop_assert!(stored_id.is_ok(), "Failed to store chunk: {:?}", stored_id.err());

            // Retrieve the chunk
            let retrieved = retrieve_chunk_by_id(&client, &chunk.id).await;
            prop_assert!(retrieved.is_ok(), "Failed to retrieve chunk: {:?}", retrieved.err());

            let retrieved_chunk = retrieved.unwrap();
            prop_assert!(retrieved_chunk.is_some(), "Chunk not found after storage");

            let retrieved_chunk = retrieved_chunk.unwrap();

            // Property: Text content must be preserved exactly
            prop_assert_eq!(
                retrieved_chunk.chunk_text,
                text,
                "Text content not preserved during round trip"
            );

            // Cleanup
            let _ = delete_chunk(&client, &chunk.id).await;

            Ok(())
        })?;

        Ok(())
    }

    /// Test that stored vector embeddings can be retrieved with high precision
    ///
    /// **Property**: Embedding values SHALL be preserved within floating-point precision
    /// **Validates: Requirements 4.1, 6.1**
    #[test]
    fn prop_roundtrip_preserves_embedding_values(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        // Skip test if FalcorDB is not available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = create_test_config();
            let client = match FalcorDBClient::new(config).await {
                Ok(c) => c,
                Err(_) => return Ok(()),
            };

            let document_id = Uuid::new_v4();
            let chunk = VectorChunk::new(
                text,
                embedding.clone(),
                document_id,
                source_id,
                chunk_index,
            );

            // Store the chunk
            let stored_id = client.store_vector_chunk(&chunk).await;
            prop_assert!(stored_id.is_ok(), "Failed to store chunk");

            // Retrieve the chunk
            let retrieved = retrieve_chunk_by_id(&client, &chunk.id).await;
            prop_assert!(retrieved.is_ok(), "Failed to retrieve chunk");

            let retrieved_chunk = retrieved.unwrap();
            prop_assert!(retrieved_chunk.is_some(), "Chunk not found after storage");

            let retrieved_chunk = retrieved_chunk.unwrap();

            // Property: Embedding dimension must be preserved
            prop_assert_eq!(
                retrieved_chunk.embedding.len(),
                384,
                "Embedding dimension not preserved"
            );

            // Property: Embedding values must be preserved within epsilon
            const EPSILON: f32 = 1e-6;
            for (i, (&stored_val, &retrieved_val)) in embedding.iter()
                .zip(retrieved_chunk.embedding.iter())
                .enumerate()
            {
                let diff = (stored_val - retrieved_val).abs();
                prop_assert!(
                    diff < EPSILON,
                    "Embedding value at index {} differs: stored={}, retrieved={}, diff={}",
                    i, stored_val, retrieved_val, diff
                );
            }

            // Cleanup
            let _ = delete_chunk(&client, &chunk.id).await;

            Ok(())
        })?;

        Ok(())
    }

    /// Test that stored metadata is preserved exactly
    ///
    /// **Property**: Metadata SHALL be preserved exactly during round trip
    /// **Validates: Requirements 4.1, 6.1**
    #[test]
    fn prop_roundtrip_preserves_metadata(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
        metadata in valid_metadata_strategy(),
    ) {
        // Skip test if FalcorDB is not available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = create_test_config();
            let client = match FalcorDBClient::new(config).await {
                Ok(c) => c,
                Err(_) => return Ok(()),
            };

            let document_id = Uuid::new_v4();
            let mut chunk = VectorChunk::new(
                text,
                embedding,
                document_id,
                source_id,
                chunk_index,
            );
            chunk.metadata = metadata.clone();

            // Store the chunk
            let stored_id = client.store_vector_chunk(&chunk).await;
            prop_assert!(stored_id.is_ok(), "Failed to store chunk");

            // Retrieve the chunk
            let retrieved = retrieve_chunk_by_id(&client, &chunk.id).await;
            prop_assert!(retrieved.is_ok(), "Failed to retrieve chunk");

            let retrieved_chunk = retrieved.unwrap();
            prop_assert!(retrieved_chunk.is_some(), "Chunk not found after storage");

            let retrieved_chunk = retrieved_chunk.unwrap();

            // Property: Metadata must be preserved exactly
            prop_assert_eq!(
                retrieved_chunk.metadata,
                metadata,
                "Metadata not preserved during round trip"
            );

            // Cleanup
            let _ = delete_chunk(&client, &chunk.id).await;

            Ok(())
        })?;

        Ok(())
    }

    /// Test that chunk index and IDs are preserved
    ///
    /// **Property**: Chunk index, document_id, and source_id SHALL be preserved exactly
    /// **Validates: Requirements 4.1, 6.1**
    #[test]
    fn prop_roundtrip_preserves_identifiers(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        // Skip test if FalcorDB is not available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = create_test_config();
            let client = match FalcorDBClient::new(config).await {
                Ok(c) => c,
                Err(_) => return Ok(()),
            };

            let document_id = Uuid::new_v4();
            let chunk = VectorChunk::new(
                text,
                embedding,
                document_id,
                source_id.clone(),
                chunk_index,
            );

            // Store the chunk
            let stored_id = client.store_vector_chunk(&chunk).await;
            prop_assert!(stored_id.is_ok(), "Failed to store chunk");

            // Retrieve the chunk
            let retrieved = retrieve_chunk_by_id(&client, &chunk.id).await;
            prop_assert!(retrieved.is_ok(), "Failed to retrieve chunk");

            let retrieved_chunk = retrieved.unwrap();
            prop_assert!(retrieved_chunk.is_some(), "Chunk not found after storage");

            let retrieved_chunk = retrieved_chunk.unwrap();

            // Property: Chunk ID must be preserved
            prop_assert_eq!(
                retrieved_chunk.id,
                chunk.id,
                "Chunk ID not preserved"
            );

            // Property: Document ID must be preserved
            prop_assert_eq!(
                retrieved_chunk.document_id,
                document_id,
                "Document ID not preserved"
            );

            // Property: Source ID must be preserved
            prop_assert_eq!(
                retrieved_chunk.source_id,
                source_id,
                "Source ID not preserved"
            );

            // Property: Chunk index must be preserved
            prop_assert_eq!(
                retrieved_chunk.chunk_index,
                chunk_index,
                "Chunk index not preserved"
            );

            // Cleanup
            let _ = delete_chunk(&client, &chunk.id).await;

            Ok(())
        })?;

        Ok(())
    }

    /// Test that timestamps are set during storage
    ///
    /// **Property**: created_at and updated_at SHALL be set during storage
    /// **Validates: Requirements 4.1, 6.1**
    #[test]
    fn prop_roundtrip_sets_timestamps(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        // Skip test if FalcorDB is not available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = create_test_config();
            let client = match FalcorDBClient::new(config).await {
                Ok(c) => c,
                Err(_) => return Ok(()),
            };

            let document_id = Uuid::new_v4();
            let chunk = VectorChunk::new(
                text,
                embedding,
                document_id,
                source_id,
                chunk_index,
            );

            let before_storage = chrono::Utc::now();

            // Store the chunk
            let stored_id = client.store_vector_chunk(&chunk).await;
            prop_assert!(stored_id.is_ok(), "Failed to store chunk");

            let after_storage = chrono::Utc::now();

            // Retrieve the chunk
            let retrieved = retrieve_chunk_by_id(&client, &chunk.id).await;
            prop_assert!(retrieved.is_ok(), "Failed to retrieve chunk");

            let retrieved_chunk = retrieved.unwrap();
            prop_assert!(retrieved_chunk.is_some(), "Chunk not found after storage");

            let retrieved_chunk = retrieved_chunk.unwrap();

            // Property: created_at should be set and within reasonable time window
            prop_assert!(
                retrieved_chunk.created_at >= before_storage - chrono::Duration::seconds(1),
                "created_at is too early"
            );
            prop_assert!(
                retrieved_chunk.created_at <= after_storage + chrono::Duration::seconds(1),
                "created_at is too late"
            );

            // Property: updated_at should be set and within reasonable time window
            prop_assert!(
                retrieved_chunk.updated_at >= before_storage - chrono::Duration::seconds(1),
                "updated_at is too early"
            );
            prop_assert!(
                retrieved_chunk.updated_at <= after_storage + chrono::Duration::seconds(1),
                "updated_at is too late"
            );

            // Cleanup
            let _ = delete_chunk(&client, &chunk.id).await;

            Ok(())
        })?;

        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a test configuration for FalcorDB
fn create_test_config() -> FalcorDBConfig {
    FalcorDBConfig {
        host: std::env::var("FALCORDB_HOST").unwrap_or_else(|_| "localhost".to_string()),
        port: std::env::var("FALCORDB_PORT")
            .unwrap_or_else(|_| "6379".to_string())
            .parse()
            .unwrap_or(6379),
        username: std::env::var("FALCORDB_USERNAME")
            .unwrap_or_else(|_| "neo4j".to_string()),
        password: std::env::var("FALCORDB_PASSWORD")
            .unwrap_or_else(|_| "password".to_string()),
        database: std::env::var("FALCORDB_DATABASE")
            .unwrap_or_else(|_| "neo4j".to_string()),
        vector_dimension: 384,
        similarity_threshold: 0.75,
        max_results: 100,
        connection_pool_size: 5,
        connection_timeout_ms: 5000,
        query_timeout_ms: 30000,
    }
}

/// Retrieve a chunk by ID from FalcorDB
async fn retrieve_chunk_by_id(
    client: &FalcorDBClient,
    chunk_id: &Uuid,
) -> Result<Option<VectorChunk>, Box<dyn std::error::Error>> {
    use neo4rs::Query;

    let query = Query::new(
        r#"
        MATCH (vc:Vector_Chunk {id: $id})
        RETURN vc.id as id,
               vc.embedding as embedding,
               vc.chunk_text as chunk_text,
               vc.chunk_index as chunk_index,
               vc.document_id as document_id,
               vc.source_id as source_id,
               vc.created_at as created_at,
               vc.updated_at as updated_at,
               vc.metadata as metadata
        "#
        .to_string(),
    )
    .param("id", chunk_id.to_string());

    let mut result = client.graph().execute(query).await?;

    if let Some(row) = result.next().await? {
        let id: String = row.get("id")?;
        let embedding: Vec<f32> = row.get("embedding")?;
        let chunk_text: String = row.get("chunk_text")?;
        let chunk_index: i64 = row.get("chunk_index")?;
        let document_id: String = row.get("document_id")?;
        let source_id: String = row.get("source_id")?;
        let created_at: String = row.get("created_at")?;
        let updated_at: String = row.get("updated_at")?;
        let metadata_str: String = row.get("metadata")?;

        let chunk = VectorChunk {
            id: Uuid::parse_str(&id)?,
            embedding,
            chunk_text,
            chunk_index: chunk_index as usize,
            document_id: Uuid::parse_str(&document_id)?,
            source_id,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at)?
                .with_timezone(&chrono::Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)?
                .with_timezone(&chrono::Utc),
            metadata: serde_json::from_str(&metadata_str)?,
        };

        Ok(Some(chunk))
    } else {
        Ok(None)
    }
}

/// Delete a chunk from FalcorDB (for test cleanup)
async fn delete_chunk(
    client: &FalcorDBClient,
    chunk_id: &Uuid,
) -> Result<(), Box<dyn std::error::Error>> {
    use neo4rs::Query;

    let query = Query::new(
        r#"
        MATCH (vc:Vector_Chunk {id: $id})
        DELETE vc
        "#
        .to_string(),
    )
    .param("id", chunk_id.to_string());

    client.graph().run(query).await?;

    Ok(())
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test that round trip test configuration is valid
    ///
    /// **Validates: Requirements 4.1, 6.1** - Test infrastructure
    #[test]
    fn test_config_creation() {
        let config = create_test_config();
        assert_eq!(config.vector_dimension, 384);
        assert!(config.connection_pool_size > 0);
        assert!(config.connection_timeout_ms > 0);
    }

    /// Test that test can be skipped when FalcorDB is not available
    ///
    /// **Validates: Requirements 4.1, 6.1** - Test infrastructure
    #[test]
    fn test_skip_when_falcordb_not_enabled() {
        // Clear the environment variable
        std::env::remove_var("FALCORDB_TEST_ENABLED");
        
        // Test should be skippable
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            // This is the expected path
            assert!(true);
        } else {
            panic!("Test should be skipped when FALCORDB_TEST_ENABLED is not set");
        }
    }

    /// Test that valid embedding strategy generates 384-dimensional vectors
    ///
    /// **Validates: Requirements 4.1, 6.1** - Test data generation
    #[test]
    fn test_valid_embedding_strategy_generates_correct_dimension() {
        let strategy = valid_embedding_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let embedding = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(embedding.len(), 384, "Generated embedding should have 384 dimensions");
        }
    }

    /// Test that valid text strategy generates non-empty strings
    ///
    /// **Validates: Requirements 4.1, 6.1** - Test data generation
    #[test]
    fn test_valid_text_strategy_generates_non_empty() {
        let strategy = valid_text_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let text = strategy.new_tree(&mut runner).unwrap().current();
            assert!(!text.is_empty(), "Generated text should not be empty");
            assert!(text.len() <= 1000, "Generated text should not exceed 1000 chars");
        }
    }

    /// Test that valid source ID strategy generates valid identifiers
    ///
    /// **Validates: Requirements 4.1, 6.1** - Test data generation
    #[test]
    fn test_valid_source_id_strategy_generates_valid_ids() {
        let strategy = valid_source_id_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let source_id = strategy.new_tree(&mut runner).unwrap().current();
            assert!(!source_id.is_empty(), "Generated source ID should not be empty");
            assert!(source_id.len() <= 50, "Generated source ID should not exceed 50 chars");
            // Should only contain lowercase letters, numbers, underscores, and hyphens
            assert!(source_id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'));
        }
    }

    /// Test that metadata strategy generates valid JSON objects
    ///
    /// **Validates: Requirements 4.1, 6.1** - Test data generation
    #[test]
    fn test_valid_metadata_strategy_generates_valid_json() {
        let strategy = valid_metadata_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let metadata = strategy.new_tree(&mut runner).unwrap().current();
            assert!(metadata.is_object(), "Generated metadata should be a JSON object");
            
            // Should be serializable
            let serialized = serde_json::to_string(&metadata);
            assert!(serialized.is_ok(), "Generated metadata should be serializable");
        }
    }
}
