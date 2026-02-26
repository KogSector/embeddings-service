//! Property-based tests for batch insert atomicity
//!
//! **Validates: Requirements 10.3, 10.5**
//!
//! This test validates that batch vector insertion operations are atomic - either all chunks
//! are stored successfully or none are stored if any validation fails.
//!
//! **Requirements:**
//! - 10.3: THE Unified_Processor SHALL support batch vector insertion for multiple chunks
//! - 10.5: WHEN inserting vectors in batch, THE Embeddings_Service SHALL use transaction 
//!         boundaries for atomicity

use super::{FalcorDBClient, FalcorDBConfig, VectorChunk};
use proptest::prelude::*;
use uuid::Uuid;

/// **Property 17: Batch Insert Atomicity**
///
/// *For all* batch insert operations, the operation SHALL be atomic:
/// - If all chunks are valid, all chunks SHALL be stored
/// - If any chunk is invalid, NO chunks SHALL be stored
/// - Partial storage SHALL NOT occur
/// - Transaction boundaries SHALL ensure all-or-nothing behavior
///
/// **Validates: Requirements 10.3, 10.5**
///
/// This property test validates that:
/// 1. Valid batches are stored completely
/// 2. Invalid batches (with any invalid chunk) result in zero storage
/// 3. No partial storage occurs when validation fails
/// 4. Transaction rollback works correctly
/// 5. Database state is consistent after failed batch operations

// Strategy for generating valid 384-dimensional embeddings
fn valid_embedding_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32, // Typical normalized embedding range
        384..=384 // Exactly 384 dimensions
    )
}

// Strategy for generating INVALID embeddings (wrong dimension)
fn invalid_embedding_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32,
        prop::sample::select(vec![0, 1, 128, 256, 512, 1024]) // Wrong dimensions
    )
}

// Strategy for generating valid chunk text (non-empty, reasonable length)
fn valid_text_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 .,!?\\-\\n]{1,500}").unwrap()
}

// Strategy for generating valid source IDs
fn valid_source_id_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z0-9_\\-]{1,30}").unwrap()
}

// Strategy for generating a batch of valid chunks
fn valid_chunk_batch_strategy(
    batch_size: usize,
) -> impl Strategy<Value = Vec<VectorChunk>> {
    let document_id = Uuid::new_v4();
    let source_id = valid_source_id_strategy();
    
    (source_id, prop::collection::vec(
        (valid_embedding_strategy(), valid_text_strategy()),
        batch_size..=batch_size
    )).prop_map(move |(source_id, chunks)| {
        chunks.into_iter().enumerate().map(|(idx, (embedding, text))| {
            VectorChunk::new(
                text,
                embedding,
                document_id,
                source_id.clone(),
                idx,
            )
        }).collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Test that valid batch inserts store all chunks atomically
    ///
    /// **Property**: All valid chunks in a batch SHALL be stored successfully
    /// **Validates: Requirements 10.3, 10.5**
    #[test]
    fn prop_valid_batch_stores_all_chunks(
        batch_size in 2usize..10usize,
    ) {
        // Skip test if FalcorDB is not available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Setup: Create client
            let config = create_test_config();
            let client = match FalcorDBClient::new(config).await {
                Ok(c) => c,
                Err(_) => return Ok(()), // Skip if connection fails
            };

            // Generate valid batch
            let mut runner = proptest::test_runner::TestRunner::default();
            let strategy = valid_chunk_batch_strategy(batch_size);
            let chunks = strategy.new_tree(&mut runner).unwrap().current();

            let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();

            // Store the batch
            let result = client.batch_store_chunks(chunks).await;
            prop_assert!(result.is_ok(), "Valid batch should store successfully: {:?}", result.err());

            let stored_ids = result.unwrap();
            prop_assert_eq!(
                stored_ids.len(),
                batch_size,
                "All chunks should be stored"
            );

            // Verify all chunks exist in database
            for chunk_id in &chunk_ids {
                let exists = chunk_exists(&client, chunk_id).await;
                prop_assert!(exists.unwrap_or(false), "Chunk {} should exist after batch insert", chunk_id);
            }

            // Cleanup
            for chunk_id in &chunk_ids {
                let _ = delete_chunk(&client, chunk_id).await;
            }

            Ok(())
        })?;

        Ok(())
    }

    /// Test that batch with invalid dimension fails atomically (no chunks stored)
    ///
    /// **Property**: If any chunk has invalid dimension, NO chunks SHALL be stored
    /// **Validates: Requirements 10.3, 10.5**
    #[test]
    fn prop_invalid_dimension_batch_stores_nothing(
        valid_count in 1usize..5usize,
        invalid_position in 0usize..5usize,
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

            // Generate batch with one invalid chunk
            let mut runner = proptest::test_runner::TestRunner::default();
            let valid_strategy = valid_chunk_batch_strategy(valid_count);
            let mut chunks = valid_strategy.new_tree(&mut runner).unwrap().current();

            // Insert invalid chunk at specified position
            let invalid_pos = invalid_position % (chunks.len() + 1);
            let invalid_embedding_strat = invalid_embedding_strategy();
            let invalid_embedding = invalid_embedding_strat.new_tree(&mut runner).unwrap().current();
            let text_strat = valid_text_strategy();
            let text = text_strat.new_tree(&mut runner).unwrap().current();
            
            let invalid_chunk = VectorChunk::new(
                text,
                invalid_embedding,
                Uuid::new_v4(),
                "test-source".to_string(),
                invalid_pos,
            );

            chunks.insert(invalid_pos, invalid_chunk);

            let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();

            // Attempt to store the batch (should fail)
            let result = client.batch_store_chunks(chunks).await;
            prop_assert!(result.is_err(), "Batch with invalid dimension should fail");

            // Property: NO chunks should be stored (atomicity)
            for chunk_id in &chunk_ids {
                let exists = chunk_exists(&client, chunk_id).await;
                prop_assert!(
                    !exists.unwrap_or(true),
                    "Chunk {} should NOT exist after failed batch insert (atomicity violation)",
                    chunk_id
                );
            }

            // Cleanup (in case of test failure)
            for chunk_id in &chunk_ids {
                let _ = delete_chunk(&client, chunk_id).await;
            }

            Ok(())
        })?;

        Ok(())
    }

    /// Test that batch with empty text fails atomically (no chunks stored)
    ///
    /// **Property**: If any chunk has empty text, NO chunks SHALL be stored
    /// **Validates: Requirements 10.3, 10.5**
    #[test]
    fn prop_empty_text_batch_stores_nothing(
        valid_count in 1usize..5usize,
        invalid_position in 0usize..5usize,
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

            // Generate batch with one chunk having empty text
            let mut runner = proptest::test_runner::TestRunner::default();
            let valid_strategy = valid_chunk_batch_strategy(valid_count);
            let mut chunks = valid_strategy.new_tree(&mut runner).unwrap().current();

            // Insert chunk with empty text at specified position
            let invalid_pos = invalid_position % (chunks.len() + 1);
            let embedding_strat = valid_embedding_strategy();
            let embedding = embedding_strat.new_tree(&mut runner).unwrap().current();
            
            let invalid_chunk = VectorChunk::new(
                "".to_string(), // Empty text
                embedding,
                Uuid::new_v4(),
                "test-source".to_string(),
                invalid_pos,
            );

            chunks.insert(invalid_pos, invalid_chunk);

            let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();

            // Attempt to store the batch (should fail)
            let result = client.batch_store_chunks(chunks).await;
            prop_assert!(result.is_err(), "Batch with empty text should fail");

            // Property: NO chunks should be stored (atomicity)
            for chunk_id in &chunk_ids {
                let exists = chunk_exists(&client, chunk_id).await;
                prop_assert!(
                    !exists.unwrap_or(true),
                    "Chunk {} should NOT exist after failed batch insert (atomicity violation)",
                    chunk_id
                );
            }

            // Cleanup (in case of test failure)
            for chunk_id in &chunk_ids {
                let _ = delete_chunk(&client, chunk_id).await;
            }

            Ok(())
        })?;

        Ok(())
    }

    /// Test that batch insert with mixed valid/invalid chunks is atomic
    ///
    /// **Property**: Batch with ANY invalid chunk SHALL store ZERO chunks
    /// **Validates: Requirements 10.3, 10.5**
    #[test]
    fn prop_mixed_batch_atomicity(
        valid_before in 1usize..3usize,
        valid_after in 1usize..3usize,
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

            // Generate batch: valid chunks, then invalid, then more valid chunks
            let mut runner = proptest::test_runner::TestRunner::default();
            
            let before_strategy = valid_chunk_batch_strategy(valid_before);
            let before_chunks = before_strategy.new_tree(&mut runner).unwrap().current();
            
            let after_strategy = valid_chunk_batch_strategy(valid_after);
            let after_chunks = after_strategy.new_tree(&mut runner).unwrap().current();

            // Create invalid chunk (wrong dimension)
            let invalid_embedding_strat = invalid_embedding_strategy();
            let invalid_embedding = invalid_embedding_strat.new_tree(&mut runner).unwrap().current();
            let text_strat = valid_text_strategy();
            let text = text_strat.new_tree(&mut runner).unwrap().current();
            
            let invalid_chunk = VectorChunk::new(
                text,
                invalid_embedding,
                Uuid::new_v4(),
                "test-source".to_string(),
                valid_before,
            );

            // Combine: before + invalid + after
            let mut chunks = before_chunks;
            chunks.push(invalid_chunk);
            chunks.extend(after_chunks);

            let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();
            let total_chunks = chunk_ids.len();

            // Attempt to store the batch (should fail)
            let result = client.batch_store_chunks(chunks).await;
            prop_assert!(
                result.is_err(),
                "Batch with invalid chunk in middle should fail"
            );

            // Property: NO chunks should be stored, including valid ones before/after
            let mut stored_count = 0;
            for chunk_id in &chunk_ids {
                if chunk_exists(&client, chunk_id).await.unwrap_or(false) {
                    stored_count += 1;
                }
            }

            prop_assert_eq!(
                stored_count,
                0,
                "Atomicity violation: {} out of {} chunks were stored despite batch failure",
                stored_count,
                total_chunks
            );

            // Cleanup (in case of test failure)
            for chunk_id in &chunk_ids {
                let _ = delete_chunk(&client, chunk_id).await;
            }

            Ok(())
        })?;

        Ok(())
    }

    /// Test that empty batch is handled correctly
    ///
    /// **Property**: Empty batch SHALL return empty result without error
    /// **Validates: Requirements 10.3, 10.5**
    #[test]
    fn prop_empty_batch_succeeds() {
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

            // Store empty batch
            let result = client.batch_store_chunks(vec![]).await;
            prop_assert!(result.is_ok(), "Empty batch should succeed");

            let stored_ids = result.unwrap();
            prop_assert_eq!(stored_ids.len(), 0, "Empty batch should return empty result");

            Ok(())
        })?;

        Ok(())
    }

    /// Test that large valid batches are stored atomically
    ///
    /// **Property**: Large batches SHALL be stored completely or not at all
    /// **Validates: Requirements 10.3, 10.5**
    #[test]
    fn prop_large_batch_atomicity(
        batch_size in 10usize..20usize,
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

            // Generate large valid batch
            let mut runner = proptest::test_runner::TestRunner::default();
            let strategy = valid_chunk_batch_strategy(batch_size);
            let chunks = strategy.new_tree(&mut runner).unwrap().current();

            let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();

            // Store the batch
            let result = client.batch_store_chunks(chunks).await;
            prop_assert!(
                result.is_ok(),
                "Large valid batch should store successfully: {:?}",
                result.err()
            );

            let stored_ids = result.unwrap();
            prop_assert_eq!(
                stored_ids.len(),
                batch_size,
                "All chunks in large batch should be stored"
            );

            // Verify all chunks exist
            let mut found_count = 0;
            for chunk_id in &chunk_ids {
                if chunk_exists(&client, chunk_id).await.unwrap_or(false) {
                    found_count += 1;
                }
            }

            prop_assert_eq!(
                found_count,
                batch_size,
                "All {} chunks should exist in database, found {}",
                batch_size,
                found_count
            );

            // Cleanup
            for chunk_id in &chunk_ids {
                let _ = delete_chunk(&client, chunk_id).await;
            }

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

/// Check if a chunk exists in FalcorDB
async fn chunk_exists(
    client: &FalcorDBClient,
    chunk_id: &Uuid,
) -> Result<bool, Box<dyn std::error::Error>> {
    use neo4rs::Query;

    let query = Query::new(
        r#"
        MATCH (vc:Vector_Chunk {id: $id})
        RETURN count(vc) as count
        "#
        .to_string(),
    )
    .param("id", chunk_id.to_string());

    let mut result = client.graph().execute(query).await?;

    if let Some(row) = result.next().await? {
        let count: i64 = row.get("count")?;
        Ok(count > 0)
    } else {
        Ok(false)
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

    /// Test that batch atomicity test configuration is valid
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test infrastructure
    #[test]
    fn test_config_creation() {
        let config = create_test_config();
        assert_eq!(config.vector_dimension, 384);
        assert!(config.connection_pool_size > 0);
        assert!(config.connection_timeout_ms > 0);
    }

    /// Test that valid embedding strategy generates 384-dimensional vectors
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test data generation
    #[test]
    fn test_valid_embedding_strategy_generates_correct_dimension() {
        let strategy = valid_embedding_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let embedding = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(embedding.len(), 384, "Generated embedding should have 384 dimensions");
        }
    }

    /// Test that invalid embedding strategy generates wrong dimensions
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test data generation
    #[test]
    fn test_invalid_embedding_strategy_generates_wrong_dimension() {
        let strategy = invalid_embedding_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let embedding = strategy.new_tree(&mut runner).unwrap().current();
            assert_ne!(
                embedding.len(),
                384,
                "Invalid embedding should not have 384 dimensions"
            );
        }
    }

    /// Test that valid text strategy generates non-empty strings
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test data generation
    #[test]
    fn test_valid_text_strategy_generates_non_empty() {
        let strategy = valid_text_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let text = strategy.new_tree(&mut runner).unwrap().current();
            assert!(!text.is_empty(), "Generated text should not be empty");
            assert!(text.len() <= 500, "Generated text should not exceed 500 chars");
        }
    }

    /// Test that valid source ID strategy generates valid identifiers
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test data generation
    #[test]
    fn test_valid_source_id_strategy_generates_valid_ids() {
        let strategy = valid_source_id_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..10 {
            let source_id = strategy.new_tree(&mut runner).unwrap().current();
            assert!(!source_id.is_empty(), "Generated source ID should not be empty");
            assert!(source_id.len() <= 30, "Generated source ID should not exceed 30 chars");
            // Should only contain lowercase letters, numbers, underscores, and hyphens
            assert!(source_id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'));
        }
    }

    /// Test that valid chunk batch strategy generates correct batch size
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test data generation
    #[test]
    fn test_valid_chunk_batch_strategy_generates_correct_size() {
        let batch_size = 5;
        let strategy = valid_chunk_batch_strategy(batch_size);
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..5 {
            let chunks = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(chunks.len(), batch_size, "Generated batch should have correct size");
            
            // All chunks should have same document_id
            let first_doc_id = chunks[0].document_id;
            for chunk in &chunks {
                assert_eq!(chunk.document_id, first_doc_id, "All chunks should have same document_id");
            }
            
            // Chunk indices should be sequential
            for (idx, chunk) in chunks.iter().enumerate() {
                assert_eq!(chunk.chunk_index, idx, "Chunk indices should be sequential");
            }
        }
    }

    /// Test that chunk_exists helper can be called
    ///
    /// **Validates: Requirements 10.3, 10.5** - Test infrastructure
    #[tokio::test]
    async fn test_chunk_exists_helper() {
        // Skip if no test database available
        if std::env::var("FALCORDB_TEST_ENABLED").is_err() {
            return;
        }

        let config = create_test_config();
        let client = FalcorDBClient::new(config).await;
        
        if let Ok(client) = client {
            let non_existent_id = Uuid::new_v4();
            let exists = chunk_exists(&client, &non_existent_id).await;
            
            // Should return Ok(false) for non-existent chunk
            assert!(exists.is_ok(), "chunk_exists should not error for non-existent chunk");
            assert!(!exists.unwrap(), "Non-existent chunk should return false");
        }
    }
}
