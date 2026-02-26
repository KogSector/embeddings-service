//! Property-based tests for embedding dimension consistency
//!
//! **Validates: Requirements 2.1**

use super::{VectorChunk};
use proptest::prelude::*;
use uuid::Uuid;

/// **Property 2: Embedding Dimension Consistency**
///
/// *For all* Vector_Chunk nodes stored in FalcorDB, the embedding vector field SHALL 
/// have exactly 384 dimensions matching the Embeddings_Service output.
///
/// **Validates: Requirements 2.1**
///
/// This property test validates that:
/// 1. Valid 384-dimensional embeddings are accepted
/// 2. Non-384 dimensional embeddings are rejected
/// 3. Dimension validation is consistent across all operations
/// 4. Empty embeddings are rejected
/// 5. Validation occurs before storage operations

// Strategy for generating valid 384-dimensional embeddings
fn valid_embedding_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32, // Typical normalized embedding range
        384..=384 // Exactly 384 dimensions
    )
}

// Strategy for generating invalid embeddings with wrong dimensions
fn invalid_dimension_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32,
        0..=2048 // Various dimensions, excluding 384
    ).prop_filter("Must not be 384 dimensions", |v| v.len() != 384)
}

// Strategy for generating valid chunk text
fn valid_text_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 .,!?-]{1,1000}").unwrap()
}

// Strategy for generating valid source IDs
fn valid_source_id_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z0-9_-]{1,50}").unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test that valid 384-dimensional embeddings pass validation
    ///
    /// **Property**: All embeddings with exactly 384 dimensions SHALL be accepted
    /// **Validates: Requirement 2.1**
    #[test]
    fn prop_valid_384_dimension_embeddings_accepted(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        let document_id = Uuid::new_v4();
        
        let chunk = VectorChunk::new(
            text,
            embedding.clone(),
            document_id,
            source_id,
            chunk_index,
        );

        // Property: 384-dimensional embeddings must pass validation
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_ok(),
            "Valid 384-dimensional embedding should pass validation, got error: {:?}",
            validation_result.err()
        );

        // Property: Embedding dimension must be exactly 384
        prop_assert_eq!(
            chunk.embedding.len(),
            384,
            "Embedding dimension must be exactly 384"
        );

        // Property: Embedding values should be preserved
        prop_assert_eq!(
            chunk.embedding,
            embedding,
            "Embedding values should be preserved exactly"
        );
    }

    /// Test that non-384 dimensional embeddings are rejected
    ///
    /// **Property**: All embeddings with dimension != 384 SHALL be rejected
    /// **Validates: Requirement 2.1**
    #[test]
    fn prop_invalid_dimension_embeddings_rejected(
        embedding in invalid_dimension_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        let document_id = Uuid::new_v4();
        
        let chunk = VectorChunk::new(
            text,
            embedding.clone(),
            document_id,
            source_id,
            chunk_index,
        );

        // Property: Non-384 dimensional embeddings must fail validation
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_err(),
            "Non-384 dimensional embedding (dim={}) should fail validation",
            embedding.len()
        );

        // Property: Error message should mention dimension mismatch
        let error_msg = validation_result.unwrap_err().to_string();
        prop_assert!(
            error_msg.contains("Invalid embedding dimension") || 
            error_msg.contains("expected 384"),
            "Error message should indicate dimension mismatch, got: {}",
            error_msg
        );
    }

    /// Test that empty embeddings are rejected
    ///
    /// **Property**: Empty embeddings (0 dimensions) SHALL be rejected
    /// **Validates: Requirement 2.1**
    #[test]
    fn prop_empty_embeddings_rejected(
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        let document_id = Uuid::new_v4();
        let empty_embedding: Vec<f32> = Vec::new();
        
        let chunk = VectorChunk::new(
            text,
            empty_embedding,
            document_id,
            source_id,
            chunk_index,
        );

        // Property: Empty embeddings must fail validation
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_err(),
            "Empty embedding should fail validation"
        );

        // Property: Dimension should be 0
        prop_assert_eq!(
            chunk.embedding.len(),
            0,
            "Empty embedding should have 0 dimensions"
        );
    }

    /// Test dimension consistency across different embedding value ranges
    ///
    /// **Property**: Dimension validation is independent of embedding values
    /// **Validates: Requirement 2.1**
    #[test]
    fn prop_dimension_validation_independent_of_values(
        value_range in -1000.0f32..=1000.0f32,
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
    ) {
        let document_id = Uuid::new_v4();
        
        // Create embedding with all values in the same range
        let embedding: Vec<f32> = vec![value_range; 384];
        
        let chunk = VectorChunk::new(
            text,
            embedding,
            document_id,
            source_id,
            0,
        );

        // Property: Dimension validation should pass regardless of values
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_ok(),
            "384-dimensional embedding should pass validation regardless of value range ({}), got error: {:?}",
            value_range,
            validation_result.err()
        );
    }

    /// Test that dimension validation is consistent across multiple chunks
    ///
    /// **Property**: All chunks with 384 dimensions pass, all others fail
    /// **Validates: Requirement 2.1**
    #[test]
    fn prop_dimension_validation_consistency(
        dimensions in prop::collection::vec(1usize..=1024usize, 1..10),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
    ) {
        let document_id = Uuid::new_v4();
        
        for (idx, &dim) in dimensions.iter().enumerate() {
            let embedding: Vec<f32> = vec![0.5; dim];
            
            let chunk = VectorChunk::new(
                text.clone(),
                embedding,
                document_id,
                source_id.clone(),
                idx,
            );

            let validation_result = chunk.validate();
            
            if dim == 384 {
                // Property: Exactly 384 dimensions should pass
                prop_assert!(
                    validation_result.is_ok(),
                    "Chunk with 384 dimensions should pass validation"
                );
            } else {
                // Property: Any other dimension should fail
                prop_assert!(
                    validation_result.is_err(),
                    "Chunk with {} dimensions should fail validation",
                    dim
                );
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test that exactly 384 dimensions pass validation
    ///
    /// **Validates: Requirement 2.1** - Vector_Chunk SHALL contain an embedding 
    /// vector field with dimensionality matching the Embeddings_Service output (384 dimensions)
    #[test]
    fn test_384_dimensions_pass_validation() {
        let embedding = vec![0.1; 384];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_ok(), "384-dimensional embedding should pass validation");
        assert_eq!(chunk.embedding.len(), 384);
    }

    /// Test that 383 dimensions fail validation
    ///
    /// **Validates: Requirement 2.1** - Non-384 dimensional vectors are rejected
    #[test]
    fn test_383_dimensions_fail_validation() {
        let embedding = vec![0.1; 383];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "383-dimensional embedding should fail validation");
        assert!(result.unwrap_err().to_string().contains("Invalid embedding dimension"));
    }

    /// Test that 385 dimensions fail validation
    ///
    /// **Validates: Requirement 2.1** - Non-384 dimensional vectors are rejected
    #[test]
    fn test_385_dimensions_fail_validation() {
        let embedding = vec![0.1; 385];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "385-dimensional embedding should fail validation");
        assert!(result.unwrap_err().to_string().contains("Invalid embedding dimension"));
    }

    /// Test that 0 dimensions (empty) fail validation
    ///
    /// **Validates: Requirement 2.1** - Empty embeddings are rejected
    #[test]
    fn test_empty_embedding_fails_validation() {
        let embedding: Vec<f32> = Vec::new();
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "Empty embedding should fail validation");
        assert_eq!(chunk.embedding.len(), 0);
    }

    /// Test that 512 dimensions fail validation
    ///
    /// **Validates: Requirement 2.1** - Only 384 dimensions are accepted
    #[test]
    fn test_512_dimensions_fail_validation() {
        let embedding = vec![0.1; 512];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "512-dimensional embedding should fail validation");
        assert!(result.unwrap_err().to_string().contains("expected 384"));
    }

    /// Test that 256 dimensions fail validation
    ///
    /// **Validates: Requirement 2.1** - Only 384 dimensions are accepted
    #[test]
    fn test_256_dimensions_fail_validation() {
        let embedding = vec![0.1; 256];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "256-dimensional embedding should fail validation");
    }

    /// Test that 1536 dimensions (OpenAI) fail validation
    ///
    /// **Validates: Requirement 2.1** - Only 384 dimensions are accepted
    #[test]
    fn test_1536_dimensions_fail_validation() {
        let embedding = vec![0.1; 1536];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "1536-dimensional embedding should fail validation");
    }

    /// Test dimension validation with extreme values
    ///
    /// **Validates: Requirement 2.1** - Dimension validation is independent of values
    #[test]
    fn test_dimension_validation_with_extreme_values() {
        // Test with very large values
        let large_embedding = vec![f32::MAX / 2.0; 384];
        let large_chunk = VectorChunk::new(
            "Test chunk".to_string(),
            large_embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );
        assert!(large_chunk.validate().is_ok(), "384-dim with large values should pass");

        // Test with very small values
        let small_embedding = vec![f32::MIN / 2.0; 384];
        let small_chunk = VectorChunk::new(
            "Test chunk".to_string(),
            small_embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );
        assert!(small_chunk.validate().is_ok(), "384-dim with small values should pass");

        // Test with zero values
        let zero_embedding = vec![0.0; 384];
        let zero_chunk = VectorChunk::new(
            "Test chunk".to_string(),
            zero_embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );
        assert!(zero_chunk.validate().is_ok(), "384-dim with zero values should pass");
    }

    /// Test dimension validation with mixed positive and negative values
    ///
    /// **Validates: Requirement 2.1** - Dimension validation is independent of value signs
    #[test]
    fn test_dimension_validation_with_mixed_values() {
        let mut mixed_embedding = Vec::with_capacity(384);
        for i in 0..384 {
            mixed_embedding.push(if i % 2 == 0 { 0.5 } else { -0.5 });
        }

        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            mixed_embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        assert!(chunk.validate().is_ok(), "384-dim with mixed values should pass");
        assert_eq!(chunk.embedding.len(), 384);
    }

    /// Test that validation error message is descriptive
    ///
    /// **Validates: Requirement 2.1** - Error messages should be clear
    #[test]
    fn test_validation_error_message_is_descriptive() {
        let embedding = vec![0.1; 512];
        let chunk = VectorChunk::new(
            "Test chunk".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid embedding dimension"));
        assert!(error_msg.contains("expected 384"));
        assert!(error_msg.contains("got 512"));
    }
}
