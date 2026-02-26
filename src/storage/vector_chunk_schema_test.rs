//! Property-based tests for Vector Chunk schema validation
//!
//! **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.2**

use super::{VectorChunk};
use proptest::prelude::*;
use uuid::Uuid;
use chrono::Utc;

/// **Property 2: Embedding Dimension Consistency**
///
/// *For any* Vector_Chunk node stored in FalcorDB, the embedding vector field SHALL 
/// have dimensionality exactly matching the Embeddings_Service output (384 dimensions).
///
/// **Validates: Requirements 2.1**
///
/// This property test validates that:
/// 1. Valid 384-dimensional embeddings pass validation
/// 2. Invalid dimensions (< 384 or > 384) fail validation
/// 3. Dimension validation is consistent across all vector chunks
/// 4. The validation error message clearly indicates the dimension mismatch

// Strategy for generating vector chunks with various embedding dimensions
fn vector_chunk_with_dimension_strategy(dimension: usize) -> impl Strategy<Value = VectorChunk> {
    (
        prop::collection::vec(prop::num::f32::NORMAL, dimension..=dimension),
        prop::string::string_regex("[a-zA-Z0-9 ]{1,100}").unwrap(),
        0usize..1000usize,
        prop::string::string_regex("[a-z0-9-]{1,50}").unwrap(),
    ).prop_map(|(embedding, text, chunk_index, source_id)| {
        VectorChunk::new(
            text,
            embedding,
            Uuid::new_v4(),
            source_id,
            chunk_index,
        )
    })
}

// Strategy for generating valid 384-dimensional vector chunks
fn valid_vector_chunk_strategy() -> impl Strategy<Value = VectorChunk> {
    vector_chunk_with_dimension_strategy(384)
}

// Strategy for generating invalid dimension sizes
fn invalid_dimension_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        1usize..384usize,      // Too small
        385usize..2048usize,   // Too large
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test that valid 384-dimensional embeddings always pass validation
    ///
    /// This validates Requirement 2.1: The Vector_Chunk SHALL contain an embedding 
    /// vector field with dimensionality matching the Embeddings_Service output (384 dimensions)
    #[test]
    fn prop_valid_384_dimension_passes_validation(chunk in valid_vector_chunk_strategy()) {
        // Property: All 384-dimensional embeddings should pass validation
        let result = chunk.validate();
        
        prop_assert!(
            result.is_ok(),
            "384-dimensional embedding should pass validation, but got error: {:?}",
            result.err()
        );
        
        // Verify the dimension is exactly 384
        prop_assert_eq!(
            chunk.embedding.len(),
            384,
            "Embedding dimension should be exactly 384"
        );
    }

    /// Test that invalid dimensions always fail validation
    ///
    /// This validates that dimension validation catches all non-384 dimensions
    #[test]
    fn prop_invalid_dimension_fails_validation(
        dimension in invalid_dimension_strategy(),
        text in prop::string::string_regex("[a-zA-Z0-9 ]{1,100}").unwrap(),
        chunk_index in 0usize..1000usize,
        source_id in prop::string::string_regex("[a-z0-9-]{1,50}").unwrap(),
    ) {
        let embedding = vec![0.5f32; dimension];
        let chunk = VectorChunk::new(
            text,
            embedding,
            Uuid::new_v4(),
            source_id,
            chunk_index,
        );
        
        // Property: All non-384 dimensional embeddings should fail validation
        let result = chunk.validate();
        
        prop_assert!(
            result.is_err(),
            "Embedding with dimension {} should fail validation",
            dimension
        );
        
        // Verify the error message mentions dimension
        let error_msg = result.unwrap_err().to_string();
        prop_assert!(
            error_msg.contains("Invalid embedding dimension") || 
            error_msg.contains("dimension"),
            "Error message should mention dimension issue, got: {}",
            error_msg
        );
    }

    /// Test that dimension validation is consistent across different chunk properties
    ///
    /// This validates that dimension validation works regardless of other chunk properties
    #[test]
    fn prop_dimension_validation_independent_of_other_fields(
        dimension in 1usize..2048usize,
        text in prop::string::string_regex("[a-zA-Z0-9 ]{1,100}").unwrap(),
        chunk_index in 0usize..1000usize,
        source_id in prop::string::string_regex("[a-z0-9-]{1,50}").unwrap(),
    ) {
        let embedding = vec![0.5f32; dimension];
        let chunk = VectorChunk::new(
            text,
            embedding,
            Uuid::new_v4(),
            source_id,
            chunk_index,
        );
        
        let result = chunk.validate();
        
        // Property: Validation result depends ONLY on dimension being 384
        if dimension == 384 {
            prop_assert!(
                result.is_ok(),
                "384-dimensional embedding should pass regardless of other fields"
            );
        } else {
            prop_assert!(
                result.is_err(),
                "Non-384 dimensional embedding should fail regardless of other fields"
            );
        }
    }

    /// Test that dimension validation error provides expected and actual dimensions
    ///
    /// This validates that error messages are informative for debugging
    #[test]
    fn prop_dimension_error_includes_expected_and_actual(
        dimension in prop_oneof![
            1usize..384usize,
            385usize..2048usize,
        ]
    ) {
        let embedding = vec![0.5f32; dimension];
        let chunk = VectorChunk::new(
            "Test text".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );
        
        let result = chunk.validate();
        
        prop_assert!(result.is_err(), "Non-384 dimension should fail");
        
        let error_msg = result.unwrap_err().to_string();
        
        // Property: Error message should include both expected (384) and actual dimension
        prop_assert!(
            error_msg.contains("384") || error_msg.contains("expected"),
            "Error should mention expected dimension 384, got: {}",
            error_msg
        );
        
        prop_assert!(
            error_msg.contains(&dimension.to_string()) || error_msg.contains("got"),
            "Error should mention actual dimension {}, got: {}",
            dimension,
            error_msg
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test exact 384 dimension passes
    ///
    /// Validates Requirement 2.1: Vector_Chunk SHALL contain 384-dimensional embedding
    #[test]
    fn test_exact_384_dimension_passes() {
        let embedding = vec![0.1f32; 384];
        let chunk = VectorChunk::new(
            "Test chunk text".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_ok(), "384-dimensional embedding should pass validation");
        assert_eq!(chunk.embedding.len(), 384);
    }

    /// Test dimension too small fails
    #[test]
    fn test_dimension_too_small_fails() {
        let embedding = vec![0.1f32; 256]; // Common alternative dimension
        let chunk = VectorChunk::new(
            "Test chunk text".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "256-dimensional embedding should fail validation");
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid embedding dimension"));
        assert!(error_msg.contains("384"));
        assert!(error_msg.contains("256"));
    }

    /// Test dimension too large fails
    #[test]
    fn test_dimension_too_large_fails() {
        let embedding = vec![0.1f32; 512]; // Common alternative dimension
        let chunk = VectorChunk::new(
            "Test chunk text".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "512-dimensional embedding should fail validation");
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid embedding dimension"));
        assert!(error_msg.contains("384"));
        assert!(error_msg.contains("512"));
    }

    /// Test empty embedding fails
    #[test]
    fn test_empty_embedding_fails() {
        let embedding = vec![];
        let chunk = VectorChunk::new(
            "Test chunk text".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "Empty embedding should fail validation");
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid embedding dimension"));
    }

    /// Test single dimension fails
    #[test]
    fn test_single_dimension_fails() {
        let embedding = vec![0.1f32];
        let chunk = VectorChunk::new(
            "Test chunk text".to_string(),
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err(), "Single-dimensional embedding should fail validation");
    }

    /// Test dimension validation with various embedding values
    #[test]
    fn test_dimension_validation_with_various_values() {
        // Test with zeros
        let chunk_zeros = VectorChunk::new(
            "Test".to_string(),
            vec![0.0f32; 384],
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_zeros.validate().is_ok());

        // Test with ones
        let chunk_ones = VectorChunk::new(
            "Test".to_string(),
            vec![1.0f32; 384],
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_ones.validate().is_ok());

        // Test with negative values
        let chunk_negative = VectorChunk::new(
            "Test".to_string(),
            vec![-0.5f32; 384],
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_negative.validate().is_ok());

        // Test with mixed values
        let mut mixed_embedding = vec![0.0f32; 384];
        for (i, val) in mixed_embedding.iter_mut().enumerate() {
            *val = (i as f32) / 384.0;
        }
        let chunk_mixed = VectorChunk::new(
            "Test".to_string(),
            mixed_embedding,
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_mixed.validate().is_ok());
    }

    /// Test that dimension validation happens before other validations
    #[test]
    fn test_dimension_validation_priority() {
        // Create chunk with both invalid dimension AND empty text
        let embedding = vec![0.1f32; 256];
        let chunk = VectorChunk::new(
            "".to_string(), // Empty text (also invalid)
            embedding,
            Uuid::new_v4(),
            "test-source".to_string(),
            0,
        );

        let result = chunk.validate();
        assert!(result.is_err());
        
        // The error should be about dimension since it's checked first
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("Invalid embedding dimension"),
            "Dimension validation should happen first, got: {}",
            error_msg
        );
    }

    /// Test common alternative embedding dimensions all fail
    #[test]
    fn test_common_alternative_dimensions_fail() {
        let common_dimensions = vec![128, 256, 512, 768, 1024, 1536];
        
        for dim in common_dimensions {
            let embedding = vec![0.1f32; dim];
            let chunk = VectorChunk::new(
                "Test".to_string(),
                embedding,
                Uuid::new_v4(),
                "source".to_string(),
                0,
            );
            
            assert!(
                chunk.validate().is_err(),
                "Dimension {} should fail validation (only 384 is valid)",
                dim
            );
        }
    }

    /// Test boundary cases around 384
    #[test]
    fn test_boundary_dimensions() {
        // 383 should fail
        let chunk_383 = VectorChunk::new(
            "Test".to_string(),
            vec![0.1f32; 383],
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_383.validate().is_err(), "383 dimensions should fail");

        // 384 should pass
        let chunk_384 = VectorChunk::new(
            "Test".to_string(),
            vec![0.1f32; 384],
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_384.validate().is_ok(), "384 dimensions should pass");

        // 385 should fail
        let chunk_385 = VectorChunk::new(
            "Test".to_string(),
            vec![0.1f32; 385],
            Uuid::new_v4(),
            "source".to_string(),
            0,
        );
        assert!(chunk_385.validate().is_err(), "385 dimensions should fail");
    }
}
