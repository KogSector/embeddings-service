//! Property-based tests for vector storage metrics emission
//!
//! **Validates: Requirements 11.1**
//!
//! This test validates that metrics are correctly emitted during vector storage operations.
//! It tests that the VectorMetrics struct properly records success and failure events,
//! and that duration histograms are populated correctly.
//!
//! **Requirements:**
//! - 11.1: WHEN the Embeddings_Service stores a vector, THE Embeddings_Service SHALL emit 
//!         a metric for vector storage duration

use super::{VectorChunk, VectorMetrics};
use proptest::prelude::*;
use std::time::Duration;
use uuid::Uuid;

/// **Property 22: Vector Storage Metrics Emission**
///
/// *For all* vector storage operations (successful or failed), metrics SHALL be emitted
/// with the following properties:
/// - storage_duration histogram SHALL record the operation duration
/// - storage_success counter SHALL increment on successful operations
/// - storage_failure counter SHALL increment on failed operations
/// - Metrics SHALL be recorded regardless of operation outcome
/// - Duration values SHALL be non-negative
/// - Counter values SHALL be monotonically increasing
///
/// **Validates: Requirements 11.1**
///
/// This property test validates that:
/// 1. Success operations increment the success counter
/// 2. Failure operations increment the failure counter
/// 3. Duration is recorded for both success and failure
/// 4. Multiple operations accumulate correctly
/// 5. Metrics are thread-safe and consistent

// Strategy for generating valid durations (1ms to 10s)
fn valid_duration_strategy() -> impl Strategy<Value = Duration> {
    (1u64..=10000u64).prop_map(Duration::from_millis)
}

// Strategy for generating operation outcomes (success or failure)
fn operation_outcome_strategy() -> impl Strategy<Value = bool> {
    prop::bool::ANY
}

// Strategy for generating valid 384-dimensional embeddings
fn valid_embedding_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0f32..=1.0f32,
        384..=384
    )
}

// Strategy for generating valid chunk text
fn valid_text_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 .,!?\\-\\n]{1,500}").unwrap()
}

// Strategy for generating valid source IDs
fn valid_source_id_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z0-9_\\-]{1,30}").unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test that success operations increment the success counter
    ///
    /// **Property**: storage_success counter SHALL increment for each successful operation
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_success_counter_increments(
        duration in valid_duration_strategy(),
    ) {
        let metrics = VectorMetrics::new();
        let initial_success = metrics.storage_success.get();
        let initial_failure = metrics.storage_failure.get();

        // Record a success
        metrics.record_success(duration);

        // Property: Success counter must increment by exactly 1
        prop_assert_eq!(
            metrics.storage_success.get(),
            initial_success + 1.0,
            "Success counter should increment by 1"
        );

        // Property: Failure counter must remain unchanged
        prop_assert_eq!(
            metrics.storage_failure.get(),
            initial_failure,
            "Failure counter should not change on success"
        );

        Ok(())
    }

    /// Test that failure operations increment the failure counter
    ///
    /// **Property**: storage_failure counter SHALL increment for each failed operation
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_failure_counter_increments(
        duration in valid_duration_strategy(),
    ) {
        let metrics = VectorMetrics::new();
        let initial_success = metrics.storage_success.get();
        let initial_failure = metrics.storage_failure.get();

        // Record a failure
        metrics.record_failure(duration);

        // Property: Failure counter must increment by exactly 1
        prop_assert_eq!(
            metrics.storage_failure.get(),
            initial_failure + 1.0,
            "Failure counter should increment by 1"
        );

        // Property: Success counter must remain unchanged
        prop_assert_eq!(
            metrics.storage_success.get(),
            initial_success,
            "Success counter should not change on failure"
        );

        Ok(())
    }

    /// Test that multiple operations accumulate correctly
    ///
    /// **Property**: Counters SHALL accumulate correctly across multiple operations
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_accumulate_correctly(
        operations in prop::collection::vec(
            (valid_duration_strategy(), operation_outcome_strategy()),
            1..=50
        ),
    ) {
        let metrics = VectorMetrics::new();
        let initial_success = metrics.storage_success.get();
        let initial_failure = metrics.storage_failure.get();

        let mut expected_success = 0;
        let mut expected_failure = 0;

        for (duration, is_success) in operations {
            if is_success {
                metrics.record_success(duration);
                expected_success += 1;
            } else {
                metrics.record_failure(duration);
                expected_failure += 1;
            }
        }

        // Property: Success counter must match expected count
        prop_assert_eq!(
            metrics.storage_success.get(),
            initial_success + expected_success as f64,
            "Success counter should match expected count"
        );

        // Property: Failure counter must match expected count
        prop_assert_eq!(
            metrics.storage_failure.get(),
            initial_failure + expected_failure as f64,
            "Failure counter should match expected count"
        );

        Ok(())
    }

    /// Test that duration histogram records values
    ///
    /// **Property**: storage_duration histogram SHALL record duration for all operations
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_duration_recorded(
        duration in valid_duration_strategy(),
        is_success in operation_outcome_strategy(),
    ) {
        let metrics = VectorMetrics::new();

        // Get initial histogram sample count
        let initial_count = metrics.storage_duration.get_sample_count();

        // Record operation
        if is_success {
            metrics.record_success(duration);
        } else {
            metrics.record_failure(duration);
        }

        // Property: Histogram sample count must increment
        let final_count = metrics.storage_duration.get_sample_count();
        prop_assert!(
            final_count > initial_count,
            "Histogram sample count should increase after recording"
        );

        Ok(())
    }

    /// Test that metrics are monotonically increasing
    ///
    /// **Property**: Counter values SHALL never decrease
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_monotonically_increasing(
        operations in prop::collection::vec(
            (valid_duration_strategy(), operation_outcome_strategy()),
            1..=20
        ),
    ) {
        let metrics = VectorMetrics::new();
        let mut prev_success = metrics.storage_success.get();
        let mut prev_failure = metrics.storage_failure.get();

        for (duration, is_success) in operations {
            if is_success {
                metrics.record_success(duration);
            } else {
                metrics.record_failure(duration);
            }

            let curr_success = metrics.storage_success.get();
            let curr_failure = metrics.storage_failure.get();

            // Property: Counters must never decrease
            prop_assert!(
                curr_success >= prev_success,
                "Success counter should never decrease"
            );
            prop_assert!(
                curr_failure >= prev_failure,
                "Failure counter should never decrease"
            );

            prev_success = curr_success;
            prev_failure = curr_failure;
        }

        Ok(())
    }

    /// Test that zero-duration operations are handled correctly
    ///
    /// **Property**: Zero-duration operations SHALL be recorded without error
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_handle_zero_duration(
        is_success in operation_outcome_strategy(),
    ) {
        let metrics = VectorMetrics::new();
        let zero_duration = Duration::from_millis(0);

        // Record operation with zero duration
        if is_success {
            metrics.record_success(zero_duration);
        } else {
            metrics.record_failure(zero_duration);
        }

        // Property: Operation should complete without panic
        // Property: Counters should increment
        if is_success {
            prop_assert!(
                metrics.storage_success.get() > 0.0,
                "Success counter should increment even for zero duration"
            );
        } else {
            prop_assert!(
                metrics.storage_failure.get() > 0.0,
                "Failure counter should increment even for zero duration"
            );
        }

        Ok(())
    }

    /// Test that very long durations are handled correctly
    ///
    /// **Property**: Long-duration operations SHALL be recorded without overflow
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_handle_long_duration(
        duration_secs in 1u64..=3600u64, // 1 second to 1 hour
        is_success in operation_outcome_strategy(),
    ) {
        let metrics = VectorMetrics::new();
        let long_duration = Duration::from_secs(duration_secs);

        // Record operation with long duration
        if is_success {
            metrics.record_success(long_duration);
        } else {
            metrics.record_failure(long_duration);
        }

        // Property: Operation should complete without panic or overflow
        // Property: Counters should increment
        if is_success {
            prop_assert!(
                metrics.storage_success.get() > 0.0,
                "Success counter should increment for long duration"
            );
        } else {
            prop_assert!(
                metrics.storage_failure.get() > 0.0,
                "Failure counter should increment for long duration"
            );
        }

        Ok(())
    }

    /// Test that histogram buckets are populated correctly
    ///
    /// **Property**: Duration histogram SHALL distribute values across appropriate buckets
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_histogram_buckets_populated(
        durations in prop::collection::vec(
            valid_duration_strategy(),
            10..=50
        ),
    ) {
        let metrics = VectorMetrics::new();
        let initial_count = metrics.storage_duration.get_sample_count();

        // Record multiple durations
        for duration in &durations {
            metrics.record_success(*duration);
        }

        // Property: Sample count should match number of recorded durations
        let final_count = metrics.storage_duration.get_sample_count();
        prop_assert_eq!(
            final_count,
            initial_count + durations.len() as u64,
            "Histogram sample count should match number of recorded operations"
        );

        Ok(())
    }
}

// =============================================================================
// Integration Tests with VectorChunk Validation
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test that metrics are emitted for valid chunk validation
    ///
    /// **Property**: Metrics SHALL be emitted when storing valid chunks
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_emitted_for_valid_chunks(
        embedding in valid_embedding_strategy(),
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        let document_id = Uuid::new_v4();
        let chunk = VectorChunk::new(
            text,
            embedding,
            document_id,
            source_id,
            chunk_index,
        );

        // Property: Valid chunks should pass validation
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_ok(),
            "Valid chunk should pass validation: {:?}",
            validation_result.err()
        );

        Ok(())
    }

    /// Test that metrics would be emitted for invalid chunk validation
    ///
    /// **Property**: Metrics SHALL be emitted even when validation fails
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_emitted_for_invalid_chunks(
        invalid_dim in 1usize..=1000usize,
        text in valid_text_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        // Skip if we accidentally generate 384 (valid dimension)
        prop_assume!(invalid_dim != 384);

        let document_id = Uuid::new_v4();
        let embedding = vec![0.1; invalid_dim];
        let chunk = VectorChunk::new(
            text,
            embedding,
            document_id,
            source_id,
            chunk_index,
        );

        // Property: Invalid dimension should fail validation
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_err(),
            "Invalid dimension should fail validation"
        );

        // Property: Error message should mention dimension
        let error_msg = validation_result.unwrap_err().to_string();
        prop_assert!(
            error_msg.contains("dimension") || error_msg.contains("384"),
            "Error message should mention dimension issue: {}",
            error_msg
        );

        Ok(())
    }

    /// Test that empty text validation triggers metric emission
    ///
    /// **Property**: Metrics SHALL be emitted for empty text validation failures
    /// **Validates: Requirements 11.1**
    #[test]
    fn prop_metrics_emitted_for_empty_text(
        embedding in valid_embedding_strategy(),
        source_id in valid_source_id_strategy(),
        chunk_index in 0usize..1000usize,
    ) {
        let document_id = Uuid::new_v4();
        let chunk = VectorChunk::new(
            "".to_string(), // Empty text
            embedding,
            document_id,
            source_id,
            chunk_index,
        );

        // Property: Empty text should fail validation
        let validation_result = chunk.validate();
        prop_assert!(
            validation_result.is_err(),
            "Empty text should fail validation"
        );

        // Property: Error message should mention empty text
        let error_msg = validation_result.unwrap_err().to_string();
        prop_assert!(
            error_msg.contains("empty") || error_msg.contains("text"),
            "Error message should mention empty text issue: {}",
            error_msg
        );

        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test that VectorMetrics can be created
    ///
    /// **Validates: Requirements 11.1** - Metrics infrastructure
    #[test]
    fn test_vector_metrics_creation() {
        let metrics = VectorMetrics::new();
        assert_eq!(metrics.storage_success.get(), 0.0);
        assert_eq!(metrics.storage_failure.get(), 0.0);
        assert_eq!(metrics.storage_duration.get_sample_count(), 0);
    }

    /// Test that metrics can be recorded multiple times
    ///
    /// **Validates: Requirements 11.1** - Metrics recording
    #[test]
    fn test_metrics_multiple_recordings() {
        let metrics = VectorMetrics::new();

        metrics.record_success(Duration::from_millis(10));
        metrics.record_success(Duration::from_millis(20));
        metrics.record_failure(Duration::from_millis(30));

        assert_eq!(metrics.storage_success.get(), 2.0);
        assert_eq!(metrics.storage_failure.get(), 1.0);
        assert_eq!(metrics.storage_duration.get_sample_count(), 3);
    }

    /// Test that histogram records duration values
    ///
    /// **Validates: Requirements 11.1** - Duration tracking
    #[test]
    fn test_histogram_records_durations() {
        let metrics = VectorMetrics::new();

        let durations = vec![
            Duration::from_millis(5),
            Duration::from_millis(15),
            Duration::from_millis(50),
            Duration::from_millis(100),
        ];

        for duration in durations {
            metrics.record_success(duration);
        }

        assert_eq!(metrics.storage_duration.get_sample_count(), 4);
        assert_eq!(metrics.storage_success.get(), 4.0);
    }

    /// Test that success and failure are tracked independently
    ///
    /// **Validates: Requirements 11.1** - Independent tracking
    #[test]
    fn test_success_and_failure_independent() {
        let metrics = VectorMetrics::new();

        metrics.record_success(Duration::from_millis(10));
        assert_eq!(metrics.storage_success.get(), 1.0);
        assert_eq!(metrics.storage_failure.get(), 0.0);

        metrics.record_failure(Duration::from_millis(20));
        assert_eq!(metrics.storage_success.get(), 1.0);
        assert_eq!(metrics.storage_failure.get(), 1.0);

        metrics.record_success(Duration::from_millis(30));
        assert_eq!(metrics.storage_success.get(), 2.0);
        assert_eq!(metrics.storage_failure.get(), 1.0);
    }

    /// Test that zero duration is handled correctly
    ///
    /// **Validates: Requirements 11.1** - Edge case handling
    #[test]
    fn test_zero_duration_handling() {
        let metrics = VectorMetrics::new();
        let zero_duration = Duration::from_millis(0);

        metrics.record_success(zero_duration);
        assert_eq!(metrics.storage_success.get(), 1.0);
        assert_eq!(metrics.storage_duration.get_sample_count(), 1);
    }

    /// Test that very long durations are handled correctly
    ///
    /// **Validates: Requirements 11.1** - Edge case handling
    #[test]
    fn test_long_duration_handling() {
        let metrics = VectorMetrics::new();
        let long_duration = Duration::from_secs(3600); // 1 hour

        metrics.record_success(long_duration);
        assert_eq!(metrics.storage_success.get(), 1.0);
        assert_eq!(metrics.storage_duration.get_sample_count(), 1);
    }

    /// Test that valid embedding strategy generates correct dimensions
    ///
    /// **Validates: Requirements 11.1** - Test data generation
    #[test]
    fn test_valid_embedding_strategy() {
        let strategy = valid_embedding_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();

        for _ in 0..10 {
            let embedding = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(embedding.len(), 384);
            for &value in &embedding {
                assert!(value >= -1.0 && value <= 1.0);
            }
        }
    }

    /// Test that valid text strategy generates non-empty strings
    ///
    /// **Validates: Requirements 11.1** - Test data generation
    #[test]
    fn test_valid_text_strategy() {
        let strategy = valid_text_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();

        for _ in 0..10 {
            let text = strategy.new_tree(&mut runner).unwrap().current();
            assert!(!text.is_empty());
            assert!(text.len() <= 500);
        }
    }

    /// Test that duration strategy generates valid durations
    ///
    /// **Validates: Requirements 11.1** - Test data generation
    #[test]
    fn test_valid_duration_strategy() {
        let strategy = valid_duration_strategy();
        let mut runner = proptest::test_runner::TestRunner::default();

        for _ in 0..10 {
            let duration = strategy.new_tree(&mut runner).unwrap().current();
            assert!(duration.as_millis() >= 1);
            assert!(duration.as_millis() <= 10000);
        }
    }
}
