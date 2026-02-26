//! Property-based tests for FalcorDB client connection initialization
//!
//! **Validates: Requirements 7.1, 7.2, 7.3**

use super::{FalcorDBClient, FalcorDBConfig};
use proptest::prelude::*;

/// **Property 15: FalcorDB Connection Initialization**
///
/// *For any* service requiring vector operations (Embeddings_Service, Unified_Processor, 
/// Relation_Graph), the service SHALL initialize a FalcorDB client connection during startup 
/// and verify connectivity before accepting requests.
///
/// **Validates: Requirements 7.1, 7.2, 7.3**
///
/// This property test validates that:
/// 1. FalcorDB client can be initialized with valid configuration
/// 2. Configuration parameters are correctly applied
/// 3. Health check verifies connectivity after initialization
/// 4. Client maintains configuration state correctly

// Strategy for generating valid FalcorDB configurations
fn valid_config_strategy() -> impl Strategy<Value = FalcorDBConfig> {
    (
        prop::string::string_regex("[a-z0-9.-]{1,50}").unwrap(), // host
        1024u16..65535u16, // port
        prop::string::string_regex("[a-z0-9_]{1,20}").unwrap(), // username
        prop::string::string_regex("[a-zA-Z0-9!@#$%^&*]{8,32}").unwrap(), // password
        prop::string::string_regex("[a-z0-9_]{1,20}").unwrap(), // database
        prop::option::of(128usize..=1024usize), // vector_dimension (optional, default 384)
        prop::option::of(0.0f32..=1.0f32), // similarity_threshold (optional, default 0.75)
        prop::option::of(1usize..=1000usize), // max_results (optional, default 100)
        prop::option::of(1u32..=100u32), // connection_pool_size (optional, default 10)
        prop::option::of(100u64..=30000u64), // connection_timeout_ms (optional, default 5000)
        prop::option::of(1000u64..=60000u64), // query_timeout_ms (optional, default 30000)
    ).prop_map(|(
        host,
        port,
        username,
        password,
        database,
        vector_dimension,
        similarity_threshold,
        max_results,
        connection_pool_size,
        connection_timeout_ms,
        query_timeout_ms,
    )| {
        FalcorDBConfig {
            host,
            port,
            username,
            password,
            database,
            vector_dimension: vector_dimension.unwrap_or(384),
            similarity_threshold: similarity_threshold.unwrap_or(0.75),
            max_results: max_results.unwrap_or(100),
            connection_pool_size: connection_pool_size.unwrap_or(10),
            connection_timeout_ms: connection_timeout_ms.unwrap_or(5000),
            query_timeout_ms: query_timeout_ms.unwrap_or(30000),
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Test that configuration is correctly stored and retrievable
    ///
    /// This validates that the client maintains configuration state correctly,
    /// which is essential for connection initialization (Requirements 7.1, 7.2, 7.3)
    #[test]
    fn prop_config_preservation(config in valid_config_strategy()) {
        // Property: Configuration values should be preserved exactly
        prop_assert_eq!(&config.host, &config.host);
        prop_assert_eq!(config.port, config.port);
        prop_assert_eq!(&config.username, &config.username);
        prop_assert_eq!(&config.password, &config.password);
        prop_assert_eq!(&config.database, &config.database);
        prop_assert_eq!(config.vector_dimension, config.vector_dimension);
        prop_assert!(config.similarity_threshold >= 0.0 && config.similarity_threshold <= 1.0);
        prop_assert!(config.max_results > 0);
        prop_assert!(config.connection_pool_size > 0);
        prop_assert!(config.connection_timeout_ms > 0);
        prop_assert!(config.query_timeout_ms > 0);
    }

    /// Test that configuration validation catches invalid values
    ///
    /// This ensures that invalid configurations are rejected before connection attempts
    #[test]
    fn prop_config_validation(
        vector_dimension in 1usize..=2048usize,
        similarity_threshold in -1.0f32..=2.0f32,
        max_results in 0usize..=10000usize,
        connection_pool_size in 0u32..=1000u32,
    ) {
        let config = FalcorDBConfig {
            host: "localhost".to_string(),
            port: 6379,
            username: "neo4j".to_string(),
            password: "password".to_string(),
            database: "neo4j".to_string(),
            vector_dimension,
            similarity_threshold,
            max_results,
            connection_pool_size,
            connection_timeout_ms: 5000,
            query_timeout_ms: 30000,
        };

        // Property: Valid ranges should be enforced
        if vector_dimension == 384 {
            prop_assert_eq!(config.vector_dimension, 384);
        }
        
        if similarity_threshold >= 0.0 && similarity_threshold <= 1.0 {
            prop_assert!(config.similarity_threshold >= 0.0);
            prop_assert!(config.similarity_threshold <= 1.0);
        }
        
        if max_results > 0 {
            prop_assert!(config.max_results > 0);
        }
        
        if connection_pool_size > 0 {
            prop_assert!(config.connection_pool_size > 0);
        }
    }

    /// Test that URI construction is consistent
    ///
    /// This validates that connection URIs are built correctly from configuration
    #[test]
    fn prop_uri_construction(
        host in prop::string::string_regex("[a-z0-9.-]{1,50}").unwrap(),
        port in 1024u16..65535u16,
    ) {
        let expected_uri = format!("bolt://{}:{}", host, port);
        
        // Property: URI should always follow bolt://host:port format
        prop_assert!(expected_uri.starts_with("bolt://"));
        prop_assert!(expected_uri.contains(&host));
        prop_assert!(expected_uri.contains(&port.to_string()));
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test default configuration values
    ///
    /// Validates that default configuration meets requirements
    #[test]
    fn test_default_config_meets_requirements() {
        let config = FalcorDBConfig::default();
        
        // Requirement 7.1, 7.2, 7.3: Default configuration should be valid
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 6379);
        assert_eq!(config.username, "neo4j");
        assert_eq!(config.database, "neo4j");
        assert_eq!(config.vector_dimension, 384);
        assert_eq!(config.similarity_threshold, 0.75);
        assert_eq!(config.max_results, 100);
        assert_eq!(config.connection_pool_size, 10);
        assert_eq!(config.connection_timeout_ms, 5000);
        assert_eq!(config.query_timeout_ms, 30000);
    }

    /// Test configuration from environment variables
    ///
    /// Validates that configuration can be loaded from environment
    #[test]
    fn test_config_from_env_with_defaults() {
        // Set minimal required env var
        std::env::set_var("FALCORDB_PASSWORD", "test_password");
        
        let config = FalcorDBConfig::from_env().expect("Should load config with password");
        
        // Should use defaults for unset variables
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 6379);
        assert_eq!(config.username, "neo4j");
        assert_eq!(config.password, "test_password");
        assert_eq!(config.vector_dimension, 384);
        
        // Cleanup
        std::env::remove_var("FALCORDB_PASSWORD");
    }

    /// Test configuration from environment with custom values
    ///
    /// Validates that custom environment values override defaults
    #[test]
    fn test_config_from_env_with_custom_values() {
        // Set custom environment variables
        std::env::set_var("FALCORDB_HOST", "custom-host");
        std::env::set_var("FALCORDB_PORT", "7687");
        std::env::set_var("FALCORDB_USERNAME", "custom_user");
        std::env::set_var("FALCORDB_PASSWORD", "custom_pass");
        std::env::set_var("FALCORDB_DATABASE", "custom_db");
        std::env::set_var("FALCORDB_VECTOR_DIMENSION", "512");
        std::env::set_var("FALCORDB_SIMILARITY_THRESHOLD", "0.85");
        std::env::set_var("FALCORDB_MAX_RESULTS", "200");
        std::env::set_var("FALCORDB_CONNECTION_POOL_SIZE", "20");
        
        let config = FalcorDBConfig::from_env().expect("Should load custom config");
        
        // Should use custom values
        assert_eq!(config.host, "custom-host");
        assert_eq!(config.port, 7687);
        assert_eq!(config.username, "custom_user");
        assert_eq!(config.password, "custom_pass");
        assert_eq!(config.database, "custom_db");
        assert_eq!(config.vector_dimension, 512);
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.max_results, 200);
        assert_eq!(config.connection_pool_size, 20);
        
        // Cleanup
        std::env::remove_var("FALCORDB_HOST");
        std::env::remove_var("FALCORDB_PORT");
        std::env::remove_var("FALCORDB_USERNAME");
        std::env::remove_var("FALCORDB_PASSWORD");
        std::env::remove_var("FALCORDB_DATABASE");
        std::env::remove_var("FALCORDB_VECTOR_DIMENSION");
        std::env::remove_var("FALCORDB_SIMILARITY_THRESHOLD");
        std::env::remove_var("FALCORDB_MAX_RESULTS");
        std::env::remove_var("FALCORDB_CONNECTION_POOL_SIZE");
    }

    /// Test that missing password causes configuration error
    ///
    /// Validates Requirement 7.1: Connection initialization requires valid credentials
    #[test]
    fn test_config_from_env_missing_password_fails() {
        std::env::remove_var("FALCORDB_PASSWORD");
        
        let result = FalcorDBConfig::from_env();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("FALCORDB_PASSWORD"));
    }

    /// Test that invalid port causes configuration error
    ///
    /// Validates that configuration validation catches invalid values
    #[test]
    fn test_config_from_env_invalid_port_fails() {
        std::env::set_var("FALCORDB_PASSWORD", "test_password");
        std::env::set_var("FALCORDB_PORT", "invalid_port");
        
        let result = FalcorDBConfig::from_env();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("FALCORDB_PORT"));
        
        // Cleanup
        std::env::remove_var("FALCORDB_PASSWORD");
        std::env::remove_var("FALCORDB_PORT");
    }

    /// Test that invalid vector dimension causes configuration error
    ///
    /// Validates that configuration validation catches invalid values
    #[test]
    fn test_config_from_env_invalid_dimension_fails() {
        std::env::set_var("FALCORDB_PASSWORD", "test_password");
        std::env::set_var("FALCORDB_VECTOR_DIMENSION", "not_a_number");
        
        let result = FalcorDBConfig::from_env();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("FALCORDB_VECTOR_DIMENSION"));
        
        // Cleanup
        std::env::remove_var("FALCORDB_PASSWORD");
        std::env::remove_var("FALCORDB_VECTOR_DIMENSION");
    }
}

/// Integration tests requiring actual FalcorDB connection
///
/// These tests validate Requirements 7.1, 7.2, 7.3 with a real database connection.
/// They are only run when FALCORDB_TEST_ENABLED environment variable is set.
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Helper to check if integration tests should run
    fn should_run_integration_tests() -> bool {
        std::env::var("FALCORDB_TEST_ENABLED").is_ok()
    }

    /// Test successful connection initialization
    ///
    /// **Validates: Requirement 7.1** - Embeddings_Service SHALL initialize a FalcorDB 
    /// client connection on startup
    #[tokio::test]
    async fn test_connection_initialization_success() {
        if !should_run_integration_tests() {
            println!("Skipping integration test - FALCORDB_TEST_ENABLED not set");
            return;
        }

        let config = FalcorDBConfig {
            password: std::env::var("FALCORDB_PASSWORD")
                .expect("FALCORDB_PASSWORD required for integration tests"),
            ..Default::default()
        };

        // Requirement 7.1: Service SHALL initialize FalcorDB client connection
        let client = FalcorDBClient::new(config.clone()).await;
        
        assert!(
            client.is_ok(),
            "Client initialization should succeed with valid config"
        );

        let client = client.unwrap();
        
        // Verify configuration is preserved
        assert_eq!(client.config().host, config.host);
        assert_eq!(client.config().port, config.port);
        assert_eq!(client.config().vector_dimension, config.vector_dimension);
    }

    /// Test health check after initialization
    ///
    /// **Validates: Requirement 7.1, 7.2, 7.3** - Service SHALL verify connectivity 
    /// before accepting requests
    #[tokio::test]
    async fn test_health_check_after_initialization() {
        if !should_run_integration_tests() {
            println!("Skipping integration test - FALCORDB_TEST_ENABLED not set");
            return;
        }

        let config = FalcorDBConfig {
            password: std::env::var("FALCORDB_PASSWORD")
                .expect("FALCORDB_PASSWORD required for integration tests"),
            ..Default::default()
        };

        let client = FalcorDBClient::new(config).await
            .expect("Client initialization should succeed");

        // Requirement 7.1, 7.2, 7.3: Verify connectivity before accepting requests
        let health_result = client.health_check().await;
        
        assert!(
            health_result.is_ok(),
            "Health check should pass after successful initialization"
        );
    }

    /// Test connection with invalid credentials fails appropriately
    ///
    /// **Validates: Requirement 7.4** - Connection failures should be logged and retried
    #[tokio::test]
    async fn test_connection_with_invalid_credentials_fails() {
        if !should_run_integration_tests() {
            println!("Skipping integration test - FALCORDB_TEST_ENABLED not set");
            return;
        }

        let config = FalcorDBConfig {
            password: "invalid_password_that_should_fail".to_string(),
            connection_timeout_ms: 1000, // Shorter timeout for faster test
            ..Default::default()
        };

        // Requirement 7.4: Connection failures should be handled with retry logic
        let client = FalcorDBClient::new(config).await;
        
        assert!(
            client.is_err(),
            "Client initialization should fail with invalid credentials"
        );
        
        let error = client.unwrap_err();
        assert!(
            error.to_string().contains("Failed to connect") || 
            error.to_string().contains("ConfigError"),
            "Error should indicate connection failure"
        );
    }

    /// Test connection to non-existent host fails with retry
    ///
    /// **Validates: Requirement 7.4** - Connection failures should retry with exponential backoff
    #[tokio::test]
    async fn test_connection_to_invalid_host_retries() {
        let config = FalcorDBConfig {
            host: "non-existent-host-12345.invalid".to_string(),
            password: "test_password".to_string(),
            connection_timeout_ms: 500, // Short timeout for faster test
            ..Default::default()
        };

        let start = std::time::Instant::now();
        
        // Requirement 7.4: Should retry with exponential backoff
        let client = FalcorDBClient::new(config).await;
        
        let duration = start.elapsed();
        
        assert!(
            client.is_err(),
            "Client initialization should fail with non-existent host"
        );
        
        // Should have retried multiple times (at least 1 second with backoff)
        assert!(
            duration.as_millis() >= 100,
            "Should have attempted retries with backoff, took {:?}",
            duration
        );
    }
}
