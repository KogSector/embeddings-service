//! Tests for embedding generation endpoints with FalcorDB integration

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::AppState;
    use crate::models::ModelManager;
    use crate::core::Config;
    use std::sync::Arc;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    use serde_json::json;

    #[tokio::test]
    async fn test_generate_request_structure() {
        // Test that GenerateRequest can be deserialized with optional fields
        let json_data = json!({
            "text": "Test text",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "document_id": "550e8400-e29b-41d4-a716-446655440000",
            "source_id": "test-source",
            "chunk_index": 0,
            "metadata": {"key": "value"}
        });

        let request: GenerateRequest = serde_json::from_value(json_data).unwrap();
        assert_eq!(request.text, "Test text");
        assert_eq!(request.document_id, Some("550e8400-e29b-41d4-a716-446655440000".to_string()));
        assert_eq!(request.source_id, Some("test-source".to_string()));
        assert_eq!(request.chunk_index, Some(0));
    }

    #[tokio::test]
    async fn test_generate_request_minimal() {
        // Test that GenerateRequest works with only required fields
        let json_data = json!({
            "text": "Test text"
        });

        let request: GenerateRequest = serde_json::from_value(json_data).unwrap();
        assert_eq!(request.text, "Test text");
        assert_eq!(request.document_id, None);
        assert_eq!(request.source_id, None);
        assert_eq!(request.chunk_index, None);
        assert_eq!(request.metadata, None);
    }

    #[tokio::test]
    async fn test_batch_generate_request_structure() {
        // Test that BatchGenerateRequest can be deserialized with optional fields
        let json_data = json!({
            "texts": ["Text 1", "Text 2"],
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 2,
            "document_id": "550e8400-e29b-41d4-a716-446655440000",
            "source_id": "test-source",
            "metadata": {"key": "value"}
        });

        let request: BatchGenerateRequest = serde_json::from_value(json_data).unwrap();
        assert_eq!(request.texts.len(), 2);
        assert_eq!(request.document_id, Some("550e8400-e29b-41d4-a716-446655440000".to_string()));
        assert_eq!(request.source_id, Some("test-source".to_string()));
    }

    #[tokio::test]
    async fn test_generate_response_structure() {
        // Test that GenerateResponse serializes correctly with optional fields
        let response = GenerateResponse {
            embedding: vec![0.1, 0.2, 0.3],
            model: "test-model".to_string(),
            dimension: 384,
            embedding_id: Some("test-id".to_string()),
            stored: Some(true),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["embedding"].as_array().unwrap().len(), 3);
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["dimension"], 384);
        assert_eq!(json["embedding_id"], "test-id");
        assert_eq!(json["stored"], true);
    }

    #[tokio::test]
    async fn test_generate_response_without_storage() {
        // Test that GenerateResponse serializes correctly without storage fields
        let response = GenerateResponse {
            embedding: vec![0.1, 0.2, 0.3],
            model: "test-model".to_string(),
            dimension: 384,
            embedding_id: None,
            stored: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["embedding"].as_array().unwrap().len(), 3);
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["dimension"], 384);
        // Optional fields should not be present when None
        assert!(json.get("embedding_id").is_none());
        assert!(json.get("stored").is_none());
    }

    #[tokio::test]
    async fn test_error_response_structure() {
        // Test that ErrorResponse includes details field
        let error = ErrorResponse {
            error: "storage_failed".to_string(),
            message: "Failed to store vector".to_string(),
            details: Some("Connection timeout".to_string()),
        };

        let json = serde_json::to_value(&error).unwrap();
        assert_eq!(json["error"], "storage_failed");
        assert_eq!(json["message"], "Failed to store vector");
        assert_eq!(json["details"], "Connection timeout");
    }

    #[tokio::test]
    async fn test_batch_generate_response_structure() {
        // Test that BatchGenerateResponse serializes correctly with storage fields
        let response = BatchGenerateResponse {
            embeddings: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            model: "test-model".to_string(),
            dimension: 384,
            processing_time_ms: 100,
            total_texts: 2,
            embedding_ids: Some(vec!["id1".to_string(), "id2".to_string()]),
            stored: Some(true),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["embeddings"].as_array().unwrap().len(), 2);
        assert_eq!(json["total_texts"], 2);
        assert_eq!(json["embedding_ids"].as_array().unwrap().len(), 2);
        assert_eq!(json["stored"], true);
    }
}
