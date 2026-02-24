//! Embeddings Service gRPC Server
//! Handles gRPC requests for text embedding generation

use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use std::collections::HashMap;

use crate::core::EmbeddingsService;
use crate::core::Config;
use crate::proto::confuse::embeddings::v1::{
    embeddings_server::Embeddings,
    EmbedRequest, EmbedResponse, BatchEmbedRequest, BatchEmbedResponse,
    ModelInfoRequest, ModelInfo, EmbeddingResult,
};

pub struct EmbeddingsGrpcService {
    service: Arc<EmbeddingsService>,
    config: Arc<Config>,
}

impl EmbeddingsGrpcService {
    pub fn new(service: Arc<EmbeddingsService>, config: Arc<Config>) -> Self {
        Self {
            service,
            config,
        }
    }
}

#[tonic::async_trait]
impl Embeddings for EmbeddingsGrpcService {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        
        tracing::info!("Generating embedding for text (length: {})", req.text.len());
        
        let model = req.model.unwrap_or_else(|| self.config.default_model.clone());
        let options = req.options;
        
        match self.service.generate_embedding(&req.text, &model, options).await {
            Ok(result) => {
                Ok(Response::new(EmbedResponse {
                    embeddings: result.embeddings,
                    dimension: result.dimension,
                    model_used: result.model_used,
                }))
            }
            Err(e) => {
                tracing::error!("Failed to generate embedding: {}", e);
                Err(Status::internal(format!("Embedding generation failed: {}", e)))
            }
        }
    }

    async fn batch_embed(
        &self,
        request: Request<BatchEmbedRequest>,
    ) -> Result<Response<BatchEmbedResponse>, Status> {
        let req = request.into_inner();
        
        tracing::info!("Generating batch embeddings for {} texts", req.texts.len());
        
        let model = req.model.unwrap_or_else(|| self.config.default_model.clone());
        let options = req.options;
        
        match self.service.generate_batch_embeddings(&req.texts, &model, options).await {
            Ok(result) => {
                let embedding_results: Vec<EmbeddingResult> = result.results.into_iter().map(|r| EmbeddingResult {
                    embeddings: r.embeddings,
                    index: r.index,
                }).collect();

                Ok(Response::new(BatchEmbedResponse {
                    results: embedding_results,
                    model_used: result.model_used,
                }))
            }
            Err(e) => {
                tracing::error!("Failed to generate batch embeddings: {}", e);
                Err(Status::internal(format!("Batch embedding generation failed: {}", e)))
            }
        }
    }

    async fn get_model_info(
        &self,
        request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfo>, Status> {
        let req = request.into_inner();
        
        let model_name = req.model_name.unwrap_or_else(|| self.config.default_model.clone());
        
        tracing::info!("Getting model info for: {}", model_name);
        
        match self.service.get_model_info(&model_name).await {
            Ok(info) => {
                Ok(Response::new(ModelInfo {
                    model_name: info.model_name,
                    dimension: info.dimension,
                    max_sequence_length: info.max_sequence_length,
                    provider: info.provider,
                }))
            }
            Err(e) => {
                tracing::error!("Failed to get model info: {}", e);
                Err(Status::internal(format!("Model info retrieval failed: {}", e)))
            }
        }
    }
}

pub async fn start_grpc_server(
    service: Arc<EmbeddingsService>,
    config: Arc<Config>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", config.grpc_host, config.grpc_port).parse()?;
    
    tracing::info!("Starting gRPC server on {}", addr);
    
    let grpc_service = EmbeddingsGrpcService::new(service, config);
    
    Server::builder()
        .add_service(
            crate::proto::confuse::embeddings::v1::embeddings_server::EmbeddingsServer::new(grpc_service)
        )
        .serve(addr)
        .await?;
    
    Ok(())
}
