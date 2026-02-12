"""
Embeddings Service gRPC Server

Implements the Embeddings gRPC service defined in proto/embeddings.proto
"""
import asyncio
import grpc
from concurrent import futures
import structlog
import sys
import os

logger = structlog.get_logger()


class EmbeddingsServicer:  # (embeddings_pb2_grpc.EmbeddingsServicer):
    """
    Implementation of Embeddings gRPC service.
    
    To generate proto code, run:
    python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto proto/embeddings.proto
    """
    
    async def Embed(self, request, context):
        """Generate embeddings for a single text."""
        logger.info("gRPC Embed called", text_length=len(request.text))
        # TODO: Implement using existing embedding generation logic
        pass
    
    async def BatchEmbed(self, request, context):
        """Generate embeddings for multiple texts."""
        logger.info("gRPC BatchEmbed called", batch_size=len(request.texts))
        # TODO: Implement batch embedding
        pass
    
    async def GetModelInfo(self, request, context):
        """Get information about the embedding model."""
        logger.info("gRPC GetModelInfo called")
        # TODO: Return model information
        pass


async def serve_grpc():
    """Start the gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add servicer
    # embeddings_pb2_grpc.add_EmbeddingsServicer_to_server(EmbeddingsServicer(), server)
    
    # Configure server
    grpc_port = int(os.getenv("GRPC_PORT", "50054"))
    server.add_insecure_port(f"[::]:{grpc_port}")
    
    logger.info(f"Starting embeddings-service gRPC server on port {grpc_port}")
    await server.start()
    logger.info("gRPC server started successfully")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        await server.stop(grace=5)


if __name__ == "__main__":
    asyncio.run(serve_grpc())
