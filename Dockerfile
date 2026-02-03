# ==============================================================================
# Embeddings Service - Dockerfile
# ==============================================================================
# Multi-stage build for Rust + Python hybrid embedding service
# Latest base images with optimized layering
# Port: 3010
# ==============================================================================

# Stage 1: Rust builder with latest stable Rust
FROM rust:1.84.0-slim AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    python3-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust files
COPY Cargo.toml Cargo.lock* ./

# Create dummy src for dependency caching
RUN mkdir -p src/api src/core src/generators src/models src/storage && \
    echo 'fn main() {}' > src/main.rs && \
    touch src/lib.rs && \
    touch src/api/mod.rs src/core/mod.rs src/generators/mod.rs src/models/mod.rs src/storage/mod.rs

# Build dependencies (cached)
RUN cargo build --release 2>/dev/null || true

# Remove dummy source
RUN rm -rf src/*

# Copy actual source
COPY src/ ./src/

# Build the application
RUN cargo build --release

# Stage 2: Python dependencies for ML models
FROM python:3.13.2-slim AS python-builder

# Install system dependencies for ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML dependencies
RUN pip install --no-cache-dir \
    sentence-transformers>=3.3.0 \
    numpy>=1.24.0 \
    torch>=2.0.0 \
    transformers>=4.35.0 \
    python-dotenv>=1.0.0 \
    structlog>=24.1.0

# Stage 3: Final runtime image
FROM python:3.13.2-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=python-builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy Rust binary
COPY --from=rust-builder /app/target/release/embeddings-service /usr/local/bin/

# Copy application source for Python modules
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Health check optimized for Azure Container Apps
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3005/health || exit 1

# Expose port
EXPOSE 3005

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start the Rust service
CMD ["embeddings-service"]
