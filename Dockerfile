# ==============================================================================
# Embeddings Service - Dockerfile
# ==============================================================================
# Multi-stage build for Rust embedding service
# Port: 3011
# ==============================================================================

# Stage 1: Rust builder
FROM rust:slim-bookworm AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    build-essential \
    cmake \
    libsasl2-dev \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust files
COPY Cargo.toml Cargo.lock* ./

# Create dummy src for dependency caching
RUN mkdir -p src/api src/core src/generators src/models src/infra && \
    echo 'fn main() {}' > src/main.rs && \
    touch src/lib.rs && \
    touch src/api/mod.rs src/core/mod.rs src/generators/mod.rs src/models/mod.rs src/infra/mod.rs

# Build dependencies (cached)
RUN cargo build --release 2>/dev/null || true

# Remove dummy source
RUN rm -rf src/*

# Copy actual source
COPY src/ ./src/

# Build the application
RUN cargo build --release --features kafka

# Stage 2: Final runtime image
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libpq5 \
    dumb-init \
    librdkafka1 \
    libsasl2-2 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy Rust binary
COPY --from=rust-builder --chown=appuser:appuser /app/target/release/embeddings-service /usr/local/bin/

# Ensure correct permissions
RUN chown -R appuser:appuser /app
USER appuser

ENV PORT=3011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-3011}/health || exit 1

# Expose port
EXPOSE 3011

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start the Rust service
CMD ["embeddings-service"]
