pub mod axum_auth;
pub mod axum_rate_limit;
pub mod security_headers;
pub mod zero_trust;

// Framework-agnostic middleware
pub mod cache;
pub mod circuit_breaker;

// Re-export Axum types
pub use axum_auth::{AuthenticatedUser, AxumAuthLayer, axum_auth_middleware, axum_optional_auth_middleware};
pub use axum_rate_limit::{AxumRateLimitConfig, axum_rate_limit_middleware};
pub use security_headers::security_headers_middleware;
pub use zero_trust::{ZeroTrustLayer, zero_trust_middleware};

// Re-export framework-agnostic types
pub use cache::{ResponseCache, CacheConfig};
pub use circuit_breaker::{CircuitBreakerRegistry, CircuitBreakerConfig, CircuitState};
