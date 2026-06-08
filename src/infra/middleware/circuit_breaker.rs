//! Circuit Breaker for downstream service calls
//!
//! Three-state circuit breaker (Closed/Open/HalfOpen) to isolate
//! failures and prevent cascading outages.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Configuration for a circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to trip the breaker
    pub failure_threshold: u32,
    /// Duration the circuit stays open before trying half-open
    pub open_duration: Duration,
    /// Number of successful probes in half-open to close the circuit
    pub half_open_successes: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(30),
            half_open_successes: 2,
        }
    }
}

/// Per-service circuit breaker state
struct BreakerState {
    consecutive_failures: AtomicU32,
    consecutive_successes: AtomicU32,
    opened_at: AtomicU64,
    half_open_probes: AtomicU32,
}

impl BreakerState {
    fn new() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            opened_at: AtomicU64::new(0),
            half_open_probes: AtomicU32::new(0),
        }
    }
}

/// Registry of circuit breakers keyed by service name
#[derive(Clone)]
pub struct CircuitBreakerRegistry {
    breakers: Arc<DashMap<String, Arc<BreakerState>>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: Arc::new(DashMap::new()),
            config,
        }
    }

    fn get_or_create(&self, service: &str) -> Arc<BreakerState> {
        self.breakers
            .entry(service.to_string())
            .or_insert_with(|| Arc::new(BreakerState::new()))
            .clone()
    }

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Get the current state of the circuit for a given service
    pub fn state(&self, service: &str) -> CircuitState {
        let breaker = self.get_or_create(service);
        let opened = breaker.opened_at.load(Ordering::Relaxed);
        if opened == 0 {
            return CircuitState::Closed;
        }
        let elapsed = Self::now_millis() - opened;
        if elapsed >= self.config.open_duration.as_millis() as u64 {
            CircuitState::HalfOpen
        } else {
            CircuitState::Open
        }
    }

    /// Check if a request is allowed through the circuit
    pub fn allow_request(&self, service: &str) -> bool {
        match self.state(service) {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => {
                let breaker = self.get_or_create(service);
                let probes = breaker.half_open_probes.fetch_add(1, Ordering::Relaxed);
                probes < self.config.half_open_successes + 1
            }
        }
    }

    /// Record a successful request
    pub fn record_success(&self, service: &str) {
        let breaker = self.get_or_create(service);
        breaker.consecutive_failures.store(0, Ordering::Relaxed);
        let successes = breaker.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;

        let opened = breaker.opened_at.load(Ordering::Relaxed);
        if opened > 0 {
            if successes >= self.config.half_open_successes {
                breaker.opened_at.store(0, Ordering::Relaxed);
                breaker.consecutive_successes.store(0, Ordering::Relaxed);
                breaker.half_open_probes.store(0, Ordering::Relaxed);
                tracing::info!(
                    service = service,
                    "Circuit breaker CLOSED — service recovered"
                );
            }
        }
    }

    /// Record a failed request
    pub fn record_failure(&self, service: &str) {
        let breaker = self.get_or_create(service);
        breaker.consecutive_successes.store(0, Ordering::Relaxed);
        let failures = breaker.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.config.failure_threshold {
            let opened = breaker.opened_at.load(Ordering::Relaxed);
            if opened == 0 {
                breaker
                    .opened_at
                    .store(Self::now_millis(), Ordering::Relaxed);
                breaker.half_open_probes.store(0, Ordering::Relaxed);
                tracing::warn!(
                    service = service,
                    failures = failures,
                    "Circuit breaker OPENED — isolating failing service"
                );
            } else {
                breaker
                    .opened_at
                    .store(Self::now_millis(), Ordering::Relaxed);
                breaker.half_open_probes.store(0, Ordering::Relaxed);
                tracing::warn!(
                    service = service,
                    "Circuit breaker re-OPENED from half-open probe failure"
                );
            }
        }
    }

    /// Get metrics for monitoring
    pub fn metrics(&self, service: &str) -> (CircuitState, u32, u32) {
        let breaker = self.get_or_create(service);
        (
            self.state(service),
            breaker.consecutive_failures.load(Ordering::Relaxed),
            breaker.consecutive_successes.load(Ordering::Relaxed),
        )
    }
}
