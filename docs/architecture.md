# embeddings-service Architecture

## Overview
This document describes the high-level architecture of `embeddings-service`.

## System Design
```mermaid
graph TD
    Client --> API[embeddings-service]
    API --> DB[(Database)]
```

## Key Components
- **API Layer**: Handles incoming requests.
- **Service Layer**: Core business logic.
- **Data Access**: Manages persistent storage.
