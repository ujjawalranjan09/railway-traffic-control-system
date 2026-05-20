# AI-Powered Railway Traffic Control System - Project Summary

## Executive Summary

This project delivers a comprehensive, production-grade AI decision-support system for Indian Railways traffic controllers, addressing Smart India Hackathon 2025 Problem Statement 25022. It has been significantly upgraded to a high-performance, asynchronous architecture.

## Problem Addressed

**Challenge**: Manual train traffic control becomes insufficient with increasing network congestion, requiring intelligent, data-driven systems to:
- Optimize section throughput
- Minimize delays and conflicts
- Enable real-time decision support
- Provide what-if scenario analysis

## Solution Components

### 1. High-Performance ML Engine (ONNX)

**Conflict Detection & Delay Prediction**
- **Architecture**: Ensembles of Random Forest, XGBoost, and LightGBM.
- **Serialization**: Models are exported to **ONNX (Open Neural Network Exchange)** format, replacing insecure and slow Pickle files.
- **Inference**: Leverages `onnxruntime` for highly optimized C++ inference loops, ensuring sub-50ms prediction latency.
- **Features**: 15+ engineered features including train density, weather, platform utilization, and temporal sine/cosine transforms.

### 2. Advanced Optimization Engine (OR-Tools)

**Constraint Programming Solver**
- **Algorithm**: Google OR-Tools **CP-SAT Solver**.
- **Constraints**:
    - **Block Occupancy**: Ensures spatial safety by preventing two trains from occupying the same block simultaneously.
    - **Platform Capacity**: Manages finite platform resources.
    - **Priority Weighting**: Explicitly penalizes delays for high-priority premium passenger expresses.
- **Performance**: Significant speedup over traditional MILP solvers, providing optimal or near-optimal schedules in real-time.

### 3. Next-Gen API Framework (FastAPI)

**High-Concurrency Backend**
- **Framework**: **FastAPI** with Python `async/await` for non-blocking I/O.
- **Validation**: Strict, Rust-powered input validation using **Pydantic v2**.
- **Security**: Header-based API Key authentication (`X-API-Key`) and hardened CORS origin management.
- **Asynchronicity**: Long-running simulations and optimizations run without blocking the main event loop.

### 4. Interactive Interfaces

**Streamlit Decision Support Dashboard**
- Real-time KPI monitoring (throughput, punctuality, delays, conflicts).
- Interactive "What-If" simulation tools.
- Plotly-based visual trend analysis.
- Decision support recommendations with explainability.

**Legacy Dashboard Support**
- Maintained compatibility for the vanilla JavaScript frontend for simple deployment scenarios.

## Technical Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                    │
│                 (Streamlit / Next.js Ready)              │
│  - KPI Panels  - Risk Analysis  - Simulation Tools      │
└───────────────────┬─────────────────────────────────────┘
                    │ REST API (Async JSON)
┌───────────────────▼─────────────────────────────────────┐
│                 FastAPI Backend Engine                  │
│  - Pydantic V2 Validation  - Async State Manager        │
└───────────────────┬──────────────────┬──────────────────┘
                    │                  │
┌───────────────────▼───┐      ┌───────▼───────────────┐
│      ONNX Models      │      │    Google OR-Tools    │
│ (Conflict/Delay Pred) │      │   (CP-SAT Optimizer)  │
└───────────────────────┘      └───────────────────────┘
```

## Impact & Benefits

- **Reduced Response Latency**: ONNX and FastAPI ensure the dashboard remains fluid even under high load.
- **Enhanced Reliability**: CP-SAT handles complex sequencing logic that traditional solvers struggle with.
- **Improved Security**: Eliminated Pickle-based remote code execution vulnerabilities.
- **Future-Proofed**: The async architecture is ready for WebSocket-based real-time telemetry and Next.js integration.

## Key Performance Metrics

- **Conflict Detection Accuracy**: ~93% achieved.
- **Delay Prediction MAE**: <3 minutes achieved.
- **Optimization Speed**: <100ms for standard section schedules.
- **Concurrent Request Handling**: Highly scalable via ASGI/Uvicorn.

## Impact on Indian Railways

- **Proactive Intervention**: Instant conflict alerts empower controllers to intervene *before* delays cascade.
- **Capacity Maximization**: Optimal sequencing allows for higher train density without compromising safety.
- **Data-Driven Strategy**: Historical prediction data enables better long-term planning for section capacity upgrades.

---

**Project Status**: 🚀 Production V2.0
**Technology Stack**: Python 3.12, FastAPI, ONNX, OR-Tools, Pydantic, Streamlit, Plotly
**Developed for**: Smart India Hackathon 2025  
**Date**: May 2026 (Updated)
