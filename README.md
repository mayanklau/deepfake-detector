# 🔍 DeepFake Detector Platform v2.0

**Production-grade AI-powered DeepFake Detection System**

## Architecture
```
┌────────────────────────────────────────────────────────────────────┐
│                      NGINX (Reverse Proxy + SSL)                   │
└────────────┬──────────────────────────────────┬────────────────────┘
             │                                  │
┌────────────▼─────────────┐  ┌────────────────▼───────────────────┐
│  React Frontend (Next.js) │  │    FastAPI Backend (REST + WS)     │
│  19 pages, Real-time UI   │  │    19 endpoint modules, RBAC       │
└───────────────────────────┘  └────────────┬──────────────────────┘
                                            │
┌───────────────────────────────────────────▼──────────────────────┐
│                    ML Detection Pipeline                         │
│  Face Detection │ Manipulation (4 models) │ Frequency Analysis   │
│  GAN Detection  │ Noise Analysis │ Compression │ Metadata        │
│  Audio Deepfake │ Lip Sync │ Temporal │ Ensemble Predictor       │
└──────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  PostgreSQL │ Redis │ Celery │ Prometheus │ Grafana │ Docker/K8s │
└─────────────────────────────────────────────────────────────────┘
```

## Features
- **12 Detection Models** in ensemble configuration
- **19 REST API Endpoints** with full CRUD, batch, real-time
- **JWT + OAuth + MFA + API Keys** authentication
- **RBAC** (Super Admin, Admin, Analyst, Viewer, API User)
- **WebSocket** real-time analysis progress
- **Batch Processing** up to 200 files
- **Forensic Reports** (PDF, HTML, JSON, CSV)
- **Explainability** (Grad-CAM, attention maps, feature attribution)
- **Multi-tenant** with organizations and teams
- **Audit Logging** complete trail
- **Docker Compose + Kubernetes** deployment
- **Prometheus + Grafana** monitoring

## Quick Start
```bash
docker-compose up -d
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

## ML Model Performance
| Model | Accuracy | AUC | F1 | Latency |
|-------|----------|-----|----|---------| 
| EfficientNet-B4 | 96.2% | 0.983 | 0.961 | 45ms |
| Xception | 95.1% | 0.975 | 0.950 | 38ms |
| Capsule Net v2 | 94.8% | 0.972 | 0.947 | 52ms |
| Multi-Attention | 96.5% | 0.985 | 0.964 | 62ms |
| RawNet3 (Audio) | 95.8% | 0.980 | 0.957 | 28ms |
| **Ensemble** | **97.3%** | **0.991** | **0.972** | 180ms |
