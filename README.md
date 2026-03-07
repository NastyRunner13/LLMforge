# 🔥 LLM Forge

**Fine-tune any open-source LLM with zero code — from a single dashboard.**

<p align="center">
  <img src="https://img.shields.io/badge/status-MVP%20In%20Progress-orange?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Next.js-16-black?style=for-the-badge&logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Node.js-20+-339933?style=for-the-badge&logo=node.js&logoColor=white" alt="Node" />
</p>

LLM Forge is a full-stack platform that lets developers upload training data, pick a base model (LLaMA, Mistral, Qwen, Gemma, etc.), fine-tune with LoRA/QLoRA, monitor training in real-time, and deploy the resulting model as an OpenAI-compatible API endpoint — all without writing ML code.

---

## 📑 Table of Contents

- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [Database Schema](#-database-schema)
- [Supported Models](#-supported-models)
- [Current Status](#-current-status--whats-implemented)
- [Roadmap — What Needs to Be Built](#-roadmap--what-needs-to-be-built)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         USERS / BROWSER                           │
└────────────────────────┬───────────────────────────────────────────┘
                         │ HTTPS
┌────────────────────────▼───────────────────────────────────────────┐
│                    apps/web  (Next.js 16)                         │
│                                                                   │
│  Landing Page ─► Auth (NextAuth v5) ─► Dashboard                  │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐  │
│  │ Projects │   │ Datasets │   │ Training  │   │  Endpoints   │  │
│  │  page    │   │  page    │   │   page    │   │    page      │  │
│  └──────────┘   └──────────┘   └───────────┘   └──────────────┘  │
│                                                                   │
│  Prisma ORM ──► PostgreSQL         API Routes ──► ML Service      │
└────────────────────────┬───────────────────────────────────────────┘
                         │ Internal HTTP (X-Internal-Key)
┌────────────────────────▼───────────────────────────────────────────┐
│                apps/ml-service  (FastAPI)                          │
│                                                                   │
│  /api/datasets   /api/training   /api/models   /api/inference     │
│  (Upload, Clean, (Launch, Pause, (Registry,    (OpenAI-compat     │
│   Preview, Format) Resume, Cancel) Deploy,      chat completions, │
│                                   Download)    endpoint mgmt)     │
│                                                                   │
│  WebSocket ─► /ws/runs/{id}/metrics  (real-time training stream)  │
└──────┬────────────────┬───────────────────────────────────────────-┘
       │                │ Task Queues
       │    ┌───────────▼─────────────┐
       │    │   Celery Workers        │
       │    │                         │
       │    │  data queue:            │
       │    │    • run_cleaning       │
       │    │    • convert_format     │
       │    │    • count_tokens       │
       │    │                         │
       │    │  training queue:        │
       │    │    • launch_training    │
       │    │    • save_checkpoint    │
       │    │    • deploy_model       │
       │    └───────────┬─────────────┘
       │                │
┌──────▼────────────────▼───────────────────────────────────────────┐
│                    Infrastructure                                 │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────────────────┐  │
│  │PostgreSQL│   │  Redis   │   │  MinIO (S3-compatible)        │  │
│  │   16     │   │    7     │   │                               │  │
│  │          │   │          │   │  Buckets:                     │  │
│  │ - Users  │   │ - Celery │   │    llmforge-datasets          │  │
│  │ - Runs   │   │   broker │   │    llmforge-checkpoints       │  │
│  │ - Models │   │ - Pub/Sub│   │    llmforge-models            │  │
│  │ - Billing│   │ - Cache  │   │                               │  │
│  └──────────┘   └──────────┘   └───────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 16, React 19, TailwindCSS 4 | Dashboard UI + SSR |
| **Auth** | NextAuth v5 (beta) | Email/password, Google, GitHub OAuth |
| **Frontend ORM** | Prisma 7.4 | Type-safe DB access from Next.js |
| **Backend API** | FastAPI + Uvicorn | ML operations REST API + WebSocket |
| **Task Queue** | Celery 5.4 + Redis | Async data processing & training jobs |
| **Training** | HuggingFace Transformers, TRL, PEFT | SFT, LoRA, QLoRA fine-tuning |
| **NLP** | spaCy 3.8 | PII redaction in data cleaning |
| **Tokenization** | tiktoken | Token counting for datasets |
| **Database** | PostgreSQL 16 | Primary data store |
| **Cache / Broker** | Redis 7 | Celery broker, Pub/Sub for real-time metrics |
| **Object Storage** | MinIO (S3-compatible) | Datasets, checkpoints, model weights |
| **Monorepo** | pnpm workspaces + Turborepo | Build orchestration |
| **Containerization** | Docker Compose | Local development infrastructure |
| **Shared Types** | `@llmforge/shared` (TypeScript) | Type-safe contracts between frontend & backend |
| **Payments** | Stripe (planned) | Credit-based billing |
| **Email** | Resend (planned) | Transactional emails |

---

## 📁 Project Structure

```
LLMForge/
├── apps/
│   ├── web/                        # Next.js 16 frontend
│   │   ├── prisma/
│   │   │   ├── schema.prisma       # Database schema (14 models)
│   │   │   └── seed.ts             # Database seed script
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── page.tsx        # Landing page
│   │   │   │   ├── login/          # Sign in page
│   │   │   │   ├── signup/         # Sign up page
│   │   │   │   ├── onboarding/     # User onboarding
│   │   │   │   ├── dashboard/
│   │   │   │   │   ├── page.tsx    # Dashboard overview
│   │   │   │   │   ├── datasets/   # Dataset management
│   │   │   │   │   ├── training/   # Training runs
│   │   │   │   │   ├── models/     # Model registry
│   │   │   │   │   ├── endpoints/  # Inference endpoints
│   │   │   │   │   ├── eval/       # Model evaluation
│   │   │   │   │   ├── billing/    # Credits & billing
│   │   │   │   │   └── settings/   # User/workspace settings
│   │   │   │   └── api/            # Next.js API routes
│   │   │   │       ├── auth/       # NextAuth endpoints
│   │   │   │       ├── projects/   # CRUD for projects
│   │   │   │       ├── users/      # User management
│   │   │   │       ├── notifications/ # Notification API
│   │   │   │       └── onboarding/ # Onboarding API
│   │   │   ├── components/         # Reusable UI components
│   │   │   ├── lib/
│   │   │   │   ├── auth.ts         # NextAuth configuration
│   │   │   │   ├── prisma.ts       # Prisma client singleton
│   │   │   │   ├── ml-service.ts   # ML Service HTTP client
│   │   │   │   └── env.ts          # Validated environment vars
│   │   │   └── types/              # TypeScript types
│   │   └── package.json
│   │
│   └── ml-service/                 # FastAPI ML backend
│       ├── app/
│       │   ├── main.py             # FastAPI app entry point
│       │   ├── api/
│       │   │   ├── health.py       # Health check endpoint
│       │   │   ├── datasets.py     # Dataset CRUD + cleaning
│       │   │   ├── training.py     # Training run orchestration
│       │   │   ├── models.py       # Model registry + deploy
│       │   │   └── inference.py    # OpenAI-compat inference API
│       │   ├── core/
│       │   │   ├── config.py       # Pydantic settings
│       │   │   ├── database.py     # SQLAlchemy engine
│       │   │   ├── celery_app.py   # Celery configuration
│       │   │   ├── security.py     # Internal API key auth
│       │   │   └── storage.py      # S3/MinIO client
│       │   ├── models/             # SQLAlchemy models
│       │   ├── services/           # Business logic layer
│       │   ├── training/           # Training orchestration
│       │   └── workers/
│       │       ├── data_tasks.py   # Celery tasks: cleaning, format, tokens
│       │       └── training_tasks.py # Celery tasks: train, checkpoint, deploy
│       ├── tests/
│       ├── Dockerfile              # Multi-stage production build
│       └── pyproject.toml
│
├── packages/
│   └── shared/                     # @llmforge/shared
│       └── src/index.ts            # Shared types, enums, model registry, GPU pricing
│
├── infra/
│   └── docker/
│       └── docker-compose.yml      # PostgreSQL, Redis, MinIO, FastAPI, Celery
│
├── package.json                    # Root workspace config
├── pnpm-workspace.yaml             # Workspace definition
├── turbo.json                      # Turborepo task pipeline
└── .prettierrc                     # Code formatting config
```

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

| Tool | Version | Purpose |
|---|---|---|
| **Node.js** | ≥ 20.0.0 | JavaScript runtime |
| **pnpm** | 9.15.0 | Package manager |
| **Python** | ≥ 3.11 | ML service runtime |
| **Docker** & **Docker Compose** | Latest | Infrastructure services |
| **Git** | Latest | Version control |

**Optional (for GPU training):**
- NVIDIA GPU with CUDA support
- `nvidia-docker2` runtime

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/LLMForge.git
cd LLMForge
```

### 2. Start Infrastructure Services

```bash
cd infra/docker
docker compose up -d
```

This starts:
- **PostgreSQL** on `localhost:5432` (user: `postgres`, pass: `postgres`, db: `llmforge`)
- **Redis** on `localhost:6379`
- **MinIO** on `localhost:9000` (API) / `localhost:9001` (Console, user: `minioadmin`, pass: `minioadmin`)

### 3. Set Up the Next.js Frontend

```bash
# Install all workspace dependencies from the root
pnpm install

# Set up environment variables
cp apps/web/.env.example apps/web/.env.local

# Generate Prisma client
cd apps/web
pnpm db:generate

# Push schema to the database (creates tables)
pnpm db:push

# (Optional) Seed the database with sample data
pnpm db:seed

# Return to root
cd ../..
```

**Important:** Edit `apps/web/.env.local` and set:
- `NEXTAUTH_SECRET` — Generate with: `openssl rand -base64 32`
- OAuth credentials (Google/GitHub) if you want social login

### 4. Set Up the ML Service

```bash
# Set up environment variables
cp apps/ml-service/.env.example apps/ml-service/.env

# Create a Python virtual environment
cd apps/ml-service
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Return to root
cd ../..
```

**Important:** Edit `apps/ml-service/.env` and set:
- `HF_TOKEN` — Your [HuggingFace access token](https://huggingface.co/settings/tokens) (required for gated models like LLaMA)

### 5. Start Development Servers

**Option A — Run everything with Turborepo (frontend only):**
```bash
pnpm dev
```

**Option B — Run services individually:**

```bash
# Terminal 1: Next.js frontend
cd apps/web && pnpm dev

# Terminal 2: FastAPI backend
cd apps/ml-service && uvicorn app.main:app --reload --port 8000

# Terminal 3: Celery data worker
cd apps/ml-service && celery -A app.core.celery_app worker -Q data -l info --concurrency=2

# Terminal 4: Celery training worker
cd apps/ml-service && celery -A app.core.celery_app worker -Q training -l info --concurrency=1
```

**Option C — Run everything with Docker Compose:**
```bash
cd infra/docker
docker compose up -d  # Starts all services including FastAPI & Celery
```

### 6. Access the Application

| Service | URL |
|---|---|
| **Web App** | http://localhost:3000 |
| **FastAPI Docs** | http://localhost:8000/docs |
| **FastAPI ReDoc** | http://localhost:8000/redoc |
| **MinIO Console** | http://localhost:9001 |

---

## 🔑 Environment Variables

### `apps/web/.env.local`

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `NEXTAUTH_URL` | ✅ | App URL (http://localhost:3000 for dev) |
| `NEXTAUTH_SECRET` | ✅ | Secret for signing sessions |
| `ML_SERVICE_URL` | ✅ | FastAPI service URL (http://localhost:8000) |
| `INTERNAL_API_SECRET` | ✅ | Shared secret for inter-service auth |
| `GOOGLE_CLIENT_ID` | ❌ | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | ❌ | Google OAuth client secret |
| `GITHUB_CLIENT_ID` | ❌ | GitHub OAuth client ID |
| `GITHUB_CLIENT_SECRET` | ❌ | GitHub OAuth client secret |
| `STRIPE_SECRET_KEY` | ❌ | Stripe API key (for billing) |
| `RESEND_API_KEY` | ❌ | Resend API key (for emails) |

### `apps/ml-service/.env`

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ✅ | Redis connection string |
| `S3_ENDPOINT_URL` | ✅ | MinIO/S3 endpoint |
| `S3_ACCESS_KEY` | ✅ | MinIO/S3 access key |
| `S3_SECRET_KEY` | ✅ | MinIO/S3 secret key |
| `CELERY_BROKER_URL` | ✅ | Redis URL for Celery broker |
| `INTERNAL_API_SECRET` | ✅ | Must match the web app's value |
| `HF_TOKEN` | ❌* | HuggingFace token (*required for gated models) |

---

## 📡 API Reference

### ML Service Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| **Datasets** | | |
| `POST` | `/api/datasets/upload` | Get presigned S3 upload URL |
| `GET` | `/api/datasets/{id}` | Get dataset metadata |
| `GET` | `/api/datasets/{id}/preview` | Preview dataset rows (paginated) |
| `POST` | `/api/datasets/{id}/clean` | Launch cleaning pipeline |
| `POST` | `/api/datasets/{id}/format` | Apply instruction format mapping |
| `DELETE` | `/api/datasets/{id}` | Delete dataset |
| **Training** | | |
| `POST` | `/api/training/runs` | Launch a training job |
| `GET` | `/api/training/runs/{id}` | Get run status & details |
| `GET` | `/api/training/runs/{id}/metrics` | Get training metrics (time-series) |
| `GET` | `/api/training/runs/{id}/logs` | Get training logs |
| `POST` | `/api/training/runs/{id}/pause` | Pause a running job |
| `POST` | `/api/training/runs/{id}/resume` | Resume from checkpoint |
| `POST` | `/api/training/runs/{id}/cancel` | Cancel a job |
| `WS` | `/ws/runs/{id}/metrics` | Real-time metrics stream |
| **Models** | | |
| `GET` | `/api/models` | List registered models |
| `GET` | `/api/models/{id}` | Get model details |
| `POST` | `/api/models/{id}/deploy` | Deploy to inference endpoint |
| `GET` | `/api/models/{id}/download` | Get signed download URL |
| `DELETE` | `/api/models/{id}` | Delete model |
| **Inference** | | |
| `GET` | `/api/inference/endpoints` | List all endpoints |
| `GET` | `/api/inference/endpoints/{id}` | Get endpoint status |
| `POST` | `/api/inference/endpoints/{id}/stop` | Stop an endpoint |
| `POST` | `/api/inference/v1/chat/completions` | OpenAI-compatible chat API |

All endpoints (except health and chat completions) require the `X-Internal-Key` header.

---

## 🗄 Database Schema

The Prisma schema defines **14 models** organized across these domains:

| Domain | Models | Description |
|---|---|---|
| **Auth** | Account, Session, VerificationToken | NextAuth v5 adapter tables |
| **Users** | User, Workspace, WorkspaceMember | Multi-tenant user management |
| **Projects** | Project | Project containers for training workflows |
| **Data** | Dataset | Training dataset metadata & status tracking |
| **Training** | TrainingRun, Checkpoint, RunMetric | Full training lifecycle with time-series metrics |
| **Models** | Model | Trained model registry |
| **Inference** | Endpoint, EndpointApiKey | Managed inference endpoint tracking |
| **Billing** | CreditLedger | Credit-based usage tracking |
| **System** | UserApiKey, Notification | API keys & in-app notifications |

---

## 🤖 Supported Models

| Model | Org | Params | VRAM | License |
|---|---|---|---|---|
| LLaMA 3 8B | Meta | 8B | 24 GB | Llama 3 Community |
| Mistral 7B | Mistral AI | 7B | 20 GB | Apache 2.0 |
| Phi-3 Mini | Microsoft | 3.8B | 12 GB | MIT |
| Qwen 2 7B | Alibaba | 7B | 20 GB | Apache 2.0 |
| Gemma 2 9B | Google | 9B | 28 GB | Gemma License |
| TinyLlama 1.1B | TinyLlama | 1.1B | 6 GB | Apache 2.0 |

**Training Methods:** SFT (Full Fine-Tuning), LoRA, QLoRA

---

## 📊 Current Status — What's Implemented

### ✅ Done

| Area | What's Built |
|---|---|
| **Monorepo Setup** | pnpm workspaces + Turborepo pipeline (dev, build, lint, typecheck) |
| **Web — Landing Page** | Full landing page with hero, features grid, model tags, CTA, footer |
| **Web — Auth Pages** | Login, signup pages + NextAuth v5 config (email/password + OAuth providers) |
| **Web — Dashboard Shell** | Dashboard layout with sidebar navigation across all sections |
| **Web — Dashboard Overview** | Stats cards, project list, activity feed, credits widget (all mock data) |
| **Web — Page Stubs** | Datasets, Training, Models, Endpoints, Eval, Billing, Settings pages |
| **Web — API Routes** | Auth, projects CRUD, users, onboarding, notifications |
| **Web — Shared Components** | DataTable, EmptyState, FileDropzone, TabNav, NotificationsPanel |
| **Web — ML Service Client** | Fully typed HTTP client for all FastAPI endpoints |
| **Database Schema** | Complete Prisma schema (14 models, all enums, indexes, relations) |
| **ML Service — API Skeleton** | All route handlers defined with Pydantic models for request/response |
| **ML Service — Core** | Config, database engine, Celery app, security middleware, S3 client |
| **ML Service — Workers** | Celery task definitions for data processing and training |
| **Shared Package** | TypeScript types, base model registry, GPU pricing, plan limits |
| **Docker Compose** | Full local dev infra (PostgreSQL, Redis, MinIO, FastAPI, 2× Celery) |
| **Dockerfile** | Multi-stage Python build for ML service |
| **Env Configuration** | `.env.example` files for both web and ML service |

### 🚧 Not Yet Implemented (TODO stubs)

All ML service API handlers return placeholder responses. The actual implementation is needed.

---

## 🗺 Roadmap — What Needs to Be Built

### 🔴 Critical (MVP Blockers)

1. **ML Service — Database Integration**
   - Implement SQLAlchemy models mirroring the Prisma schema
   - Run Alembic migrations to create tables
   - Wire up CRUD operations in all API route handlers

2. **ML Service — Dataset Processing Pipeline**
   - Implement presigned URL generation for direct S3 uploads
   - Build the cleaning pipeline workers (dedup, language filter, PII redact, etc.)
   - Implement file format conversion (CSV, PDF, DOCX → JSONL)
   - Wire up token counting with tiktoken

3. **ML Service — Training Engine**
   - Implement the `launch_training_job` Celery task using HuggingFace TRL + PEFT
   - Download base models from HuggingFace Hub
   - Configure SFT Trainer with LoRA/QLoRA adapters
   - Save checkpoints to S3 at configured intervals
   - Stream real-time metrics via Redis Pub/Sub → WebSocket

4. **ML Service — Model Registry**
   - Register trained models after training completes
   - Generate presigned download URLs for model weights
   - Implement model deletion with S3 cleanup

5. **Web — Connect to Real Data**
   - Replace all mock/hardcoded data in dashboard with Prisma queries
   - Wire up project creation flow
   - Build dataset upload UI → presigned URL → S3 direct upload
   - Build training configuration form → launch via ML service
   - Build real-time training monitoring (WebSocket metrics charts)

6. **Web — Auth Flow Completion**
   - Test and verify the NextAuth sign-up/sign-in flow end-to-end
   - Implement session-based route protection
   - Wire up workspace creation during onboarding

### 🟡 Important (Post-MVP)

7. **Inference Deployment**
   - Integrate vLLM or TGI for model serving
   - Implement container lifecycle management (deploy, stop, scale)
   - Build the OpenAI-compatible `/v1/chat/completions` proxy

8. **Billing & Credits**
   - Integrate Stripe for credit purchases
   - Implement credit deduction during training (GPU-seconds tracking)
   - Build credit balance checks before launching jobs

9. **Evaluation**
   - Build model evaluation page with benchmark datasets
   - Implement evaluation metrics (perplexity, accuracy, etc.)

10. **Email Notifications**
    - Integrate Resend for transactional emails
    - Training complete/failed notifications
    - Low credit warnings

### 🟢 Nice to Have

11. **CI/CD Pipeline** — GitHub Actions for lint, typecheck, tests, Docker builds
12. **Rate Limiting** — API rate limiting on inference endpoints
13. **Multi-GPU Training** — Distributed training support with DeepSpeed/FSDP
14. **Model Comparison** — Side-by-side model evaluation playground
15. **Team Features** — Workspace invitations, role-based access control
16. **Prompt Playground** — Interactive testing UI for deployed models

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feat/my-feature`
5. Open a Pull Request

### Development Commands

```bash
# Install dependencies
pnpm install

# Start all dev servers
pnpm dev

# Lint all packages
pnpm lint

# Type-check all packages
pnpm typecheck

# Format code
pnpm format

# Database management (from apps/web/)
pnpm db:generate     # Generate Prisma client
pnpm db:push         # Push schema to DB
pnpm db:migrate      # Create a migration
pnpm db:seed         # Seed with sample data
pnpm db:studio       # Open Prisma Studio GUI
```

---

## 📄 License

MIT
---

<p align="center">
  Built with ❤️ using Next.js, FastAPI, and HuggingFace
</p>
