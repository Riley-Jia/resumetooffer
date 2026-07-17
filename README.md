# Resume to Offer

Resume to Offer is an AI-assisted job application workspace for turning a user's
profile and project experience into career direction recommendations, tailored
resume drafts, job matches, skill gap analysis, and a short next-step learning
plan.

The project is currently an MVP for local development. It includes a FastAPI
backend, a React TypeScript frontend, PostgreSQL persistence, seeded job data,
and OpenAI-powered extraction, reasoning, ranking, and resume generation flows.

## Features

- Profile and project management
- Natural-language project experience extraction
- Profile/project edit preview before applying changes
- Career direction recommendations
- Targeted Chinese resume generation
- Seeded junior IT job database
- Job matching with metadata filters, BM25 retrieval, vector retrieval, rule
  reranking, and optional LLM reranking
- Top 3 job skill gap analysis
- 3-week next-step plan generation
- Agent-style one-click workflow that can orchestrate the main steps

## Tech Stack

- Backend: FastAPI, SQLAlchemy, Pydantic
- Database: PostgreSQL
- AI/model integration: LangChain OpenAI
- Vector search: ChromaDB
- Frontend: React, TypeScript, Vite
- Local infrastructure: Docker Compose

## Project Structure

```text
.
├── backend
│   ├── app
│   │   ├── main.py
│   │   ├── agent_orchestrator.py
│   │   ├── career_direction.py
│   │   ├── job_matching.py
│   │   ├── profiling.py
│   │   ├── resume_generation.py
│   │   └── skill_gap.py
│   ├── data
│   │   ├── jobs_seed.json
│   │   └── profile_projects.json
│   ├── migrate_json_to_postgres.py
│   ├── seed_profile_projects.py
│   └── requirements.txt
├── frontend
│   ├── src
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
└── README.md
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop or another Docker Compose-compatible runtime
- OpenAI API key

## Environment Variables

Create a backend environment file from the example:

```bash
cp backend/.env.example backend/.env
```

Required backend variables:

```bash
DATABASE_URL=postgresql+psycopg://resume_user:resume_password@localhost:5432/resumetooffer
OPENAI_API_KEY=your_openai_api_key
```

Optional model overrides:

```bash
PROJECT_PROFILING_MODEL=gpt-5-nano
CAREER_DIRECTION_MODEL=gpt-5-nano
RESUME_GENERATION_MODEL=gpt-5-mini
JOB_MATCHING_MODEL=gpt-5-nano
NEXT_STEP_PLAN_MODEL=gpt-5-nano
AGENT_ORCHESTRATOR_MODEL=gpt-5-nano
PROFILE_PROJECT_EDITING_MODEL=gpt-5-nano
INPUT_ROUTER_MODEL=gpt-5-nano
```

If your API key does not have access to GPT-5 models, use the fallback values in
`backend/.env.example`, such as `gpt-4.1-nano` and `gpt-4.1-mini`.

The frontend defaults to `http://localhost:8000`. To override it:

```bash
cp frontend/.env.example frontend/.env
```

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Run Locally

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Set up and run the backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python migrate_json_to_postgres.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal, run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal. By default, the frontend talks to the
backend at `http://localhost:8000`.

## Useful API Endpoints

Health and config:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-status
```

Profile and projects:

```bash
curl http://localhost:8000/profile-projects
curl http://localhost:8000/profile
curl http://localhost:8000/projects
```

Project profiling:

```bash
curl -X POST http://localhost:8000/projects/profile/preview \
  -H 'Content-Type: application/json' \
  -d '{"text":"Built a React and FastAPI resume app from June 2026..."}'
```

Career directions:

```bash
curl -X POST http://localhost:8000/career-directions/generate
curl http://localhost:8000/career-directions
```

Resume generation:

```bash
curl -X POST http://localhost:8000/resumes/generate \
  -H 'Content-Type: application/json' \
  -d '{"target_direction":"AI Application Developer"}'

curl http://localhost:8000/resumes
```

Job matching:

```bash
curl -X POST http://localhost:8000/job-matches/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "target_direction":"Backend Developer",
    "locations":["北京","上海","深圳"],
    "levels":["实习","校招","初级","应届","1年以内"],
    "role_families":["Backend","AI Application","Graduate Software Engineer"],
    "status":"active",
    "top_k":10,
    "llm_candidate_count":20
  }'
```

Skill gap analysis:

```bash
curl -X POST http://localhost:8000/skill-gap/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "target_direction":"Backend Developer",
    "user_skills":["Python","FastAPI","SQL"],
    "jobs":[]
  }'
```

Agent workflow:

```bash
curl -X POST http://localhost:8000/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我面向后端开发岗位优化简历，并分析还缺什么技能"}'
```

## Job Matching Logic

The backend seeds 100 junior domestic IT jobs into PostgreSQL on startup from
`backend/data/jobs_seed.json`. Matching combines:

1. metadata filtering
2. BM25 keyword retrieval
3. ChromaDB vector retrieval
4. candidate merge
5. rule-based reranking
6. LLM reranking on the shortlist

The rule score is weighted toward skill coverage:

```text
rule_score = skill_coverage * 50
           + location_score * 20
           + level_score * 15
           + role_family_score * 15

final_score = rule_score * 0.7 + llm_score * 0.3
```

## Data Model Notes

- `profiles.id = 1` is used as the default local profile.
- `projects.profile_id` points to `profiles.id`.
- Generated resumes are stored in `generated_resumes`.
- Seeded jobs are stored in `jobs`.
- `backend/data/profile_projects.json` is kept as a migration source for local
  seed data.

## Development Checks

Frontend build:

```bash
cd frontend
npm run build
```

Backend syntax check:

```bash
python3 -m compileall backend/app backend/*.py
```

## GitHub Readiness

This repository is safe to publish after initializing Git, as long as generated
and local files stay ignored. The current `.gitignore` excludes:

- `.env` and `backend/.env`
- `.venv`
- `node_modules`
- `frontend/dist`
- Python `__pycache__` and bytecode files

Before pushing, check:

```bash
git status
```

Make sure no secrets, virtual environments, dependency directories, or build
artifacts are staged.

## Current Status

This is a functional local MVP. The main product workflow is implemented, and
the frontend production build passes. The biggest remaining gap before treating
it as production-ready is automated test coverage for backend routes, model
fallback behavior, job matching, and frontend critical flows.
