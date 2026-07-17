# Resume to Offer

Resume to Offer 是一个 AI 辅助求职工作台，用来把用户的个人画像和项目经历转化为职业方向推荐、定制简历、岗位匹配、技能差距分析和短期行动计划。

Resume to Offer is an AI-assisted job application workspace for turning a user's profile and project experience into career direction recommendations, tailored resume drafts, job matches, skill gap analysis, and a short next-step plan.

当前项目是一个本地开发版 MVP，包含 FastAPI 后端、React TypeScript 前端、PostgreSQL 数据持久化、种子岗位数据，以及基于 OpenAI 的信息抽取、推理、排序和简历生成流程。

This project is currently a local-development MVP. It includes a FastAPI backend, a React TypeScript frontend, PostgreSQL persistence, seeded job data, and OpenAI-powered extraction, reasoning, ranking, and resume generation flows.

## 功能 / Features

- 个人画像和项目经历管理 / Profile and project management
- 自然语言项目经历抽取 / Natural-language project experience extraction
- 应用修改前的画像/项目编辑预览 / Profile/project edit preview before applying changes
- 职业方向推荐 / Career direction recommendations
- 面向目标方向的中文简历生成 / Targeted Chinese resume generation
- 本地种子初级 IT 岗位库 / Seeded junior IT job database
- 岗位匹配：元数据过滤、BM25 检索、向量检索、规则重排、可选 LLM 重排 / Job matching with metadata filters, BM25 retrieval, vector retrieval, rule reranking, and optional LLM reranking
- Top 3 岗位技能差距分析 / Top 3 job skill gap analysis
- 3 周下一步行动计划 / 3-week next-step plan generation
- 一键 Agent 工作流编排主要求职步骤 / Agent-style one-click workflow for orchestrating the main steps

## 技术栈 / Tech Stack

- 后端 / Backend: FastAPI, SQLAlchemy, Pydantic
- 数据库 / Database: PostgreSQL
- AI/模型集成 / AI and model integration: LangChain OpenAI
- 向量检索 / Vector search: ChromaDB
- 前端 / Frontend: React, TypeScript, Vite
- 本地基础设施 / Local infrastructure: Docker Compose

## 项目结构 / Project Structure

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

## 环境要求 / Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop 或其他兼容 Docker Compose 的运行环境 / Docker Desktop or another Docker Compose-compatible runtime
- OpenAI API key

## 环境变量 / Environment Variables

从示例文件创建后端环境变量文件：

Create a backend environment file from the example:

```bash
cp backend/.env.example backend/.env
```

后端必需变量：

Required backend variables:

```bash
DATABASE_URL=postgresql+psycopg://resume_user:resume_password@localhost:5432/resumetooffer
OPENAI_API_KEY=your_openai_api_key
```

可选模型覆盖配置：

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

如果你的 API key 暂时没有 GPT-5 模型权限，可以使用 `backend/.env.example` 里的 fallback 配置，例如 `gpt-4.1-nano` 和 `gpt-4.1-mini`。

If your API key does not have access to GPT-5 models, use the fallback values in `backend/.env.example`, such as `gpt-4.1-nano` and `gpt-4.1-mini`.

前端默认请求 `http://localhost:8000`。如需覆盖：

The frontend defaults to `http://localhost:8000`. To override it:

```bash
cp frontend/.env.example frontend/.env
```

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## 本地运行 / Run Locally

启动 PostgreSQL：

Start PostgreSQL:

```bash
docker compose up -d postgres
```

安装并启动后端：

Set up and run the backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python migrate_json_to_postgres.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

在另一个终端启动前端：

In another terminal, run the frontend:

```bash
cd frontend
npm install
npm run dev
```

打开终端里打印出的 Vite URL。默认情况下，前端会请求 `http://localhost:8000` 上的后端服务。

Open the Vite URL printed in the terminal. By default, the frontend talks to the backend at `http://localhost:8000`.

## 常用 API / Useful API Endpoints

健康检查和模型配置：

Health and config:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-status
```

个人画像和项目：

Profile and projects:

```bash
curl http://localhost:8000/profile-projects
curl http://localhost:8000/profile
curl http://localhost:8000/projects
```

项目经历抽取：

Project profiling:

```bash
curl -X POST http://localhost:8000/projects/profile/preview \
  -H 'Content-Type: application/json' \
  -d '{"text":"Built a React and FastAPI resume app from June 2026..."}'
```

职业方向推荐：

Career directions:

```bash
curl -X POST http://localhost:8000/career-directions/generate
curl http://localhost:8000/career-directions
```

简历生成：

Resume generation:

```bash
curl -X POST http://localhost:8000/resumes/generate \
  -H 'Content-Type: application/json' \
  -d '{"target_direction":"AI Application Developer"}'

curl http://localhost:8000/resumes
```

岗位匹配：

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

技能差距分析：

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

Agent 工作流：

Agent workflow:

```bash
curl -X POST http://localhost:8000/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我面向后端开发岗位优化简历，并分析还缺什么技能"}'
```

## 岗位匹配逻辑 / Job Matching Logic

后端启动时会从 `backend/data/jobs_seed.json` 向 PostgreSQL 写入 100 条国内初级 IT 岗位数据。匹配流程包括：

The backend seeds 100 junior domestic IT jobs into PostgreSQL on startup from `backend/data/jobs_seed.json`. Matching combines:

1. 元数据过滤 / metadata filtering
2. BM25 关键词检索 / BM25 keyword retrieval
3. ChromaDB 向量检索 / ChromaDB vector retrieval
4. 候选结果合并 / candidate merge
5. 规则重排 / rule-based reranking
6. 对短名单进行 LLM 重排 / LLM reranking on the shortlist

规则分数更强调技能覆盖度：

The rule score is weighted toward skill coverage:

```text
rule_score = skill_coverage * 50
           + location_score * 20
           + level_score * 15
           + role_family_score * 15

final_score = rule_score * 0.7 + llm_score * 0.3
```

## 数据模型说明 / Data Model Notes

- `profiles.id = 1` 作为默认本地用户画像 / `profiles.id = 1` is used as the default local profile.
- `projects.profile_id` 指向 `profiles.id` / `projects.profile_id` points to `profiles.id`.
- 生成的简历存储在 `generated_resumes` / Generated resumes are stored in `generated_resumes`.
- 种子岗位存储在 `jobs` / Seeded jobs are stored in `jobs`.
- `backend/data/profile_projects.json` 保留为本地种子数据迁移来源 / `backend/data/profile_projects.json` is kept as a migration source for local seed data.

## 开发检查 / Development Checks

前端构建：

Frontend build:

```bash
cd frontend
npm run build
```

后端语法检查：

Backend syntax check:

```bash
python3 -m compileall backend/app backend/*.py
```

## GitHub 上传准备 / GitHub Readiness

初始化 Git 后，只要生成文件和本地私密文件仍被忽略，这个仓库就可以发布。当前 `.gitignore` 会排除：

This repository is safe to publish after initializing Git, as long as generated and local files stay ignored. The current `.gitignore` excludes:

- `.env` and `backend/.env`
- `.venv`
- `node_modules`
- `frontend/dist`
- Python `__pycache__` and bytecode files

推送前检查：

Before pushing, check:

```bash
git status
```

确认没有密钥、虚拟环境、依赖目录或构建产物被加入暂存区。

Make sure no secrets, virtual environments, dependency directories, or build artifacts are staged.

## 当前状态 / Current Status

这是一个可运行的本地 MVP。主要产品流程已经实现，前端生产构建可以通过。距离生产级项目最大的缺口是自动化测试覆盖，包括后端路由、模型 fallback 行为、岗位匹配逻辑和前端关键流程。

This is a functional local MVP. The main product workflow is implemented, and the frontend production build passes. The biggest remaining gap before treating it as production-ready is automated test coverage for backend routes, model fallback behavior, job matching, and frontend critical flows.
