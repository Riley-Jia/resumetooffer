# AGENTS.md

## Project Background

This repository implements an AI Career Planning Agent.

Users fill in their Profile and Projects, then enter a job-search goal. The
system should help complete the core career-planning workflow:

- project profiling
- resume generation
- job retrieval and ranking
- skill gap analysis
- learning plan generation

## Core Architecture

- Backend uses FastAPI.
- Frontend uses React + TypeScript.
- The Agent layer is responsible for intent recognition, task planning, tool
  routing, and state management.
- The Tools layer is responsible for concrete business capabilities such as
  project profiling, resume generation, job matching, skill gap analysis, and
  next-step plan generation.
- The Ranking layer is responsible for job retrieval and ranking.
- The Schemas layer uses Pydantic to constrain API inputs, tool outputs, and
  structured response contracts.

## Development Principles

- Change only the files necessary for the current task.
- Do not perform broad refactors unless the user explicitly asks for them or
  the current task cannot be completed safely without them.
- New features must preserve the existing API behavior unless the user
  explicitly requests a breaking change.
- Agent and Tool outputs should use structured JSON whenever possible.
- Do not let the LLM invent user experience, projects, skills, education, or
  work history. Generated content must be grounded in user-provided Profile and
  Projects data.
- Job ranking should prioritize explainable rule-based logic. LLM reranking
  should only be used for fine-ranking an already selected candidate set.
- Keep model prompts explicit about data boundaries, required output shape, and
  fallback behavior.
- Prefer Pydantic schemas for cross-layer contracts instead of loosely shaped
  dictionaries.
- Keep frontend changes consistent with the current React + TypeScript style
  and avoid adding new UI libraries unless needed.

## Test and Check Commands

There is currently no dedicated automated test command in the project.

Use these checks before handing off changes:

```bash
cd frontend
npm run build
```

```bash
python3 -m compileall backend/app backend/*.py
```

TODO:

- Add backend API tests for FastAPI routes.
- Add tests for Agent planning, tool routing, and state transitions.
- Add tests for job ranking and LLM fallback behavior.
- Add frontend tests for critical user flows.
