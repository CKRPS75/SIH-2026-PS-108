# StandardWise Backend

FastAPI foundation for StandardWise (Manak-AI), an auditable procurement assistant for verified Indian Standards.

## Prerequisites

- Python 3.11
- Docker Desktop for PostgreSQL, Qdrant, and Neo4j

## Local development

From the repository root in PowerShell:

```powershell
cd Backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d postgres qdrant neo4j
alembic upgrade head
uvicorn app.main:app --reload
```

The API is served at `http://localhost:8000` and OpenAPI documentation is at `http://localhost:8000/docs`.

Run the whole local stack with `docker compose up --build`.

## Verification commands

```powershell
pytest -q
ruff check .
alembic upgrade head
```

## Implemented API

- `GET /health` reports process liveness.
- `GET /ready` reports PostgreSQL, Qdrant, and Neo4j. PostgreSQL is required for a ready response.
- `POST /api/v1/standards/validate` resolves a stored IS identifier. Missing identifiers return `STANDARD_NOT_FOUND`; the API does not invent replacement standards.

```json
{
  "standard_id": "IS 16910 (Part 1): 2020"
}
```

## Phase 2 scope

- Import a curated BIS catalog into PostgreSQL.
- Index verified standards in Qdrant and construct the BM25 corpus.
- Implement reciprocal-rank fusion, reranking, and `POST /api/v1/search`.
- Load Neo4j normative-reference edges and QCO rules.