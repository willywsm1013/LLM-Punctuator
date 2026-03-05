FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY llm_punctuator/ llm_punctuator/

RUN uv sync --frozen --no-dev --no-editable

EXPOSE 8000

CMD [".venv/bin/python", "-m", "llm_punctuator.server"]
