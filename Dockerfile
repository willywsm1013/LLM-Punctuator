FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY llm_punctuator/ llm_punctuator/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "llm_punctuator"]
