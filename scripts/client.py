"""Simple client script for testing the LLM Punctuator API.

Usage:
    python scripts/client.py "你好世界今天天氣真好"
    python scripts/client.py "hello world how are you" --language en
    python scripts/client.py "你好世界" --base-url http://localhost:9000
"""

import argparse
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Punctuator API client")
    parser.add_argument("text", help="Text to punctuate")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # Health check
    try:
        health = httpx.get(f"{base}/health", timeout=5)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {base}", file=sys.stderr)
        sys.exit(1)

    if health.status_code != 200:
        print(f"Server unhealthy: {health.json()}", file=sys.stderr)
        sys.exit(1)

    # Punctuate
    resp = httpx.post(
        f"{base}/api/v1/punctuate",
        json={"text": args.text, "language": args.language, "chunk_size": args.chunk_size},
        timeout=120,
    )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    print(data["text"])


if __name__ == "__main__":
    main()
