"""Entry point for running the server with `python -m llm_punctuator`."""

import uvicorn

from llm_punctuator.server import app, get_settings


def main() -> None:
    """Run the server."""
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
