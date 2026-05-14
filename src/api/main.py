"""FastAPI application entry point.

Start with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import initialise, router
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup/shutdown hooks.

    Loads the scaler, feature names, and any existing model checkpoints into
    memory before the server starts accepting requests.
    """
    log.info("API startup — loading models and artefacts …")
    try:
        initialise()
        log.info("API startup complete.")
    except Exception as exc:
        log.warning("Startup artefact loading failed (non-fatal): %s", exc)
    yield
    log.info("API shutdown.")


app = FastAPI(
    title="Market Pulse Predictor API",
    description=(
        "Real-time stock market direction and sentiment prediction.\n\n"
        "Supports AAPL, MSFT, GOOGL, AMZN, TSLA, META using "
        "RNN / LSTM / GRU / BiLSTM-Attention models trained on "
        "financial news sentiment and technical indicators."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ─────────────────────────────────────────────────────────────────── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────── #
app.include_router(router)


# ── Global exception handler ──────────────────────────────────────────────── #
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "path": str(request.url)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
