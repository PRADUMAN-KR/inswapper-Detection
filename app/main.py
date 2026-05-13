from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import reload_model
from app.routers import admin, detection, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    reload_model(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(detection.router, prefix="/detect", tags=["detection"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])
    return app


app = create_app()

