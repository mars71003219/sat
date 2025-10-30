from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config.settings import settings
from api.routes.inference import router as inference_router
from api.routes.query import router as query_router
from database.postgres_client import postgres_client
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Operation Server...")
    
    try:
        postgres_client.init_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
        logger.warning("Redis connection failed")
    
    logger.info(f"Operation Server started - {settings.APP_VERSION}")
    
    yield
    
    logger.info("Shutting down Operation Server...")


app = FastAPI(
    title="Operation Server",
    version=settings.APP_VERSION,
    description="Operation server for managing inference requests and results",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "service": "Operation Server",
        "version": settings.APP_VERSION,
        "status": "running",
        "type": "api_server"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server_type": "operation_server"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Search router 추가
from api.routes.search import router as search_router
app.include_router(search_router, prefix="/api/v1")

# Dashboard router 추가
from api.routes.dashboard import router as dashboard_router
app.include_router(dashboard_router, prefix="/api/v1")

# Trends router 추가
from api.routes.trends import router as trends_router
app.include_router(trends_router, prefix="/api/v1")
