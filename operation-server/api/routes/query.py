from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from database.postgres_client import postgres_client
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/history")
async def get_results_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    model_name: Optional[str] = None
):
    """Get inference results history"""
    try:
        results = postgres_client.get_results_history(
            limit=limit,
            offset=offset,
            model_name=model_name
        )
        return {
            "results": results,
            "count": len(results),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")


@router.get("/{job_id}")
async def get_result_by_id(job_id: str):
    """Get specific inference result by job ID"""
    try:
        result = postgres_client.get_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching result: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch result")


@router.get("/stats/summary")
async def get_statistics():
    """Get inference statistics summary"""
    try:
        results = postgres_client.get_results_history(limit=1000)
        
        total_jobs = len(results)
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        models_used = {}
        for r in results:
            model = r.get('model_name')
            if model:
                models_used[model] = models_used.get(model, 0) + 1
        
        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total_jobs * 100) if total_jobs > 0 else 0,
            "models_used": models_used
        }
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")
