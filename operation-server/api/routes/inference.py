from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import sys
import os
import uuid
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.schemas import InferenceRequest, InferenceResponse, InferenceResult, JobStatus
from celery_tasks.client import submit_inference_task
from database.redis_client import redis_client
from database.postgres_client import postgres_client
from messaging.websocket_manager import websocket_manager
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/submit", response_model=InferenceResponse)
async def submit_inference(request: InferenceRequest):
    """Submit an inference job to analysis server"""
    try:
        job_id = str(uuid.uuid4())
        metadata = request.metadata or {}
        metadata["created_at"] = datetime.utcnow().isoformat()
        metadata["submitted_from"] = "operation_server"
        
        redis_client.set_job_status(job_id, JobStatus.PENDING)
        
        submit_inference_task(
            job_id=job_id,
            model_name=request.model_name,
            data=request.data,
            config=request.config or {},
            metadata=metadata
        )
        
        logger.info(f"Submitted inference job {job_id} to analysis server")
        
        return InferenceResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Inference job submitted to analysis server"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting inference job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get inference job status"""
    status = redis_client.get_job_status(job_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/result/{job_id}", response_model=InferenceResult)
async def get_result(job_id: str):
    """Get inference job result"""
    result = redis_client.get_job_result(job_id)
    if not result:
        db_result = postgres_client.get_result(job_id)
        if not db_result:
            raise HTTPException(status_code=404, detail="Result not found")
        return db_result
    return result


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await websocket_manager.connect(websocket, job_id)
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "status":
                status = redis_client.get_job_status(job_id)
                await websocket_manager.send_personal_message(status, websocket)
            elif data == "result":
                result = redis_client.get_job_result(job_id)
                if result:
                    await websocket_manager.send_result(job_id, result)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket, job_id)
