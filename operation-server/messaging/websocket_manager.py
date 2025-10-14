from fastapi import WebSocket
from typing import Dict, List
import json
import asyncio
from utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove a WebSocket connection"""
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    async def broadcast_to_job(self, job_id: str, message: dict):
        """Broadcast message to all connections for a job"""
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast message: {e}")
    
    async def send_progress(self, job_id: str, progress: int, message: str = ""):
        """Send progress update"""
        await self.broadcast_to_job(job_id, {
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "message": message
        })
    
    async def send_result(self, job_id: str, result: dict):
        """Send final result"""
        await self.broadcast_to_job(job_id, {
            "type": "result",
            "job_id": job_id,
            "result": result
        })
    
    async def send_error(self, job_id: str, error: str):
        """Send error message"""
        await self.broadcast_to_job(job_id, {
            "type": "error",
            "job_id": job_id,
            "error": error
        })


websocket_manager = WebSocketManager()
