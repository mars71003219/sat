from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
from decimal import Decimal
import sys
import os
import asyncio
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from database.postgres_client import postgres_client
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class DashboardManager:
    """대시보드 WebSocket 연결 관리"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Dashboard client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """모든 연결된 클라이언트에게 메시지 전송"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(connection)

        # 연결 끊긴 클라이언트 제거
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


dashboard_manager = DashboardManager()


@router.websocket("/ws")
async def dashboard_websocket(websocket: WebSocket):
    """대시보드 WebSocket 엔드포인트"""
    await dashboard_manager.connect(websocket)

    try:
        while True:
            # 주기적으로 최신 데이터 전송
            await asyncio.sleep(2)

            try:
                # 최근 결과 조회
                recent_results = postgres_client.get_recent_results(limit=10)
                logger.info(f"Fetched {len(recent_results)} recent results")

                # datetime 객체와 JSONB 필드를 JSON 직렬화 가능하게 변환
                for result in recent_results:
                    if result.get('created_at'):
                        result['created_at'] = str(result['created_at'])
                    if result.get('completed_at'):
                        result['completed_at'] = str(result['completed_at'])

                    # JSONB 필드 처리 (이미 dict이지만 안전하게 변환)
                    for field in ['predictions', 'confidence', 'metrics', 'metadata']:
                        if field in result and result[field] is not None:
                            if isinstance(result[field], str):
                                result[field] = json.loads(result[field])

                # 통계 조회
                stats = postgres_client.get_statistics()
                logger.info(f"Fetched stats: {stats}")

                # Decimal을 float로 변환
                if 'success_rate' in stats and isinstance(stats['success_rate'], Decimal):
                    stats['success_rate'] = float(stats['success_rate'])

                # 실시간 데이터 전송
                message = {
                    "type": "update",
                    "timestamp": str(postgres_client._get_timestamp()),
                    "recent_results": recent_results,
                    "statistics": stats
                }

                await websocket.send_json(message)
                logger.info("Successfully sent WebSocket message")

            except Exception as e:
                logger.error(f"Error preparing WebSocket data: {e}", exc_info=True)
                # 에러가 발생해도 연결은 유지

    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        dashboard_manager.disconnect(websocket)


@router.get("/recent")
async def get_recent_results(limit: int = 20):
    """최근 추론 결과 조회"""
    try:
        results = postgres_client.get_recent_results(limit=limit)
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error getting recent results: {e}")
        return {"results": [], "count": 0}


@router.get("/live-stats")
async def get_live_statistics():
    """실시간 통계 조회"""
    try:
        stats = postgres_client.get_statistics()

        # 최근 5분간의 처리량 계산
        recent_results = postgres_client.get_recent_results(limit=100)
        throughput_1min = len([r for r in recent_results
                               if (postgres_client._get_timestamp() -
                                   postgres_client._parse_timestamp(r.get('created_at', ''))).total_seconds() < 60])

        stats['throughput_per_minute'] = throughput_1min

        # Decimal을 float로 변환
        if 'success_rate' in stats and isinstance(stats['success_rate'], Decimal):
            stats['success_rate'] = float(stats['success_rate'])

        return stats
    except Exception as e:
        logger.error(f"Error getting live stats: {e}")
        return {
            "total_jobs": 0,
            "completed": 0,
            "failed": 0,
            "success_rate": 0,
            "models_used": {},
            "throughput_per_minute": 0
        }


@router.get("/model-comparison")
async def get_model_comparison():
    """모델 비교 분석"""
    try:
        # 각 모델별 성능 메트릭 조회
        comparison = {}

        for model in ["lstm_timeseries", "moving_average"]:
            results = postgres_client.query(
                """
                SELECT
                    COUNT(*) as total,
                    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_time,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                FROM inference_results
                WHERE model_name = %s
                    AND created_at > NOW() - INTERVAL '1 hour'
                """,
                (model,)
            )

            if results:
                comparison[model] = results[0]

        return {"comparison": comparison}
    except Exception as e:
        logger.error(f"Error getting model comparison: {e}")
        return {"comparison": {}}


@router.get("/patterns")
async def get_pattern_distribution():
    """패턴별 분포 조회"""
    try:
        results = postgres_client.query(
            """
            SELECT
                metadata->>'pattern' as pattern,
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_time
            FROM inference_results
            WHERE metadata->>'pattern' IS NOT NULL
                AND created_at > NOW() - INTERVAL '1 hour'
            GROUP BY metadata->>'pattern'
            ORDER BY count DESC
            """
        )

        return {
            "patterns": [
                {
                    "pattern": r["pattern"],
                    "count": r["count"],
                    "avg_time": float(r["avg_time"]) if r["avg_time"] else 0
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Error getting pattern distribution: {e}")
        return {"patterns": []}
