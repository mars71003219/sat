from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from database.elasticsearch_client import elasticsearch_client
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/inference")
async def search_inference_logs(
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """추론 로그 검색 (Elasticsearch)"""
    try:
        results = elasticsearch_client.search_inference_logs(
            model_name=model_name,
            status=status,
            from_date=from_date,
            to_date=to_date,
            limit=limit
        )
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Inference search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/sensor")
async def search_sensor_data(
    sensor_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    location: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: int = Query(1000, ge=1, le=10000)
):
    """센서 데이터 검색 (Elasticsearch)"""
    try:
        results = elasticsearch_client.search_sensor_data(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            location=location,
            from_time=from_time,
            to_time=to_time,
            limit=limit
        )
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Sensor search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/analytics/models")
async def get_model_analytics():
    """모델별 통계 분석 (Elasticsearch aggregation)"""
    try:
        aggregations = elasticsearch_client.aggregate_by_model()
        
        # 결과 포맷팅
        model_stats = []
        if 'models' in aggregations:
            for bucket in aggregations['models']['buckets']:
                model_stats.append({
                    "model_name": bucket['key'],
                    "total_inferences": bucket['doc_count'],
                    "avg_inference_time": bucket.get('avg_inference_time', {}).get('value'),
                    "status_breakdown": {
                        status_bucket['key']: status_bucket['doc_count']
                        for status_bucket in bucket.get('status_breakdown', {}).get('buckets', [])
                    }
                })
        
        return {
            "model_statistics": model_stats
        }
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics failed")
