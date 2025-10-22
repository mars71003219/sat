"""
트렌드 조회 API
VictoriaMetrics와 PostgreSQL에서 시계열 데이터를 조회하고 비교 분석
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import requests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '..'))

from database.postgres_client import postgres_client
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["trends"])

# VictoriaMetrics 설정
VICTORIA_METRICS_URL = os.environ.get('VICTORIA_METRICS_URL', 'http://victoria-metrics:8428')


class TimeRange(BaseModel):
    """시간 범위"""
    start_time: datetime
    end_time: datetime


class TrendDataPoint(BaseModel):
    """트렌드 데이터 포인트"""
    timestamp: datetime
    value: float


class TrendResponse(BaseModel):
    """트렌드 응답"""
    metric_name: str
    satellite_id: Optional[str] = None
    data_points: List[TrendDataPoint]
    summary: Dict[str, Any]


class ComparisonResponse(BaseModel):
    """비교 응답"""
    metric_name: str
    raw_data: List[TrendDataPoint]
    prediction_data: List[TrendDataPoint]
    correlation: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None


def query_victoria_metrics(metric_name: str, start_time: datetime, end_time: datetime,
                          satellite_id: Optional[str] = None) -> List[TrendDataPoint]:
    """
    VictoriaMetrics에서 시계열 데이터 조회

    Args:
        metric_name: 메트릭 이름 (예: satellite_temperature)
        start_time: 시작 시간
        end_time: 종료 시간
        satellite_id: 위성 ID (선택사항)

    Returns:
        데이터 포인트 리스트
    """
    try:
        # PromQL 쿼리 작성
        if satellite_id:
            query = f'{metric_name}{{satellite_id="{satellite_id}"}}'
        else:
            query = metric_name

        # API 호출
        params = {
            'query': query,
            'start': int(start_time.timestamp()),
            'end': int(end_time.timestamp()),
            'step': '60s'  # 1분 간격
        }

        response = requests.get(
            f'{VICTORIA_METRICS_URL}/api/v1/query_range',
            params=params,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        # 결과 파싱
        data_points = []
        if data['status'] == 'success' and data['data']['result']:
            result = data['data']['result'][0]  # 첫 번째 시계열 사용
            for timestamp, value in result['values']:
                data_points.append(TrendDataPoint(
                    timestamp=datetime.fromtimestamp(timestamp),
                    value=float(value)
                ))

        return data_points

    except requests.RequestException as e:
        logger.error(f"VictoriaMetrics 조회 실패: {e}")
        raise HTTPException(status_code=503, detail=f"VictoriaMetrics 서비스 오류: {str(e)}")
    except Exception as e:
        logger.error(f"데이터 파싱 오류: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 처리 오류: {str(e)}")


def calculate_statistics(data_points: List[TrendDataPoint]) -> Dict[str, Any]:
    """통계 계산"""
    if not data_points:
        return {
            'count': 0,
            'mean': None,
            'min': None,
            'max': None,
            'std': None
        }

    values = [dp.value for dp in data_points]
    count = len(values)
    mean = sum(values) / count
    min_val = min(values)
    max_val = max(values)

    # 표준편차 계산
    variance = sum((x - mean) ** 2 for x in values) / count
    std = variance ** 0.5

    return {
        'count': count,
        'mean': round(mean, 4),
        'min': round(min_val, 4),
        'max': round(max_val, 4),
        'std': round(std, 4)
    }


@router.get("/trends/raw", response_model=TrendResponse)
async def get_raw_data_trend(
    metric: str = Query(..., description="메트릭 이름 (예: satellite_temperature)"),
    start_time: datetime = Query(..., description="시작 시간 (ISO 8601)"),
    end_time: datetime = Query(..., description="종료 시간 (ISO 8601)"),
    satellite_id: Optional[str] = Query(None, description="위성 ID (선택사항)")
):
    """
    원본 시계열 데이터 조회 (VictoriaMetrics)

    Example:
        GET /api/v1/trends/raw?metric=satellite_temperature&start_time=2025-10-22T00:00:00Z&end_time=2025-10-22T12:00:00Z&satellite_id=SAT-001
    """
    logger.info(f"원본 데이터 조회: {metric}, {start_time} ~ {end_time}")

    # VictoriaMetrics 쿼리
    data_points = query_victoria_metrics(metric, start_time, end_time, satellite_id)

    # 통계 계산
    summary = calculate_statistics(data_points)

    return TrendResponse(
        metric_name=metric,
        satellite_id=satellite_id,
        data_points=data_points,
        summary=summary
    )


@router.get("/trends/prediction", response_model=TrendResponse)
async def get_prediction_trend(
    model_name: str = Query(..., description="모델 이름 (예: vae_timeseries, transformer_timeseries)"),
    start_time: datetime = Query(..., description="시작 시간 (ISO 8601)"),
    end_time: datetime = Query(..., description="종료 시간 (ISO 8601)"),
    satellite_id: Optional[str] = Query(None, description="위성 ID (선택사항)")
):
    """
    예측 결과 트렌드 조회 (PostgreSQL)

    Example:
        GET /api/v1/trends/prediction?model_name=vae_timeseries&start_time=2025-10-22T00:00:00Z&end_time=2025-10-22T12:00:00Z
    """
    logger.info(f"예측 데이터 조회: {model_name}, {start_time} ~ {end_time}")

    try:
        # PostgreSQL 쿼리
        query = """
            SELECT
                created_at as timestamp,
                CAST(result->>'mean_prediction' AS FLOAT) as value
            FROM inference_results
            WHERE model_name = %s
              AND status = 'completed'
              AND created_at BETWEEN %s AND %s
        """

        params = [model_name, start_time, end_time]

        if satellite_id:
            query += " AND satellite_id = %s"
            params.append(satellite_id)

        query += " ORDER BY created_at ASC"

        results = postgres_client.execute_query(query, params)

        # 데이터 포인트 생성
        data_points = []
        for row in results:
            data_points.append(TrendDataPoint(
                timestamp=row['timestamp'],
                value=row['value']
            ))

        # 통계 계산
        summary = calculate_statistics(data_points)

        return TrendResponse(
            metric_name=f"{model_name}_prediction",
            satellite_id=satellite_id,
            data_points=data_points,
            summary=summary
        )

    except Exception as e:
        logger.error(f"예측 데이터 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"데이터베이스 오류: {str(e)}")


@router.get("/trends/compare", response_model=ComparisonResponse)
async def compare_trends(
    raw_metric: str = Query(..., description="원본 메트릭 이름"),
    model_name: str = Query(..., description="예측 모델 이름"),
    start_time: datetime = Query(..., description="시작 시간"),
    end_time: datetime = Query(..., description="종료 시간"),
    satellite_id: Optional[str] = Query(None, description="위성 ID")
):
    """
    원본 데이터와 예측 결과 비교

    Example:
        GET /api/v1/trends/compare?raw_metric=satellite_temperature&model_name=vae_timeseries&start_time=2025-10-22T00:00:00Z&end_time=2025-10-22T12:00:00Z
    """
    logger.info(f"트렌드 비교: {raw_metric} vs {model_name}")

    # 원본 데이터 조회
    raw_data = query_victoria_metrics(raw_metric, start_time, end_time, satellite_id)

    # 예측 데이터 조회
    prediction_response = await get_prediction_trend(model_name, start_time, end_time, satellite_id)
    prediction_data = prediction_response.data_points

    # 상관계수 및 오차 계산
    correlation = None
    mae = None
    rmse = None

    if raw_data and prediction_data:
        # 타임스탬프 매칭 (간단한 구현)
        # 실제로는 더 정교한 매칭 로직 필요
        raw_values = [dp.value for dp in raw_data]
        pred_values = [dp.value for dp in prediction_data]

        min_len = min(len(raw_values), len(pred_values))
        if min_len > 0:
            raw_values = raw_values[:min_len]
            pred_values = pred_values[:min_len]

            # MAE (Mean Absolute Error)
            mae = sum(abs(r - p) for r, p in zip(raw_values, pred_values)) / min_len

            # RMSE (Root Mean Square Error)
            mse = sum((r - p) ** 2 for r, p in zip(raw_values, pred_values)) / min_len
            rmse = mse ** 0.5

            # 상관계수 (Pearson correlation)
            mean_raw = sum(raw_values) / min_len
            mean_pred = sum(pred_values) / min_len

            numerator = sum((r - mean_raw) * (p - mean_pred) for r, p in zip(raw_values, pred_values))
            denominator_raw = sum((r - mean_raw) ** 2 for r in raw_values) ** 0.5
            denominator_pred = sum((p - mean_pred) ** 2 for p in pred_values) ** 0.5

            if denominator_raw > 0 and denominator_pred > 0:
                correlation = numerator / (denominator_raw * denominator_pred)

    return ComparisonResponse(
        metric_name=raw_metric,
        raw_data=raw_data,
        prediction_data=prediction_data,
        correlation=round(correlation, 4) if correlation else None,
        mae=round(mae, 4) if mae else None,
        rmse=round(rmse, 4) if rmse else None
    )


@router.get("/trends/metrics")
async def get_available_metrics():
    """
    사용 가능한 메트릭 목록 조회
    """
    # VictoriaMetrics에서 메트릭 목록 조회
    try:
        response = requests.get(
            f'{VICTORIA_METRICS_URL}/api/v1/label/__name__/values',
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        metrics = []
        if data['status'] == 'success':
            # satellite_ 로 시작하는 메트릭만 필터링
            metrics = [m for m in data['data'] if m.startswith('satellite_')]

        return {
            'metrics': metrics,
            'count': len(metrics)
        }

    except Exception as e:
        logger.error(f"메트릭 목록 조회 실패: {e}")
        return {
            'metrics': [
                'satellite_temperature',
                'satellite_altitude',
                'satellite_velocity',
                'satellite_battery_voltage',
                'satellite_solar_power',
                'satellite_latitude',
                'satellite_longitude'
            ],
            'count': 7,
            'source': 'default'
        }


@router.get("/trends/satellites")
async def get_available_satellites():
    """
    사용 가능한 위성 목록 조회
    """
    try:
        # VictoriaMetrics에서 satellite_id 라벨 값 조회
        response = requests.get(
            f'{VICTORIA_METRICS_URL}/api/v1/label/satellite_id/values',
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        satellites = []
        if data['status'] == 'success':
            satellites = data['data']

        return {
            'satellites': satellites,
            'count': len(satellites)
        }

    except Exception as e:
        logger.error(f"위성 목록 조회 실패: {e}")
        return {
            'satellites': [],
            'count': 0,
            'error': str(e)
        }
