# 위성 텔레메트리 분석 시스템 - API 레퍼런스

## 기본 정보

**Base URL**: `http://localhost:8000/api/v1`

**인증**: 현재 버전에서는 인증 없음

**Content-Type**: `application/json`

**응답 형식**: JSON

---

## 목차

1. [Inference API](#1-inference-api) - 추론 작업 관리
2. [Dashboard API](#2-dashboard-api) - 실시간 모니터링
3. [Trends API](#3-trends-api) - 시계열 데이터 조회
4. [Query API](#4-query-api) - 결과 조회
5. [Search API](#5-search-api) - 검색 기능

---

## 1. Inference API

추론 작업 제출, 상태 조회, 결과 조회를 위한 API

### 1.1 추론 작업 제출

**Endpoint**: `POST /inference/submit`

**설명**: 새로운 추론 작업을 제출합니다.

**Request Body**:
```json
{
  "model_name": "lstm_timeseries",
  "data": [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
  "config": {
    "forecast_horizon": 5,
    "window_size": 10
  },
  "metadata": {
    "satellite_id": "SAT-001",
    "subsystem": "thermal",
    "source": "manual"
  }
}
```

**Request Parameters**:

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| model_name | string | ✅ | 모델 이름 (`lstm_timeseries`, `moving_average`) |
| data | array[float] | ✅ | 입력 데이터 (최소 10개) |
| config | object | ❌ | 모델 설정 |
| metadata | object | ❌ | 추가 메타데이터 |

**Response** (200 OK):
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "pending",
  "message": "Inference job submitted to analysis server"
}
```

**Error Responses**:

- `400 Bad Request`: 잘못된 요청 데이터
  ```json
  {
    "detail": "data must contain at least 10 elements"
  }
  ```

- `500 Internal Server Error`: 서버 오류
  ```json
  {
    "detail": "Internal server error"
  }
  ```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
    "config": {"forecast_horizon": 5}
  }'
```

---

### 1.2 작업 상태 조회

**Endpoint**: `GET /inference/status/{job_id}`

**설명**: 제출된 작업의 현재 상태를 조회합니다.

**Path Parameters**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| job_id | string | 작업 ID (UUID) |

**Response** (200 OK):
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "model_name": "lstm_timeseries",
  "status": "completed",
  "created_at": "2025-10-30T12:34:56.789Z",
  "completed_at": "2025-10-30T12:35:01.234Z",
  "error_message": null
}
```

**Status Values**:

| 상태 | 설명 |
|------|------|
| `pending` | 작업이 큐에 대기 중 |
| `processing` | 작업 처리 중 |
| `completed` | 작업 완료 |
| `failed` | 작업 실패 |

**Error Responses**:

- `404 Not Found`: 작업을 찾을 수 없음
  ```json
  {
    "detail": "Job not found"
  }
  ```

**cURL Example**:
```bash
curl http://localhost:8000/api/v1/inference/status/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

---

### 1.3 추론 결과 조회

**Endpoint**: `GET /inference/result/{job_id}`

**설명**: 완료된 작업의 추론 결과를 조회합니다.

**Path Parameters**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| job_id | string | 작업 ID (UUID) |

**Response** (200 OK):
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "model_name": "lstm_timeseries",
  "status": "completed",
  "predictions": [35.2, 36.1, 37.0, 37.8, 38.5],
  "confidence": [0.92, 0.89, 0.85, 0.81, 0.78],
  "metrics": {
    "inference_time": 0.125,
    "model_type": "LSTM",
    "sequence_length": 10,
    "forecast_steps": 5
  },
  "metadata": {
    "satellite_id": "SAT-001",
    "subsystem": "thermal",
    "processed_by": "triton_server",
    "triton_client": "grpc"
  },
  "created_at": "2025-10-30T12:34:56.789Z",
  "completed_at": "2025-10-30T12:35:01.234Z"
}
```

**Response Fields**:

| 필드 | 타입 | 설명 |
|------|------|------|
| predictions | array[float] | 예측값 배열 |
| confidence | array[float] | 신뢰도 배열 (0~1) |
| metrics | object | 추론 메트릭 |
| metadata | object | 작업 메타데이터 |

**Error Responses**:

- `404 Not Found`: 결과를 찾을 수 없음

**cURL Example**:
```bash
curl http://localhost:8000/api/v1/inference/result/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

---

### 1.4 실시간 WebSocket (작업별)

**Endpoint**: `WS /inference/ws/{job_id}`

**설명**: 특정 작업의 실시간 상태 업데이트를 수신합니다.

**Path Parameters**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| job_id | string | 작업 ID (UUID) |

**Client → Server Messages**:

- `"status"`: 현재 상태 요청
- `"result"`: 결과 요청

**Server → Client Messages**:

상태 업데이트:
```json
{
  "type": "status",
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "processing",
  "timestamp": "2025-10-30T12:35:00.000Z"
}
```

결과 업데이트:
```json
{
  "type": "result",
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "predictions": [35.2, 36.1, 37.0],
  "confidence": [0.92, 0.89, 0.85]
}
```

**JavaScript Example**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/inference/ws/f47ac10b-...');

ws.onopen = () => {
  ws.send('status');  // 상태 요청
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

---

## 2. Dashboard API

실시간 대시보드 데이터를 위한 API

### 2.1 최근 결과 조회

**Endpoint**: `GET /dashboard/recent`

**설명**: 최근 추론 결과를 조회합니다.

**Query Parameters**:

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| limit | integer | 20 | 조회할 결과 수 (최대 100) |

**Response** (200 OK):
```json
{
  "results": [
    {
      "job_id": "abc123",
      "model_name": "lstm_timeseries",
      "status": "completed",
      "predictions": [35.2, 36.1],
      "created_at": "2025-10-30T12:34:56Z",
      "completed_at": "2025-10-30T12:35:01Z"
    }
  ],
  "count": 1
}
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/dashboard/recent?limit=10"
```

---

### 2.2 실시간 통계

**Endpoint**: `GET /dashboard/live-stats`

**설명**: 시스템 전체 통계를 조회합니다.

**Response** (200 OK):
```json
{
  "total_jobs": 1250,
  "completed": 1180,
  "failed": 70,
  "success_rate": 94.4,
  "models_used": {
    "lstm_timeseries": 800,
    "moving_average": 450
  },
  "throughput_per_minute": 15
}
```

**Response Fields**:

| 필드 | 타입 | 설명 |
|------|------|------|
| total_jobs | integer | 총 작업 수 |
| completed | integer | 완료된 작업 수 |
| failed | integer | 실패한 작업 수 |
| success_rate | float | 성공률 (%) |
| models_used | object | 모델별 사용 횟수 |
| throughput_per_minute | integer | 분당 처리량 |

**cURL Example**:
```bash
curl http://localhost:8000/api/v1/dashboard/live-stats
```

---

### 2.3 모델 비교

**Endpoint**: `GET /dashboard/model-comparison`

**설명**: 모델별 성능을 비교합니다 (최근 1시간 기준).

**Response** (200 OK):
```json
{
  "comparison": {
    "lstm_timeseries": {
      "total": 45,
      "avg_time": 0.142,
      "completed": 43,
      "failed": 2
    },
    "moving_average": {
      "total": 30,
      "avg_time": 0.058,
      "completed": 30,
      "failed": 0
    }
  }
}
```

---

### 2.4 실시간 대시보드 WebSocket

**Endpoint**: `WS /dashboard/ws`

**설명**: 전체 시스템의 실시간 업데이트를 수신합니다.

**Server → Client Messages** (2초 간격):

```json
{
  "type": "update",
  "timestamp": "2025-10-30T12:35:00.000Z",
  "recent_results": [
    {
      "job_id": "abc123",
      "status": "completed",
      "model_name": "lstm_timeseries",
      "created_at": "2025-10-30T12:34:56Z"
    }
  ],
  "statistics": {
    "total_jobs": 1250,
    "completed": 1180,
    "failed": 70,
    "success_rate": 94.4
  }
}
```

**JavaScript Example**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/dashboard/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  // 최근 결과 업데이트
  updateRecentResults(data.recent_results);

  // 통계 업데이트
  updateStatistics(data.statistics);
};
```

---

## 3. Trends API

VictoriaMetrics와 연동한 시계열 데이터 조회

### 3.1 시계열 트렌드 조회

**Endpoint**: `GET /trends/satellite/{satellite_id}/metric/{metric_name}`

**설명**: 특정 위성의 메트릭 트렌드를 조회합니다.

**Path Parameters**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| satellite_id | string | 위성 ID (예: SAT-001) |
| metric_name | string | 메트릭 이름 |

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| start_time | string (ISO8601) | ✅ | 시작 시간 |
| end_time | string (ISO8601) | ✅ | 종료 시간 |

**Response** (200 OK):
```json
{
  "metric_name": "satellite_temperature",
  "satellite_id": "SAT-001",
  "data_points": [
    {
      "timestamp": "2025-10-30T12:00:00Z",
      "value": 25.3
    },
    {
      "timestamp": "2025-10-30T12:00:03Z",
      "value": 25.5
    }
  ],
  "summary": {
    "count": 100,
    "min": 24.8,
    "max": 26.2,
    "mean": 25.4,
    "std": 0.35
  }
}
```

**Available Metrics**:

| 메트릭 이름 | 설명 | 단위 |
|-----------|------|------|
| `satellite_battery_voltage` | 배터리 전압 | V |
| `satellite_battery_soc` | 배터리 충전율 | % |
| `satellite_temperature` | 온도 | °C |
| `satellite_solar_panel_voltage` | 태양전지 전압 | V |
| `satellite_power_consumption` | 전력 소비 | W |

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/trends/satellite/SAT-001/metric/satellite_temperature?start_time=2025-10-30T00:00:00Z&end_time=2025-10-30T23:59:59Z"
```

---

### 3.2 예측 vs 실제 비교

**Endpoint**: `GET /trends/comparison/{satellite_id}/{metric_name}`

**설명**: 예측값과 실제값을 비교합니다.

**Response** (200 OK):
```json
{
  "metric_name": "satellite_temperature",
  "raw_data": [
    {"timestamp": "2025-10-30T12:00:00Z", "value": 25.3},
    {"timestamp": "2025-10-30T12:00:03Z", "value": 25.5}
  ],
  "prediction_data": [
    {"timestamp": "2025-10-30T12:00:00Z", "value": 25.2},
    {"timestamp": "2025-10-30T12:00:03Z", "value": 25.4}
  ],
  "correlation": 0.985,
  "mae": 0.15,
  "rmse": 0.22
}
```

**Response Fields**:

| 필드 | 타입 | 설명 |
|------|------|------|
| correlation | float | 상관계수 (-1~1) |
| mae | float | 평균 절대 오차 |
| rmse | float | 평균 제곱근 오차 |

---

## 4. Query API

고급 쿼리 및 필터링

### 4.1 결과 이력 조회

**Endpoint**: `GET /query/history`

**설명**: 추론 결과 이력을 조회합니다.

**Query Parameters**:

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| limit | integer | 100 | 조회 개수 |
| offset | integer | 0 | 시작 위치 |
| model_name | string | null | 모델 필터 |

**Response** (200 OK):
```json
{
  "results": [...],
  "total": 1250,
  "limit": 100,
  "offset": 0
}
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/query/history?limit=50&model_name=lstm_timeseries"
```

---

## 5. Search API

Elasticsearch 기반 검색

### 5.1 전체 텍스트 검색

**Endpoint**: `GET /search`

**설명**: 메타데이터 기반 전체 텍스트 검색을 수행합니다.

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| q | string | ✅ | 검색 쿼리 |
| limit | integer | ❌ | 결과 개수 (기본: 20) |

**Response** (200 OK):
```json
{
  "hits": [
    {
      "job_id": "abc123",
      "model_name": "lstm_timeseries",
      "metadata": {
        "satellite_id": "SAT-001",
        "subsystem": "thermal"
      },
      "score": 0.95
    }
  ],
  "total": 15,
  "took_ms": 12
}
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/search?q=SAT-001+thermal"
```

---

## 에러 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 404 | 리소스를 찾을 수 없음 |
| 500 | 서버 내부 오류 |

**표준 에러 응답**:
```json
{
  "detail": "Error message description"
}
```

---

## Rate Limiting

현재 버전에서는 Rate Limiting이 적용되지 않습니다.

---

## Pagination

이력 조회 API는 `limit`과 `offset` 파라미터를 사용합니다:

```bash
# 첫 번째 페이지 (0-99)
GET /query/history?limit=100&offset=0

# 두 번째 페이지 (100-199)
GET /query/history?limit=100&offset=100
```

---

## WebSocket 프로토콜

### 연결

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/dashboard/ws');
```

### 메시지 형식

모든 WebSocket 메시지는 JSON 형식입니다.

### Heartbeat

서버는 2초마다 자동으로 업데이트를 전송합니다. 별도의 ping/pong은 필요하지 않습니다.

### 재연결

연결이 끊긴 경우 클라이언트는 자동으로 재연결을 시도해야 합니다.

---

## 예제 코드

### Python

```python
import requests

# 추론 작업 제출
response = requests.post(
    'http://localhost:8000/api/v1/inference/submit',
    json={
        'model_name': 'lstm_timeseries',
        'data': [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
        'config': {'forecast_horizon': 5}
    }
)

job_id = response.json()['job_id']
print(f'Job ID: {job_id}')

# 상태 확인
status = requests.get(f'http://localhost:8000/api/v1/inference/status/{job_id}')
print(status.json())

# 결과 조회
result = requests.get(f'http://localhost:8000/api/v1/inference/result/{job_id}')
print(result.json())
```

### JavaScript (Fetch API)

```javascript
// 추론 작업 제출
const submitInference = async () => {
  const response = await fetch('http://localhost:8000/api/v1/inference/submit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_name: 'lstm_timeseries',
      data: [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
      config: { forecast_horizon: 5 }
    })
  });

  const data = await response.json();
  return data.job_id;
};

// 결과 조회
const getResult = async (jobId) => {
  const response = await fetch(`http://localhost:8000/api/v1/inference/result/${jobId}`);
  const result = await response.json();
  return result;
};

// 사용 예시
(async () => {
  const jobId = await submitInference();
  console.log('Job ID:', jobId);

  // 2초 후 결과 조회
  setTimeout(async () => {
    const result = await getResult(jobId);
    console.log('Result:', result);
  }, 2000);
})();
```

### cURL

```bash
# 추론 제출
curl -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
    "config": {"forecast_horizon": 5}
  }'

# 상태 조회
curl http://localhost:8000/api/v1/inference/status/{job_id}

# 결과 조회
curl http://localhost:8000/api/v1/inference/result/{job_id}

# 통계 조회
curl http://localhost:8000/api/v1/dashboard/live-stats
```

---

## Postman Collection

Postman 컬렉션은 `docs/postman/` 디렉토리에서 다운로드할 수 있습니다.

---

## 변경 이력

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| 1.0.0 | 2025-10-30 | 초기 API 문서 작성 |
