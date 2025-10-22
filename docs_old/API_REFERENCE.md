# API Reference

이 문서는 시스템의 모든 API 엔드포인트와 WebSocket 연결에 대한 상세한 설명을 제공합니다.

## 목차

1. [Base URL](#base-url)
2. [인증](#인증)
3. [추론 API](#추론-api)
4. [대시보드 API](#대시보드-api)
5. [WebSocket API](#websocket-api)
6. [에러 코드](#에러-코드)
7. [데이터 스키마](#데이터-스키마)

## Base URL

### Development
```
http://localhost:8000/api/v1
```

### Nginx Proxy (Production)
```
http://localhost/api/v1
```

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 인증

현재 버전은 인증이 비활성화되어 있습니다. 프로덕션 환경에서는 다음을 구현해야 합니다:
- API Key 인증
- JWT Bearer Token
- Rate Limiting

## 추론 API

### 1. Submit Inference Job

추론 작업을 제출합니다.

**Endpoint**: `POST /api/v1/inference/submit`

**Request Body**:
```json
{
  "model_name": "lstm_timeseries",
  "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5],
  "config": {
    "forecast_steps": 5,
    "confidence_level": 0.95
  },
  "metadata": {
    "source": "sensor_01",
    "timestamp": "2025-01-15T10:30:00Z",
    "pattern": "seasonal"
  }
}
```

**Request Parameters**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| model_name | string | Yes | 사용할 모델 이름 (`lstm_timeseries`, `moving_average`) |
| data | array[float] | Yes | 시계열 데이터 포인트 배열 (최소 3개 이상) |
| config | object | No | 모델별 설정 |
| metadata | object | No | 추가 메타데이터 |

**Response**:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "message": "Inference job submitted to analysis server"
}
```

**Status Codes**:
- `200 OK`: 작업이 성공적으로 제출됨
- `400 Bad Request`: 잘못된 요청 (유효성 검사 실패)
- `500 Internal Server Error`: 서버 내부 오류

**cURL Example**:
```bash
curl -X POST "http://localhost/api/v1/inference/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5],
    "metadata": {"pattern": "seasonal"}
  }'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost/api/v1/inference/submit",
    json={
        "model_name": "lstm_timeseries",
        "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5],
        "metadata": {"pattern": "seasonal"}
    }
)

result = response.json()
job_id = result["job_id"]
print(f"Job ID: {job_id}")
```

---

### 2. Get Job Status

추론 작업의 현재 상태를 조회합니다.

**Endpoint**: `GET /api/v1/inference/status/{job_id}`

**Path Parameters**:
| 필드 | 타입 | 설명 |
|------|------|------|
| job_id | string | 작업 고유 식별자 |

**Response**:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress": 100,
  "message": "Inference completed successfully"
}
```

**Status Values**:
- `pending`: 대기 중
- `running`: 실행 중
- `completed`: 완료
- `failed`: 실패

**Status Codes**:
- `200 OK`: 상태 조회 성공
- `404 Not Found`: 작업을 찾을 수 없음

**cURL Example**:
```bash
curl -X GET "http://localhost/api/v1/inference/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

---

### 3. Get Inference Result

완료된 추론 작업의 결과를 조회합니다.

**Endpoint**: `GET /api/v1/inference/result/{job_id}`

**Path Parameters**:
| 필드 | 타입 | 설명 |
|------|------|------|
| job_id | string | 작업 고유 식별자 |

**Response**:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "model_name": "lstm_timeseries",
  "predictions": [4.8, 5.2, 5.5, 5.3, 5.7],
  "confidence": [0.95, 0.94, 0.92, 0.93, 0.91],
  "metrics": {
    "inference_time": 0.125,
    "mse": 0.032,
    "mae": 0.15
  },
  "metadata": {
    "pattern": "seasonal",
    "source": "sensor_01"
  },
  "created_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:01Z"
}
```

**Status Codes**:
- `200 OK`: 결과 조회 성공
- `404 Not Found`: 결과를 찾을 수 없음

**cURL Example**:
```bash
curl -X GET "http://localhost/api/v1/inference/result/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

---

## 대시보드 API

### 4. Get Recent Results

최근 추론 결과 목록을 조회합니다.

**Endpoint**: `GET /api/v1/dashboard/recent`

**Query Parameters**:
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| limit | integer | 20 | 조회할 결과 수 (최대 100) |

**Response**:
```json
{
  "results": [
    {
      "job_id": "job_001",
      "model_name": "lstm_timeseries",
      "status": "completed",
      "predictions": [4.8, 5.2, 5.5],
      "created_at": "2025-01-15T10:30:00Z",
      "completed_at": "2025-01-15T10:30:01Z"
    }
  ],
  "count": 1
}
```

**Status Codes**:
- `200 OK`: 조회 성공

**cURL Example**:
```bash
curl -X GET "http://localhost/api/v1/dashboard/recent?limit=10"
```

---

### 5. Get Live Statistics

실시간 통계 정보를 조회합니다.

**Endpoint**: `GET /api/v1/dashboard/live-stats`

**Response**:
```json
{
  "total_jobs": 2144,
  "completed": 2144,
  "failed": 0,
  "success_rate": 100.0,
  "models_used": {
    "moving_average": 2142,
    "lstm_timeseries": 2
  },
  "throughput_per_minute": 100
}
```

**Status Codes**:
- `200 OK`: 조회 성공

**cURL Example**:
```bash
curl -X GET "http://localhost/api/v1/dashboard/live-stats"
```

---

### 6. Get Model Comparison

모델 간 성능 비교 정보를 조회합니다.

**Endpoint**: `GET /api/v1/dashboard/model-comparison`

**Response**:
```json
{
  "comparison": {
    "lstm_timeseries": {
      "total": 100,
      "avg_time": 0.125,
      "completed": 98,
      "failed": 2
    },
    "moving_average": {
      "total": 1000,
      "avg_time": 0.015,
      "completed": 1000,
      "failed": 0
    }
  }
}
```

**Status Codes**:
- `200 OK`: 조회 성공

---

### 7. Get Pattern Distribution

데이터 패턴별 분포를 조회합니다.

**Endpoint**: `GET /api/v1/dashboard/patterns`

**Response**:
```json
{
  "patterns": [
    {
      "pattern": "seasonal",
      "count": 500,
      "avg_time": 0.082
    },
    {
      "pattern": "linear",
      "count": 300,
      "avg_time": 0.075
    }
  ]
}
```

**Status Codes**:
- `200 OK`: 조회 성공

---

## WebSocket API

### 8. Job Status WebSocket

특정 작업의 실시간 상태 업데이트를 수신합니다.

**Endpoint**: `ws://localhost/api/v1/inference/ws/{job_id}`

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost/api/v1/inference/ws/job_123');

ws.onopen = () => {
    console.log('Connected');

    // Request status
    ws.send('status');

    // Request result
    ws.send('result');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

ws.onclose = () => {
    console.log('Disconnected');
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

**Messages from Client**:
- `"status"`: 현재 상태 요청
- `"result"`: 결과 요청

**Messages from Server**:
```json
{
  "job_id": "job_123",
  "status": "completed",
  "progress": 100,
  "message": "Inference completed"
}
```

---

### 9. Dashboard WebSocket

실시간 대시보드 데이터 업데이트를 수신합니다.

**Endpoint**: `ws://localhost/api/v1/dashboard/ws`

**Connection**:
```javascript
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/api/v1/dashboard/ws`;

const ws = new WebSocket(wsUrl);

ws.onopen = () => {
    console.log('Dashboard WebSocket connected');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'update') {
        updateDashboard(data);
    }
};

function updateDashboard(data) {
    // Update statistics
    const stats = data.statistics;
    document.getElementById('totalJobs').textContent = stats.total_jobs;
    document.getElementById('completedJobs').textContent = stats.completed;

    // Update recent results
    const results = data.recent_results;
    // ... update UI
}
```

**Update Frequency**: 2초마다 자동 전송

**Message Format**:
```json
{
  "type": "update",
  "timestamp": "2025-01-15T10:30:00Z",
  "recent_results": [
    {
      "job_id": "job_001",
      "model_name": "lstm_timeseries",
      "status": "completed",
      "predictions": [4.8, 5.2],
      "confidence": {"mean": 0.95},
      "metrics": {"inference_time": 0.125},
      "metadata": {"pattern": "seasonal"},
      "created_at": "2025-01-15T10:29:58Z",
      "completed_at": "2025-01-15T10:29:59Z"
    }
  ],
  "statistics": {
    "total_jobs": 2144,
    "completed": 2144,
    "failed": 0,
    "success_rate": 100.0,
    "models_used": {
      "moving_average": 2142,
      "lstm_timeseries": 2
    },
    "throughput_per_minute": 100
  }
}
```

**Python Example (websockets library)**:
```python
import asyncio
import websockets
import json

async def receive_updates():
    uri = "ws://localhost/api/v1/dashboard/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to dashboard")

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "update":
                stats = data["statistics"]
                print(f"Total Jobs: {stats['total_jobs']}")
                print(f"Success Rate: {stats['success_rate']}%")

asyncio.run(receive_updates())
```

---

## 에러 코드

### HTTP Status Codes

| 코드 | 의미 | 설명 |
|------|------|------|
| 200 | OK | 요청 성공 |
| 400 | Bad Request | 잘못된 요청 (유효성 검사 실패) |
| 404 | Not Found | 리소스를 찾을 수 없음 |
| 500 | Internal Server Error | 서버 내부 오류 |
| 503 | Service Unavailable | 서비스 일시 중단 |

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Errors

#### 1. Invalid Model Name
```json
{
  "detail": "Model 'invalid_model' not found. Available models: lstm_timeseries, moving_average"
}
```

#### 2. Invalid Data Format
```json
{
  "detail": "Field 'data' must be an array of numbers with at least 3 elements"
}
```

#### 3. Job Not Found
```json
{
  "detail": "Job not found"
}
```

#### 4. WebSocket Connection Error
```json
{
  "detail": "WebSocket connection failed: Unable to establish connection"
}
```

---

## 데이터 스키마

### InferenceRequest

```typescript
interface InferenceRequest {
  model_name: string;           // Required: "lstm_timeseries" | "moving_average"
  data: number[];               // Required: Array of floats, min length 3
  config?: {                    // Optional: Model-specific configuration
    forecast_steps?: number;
    confidence_level?: number;
    window_size?: number;
  };
  metadata?: {                  // Optional: Additional metadata
    [key: string]: any;
  };
}
```

### InferenceResponse

```typescript
interface InferenceResponse {
  job_id: string;               // UUID format
  status: "pending" | "running" | "completed" | "failed";
  message: string;              // Status message
}
```

### InferenceResult

```typescript
interface InferenceResult {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  model_name: string;
  predictions: number[];        // Predicted values
  confidence?: number[];        // Confidence intervals
  metrics?: {                   // Performance metrics
    inference_time: number;     // Seconds
    mse?: number;               // Mean Squared Error
    mae?: number;               // Mean Absolute Error
  };
  metadata?: {
    [key: string]: any;
  };
  created_at: string;           // ISO 8601 format
  completed_at?: string;        // ISO 8601 format
  error_message?: string;       // Error description if failed
}
```

### Statistics

```typescript
interface Statistics {
  total_jobs: number;
  completed: number;
  failed: number;
  success_rate: number;         // Percentage (0-100)
  models_used: {
    [model_name: string]: number;
  };
  throughput_per_minute: number;
}
```

### DashboardUpdate (WebSocket)

```typescript
interface DashboardUpdate {
  type: "update";
  timestamp: string;            // ISO 8601 format
  recent_results: InferenceResult[];
  statistics: Statistics;
}
```

---

## Rate Limiting (향후 구현)

프로덕션 환경에서는 다음 제한을 권장합니다:

- **추론 API**: 분당 60 요청
- **조회 API**: 분당 120 요청
- **WebSocket**: IP당 동시 연결 3개

**Rate Limit Headers**:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248000
```

---

## API Versioning

현재 버전: `v1`

API URL 형식: `/api/v{version}/{endpoint}`

버전별 URL:
- v1: `/api/v1/inference/submit`
- v2: `/api/v2/inference/submit` (향후)

---

## Best Practices

### 1. 비동기 처리
추론 작업은 비동기로 처리됩니다. 결과를 즉시 받을 수 없으므로:
- Job ID를 저장하고 나중에 조회
- WebSocket으로 실시간 업데이트 수신
- 폴링 간격은 최소 1초 이상 유지

### 2. 에러 처리
```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(url, json=data, timeout=10)
    response.raise_for_status()
    result = response.json()
except RequestException as e:
    print(f"Request failed: {e}")
```

### 3. WebSocket 재연결
```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

function connectWebSocket() {
    const ws = new WebSocket(wsUrl);

    ws.onclose = () => {
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            setTimeout(connectWebSocket, 5000);
        }
    };

    ws.onopen = () => {
        reconnectAttempts = 0;
    };
}
```

### 4. 데이터 유효성 검사
요청 전 데이터 유효성을 검사하여 불필요한 API 호출을 방지합니다:
```python
def validate_data(data):
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    if len(data) < 3:
        raise ValueError("Data must have at least 3 elements")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All data elements must be numbers")
    return True
```

---

## 추가 리소스

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Architecture Documentation**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **UML Diagrams**: [UML_DIAGRAMS.md](./UML_DIAGRAMS.md)
