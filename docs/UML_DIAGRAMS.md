# 위성 텔레메트리 분석 시스템 - UML 다이어그램

## 1. 시퀀스 다이어그램

### 1.1 자동 추론 트리거 시퀀스 (Kafka → Triton)

```
┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐
│  Satellite  │  │  Kafka   │  │   Kafka      │  │RabbitMQ │  │ Analysis │  │ Triton │  │PostgreSQL│
│  Simulator  │  │  Topic   │  │  Inference   │  │  Queue  │  │  Worker  │  │ Server │  │          │
│             │  │          │  │  Trigger     │  │         │  │          │  │        │  │          │
└──────┬──────┘  └────┬─────┘  └──────┬───────┘  └────┬────┘  └────┬─────┘  └───┬────┘  └────┬─────┘
       │              │               │               │            │            │           │
       │ telemetry    │               │               │            │            │           │
       ├─────────────►│               │               │            │            │           │
       │ (JSON/3s)    │               │               │            │            │           │
       │              │               │               │            │            │           │
       │              │ poll()        │               │            │            │           │
       │              ├──────────────►│               │            │            │           │
       │              │               │               │            │            │           │
       │              │ telemetry msg │               │            │            │           │
       │              │◄──────────────┤               │            │            │           │
       │              │               │               │            │            │           │
       │              │               │ analyze       │            │            │           │
       │              │               │ conditions    │            │            │           │
       │              │               │ (battery<30%, │            │            │           │
       │              │               │  temp>50°C,   │            │            │           │
       │              │               │  thruster on) │            │            │           │
       │              │               │               │            │            │           │
       │              │               │ if triggered: │            │            │           │
       │              │               │ submit 4 tasks│            │            │           │
       │              │               │ (EPS,Thermal, │            │            │           │
       │              │               │  AOCS,Comm)   │            │            │           │
       │              │               ├──────────────────────────► │            │           │
       │              │               │               │  Celery    │            │           │
       │              │               │               │  Task x4   │            │           │
       │              │               │               │            │            │           │
       │              │               │               │            │ poll task  │           │
       │              │               │               │            │◄───────────┤           │
       │              │               │               │            │            │           │
       │              │               │               │            │ INSERT job │           │
       │              │               │               │            ├───────────────────────►│
       │              │               │               │            │ ON CONFLICT│           │
       │              │               │               │            │ DO UPDATE  │           │
       │              │               │               │            │            │           │
       │              │               │               │            │        job_id,         │
       │              │               │               │            │ satellite_id,status    │
       │              │               │               │            │◄───────────────────────┤
       │              │               │               │            │            │           │
       │              │               │               │            │ gRPC Infer │           │
       │              │               │               │            │ (model_name│           │
       │              │               │               │            │  data,     │           │
       │              │               │               │            │  config)   │           │
       │              │               │               │            ├───────────►│           │
       │              │               │               │            │            │           │
       │              │               │               │            │            │ Preprocess│
       │              │               │               │            │            │ LSTM Infer│
       │              │               │               │            │            │ Postproc. │
       │              │               │               │            │            │           │
       │              │               │               │            │ predictions│           │
       │              │               │               │            │ confidence │           │
       │              │               │               │            │◄───────────┤           │
       │              │               │               │            │            │           │
       │              │               │               │            │ calculate  │           │
       │              │               │               │            │ anomaly_   │           │
       │              │               │               │            │ score      │           │
       │              │               │               │            │            │           │
       │              │               │               │            │ UPDATE     │           │
       │              │               │               │            │ subsystem_ │           │
       │              │               │               │            │ inferences │           │
       │              │               │               │            ├───────────────────────►│
       │              │               │               │            │ SET status,│           │
       │              │               │               │            │ predictions│           │
       │              │               │               │            │ anomaly_   │           │
       │              │               │               │            │ detected   │           │
       │              │               │               │            │◄───────────────────────┤
       │              │               │               │            │            │           │
       │              │               │               │ ack task   │            │           │
       │              │               │               │◄───────────┤            │           │
       │              │               │               │            │            │           │
       │              │               │◄──log: inference completed─┤            │           │
       │              │               │               │            │            │           │
       │              │               │               │            │            │           │
       │              │ (4 tasks complete)            │            │            │           │
       │              │               │               │            │            │           │

┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 병렬 처리: 4개 서브시스템 (EPS, Thermal, AOCS, Comm)이 독립적으로 동시 실행           │
│ • 각 서브시스템은 별도의 Celery Task로 RabbitMQ를 통해 전송                          │
│ • Analysis Worker가 concurrency=2로 동시에 2개 작업 처리                             │
│ • PostgreSQL ON CONFLICT로 동시 INSERT 문제 해결                                    │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 수동 추론 API 시퀀스

```
┌────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐
│ Client │  │Operation │  │RabbitMQ │  │ Analysis │  │ Triton │  │PostgreSQL│
│  (Web) │  │  Server  │  │         │  │  Worker  │  │ Server │  │          │
└───┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  └───┬────┘  └────┬─────┘
    │            │             │            │            │           │
    │ POST /api/v1/inference/submit         │            │           │
    │ {                                     │            │           │
    │   "model_name": "lstm_timeseries",    │            │           │
    │   "data": [...],                      │            │           │
    │   "config": {...}                     │            │           │
    │ }                                     │            │           │
    ├───────────►│             │            │            │           │
    │            │             │            │            │           │
    │            │ generate    │            │            │           │
    │            │ job_id      │            │            │           │
    │            │             │            │            │           │
    │            │ INSERT      │            │            │           │
    │            │ status=     │            │            │           │
    │            │ 'pending'   │            │            │           │
    │            ├─────────────────────────────────────────────────►│
    │            │             │            │            │           │
    │            │ submit_     │            │            │           │
    │            │ inference_  │            │            │           │
    │            │ task()      │            │            │           │
    │            ├────────────►│            │            │           │
    │            │             │            │            │           │
    │ 200 OK     │             │            │            │           │
    │ {          │             │            │            │           │
    │  "job_id": "uuid",       │            │            │           │
    │  "status": "pending"     │            │            │           │
    │ }          │             │            │            │           │
    │◄───────────┤             │            │            │           │
    │            │             │            │            │           │
    │            │             │ consume    │            │           │
    │            │             │ task       │            │           │
    │            │             ├───────────►│            │           │
    │            │             │            │            │           │
    │            │             │            │ gRPC infer │           │
    │            │             │            ├───────────►│           │
    │            │             │            │            │           │
    │            │             │            │ predictions│           │
    │            │             │            │◄───────────┤           │
    │            │             │            │            │           │
    │            │             │            │ UPDATE     │           │
    │            │             │            │ status=    │           │
    │            │             │            │ 'completed'│           │
    │            │             │            ├───────────────────────►│
    │            │             │            │            │           │
    │            │             │ ack        │            │           │
    │            │             │◄───────────┤            │           │
    │            │             │            │            │           │
    │ GET /api/v1/inference/result/{job_id} │            │           │
    ├───────────►│             │            │            │           │
    │            │             │            │            │           │
    │            │ SELECT      │            │            │           │
    │            │ FROM        │            │            │           │
    │            │ inference_  │            │            │           │
    │            │ results     │            │            │           │
    │            ├─────────────────────────────────────────────────►│
    │            │             │            │            │           │
    │            │ result      │            │            │           │
    │            │◄─────────────────────────────────────────────────┤
    │            │             │            │            │           │
    │ 200 OK     │             │            │            │           │
    │ {predictions, confidence, metrics}    │            │           │
    │◄───────────┤             │            │            │           │
    │            │             │            │            │           │
```

### 1.3 실시간 대시보드 WebSocket 시퀀스

```
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Client │  │Operation │  │PostgreSQL│  │  Kafka   │
│  (Web) │  │  Server  │  │          │  │  Topic   │
└───┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
    │            │             │            │
    │ WS CONNECT │             │            │
    │ /api/v1/dashboard/ws     │            │
    ├───────────►│             │            │
    │            │             │            │
    │ WS ACCEPT  │             │            │
    │◄───────────┤             │            │
    │            │             │            │
    │            │ (2초마다 반복)           │
    │            │             │            │
    │            │ SELECT      │            │
    │            │ recent_     │            │
    │            │ results     │            │
    │            ├────────────►│            │
    │            │             │            │
    │            │ results     │            │
    │            │◄────────────┤            │
    │            │             │            │
    │            │ SELECT      │            │
    │            │ statistics  │            │
    │            ├────────────►│            │
    │            │             │            │
    │            │ stats       │            │
    │            │◄────────────┤            │
    │            │             │            │
    │ WS MESSAGE │             │            │
    │ {          │             │            │
    │   "type": "update",      │            │
    │   "recent_results": [...],            │
    │   "statistics": {...}    │            │
    │ }          │             │            │
    │◄───────────┤             │            │
    │            │             │            │
    │            │             │ (새 결과가 │
    │            │             │  저장될 때) │
    │            │             │◄───────────┤
    │            │             │            │
    │            │ 즉시 조회   │            │
    │            ├────────────►│            │
    │            │             │            │
    │ WS MESSAGE │             │            │
    │ (실시간 업데이트)         │            │
    │◄───────────┤             │            │
    │            │             │            │
```

## 2. 클래스 다이어그램

### 2.1 핵심 도메인 모델

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Domain Models                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────┐
│   InferenceRequest        │
├───────────────────────────┤
│ + model_name: str         │
│ + data: List[float]       │
│ + config: Dict[str, Any]  │
│ + metadata: Dict[str, Any]│
└───────────────────────────┘
          │
          │ creates
          ▼
┌───────────────────────────┐         ┌────────────────────────────┐
│   InferenceJob            │◄────────┤  SubsystemInference        │
├───────────────────────────┤ 1    * ├────────────────────────────┤
│ + job_id: str (PK)        │         │ + id: int (PK)             │
│ + satellite_id: str       │         │ + job_id: str (FK)         │
│ + source: str             │         │ + subsystem: str           │
│ + trigger_reason: str     │         │ + model_name: str          │
│ + status: str             │         │ + status: str              │
│ + total_subsystems: int   │         │ + input_data: JSONB        │
│ + completed_subsystems:int│         │ + input_features: str[]    │
│ + created_at: datetime    │         │ + predictions: JSONB       │
│ + completed_at: datetime  │         │ + confidence: JSONB        │
│ + metadata: JSONB         │         │ + anomaly_score: float     │
│ + error_message: str      │         │ + anomaly_detected: bool   │
└───────────────────────────┘         │ + inference_time_ms: float │
          │                           │ + sequence_length: int     │
          │                           │ + forecast_horizon: int    │
          │                           │ + processed_by: str        │
          │                           │ + created_at: datetime     │
          │                           │ + completed_at: datetime   │
          │                           └────────────────────────────┘
          │
          │ 1
          ▼
┌───────────────────────────┐
│   InferenceResponse       │
├───────────────────────────┤
│ + job_id: str             │
│ + status: JobStatus       │
│ + message: str            │
└───────────────────────────┘
          │
          │ enum
          ▼
┌───────────────────────────┐
│   JobStatus (Enum)        │
├───────────────────────────┤
│ • PENDING                 │
│ • PROCESSING              │
│ • COMPLETED               │
│ • FAILED                  │
└───────────────────────────┘
```

### 2.2 서비스 계층 클래스 다이어그램

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Service Layer                                │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐
│   TritonClient          │
├─────────────────────────┤
│ - server_url: str       │
│ - grpc_client: InferenceServerClient
├─────────────────────────┤
│ + __init__(url: str)    │
│ + infer(               │
│     model_name: str,    │
│     data: List[float],  │
│     config: Dict        │
│   ) -> Dict             │
│ + _prepare_input(...)   │
│ + _parse_output(...)    │
└───────────┬─────────────┘
            │ uses
            ▼
┌─────────────────────────────────┐
│  tritonclient.grpc              │
│  .InferenceServerClient         │
├─────────────────────────────────┤
│ + infer(model_name, inputs, ...)│
│ + is_server_live()              │
│ + is_server_ready()             │
└─────────────────────────────────┘


┌─────────────────────────┐
│   PostgresClient        │
├─────────────────────────┤
│ - conn_params: Dict     │
├─────────────────────────┤
│ + get_connection()      │
│ + init_tables()         │
│ + save_result(...)      │
│ + get_result(job_id)    │
│ + get_statistics()      │
│ + update_job_status(...) │
│ + query(sql, params)    │
└─────────────────────────┘
            │ manages
            ▼
┌─────────────────────────┐
│   psycopg2.connection   │
├─────────────────────────┤
│ + cursor()              │
│ + commit()              │
│ + rollback()            │
│ + close()               │
└─────────────────────────┘


┌──────────────────────────┐
│   CeleryApp              │
├──────────────────────────┤
│ - broker_url: str        │
│ - backend_url: str       │
├──────────────────────────┤
│ + send_task(            │
│     name: str,           │
│     args: List,          │
│     queue: str           │
│   ) -> AsyncResult       │
└──────────────────────────┘
            │ creates
            ▼
┌──────────────────────────┐
│   InferenceTask          │
│   (Celery Task)          │
├──────────────────────────┤
│ + run_inference(...)     │
│ + run_subsystem_         │
│   inference(...)         │
│ + on_failure(...)        │
│ + on_success(...)        │
└──────────────────────────┘
```

### 2.3 API 라우터 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Routers                            │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────────┐
│   FastAPI App                  │
├────────────────────────────────┤
│ + include_router(router, ...)  │
└────────────────┬───────────────┘
                 │ includes
       ┌─────────┼──────────┬──────────┬─────────────┐
       │         │          │          │             │
       ▼         ▼          ▼          ▼             ▼
┌──────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐
│Inference │ │Dashboard│ │  Trends  │ │  Query  │ │ Search  │
│  Router  │ │ Router  │ │  Router  │ │ Router  │ │ Router  │
└──────────┘ └─────────┘ └──────────┘ └─────────┘ └─────────┘


┌─────────────────────────────────────────────────┐
│   InferenceRouter                               │
├─────────────────────────────────────────────────┤
│ + POST   /submit                                │
│     → submit_inference(request)                 │
│                                                 │
│ + GET    /status/{job_id}                       │
│     → get_status(job_id)                        │
│                                                 │
│ + GET    /result/{job_id}                       │
│     → get_result(job_id)                        │
│                                                 │
│ + WS     /ws/{job_id}                           │
│     → websocket_endpoint(websocket, job_id)     │
└─────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────┐
│   DashboardRouter                               │
├─────────────────────────────────────────────────┤
│ + WS     /ws                                    │
│     → dashboard_websocket(websocket)            │
│                                                 │
│ + GET    /recent?limit=20                       │
│     → get_recent_results(limit)                 │
│                                                 │
│ + GET    /live-stats                            │
│     → get_live_statistics()                     │
│                                                 │
│ + GET    /model-comparison                      │
│     → get_model_comparison()                    │
└─────────────────────────────────────────────────┘
            │ uses
            ▼
┌─────────────────────────────────────────────────┐
│   DashboardManager                              │
├─────────────────────────────────────────────────┤
│ - active_connections: List[WebSocket]           │
├─────────────────────────────────────────────────┤
│ + connect(websocket: WebSocket)                 │
│ + disconnect(websocket: WebSocket)              │
│ + broadcast(message: dict)                      │
└─────────────────────────────────────────────────┘
```

## 3. 컴포넌트 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Component Architecture                             │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│   Frontend Component     │
│  ┌────────────────────┐  │
│  │  React App         │  │
│  │  ├─ Dashboard      │  │
│  │  ├─ Monitoring     │  │
│  │  └─ Trends View    │  │
│  └────────────────────┘  │
└──────────┬───────────────┘
           │ HTTP/WS
           ▼
┌──────────────────────────┐
│   Nginx Component        │
│  ┌────────────────────┐  │
│  │  Reverse Proxy     │  │
│  │  ├─ /api/* → API   │  │
│  │  └─ /* → Frontend  │  │
│  └────────────────────┘  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│   Operation Server Component                 │
│  ┌────────────────────────────────────────┐  │
│  │  FastAPI Application                   │  │
│  │  ├─ InferenceRouter                    │  │
│  │  ├─ DashboardRouter                    │  │
│  │  ├─ TrendsRouter                       │  │
│  │  ├─ QueryRouter                        │  │
│  │  └─ SearchRouter                       │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  Celery Client (Task Producer)        │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  Database Clients                      │  │
│  │  ├─ PostgresClient                     │  │
│  │  └─ ElasticsearchClient                │  │
│  └────────────────────────────────────────┘  │
└──────────┬─────────────┬─────────────────────┘
           │             │
    ┌──────┘             └──────────┐
    │                               │
    ▼                               ▼
┌─────────────────┐      ┌──────────────────────┐
│  RabbitMQ       │      │  PostgreSQL          │
│  Component      │      │  Component           │
│  ┌───────────┐  │      │  ┌────────────────┐  │
│  │  Queue:   │  │      │  │ inference_jobs │  │
│  │  inference│  │      │  │ subsystem_     │  │
│  └───────────┘  │      │  │   inferences   │  │
└────────┬────────┘      │  └────────────────┘  │
         │               └──────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│   Analysis Server Component                  │
│  ┌────────────────────────────────────────┐  │
│  │  Celery Worker (Task Consumer)        │  │
│  │  ├─ run_inference                     │  │
│  │  └─ run_subsystem_inference           │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  Triton Client (gRPC)                 │  │
│  └────────────────────────────────────────┘  │
└──────────┬───────────────────────────────────┘
           │ gRPC
           ▼
┌──────────────────────────────────────────────┐
│   Triton Inference Server Component          │
│  ┌────────────────────────────────────────┐  │
│  │  Model Repository                      │  │
│  │  ├─ lstm_timeseries (EPS)              │  │
│  │  ├─ lstm_timeseries (Thermal)          │  │
│  │  ├─ lstm_timeseries (AOCS)             │  │
│  │  └─ lstm_timeseries (Comm)             │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  Python Backend Runtime                │  │
│  │  ├─ Preprocessing                      │  │
│  │  ├─ LSTM Inference (PyTorch)           │  │
│  │  └─ Postprocessing                     │  │
│  └────────────────────────────────────────┘  │
│  GPU: NVIDIA CUDA 13.0                       │
└──────────────────────────────────────────────┘


┌──────────────────────────────────────────────┐
│   Kafka Component                            │
│  ┌────────────────────────────────────────┐  │
│  │  Topic: satellite-telemetry            │  │
│  └────────┬───────────────────────────────┘  │
│           │                                   │
└───────────┼───────────────────────────────────┘
            │
       ┌────┴────┐
       │         │
       ▼         ▼
┌──────────┐  ┌────────────────────┐
│ Victoria │  │ Kafka Inference    │
│ Consumer │  │ Trigger Component  │
└────┬─────┘  └─────────┬──────────┘
     │                  │
     ▼                  │
┌──────────┐            │
│Victoria  │            │
│ Metrics  │            │
│  (TSDB)  │            │
└──────────┘            │
                        └──► (RabbitMQ로 Task 전송)
```

## 4. 상태 다이어그램

### 4.1 Inference Job 상태 전이

```
                  ┌─────────┐
                  │         │
                  │ START   │
                  │         │
                  └────┬────┘
                       │ API 요청 또는 Kafka 트리거
                       ▼
              ┌────────────────┐
              │                │
              │    PENDING     │◄──┐
              │                │   │ 재시도
              └────┬───────────┘   │
                   │ Worker가 Task Pickup
                   ▼               │
              ┌────────────────┐   │
              │                │   │
              │   PROCESSING   │   │
              │                │   │
              └────┬───────────┘   │
                   │                │
      ┌────────────┼────────────┐   │
      │            │            │   │
      │ 성공       │ 실패       │   │
      │            │            │   │
      ▼            ▼            │   │
┌──────────┐ ┌──────────┐      │   │
│          │ │          │      │   │
│COMPLETED │ │  FAILED  │──────┘   │
│          │ │          │  retry   │
└──────────┘ └──────────┘          │
                   │                │
                   │ 최종 실패       │
                   ▼                │
              ┌──────────┐          │
              │          │          │
              │   ERROR  │──────────┘
              │          │
              └──────────┘
```

### 4.2 Subsystem Inference 상태 전이

```
                  ┌─────────┐
                  │  START  │
                  └────┬────┘
                       │ Celery Task 생성
                       ▼
              ┌────────────────┐
              │                │
              │    PENDING     │
              │                │
              └────┬───────────┘
                   │ Worker 처리 시작
                   ▼
              ┌────────────────┐
              │                │
              │   PROCESSING   │
              │                │
              └────┬───────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      │            ▼            │
      │    ┌──────────────┐    │
      │    │   Triton     │    │
      │    │   Inference  │    │
      │    └──────┬───────┘    │
      │           │            │
      │  성공     │  실패      │
      │           │            │
      ▼           ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│          │ │          │ │          │
│COMPLETED │ │  FAILED  │ │  TIMEOUT │
│          │ │          │ │          │
└──────────┘ └──────────┘ └──────────┘
      │
      │ anomaly_detected 업데이트
      ▼
┌──────────────────┐
│ Anomaly Analysis │
│ (score > 0.7)    │
└──────────────────┘
```

## 5. 배포 다이어그램

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Docker Host                                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                   Docker Network: webnet                           │  │
│  │                                                                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
│  │  │  nginx   │  │ frontend │  │operation │  │analysis- │         │  │
│  │  │  :80     │  │          │  │ server   │  │ worker-1 │         │  │
│  │  │          │  │          │  │  :8000   │  │          │         │  │
│  │  └────┬─────┘  └──────────┘  └────┬─────┘  └────┬─────┘         │  │
│  │       │                           │             │                │  │
│  │  ┌────┴─────────────────────────────────────────┴─────┐          │  │
│  │  │                                                     │          │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │          │  │
│  │  │  │ postgres │  │ rabbitmq │  │  kafka   │         │          │  │
│  │  │  │  :5432   │  │  :5672   │  │  :9092   │         │          │  │
│  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │          │  │
│  │  │       │             │             │                │          │  │
│  │  │       │             │             │                │          │  │
│  │  │  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐         │          │  │
│  │  │  │postgres_ │  │          │  │ kafka_   │         │          │  │
│  │  │  │  data    │  │          │  │  data    │         │          │  │
│  │  │  │ (volume) │  │          │  │(volume)  │         │          │  │
│  │  │  └──────────┘  │          │  └──────────┘         │          │  │
│  │  │                │          │                       │          │  │
│  │  │  ┌──────────┐  │          │  ┌──────────┐        │          │  │
│  │  │  │  triton  │  │          │  │victoria- │        │          │  │
│  │  │  │ -server  │  │          │  │ metrics  │        │          │  │
│  │  │  │:8500-8502│  │          │  │  :8428   │        │          │  │
│  │  │  │   GPU    │  │          │  └──────────┘        │          │  │
│  │  │  └──────────┘  │          │                       │          │  │
│  │  │                │          │                       │          │  │
│  │  └────────────────┴──────────┴───────────────────────┘          │  │
│  │                                                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  External Access:                                                     │
│  • :80     → Nginx (Web)                                              │
│  • :8000   → Operation Server API                                     │
│  • :8080   → Kafka UI                                                 │
│  • :8428   → VictoriaMetrics                                          │
│  • :5555   → Flower (Celery Monitor)                                  │
│  • :15672  → RabbitMQ Management                                      │
└────────────────────────────────────────────────────────────────────────┘
```
