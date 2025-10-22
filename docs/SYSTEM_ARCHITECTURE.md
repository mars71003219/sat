# 인공위성 시계열 데이터 모니터링 시스템 - 시스템 구성도

## 📊 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         인공위성 시계열 모니터링 시스템                              │
│                    Satellite Time-Series Monitoring System                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              1. 데이터 수집 계층                                   │
│                           Data Collection Layer                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │  Satellite Data Simulator           │
    │  tests/satellite_simulator.py       │
    │                                     │
    │  • Python 3.10                      │
    │  • confluent-kafka                  │
    │  • 5가지 센서 시뮬레이션               │
    │    - Temperature: -50°C ~ 50°C      │
    │    - Altitude: 400km ~ 450km        │
    │    - Velocity: 7.6 ~ 7.8 km/s       │
    │    - Battery: 3.0V ~ 4.2V           │
    │    - Solar Power: 0W ~ 100W         │
    │  • 위치: 위도/경도 (궤도 51.6°)        │
    └─────────────────┬───────────────────┘
                      │
                      │ Kafka Produce
                      │ (JSON Messages)
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              2. 메시지 큐 계층                                    │
│                           Message Queue Layer                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  Apache Kafka (KRaft Mode)                                  │
    │  Container: kafka                                           │
    │  Image: confluentinc/cp-kafka:latest                        │
    │  Port: 9092                                                 │
    │                                                             │
    │  Topics:                                                    │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ 1. satellite-telemetry                                │ │
    │  │    - Partitions: 1                                    │ │
    │  │    - Replication: 1                                   │ │
    │  │    - Producer: satellite_simulator                    │ │
    │  │    - Consumer: victoria-consumer                      │ │
    │  └───────────────────────────────────────────────────────┘ │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ 2. inference.results                                  │ │
    │  │    - Partitions: 1                                    │ │
    │  │    - Producer: analysis-server                        │ │
    │  │    - Consumer: operation-server                       │ │
    │  └───────────────────────────────────────────────────────┘ │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ 3. __consumer_offsets (System)                        │ │
    │  │    - Partitions: 50                                   │ │
    │  └───────────────────────────────────────────────────────┘ │
    └─────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            3. 데이터 저장 계층                                   │
│                          Data Storage Layer                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐        ┌──────────────────────────────┐
│  Victoria Consumer       │        │  Analysis Server             │
│  victoria-consumer/      │        │  analysis-server/            │
│                          │        │                              │
│  • Python 3.10           │        │  • Celery Worker             │
│  • confluent-kafka       │        │  • Triton Client             │
│  • requests              │        │  • Python Backend            │
│                          │        │                              │
│  Process:                │        │  Process:                    │
│  1. Consume from Kafka   │        │  1. Consume inference tasks  │
│  2. Parse telemetry      │        │  2. Call Triton Server       │
│  3. Format Prometheus    │        │  3. Get predictions          │
│  4. Write VictoriaMetrics│        │  4. Store in PostgreSQL      │
└──────────┬───────────────┘        └──────────┬───────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────────┐        ┌──────────────────────────────┐
│  VictoriaMetrics         │        │  PostgreSQL                  │
│  Time-Series Database    │        │  Relational Database         │
│                          │        │                              │
│  • Port: 8428            │        │  • Port: 5432                │
│  • Storage: 1 year       │        │  • Database: orders_db       │
│  • Format: Prometheus    │        │  • User: admin               │
│                          │        │                              │
│  Metrics:                │        │  Tables:                     │
│  • satellite_temperature │        │  • inference_results         │
│  • satellite_altitude    │        │  • (job metadata)            │
│  • satellite_velocity    │        │                              │
│  • satellite_battery     │        └──────────────────────────────┘
│  • satellite_solar_power │
│  • satellite_latitude    │
│  • satellite_longitude   │
└──────────┬───────────────┘
           │
           │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            4. AI 추론 계층                                       │
│                         AI Inference Layer                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────────────┐
           │  Triton Inference Server                     │
           │  Container: triton-server                    │
           │  Image: triton-server-pytorch:latest         │
           │                                              │
           │  • Port: 8500 (HTTP), 8501 (gRPC)            │
           │  • GPU: NVIDIA RTX 5060 (sm_100)             │
           │  • PyTorch: 2.8.0                            │
           │  • Python: 3.12.3                            │
           │                                              │
           │  Models:                                     │
           │  ┌────────────────────────────────────────┐  │
           │  │ 1. vae_timeseries                      │  │
           │  │    - Backend: Python                   │  │
           │  │    - Input: [1, 20] (sequence)         │  │
           │  │    - Output: [10] (forecast)           │  │
           │  │    - Latent Dim: 32                    │  │
           │  └────────────────────────────────────────┘  │
           │  ┌────────────────────────────────────────┐  │
           │  │ 2. transformer_timeseries              │  │
           │  │    - Backend: Python                   │  │
           │  │    - Input: [1, 20] (sequence)         │  │
           │  │    - Output: [10] (forecast)           │  │
           │  │    - d_model: 64, nhead: 4             │  │
           │  └────────────────────────────────────────┘  │
           └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            5. API 계층                                           │
│                           API Layer                                              │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  Operation Server                                           │
    │  operation-server/                                          │
    │                                                             │
    │  • Framework: FastAPI                                       │
    │  • Port: 8000                                               │
    │  • Python: 3.10                                             │
    │                                                             │
    │  API Endpoints:                                             │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ Inference APIs                                        │ │
    │  │ • POST /api/v1/inference/submit                       │ │
    │  │ • GET  /api/v1/inference/result/{job_id}              │ │
    │  └───────────────────────────────────────────────────────┘ │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ Trend APIs (NEW!)                                     │ │
    │  │ • GET  /api/v1/trends/raw                             │ │
    │  │ • GET  /api/v1/trends/prediction                      │ │
    │  │ • GET  /api/v1/trends/compare                         │ │
    │  │ • GET  /api/v1/trends/metrics                         │ │
    │  │ • GET  /api/v1/trends/satellites                      │ │
    │  └───────────────────────────────────────────────────────┘ │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ Query APIs                                            │ │
    │  │ • GET  /api/v1/query/jobs                             │ │
    │  │ • GET  /api/v1/query/stats                            │ │
    │  └───────────────────────────────────────────────────────┘ │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ Search APIs                                           │ │
    │  │ • GET  /api/v1/search/results                         │ │
    │  └───────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            6. 프론트엔드 계층                                     │
│                          Frontend Layer                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  Nginx Reverse Proxy                                        │
    │  Container: web_server                                      │
    │  Port: 80                                                   │
    │                                                             │
    │  Routes:                                                    │
    │  • /          → frontend (React SPA)                        │
    │  • /api       → operation-server:8000                       │
    │  • /dashboard.html → static files                           │
    └─────────────────────┬───────────────────────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
    ┌──────────────┐            ┌──────────────────────┐
    │  Frontend    │            │  Operation Server    │
    │              │            │  API Backend         │
    │  • React 19  │            │  FastAPI             │
    │  • Recharts  │            └──────────────────────┘
    │  • Dark Theme│
    │              │
    │  Components: │
    │  • TrendDashboard.js                              │
    │    - Metric Cards                                 │
    │    - Time Range Selector (1h, 6h, 1d, 1w)        │
    │    - Line Charts (Raw vs Prediction)             │
    │    - Statistics Panel                             │
    │    - Real-time Updates (30s polling)             │
    │                                                   │
    │  Design:                                          │
    │  • Background: #191a1f (Dark)                     │
    │  • Text: #ffffff                                  │
    │  • Accent: #4a9eff (Blue), #4ade80 (Green)       │
    └───────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            7. 보조 서비스 계층                                    │
│                        Supporting Services Layer                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Redis       │  │ Elasticsearch│  │ Kafka UI     │  │ Flower       │
│              │  │              │  │              │  │              │
│ • Port: 6379 │  │ • Port: 9200 │  │ • Port: 8080 │  │ • Port: 5555 │
│ • Cache      │  │ • Search     │  │ • Monitoring │  │ • Celery UI  │
│ • Sessions   │  │ • Logs       │  │ • Topics     │  │ • Tasks      │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            8. 네트워크 구성                                       │
│                          Network Configuration                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────┐
                        │  Docker Network      │
                        │  Name: webnet        │
                        │  Driver: bridge      │
                        └──────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   All containers communicate via Docker internal DNS
   (kafka:9092, victoria-metrics:8428, postgres:5432, etc.)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            9. 볼륨 구성                                          │
│                          Volume Configuration                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ kafka_data       │  │ postgres_data    │  │ victoria_data    │
│                  │  │                  │  │                  │
│ • Kafka logs     │  │ • DB files       │  │ • Time-series    │
│ • Topic data     │  │ • Tables         │  │ • Metrics        │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          10. 전체 데이터 흐름                                    │
│                        Complete Data Flow                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│   사용자     │
│   (User)    │
└──────┬──────┘
       │
       │ 1. 브라우저 접속 (http://localhost)
       ▼
┌─────────────────────┐
│   Nginx (Port 80)   │
└──────┬──────────────┘
       │
       ├─── 2a. Static Files → React Frontend
       │                              │
       │                              │ 3. API 호출
       │                              ▼
       └─── 2b. /api/* → Operation Server (Port 8000)
                              │
                              ├─── 4a. GET /trends/raw
                              │        ↓
                              │    VictoriaMetrics 쿼리
                              │        ↓
                              │    원본 시계열 데이터 반환
                              │
                              ├─── 4b. GET /trends/prediction
                              │        ↓
                              │    PostgreSQL 쿼리
                              │        ↓
                              │    예측 결과 반환
                              │
                              └─── 4c. POST /inference/submit
                                       ↓
                                   Celery Task 생성
                                       ↓
                                   Analysis Server
                                       ↓
                                   Triton Inference
                                       ↓
                                   Result → PostgreSQL

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          11. 컨테이너 목록                                       │
│                        Container List                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

| # | Container Name      | Image                        | Port Mapping         |
|---|---------------------|------------------------------|----------------------|
| 1 | kafka               | confluentinc/cp-kafka        | 9092:9092            |
| 2 | victoria-metrics    | victoriametrics/victoria     | 8428:8428            |
| 3 | victoria-consumer   | satellite-victoria-consumer  | -                    |
| 4 | operation-server    | satellite-operation-server   | 8000:8000            |
| 5 | triton-server       | triton-server-pytorch        | 8500-8502:8000-8002  |
| 6 | analysis-worker-1   | satellite-analysis-worker    | -                    |
| 7 | postgres            | postgres:latest              | 5432:5432            |
| 8 | redis               | redis/redis-stack            | 6379:6379, 8001:8001 |
| 9 | elasticsearch       | elasticsearch:8.5.0          | 9200:9200            |
|10 | kafka-ui            | provectuslabs/kafka-ui       | 8080:8080            |
|11 | flower              | satellite-flower             | 5555:5555            |
|12 | web_frontend        | satlas-ui                    | 80 (internal)        |
|13 | web_server          | nginx:alpine                 | 80:80                |

**총 컨테이너 수: 13개**

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          12. 포트 매핑                                           │
│                        Port Mapping                                              │
└─────────────────────────────────────────────────────────────────────────────────┘

| Port  | Service              | Protocol | Purpose                      |
|-------|----------------------|----------|------------------------------|
| 80    | Nginx                | HTTP     | 프론트엔드 + API Gateway     |
| 5432  | PostgreSQL           | TCP      | 데이터베이스                 |
| 5555  | Flower               | HTTP     | Celery 모니터링 UI           |
| 6379  | Redis                | TCP      | 캐시 + 세션                  |
| 8000  | Operation Server     | HTTP     | FastAPI REST API             |
| 8001  | RedisInsight         | HTTP     | Redis 관리 UI                |
| 8080  | Kafka UI             | HTTP     | Kafka 모니터링 UI            |
| 8428  | VictoriaMetrics      | HTTP     | 시계열 DB API                |
| 8500  | Triton HTTP          | HTTP     | 추론 요청 (HTTP)             |
| 8501  | Triton gRPC          | gRPC     | 추론 요청 (gRPC)             |
| 8502  | Triton Metrics       | HTTP     | Triton 메트릭                |
| 9092  | Kafka                | TCP      | Kafka 브로커                 |
| 9200  | Elasticsearch        | HTTP     | 검색 엔진 API                |

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          13. 기술 스택                                           │
│                        Technology Stack                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

**Backend:**
- Python 3.10/3.12
- FastAPI 0.109.0
- Celery 5.3.6
- PyTorch 2.8.0
- Triton Inference Server 25.08

**Frontend:**
- React 19.2.0
- Recharts 2.10.3
- date-fns 3.0.6

**Data Storage:**
- VictoriaMetrics (Time-Series)
- PostgreSQL 17
- Redis Stack

**Message Queue:**
- Apache Kafka (KRaft Mode)
- confluent-kafka-python 2.3.0

**Infrastructure:**
- Docker & Docker Compose
- Nginx (Reverse Proxy)
- NVIDIA Container Toolkit (GPU)

**Monitoring:**
- Flower (Celery)
- Kafka UI
- RedisInsight
- Elasticsearch

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          14. 시스템 특징                                         │
│                        System Features                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

✅ **마이크로서비스 아키텍처**
- 13개의 독립적인 서비스
- 각 서비스는 단일 책임 원칙 준수
- Docker Compose 오케스트레이션

✅ **실시간 데이터 파이프라인**
- Kafka 기반 이벤트 스트리밍
- VictoriaMetrics 시계열 저장
- 30초 자동 새로고침

✅ **AI/ML 추론**
- NVIDIA GPU 가속
- Triton Inference Server
- PyTorch 2.8.0 (RTX 5060 지원)
- VAE + Transformer 모델

✅ **확장 가능한 설계**
- 수평 확장 가능 (Consumer, Worker)
- Kafka 파티셔닝
- 독립적 서비스 배포

✅ **모니터링 & 관찰성**
- Flower (Celery 작업 모니터링)
- Kafka UI (메시지 큐 모니터링)
- Elasticsearch (로그 검색)
- VictoriaMetrics (메트릭)

✅ **다크 테마 UI**
- React 기반 SPA
- Recharts 시각화
- 반응형 디자인
- 실시간 데이터 업데이트
