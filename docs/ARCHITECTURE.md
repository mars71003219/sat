# 시스템 아키텍처

## 개요

위성 이미지 분석을 위한 마이크로서비스 기반 AI 추론 시스템입니다. Operation Server와 Analysis Server로 분리된 아키텍처를 사용하며, Celery를 통한 비동기 작업 처리와 실시간 모니터링 기능을 제공합니다.

## 전체 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Web Browser                                                     │
│  ├─ Dashboard (http://localhost/)                               │
│  ├─ Kafka UI (http://localhost:8080)                            │
│  ├─ Redis Insight (http://localhost:8001)                       │
│  └─ Celery Flower (http://localhost:5555)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Proxy Layer (Nginx)                         │
├─────────────────────────────────────────────────────────────────┤
│  Port 80                                                         │
│  ├─ / → Static HTML (Dashboard)                                 │
│  ├─ /api → Operation Server (8000)                              │
│  └─ /ws → WebSocket (Operation Server)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐          ┌─────────────────────┐       │
│  │ Operation Server   │          │ Analysis Worker     │       │
│  │ (FastAPI)          │          │ (Celery Consumer)   │       │
│  │ Port: 8000         │          │ GPU Enabled         │       │
│  │                    │          │                     │       │
│  │ - REST API         │◄────────►│ - LSTM Model        │       │
│  │ - WebSocket        │  Celery  │ - Moving Average    │       │
│  │ - Task Producer    │          │ - Task Processing   │       │
│  └────────────────────┘          └─────────────────────┘       │
│           │                                  │                  │
│           │                                  │                  │
└───────────┼──────────────────────────────────┼──────────────────┘
            │                                  │
            ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Redis    │  │PostgreSQL│  │  Kafka   │  │Elastic   │       │
│  │          │  │          │  │          │  │Search    │       │
│  │ DB 0:    │  │ Orders   │  │ Topics:  │  │          │       │
│  │  Cache   │  │ Database │  │ - infer  │  │ Logs &   │       │
│  │ DB 1:    │  │          │  │   ence_  │  │ Metrics  │       │
│  │  Broker  │  │ Tables:  │  │   results│  │          │       │
│  │ DB 2:    │  │ - infer  │  │          │  │          │       │
│  │  Backend │  │   ence_  │  │          │  │          │       │
│  │          │  │   results│  │          │  │          │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│  Port: 6379    Port: 5432    Port: 9092    Port: 9200         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 컴포넌트 구성

### 1. Application Layer

#### Operation Server (FastAPI)
- **역할**: API Gateway 및 작업 관리
- **포트**: 8000
- **주요 기능**:
  - REST API 제공 (/api/v1/inference/submit)
  - WebSocket 실시간 대시보드 (/api/v1/dashboard/ws)
  - Celery Producer (작업 큐 전송)
  - 데이터베이스 조회 및 저장

#### Analysis Worker (Celery Consumer)
- **역할**: AI 모델 추론 실행
- **주요 기능**:
  - GPU 기반 LSTM 모델 추론
  - Moving Average 계산
  - 비동기 작업 처리
  - 결과 저장 (PostgreSQL, Redis, Kafka)

### 2. Infrastructure Layer

#### Redis (Cache & Message Broker)
- **이미지**: redis/redis-stack:latest
- **포트**: 6379 (Redis), 8001 (RedisInsight UI)
- **데이터베이스 구성**:
  - DB 0: 애플리케이션 캐시
  - DB 1: Celery Broker (작업 큐)
  - DB 2: Celery Result Backend

#### PostgreSQL (Primary Database)
- **이미지**: postgres:latest
- **포트**: 5432
- **데이터베이스**: orders_db
- **주요 테이블**:
  - `inference_results`: 추론 결과 저장

#### Kafka (Event Stream)
- **이미지**: confluentinc/cp-kafka:latest
- **포트**: 9092
- **모드**: KRaft (Zookeeper 불필요)
- **주요 토픽**:
  - `inference_results`: 추론 결과 이벤트

#### Elasticsearch (Search & Analytics)
- **이미지**: elasticsearch:8.5.0
- **포트**: 9200
- **용도**: 로그 수집 및 검색

### 3. Monitoring Layer

#### Kafka UI
- **포트**: 8080
- **기능**: Kafka 토픽, 메시지, 컨슈머 모니터링

#### Redis Insight
- **포트**: 8001
- **기능**: Redis 데이터 조회 및 관리

#### Celery Flower
- **포트**: 5555
- **기능**: Celery 워커 및 작업 모니터링

#### AI Dashboard
- **포트**: 80 (Nginx)
- **기능**: 실시간 추론 결과 대시보드

### 4. Proxy Layer

#### Nginx
- **포트**: 80
- **설정 파일**:
  - `nginx/nginx.conf`
  - `nginx/conf.d/default.conf`
- **라우팅**:
  - `/` → Static HTML (Dashboard)
  - `/api` → Operation Server
  - `/ws` → WebSocket Proxy

## 데이터 플로우

### 추론 요청 플로우

```
1. Client → Operation Server
   POST /api/v1/inference/submit
   {
     "model_name": "lstm_timeseries",
     "data": [1.0, 2.0, 3.0, ...],
     "metadata": {"pattern": "seasonal"}
   }

2. Operation Server → Celery (Redis)
   Task: inference_task.apply_async()
   Queue: inference

3. Analysis Worker ← Celery
   Worker receives task

4. Analysis Worker → AI Model
   LSTM or Moving Average inference

5. Analysis Worker → Storage
   ├─ PostgreSQL: INSERT inference_results
   ├─ Redis: SET cache key
   └─ Kafka: SEND inference_results topic

6. Operation Server ← PostgreSQL
   SELECT recent results for dashboard

7. Client ← WebSocket
   Real-time update every 2 seconds
```

### WebSocket 실시간 업데이트 플로우

```
1. Client → Nginx → Operation Server
   WebSocket: ws://localhost/api/v1/dashboard/ws

2. Operation Server (Loop every 2s)
   ├─ Query PostgreSQL (recent_results)
   ├─ Query PostgreSQL (statistics)
   └─ Send JSON to client

3. Client ← WebSocket
   {
     "type": "update",
     "timestamp": "2025-10-14T12:00:00",
     "recent_results": [...],
     "statistics": {
       "total_jobs": 1000,
       "completed": 995,
       "failed": 5,
       "success_rate": 99.5
     }
   }
```

## 네트워크 구성

### Docker Network: webnet (bridge)
모든 컨테이너가 동일한 네트워크에서 통신:

- `kafka`: Kafka 브로커
- `redis`: Redis 서버
- `postgres`: PostgreSQL 서버
- `elasticsearch`: Elasticsearch 서버
- `operation-server`: Operation Server
- `analysis-worker-1`: Analysis Worker
- `kafka-ui`: Kafka UI
- `flower`: Celery Flower
- `nginx`: Nginx 프록시

### 서비스 간 통신

```
Operation Server → Redis:        redis://redis:6379
Operation Server → PostgreSQL:   postgresql://postgres:5432
Operation Server → Kafka:        kafka:9092
Operation Server → Elasticsearch: http://elasticsearch:9200

Analysis Worker → Redis:          redis://redis:6379
Analysis Worker → PostgreSQL:     postgresql://postgres:5432
Analysis Worker → Kafka:          kafka:9092

Kafka UI → Kafka:                 kafka:9092
Nginx → Operation Server:         http://operation-server:8000
```

## 스케일링 전략

### 수평 스케일링 (Horizontal Scaling)

1. **Analysis Worker 확장**:
   ```bash
   docker compose up -d --scale analysis-worker-1=3
   ```
   - GPU 리소스가 허용하는 만큼 워커 추가 가능
   - Celery가 자동으로 작업 분산

2. **Kafka 파티션 증가**:
   - 토픽 파티션 수 증가로 처리량 향상
   - 컨슈머 그룹 추가

### 수직 스케일링 (Vertical Scaling)

1. **Redis 메모리 증가**:
   - Docker 컨테이너 메모리 제한 조정

2. **PostgreSQL 성능 튜닝**:
   - connection pool 크기 조정
   - 인덱스 최적화

3. **GPU 리소스 할당**:
   - CUDA_VISIBLE_DEVICES 설정
   - Multi-GPU 지원

## 보안 고려사항

### 현재 구성
- 개발 환경용 설정 (인증 없음)
- 모든 포트가 localhost에 바인딩

### 프로덕션 권장사항

1. **인증 및 권한**:
   - API 키 또는 JWT 토큰 인증
   - PostgreSQL/Redis 패스워드 강화
   - Kafka SASL/SSL 활성화

2. **네트워크 격리**:
   - 내부 네트워크와 외부 네트워크 분리
   - 방화벽 규칙 설정

3. **데이터 암호화**:
   - TLS/SSL 적용
   - 데이터베이스 암호화

4. **환경 변수 관리**:
   - Docker Secrets 사용
   - .env 파일 버전 관리 제외

## 모니터링 및 로깅

### 메트릭 수집
- Celery Flower: 작업 성능 메트릭
- Redis Insight: 메모리 사용량
- Kafka UI: 메시지 처리량

### 로그 관리
- Elasticsearch: 중앙 로그 수집
- Docker logs: 컨테이너 로그 확인

### 알림 설정 (향후 추가 권장)
- 작업 실패 시 알림
- 시스템 리소스 임계치 알림
- 성능 저하 감지

## 배포 환경

### 개발 환경
- docker-compose-dev.yml 사용
- 볼륨 마운트로 코드 hot-reload
- 디버그 모드 활성화

### 프로덕션 환경
- docker-compose.yml 사용
- 사전 빌드된 이미지 사용
- 리소스 제한 설정
- 헬스 체크 활성화

## 성능 특성

### 처리량
- 현재 테스트 결과: 분당 100개 작업 처리
- GPU 기반 LSTM: 평균 추론 시간 < 100ms
- Moving Average: 평균 추론 시간 < 10ms

### 가용성
- 작업 큐 기반 비동기 처리로 부하 분산
- PostgreSQL 영구 저장으로 데이터 손실 방지
- Kafka 이벤트 스트림으로 재처리 가능

### 확장성
- Celery 워커 추가로 선형 확장 가능
- Kafka 파티션 증가로 처리량 증가
- Redis 클러스터링 지원 (향후)
