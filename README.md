# Satellite - 분산 AI 추론 시스템

마이크로서비스 아키텍처 기반 GPU 가속 AI 추론 시스템

## 시스템 아키텍처

### 서버 구성

```
┌─────────────────┐
│   Frontend UI   │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Nginx  │
    └────┬────┘
         │
┌────────▼────────────┐         ┌──────────────────────┐
│ Operation Server    │────────▶│ Redis (Broker)       │
│ - FastAPI           │         └──────────┬───────────┘
│ - Celery Producer   │                    │
│ - WebSocket         │         ┌──────────▼───────────┐
│ - DB 관리           │         │ Analysis Workers     │
└─────────────────────┘         │ - Celery Consumer    │
                                │ - GPU Inference      │
                                │ - Batch Processing   │
                                └──────────────────────┘
```

### 운영서버 (Operation Server)
- FastAPI 기반 REST API
- Celery Producer (작업 제출)
- WebSocket 실시간 통신
- PostgreSQL/Redis 데이터 관리
- 클라이언트 직접 통신

### 분석서버 (Analysis Server)
- Celery Consumer (작업 처리)
- GPU 기반 AI 모델 추론
- 배치 처리 (8개 또는 2초)
- 팩토리 패턴 모델 관리
- 독립적 스케일 아웃

## 주요 기능

### 1. 배치 처리 (Batch Processing)
GPU 활용성 향상을 위한 자동 배치 큐잉
- 배치 크기: 8개 요청 도달 시 즉시 처리
- 시간 제한: 2초 경과 시 자동 처리
- 모델별 독립적 큐 관리

### 2. 팩토리 패턴 모델 관리
- 데코레이터 기반 모델 등록
- 동적 모델 로딩
- LRU 캐싱

### 3. 분산 처리
- Celery 작업 큐
- 다중 워커 지원
- 수평 확장 가능

## 빠른 시작

### 1. 환경 설정

```bash
# Kafka 초기화
./init_kafka.sh

# 서비스 시작
docker compose up -d

# 상태 확인
docker compose ps
```

### 2. API 사용

```bash
# 추론 작업 제출
curl -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5],
    "config": {"forecast_steps": 5}
  }'

# 결과 조회
curl http://localhost:8000/api/v1/inference/result/{job_id}
```

### 3. 모니터링

- Flower (Celery): http://localhost:5555
- Redis Insight: http://localhost:8001
- Operation Server: http://localhost:8000/health

## 워커 스케일 아웃

### Docker Compose

```bash
# 워커 추가
docker compose up -d --scale analysis-worker-1=5

# 상태 확인
docker compose ps
```

### Kubernetes

```bash
# HPA 자동 스케일링 (CPU 70% 기준)
kubectl apply -f k8s/analysis-server.yaml

# 수동 스케일링
kubectl scale deployment analysis-worker --replicas=10
```

## 새 모델 추가

### 1. 모델 클래스 작성

```python
# analysis-server/models/my_models/new_model.py
from analysis_server.core.model_factory import model_factory

@model_factory.register_decorator("new_model")
class NewModel(BaseModel):
    # 구현...
```

### 2. Import

```python
# analysis-server/tasks.py에 추가
from analysis_server.models.my_models import NewModel
```

끝! 자동으로 사용 가능합니다.

자세한 내용은 `analysis-server/README.md` 참조

## 프로젝트 구조

```
satellite/
├── operation-server/       # 운영서버 (API)
│   ├── api/               # FastAPI routes
│   ├── database/          # DB clients
│   ├── messaging/         # Kafka, WebSocket
│   └── celery_tasks/      # Producer
├── analysis-server/       # 분석서버 (AI)
│   ├── core/             # Factory, Batch, Loader
│   ├── models/           # AI 모델들
│   └── tasks.py          # Consumer
├── shared/               # 공통 모듈
│   ├── schemas/          # Pydantic 스키마
│   └── config/           # 설정
├── frontend/             # React UI
├── nginx/                # Reverse proxy
└── docker-compose.yml
```

## 서비스 포트

- Nginx: 80
- Operation Server: 8000
- Flower (Celery 모니터링): 5555
- Redis: 6379, 8001 (Insight)
- PostgreSQL: 5432
- Kafka: 9092

## 배포

자세한 배포 가이드는 `DEPLOYMENT.md` 참조

### Docker Compose (개발/소규모)

```bash
docker compose up -d
```

### Kubernetes (프로덕션)

```bash
kubectl apply -f k8s/
```

## 문서

### 주요 문서
- [빠른 시작 가이드](docs/QUICKSTART.md) - 5분 만에 시작하기
- [시스템 아키텍처](docs/ARCHITECTURE.md) - 전체 시스템 구조
- [API 레퍼런스](docs/API_REFERENCE.md) - REST API 및 WebSocket
- [모니터링 UI](docs/MONITORING_UI.md) - 웹 UI 접속 가이드

### 상세 문서
- [UML 다이어그램](docs/UML_DIAGRAMS.md) - 시퀀스, 클래스, 컴포넌트 다이어그램
- [배포 가이드](docs/DEPLOYMENT.md) - 배포 및 운영
- [시뮬레이터 가이드](docs/SIMULATOR_README.md) - 데이터 시뮬레이터 사용법
- [테스트 결과](docs/TEST_RESULTS.md) - 시스템 테스트 보고서

### 개발자 문서
- [분석서버 상세](analysis-server/README.md)
- [Claude 가이드](CLAUDE.md)
- [전체 문서 목록](docs/README.md)
