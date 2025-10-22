# 분리된 아키텍처 배포 가이드

## 아키텍처 개요

시스템은 물리적/논리적으로 분리된 2개의 서버로 구성됩니다:

### 1. 운영서버 (Operation Server)
- FastAPI 기반 API 서버
- Celery Producer (작업 제출)
- WebSocket 실시간 통신
- PostgreSQL에 결과 저장
- 클라이언트와 직접 통신

### 2. 분석서버 (Analysis Server)  
- Celery Consumer (작업 처리)
- GPU 기반 AI 모델 추론
- 다중 워커 지원
- 독립적으로 스케일 아웃 가능

### 통신 흐름

```
Client → Nginx → Operation Server → Redis (Celery Broker) → Analysis Workers
                      ↓                                              ↓
                  PostgreSQL ←──────────────────────────────────────┘
                      ↓
                   Kafka (선택적 메시지 큐)
```

## 프로젝트 구조

```
satellite/
├── operation-server/          # 운영서버
│   ├── api/                   # FastAPI routes
│   ├── database/              # PostgreSQL, Redis clients
│   ├── messaging/             # Kafka, WebSocket
│   ├── celery_tasks/          # Celery producer
│   ├── utils/                 # Utilities
│   ├── main.py                # FastAPI 애플리케이션
│   ├── Dockerfile
│   └── requirements.txt
│
├── analysis-server/           # 분석서버
│   ├── models/                # AI 모델 구현
│   │   └── timeseries/        # 시계열 모델
│   ├── core/                  # 모델 로더, 레지스트리
│   ├── utils/                 # Utilities
│   ├── tasks.py               # Celery consumer tasks
│   ├── Dockerfile
│   └── requirements.txt
│
├── shared/                    # 공유 모듈
│   ├── schemas/               # Pydantic 스키마
│   └── config/                # 설정
│
├── frontend/                  # React UI
├── nginx/                     # Reverse proxy
├── docker-compose-new.yml     # 새로운 Docker Compose 설정
└── DEPLOYMENT.md              # 이 파일
```

## 배포 방법

### 1. Docker Compose로 배포

```bash
# Kafka 초기화
./init_kafka.sh

# 서비스 시작
docker compose -f docker-compose-new.yml up -d

# 로그 확인
docker compose -f docker-compose-new.yml logs -f operation-server
docker compose -f docker-compose-new.yml logs -f analysis-worker-1
```

### 2. 서비스 확인

```bash
# 운영서버 헬스 체크
curl http://localhost:8000/health

# Celery 모니터링 (Flower)
open http://localhost:5555

# Redis 모니터링
open http://localhost:8001
```

### 3. API 사용 예시

```bash
# 추론 작업 제출
curl -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5],
    "config": {"forecast_steps": 5}
  }'

# 작업 상태 조회
curl http://localhost:8000/api/v1/inference/status/{job_id}

# 결과 조회
curl http://localhost:8000/api/v1/inference/result/{job_id}
```

## 스케일 아웃

### Docker Compose로 워커 추가

docker-compose-new.yml에서 `analysis-worker` 서비스를 복제:

```yaml
analysis-worker-3:
  build:
    context: .
    dockerfile: analysis-server/Dockerfile
  container_name: analysis-worker-3
  command: celery -A analysis_server.tasks worker --loglevel=info --concurrency=2 --queue=inference
  # ... (환경변수 동일)
```

### 동적 스케일링

```bash
# 워커 수 증가
docker compose -f docker-compose-new.yml up -d --scale analysis-worker-1=5

# 워커 상태 확인
docker compose -f docker-compose-new.yml ps
```

## Kubernetes 배포

### 1. 이미지 빌드

```bash
# 운영서버 이미지
docker build -f operation-server/Dockerfile -t operation-server:latest .

# 분석서버 이미지  
docker build -f analysis-server/Dockerfile -t analysis-server:latest .
```

### 2. Kubernetes 매니페스트

operation-server deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: operation-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: operation-server
  template:
    metadata:
      labels:
        app: operation-server
    spec:
      containers:
      - name: operation-server
        image: operation-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: CELERY_BROKER_URL
          value: "redis://redis-service:6379/1"
        # ... 기타 환경변수
---
apiVersion: v1
kind: Service
metadata:
  name: operation-server
spec:
  selector:
    app: operation-server
  ports:
  - port: 8000
    targetPort: 8000
```

analysis-server deployment with HPA:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analysis-worker
  template:
    metadata:
      labels:
        app: analysis-worker
    spec:
      containers:
      - name: analysis-worker
        image: analysis-server:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "2Gi"
            cpu: "1"
          limits:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: CELERY_BROKER_URL
          value: "redis://redis-service:6379/1"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: analysis-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: analysis-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. 배포

```bash
kubectl apply -f k8s/operation-server.yaml
kubectl apply -f k8s/analysis-server.yaml
kubectl apply -f k8s/hpa.yaml
```

## 모니터링

### Celery 작업 모니터링 (Flower)

- URL: http://localhost:5555
- 실시간 작업 상태, 워커 상태, 통계 확인

### 통계 API

```bash
# 추론 통계 요약
curl http://localhost:8000/api/v1/results/stats/summary
```

### 로그 확인

```bash
# 운영서버 로그
docker logs -f operation-server

# 분석서버 로그
docker logs -f analysis-worker-1
docker logs -f analysis-worker-2
```

## 새 모델 추가

1. `analysis-server/models/` 에 새 모델 클래스 작성
2. `BaseModel` 상속 및 필수 메서드 구현
3. `@model_registry.register("model_name")` 데코레이터 추가
4. `analysis-server/tasks.py`에서 import

예시:

```python
from analysis_server.models.base_model import BaseModel
from analysis_server.core.model_registry import model_registry

@model_registry.register("my_new_model")
class MyNewModel(BaseModel):
    def load(self):
        # 모델 로드
        pass
    
    def preprocess(self, data):
        # 전처리
        return data
    
    def predict(self, data):
        # 추론
        return predictions
    
    def postprocess(self, predictions):
        # 후처리
        return {"predictions": predictions}
```

## 트러블슈팅

### 1. Celery 워커가 작업을 받지 못함

- Redis 연결 확인: `redis-cli -h localhost ping`
- Celery 브로커 URL 확인
- 큐 이름 확인 (`inference` 큐 사용 중)

### 2. GPU 인식 안됨

```bash
# Docker GPU 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# NVIDIA Docker runtime 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. PostgreSQL 연결 오류

```bash
# PostgreSQL 상태 확인
docker exec -it postgres psql -U admin -d orders_db

# 테이블 초기화
docker exec -it operation-server python -c "from operation_server.database.postgres_client import postgres_client; postgres_client.init_tables()"
```

## 성능 최적화

### 1. Celery 동시성 조정

`--concurrency` 옵션으로 워커당 동시 작업 수 조절:

```bash
celery -A analysis_server.tasks worker --concurrency=4
```

### 2. 모델 캐싱

`settings.py`에서 캐시 설정 조정:

```python
MODEL_MAX_LOADED = 5  # 메모리에 로드할 최대 모델 수
```

### 3. GPU 메모리 관리

```python
GPU_MEMORY_FRACTION = 0.8  # GPU 메모리 사용률
```

## 보안 고려사항

1. 프로덕션 환경에서는 환경변수를 secrets로 관리
2. PostgreSQL 비밀번호 변경
3. Redis 인증 활성화
4. Nginx SSL/TLS 설정
5. API 인증/인가 추가

## 백업 및 복구

### 데이터베이스 백업

```bash
docker exec postgres pg_dump -U admin orders_db > backup.sql
```

### 데이터베이스 복구

```bash
docker exec -i postgres psql -U admin orders_db < backup.sql
```
