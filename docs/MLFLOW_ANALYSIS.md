# MLflow 적용 분석 및 제안

## 목차
1. [현재 시스템 분석](#현재-시스템-분석)
2. [MLflow 개요](#mlflow-개요)
3. [적용 시 장점](#적용-시-장점)
4. [적용 시 단점](#적용-시-단점)
5. [적용 방안](#적용-방안)
6. [아키텍처 변경](#아키텍처-변경)
7. [구현 예시](#구현-예시)
8. [비용 분석](#비용-분석)
9. [최종 권장사항](#최종-권장사항)

---

## 현재 시스템 분석

### 현재 모델 관리 방식

```python
# analysis-server/core/model_factory.py
class ModelFactory:
    def __init__(self):
        self._models = {}  # 메모리 내 모델 저장
        self._model_cache = {}  # 로컬 캐시

    @lru_cache(maxsize=10)
    def load_model(self, model_name: str):
        # 파일 시스템에서 직접 로드
        return self._models[model_name]
```

### 현재 메트릭 저장 방식

```python
# PostgreSQL에 JSON으로 저장
{
    "job_id": "abc123",
    "metrics": {
        "inference_time": 0.125,
        "mse": 0.032,
        "mae": 0.15
    }
}
```

### 현재 시스템의 제약사항

1. **모델 버전 관리 부재**
   - 모델 파일명으로만 구분
   - 변경 이력 추적 불가
   - 롤백 어려움

2. **실험 관리 어려움**
   - 하이퍼파라미터 튜닝 이력 없음
   - A/B 테스트 인프라 부족
   - 모델 성능 비교 수동 작업

3. **메트릭 시각화 제한**
   - PostgreSQL 쿼리로만 조회
   - 대시보드는 실시간 데이터만 표시
   - 장기 트렌드 분석 어려움

4. **모델 배포 프로세스**
   - 수동으로 파일 교체
   - 배포 이력 추적 불가
   - 프로덕션/스테이징 구분 없음

---

## MLflow 개요

### MLflow 주요 컴포넌트

#### 1. MLflow Tracking
실험 추적 및 메트릭 로깅
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("mse", 0.032)
    mlflow.log_artifact("model.pkl")
```

#### 2. MLflow Models
모델 패키징 및 배포
```python
mlflow.pytorch.log_model(model, "lstm_model")
loaded_model = mlflow.pytorch.load_model("models:/lstm_timeseries/production")
```

#### 3. MLflow Model Registry
모델 버전 관리 및 스테이징
```
None → Staging → Production → Archived
```

#### 4. MLflow Projects
재현 가능한 실험 환경

---

## 적용 시 장점

### 1. 모델 버전 관리 (★★★★★)

**현재 문제:**
```bash
# 어떤 모델이 현재 프로덕션인지 불명확
analysis-server/models/
├── lstm_v1.pth
├── lstm_v2.pth
├── lstm_final.pth
├── lstm_final_final.pth  # ???
```

**MLflow 적용 후:**
```python
# 명확한 버전 관리
client = mlflow.tracking.MlflowClient()

# 스테이지별 모델 조회
production_model = client.get_latest_versions(
    "lstm_timeseries",
    stages=["Production"]
)[0]

# 버전: 1, 2, 3, ...
# 스테이지: None, Staging, Production, Archived
```

**효과:**
- 모델 이력 완전 추적
- 프로덕션 모델 명확 식별
- 버전별 성능 비교 가능
- 롤백 즉시 가능

---

### 2. 실험 추적 및 비교 (★★★★★)

**현재 문제:**
하이퍼파라미터 튜닝 시 결과를 수동으로 기록

**MLflow 적용 후:**
```python
# 여러 실험 자동 추적
for lr in [0.001, 0.01, 0.1]:
    with mlflow.start_run():
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_size", 64)

        model = train_model(lr=lr)
        mse = evaluate(model)

        mlflow.log_metric("mse", mse)
        mlflow.pytorch.log_model(model, "model")
```

**MLflow UI에서 자동 비교:**
```
Run ID  | learning_rate | hidden_size | mse    | duration
--------|---------------|-------------|--------|----------
abc123  | 0.001         | 64          | 0.045  | 120s
def456  | 0.01          | 64          | 0.032  | 118s  ← Best
ghi789  | 0.1           | 64          | 0.089  | 115s
```

**효과:**
- 자동 실험 추적
- 시각적 비교 (그래프, 테이블)
- 최적 하이퍼파라미터 쉽게 식별
- 실험 재현 가능

---

### 3. 메트릭 시각화 및 분석 (★★★★☆)

**현재 시스템:**
```sql
-- PostgreSQL 쿼리로만 조회
SELECT AVG(metrics->>'inference_time')
FROM inference_results
WHERE model_name = 'lstm_timeseries';
```

**MLflow 적용 후:**
- **MLflow UI**: 실시간 그래프 및 비교
- **메트릭 히스토리**: 시간에 따른 성능 변화
- **파라미터 상관관계**: 어떤 파라미터가 성능에 영향을 주는지

```python
# 메트릭 히스토리
mlflow.log_metric("mse", 0.045, step=1)
mlflow.log_metric("mse", 0.032, step=2)
mlflow.log_metric("mse", 0.028, step=3)
```

**효과:**
- 즉시 시각화
- 장기 트렌드 분석
- 모델 성능 저하 조기 감지

---

### 4. 모델 배포 자동화 (★★★★☆)

**현재 프로세스:**
```bash
# 수동 배포
1. 모델 학습
2. 파일 복사: cp model.pth analysis-server/models/
3. 코드 수정: MODEL_PATH = "models/new_model.pth"
4. 서버 재시작: docker compose restart
```

**MLflow 적용 후:**
```python
# 1. 모델 등록
mlflow.pytorch.log_model(model, "lstm_model")

# 2. 스테이징으로 승격
client.transition_model_version_stage(
    name="lstm_timeseries",
    version=5,
    stage="Staging"
)

# 3. 스테이징 테스트 통과 후 프로덕션 승격
client.transition_model_version_stage(
    name="lstm_timeseries",
    version=5,
    stage="Production"
)

# 4. Analysis Worker가 자동으로 최신 Production 모델 로드
model = mlflow.pytorch.load_model(
    "models:/lstm_timeseries/Production"
)
```

**효과:**
- 배포 자동화
- 스테이징 환경 테스트
- 프로덕션 배포 승인 프로세스
- 즉시 롤백 가능

---

### 5. A/B 테스트 지원 (★★★★☆)

**MLflow 적용 후:**
```python
# 두 모델 버전 동시 운영
model_a = mlflow.pytorch.load_model("models:/lstm_timeseries/1")
model_b = mlflow.pytorch.load_model("models:/lstm_timeseries/2")

# 트래픽 분할
if job_id.hash() % 2 == 0:
    result = model_a.predict(data)
    mlflow.log_metric("model_a_latency", latency)
else:
    result = model_b.predict(data)
    mlflow.log_metric("model_b_latency", latency)
```

**효과:**
- 안전한 모델 업데이트
- 실시간 성능 비교
- 데이터 기반 의사결정

---

### 6. 협업 개선 (★★★☆☆)

**MLflow UI 공유:**
- 데이터 과학자: 실험 결과 공유
- 엔지니어: 프로덕션 모델 성능 모니터링
- PM: 모델 개선 추이 확인

---

## 적용 시 단점

### 1. 인프라 복잡도 증가 (★★★★☆)

**추가 컴포넌트:**
```yaml
services:
  mlflow-server:
    image: mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://...
      - DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/
    depends_on:
      - postgres
      - minio  # S3 호환 스토리지
```

**복잡도:**
- MLflow Server 운영
- Artifact Store (S3/MinIO) 필요
- Backend Store (PostgreSQL) 설정
- 네트워크 설정 추가

---

### 2. 러닝 커브 (★★★☆☆)

**학습 필요 사항:**
- MLflow API 사용법
- 모델 등록 및 스테이징
- Artifact 관리
- MLflow UI 사용

**예상 학습 시간:**
- 기본 사용: 1-2일
- 고급 기능: 1주일

---

### 3. 성능 오버헤드 (★★☆☆☆)

**추가 네트워크 호출:**
```python
# 추론 시마다 MLflow와 통신
with mlflow.start_run():
    mlflow.log_metric("inference_time", 0.125)  # HTTP 요청
    mlflow.log_param("model_version", "1")      # HTTP 요청
```

**오버헤드:**
- 메트릭 로깅: ~10-50ms
- 모델 로딩: 첫 로딩 시 ~100-500ms 증가

**완화 방법:**
- 비동기 로깅
- 배치 로깅
- 로컬 캐싱

---

### 4. 비용 증가 (★★☆☆☆)

**추가 리소스:**
- MLflow Server: CPU 1-2 코어, 메모리 2-4GB
- Artifact Storage: 모델 크기에 따라 가변
  - LSTM 모델: ~50-100MB per version
  - 10개 버전: ~1GB

**비용 예상:**
- 개발 환경: 무시 가능 (Docker Compose)
- 프로덕션: 월 $20-50 (클라우드 기준)

---

### 5. 현재 시스템과의 통합 (★★★☆☆)

**변경 필요 부분:**
1. Model Factory 전면 수정
2. Analysis Worker 코드 수정
3. 배포 프로세스 변경
4. 모니터링 대시보드 연동

**마이그레이션 리스크:**
- 기존 모델 이전 필요
- 메트릭 데이터 마이그레이션
- 다운타임 발생 가능

---

## 적용 방안

### 옵션 1: 전체 도입 (Aggressive)

**적용 범위:**
- MLflow Tracking
- MLflow Models
- MLflow Model Registry

**장점:**
- 최대 효과
- 완전한 MLOps 인프라

**단점:**
- 높은 초기 비용
- 긴 마이그레이션 기간 (2-4주)
- 높은 리스크

**권장 시나리오:**
- 모델 업데이트 빈번
- 여러 데이터 과학자 협업
- 장기 프로젝트

---

### 옵션 2: 점진적 도입 (Recommended)

**Phase 1: MLflow Tracking (1주)**
```python
# 실험 추적만 먼저 도입
with mlflow.start_run():
    mlflow.log_param("model_name", "lstm_timeseries")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("inference_time", inference_time)

# 기존 모델 로딩 방식 유지
model = model_factory.get_model("lstm_timeseries")
```

**Phase 2: MLflow Models (2주)**
```python
# 모델 로깅 추가
mlflow.pytorch.log_model(model, "lstm_model")

# 하지만 로딩은 여전히 기존 방식
model = model_factory.get_model("lstm_timeseries")
```

**Phase 3: MLflow Model Registry (2주)**
```python
# 완전 전환
model = mlflow.pytorch.load_model(
    "models:/lstm_timeseries/Production"
)
```

**장점:**
- 단계별 검증
- 낮은 리스크
- 빠른 ROI (Phase 1부터 효과)

**단점:**
- 긴 전체 기간 (5주)
- 중간 단계에서 중복 코드

---

### 옵션 3: 선택적 도입 (Conservative)

**적용:**
MLflow Tracking만 도입

**장점:**
- 최소 변경
- 빠른 도입 (1주)
- 낮은 리스크

**단점:**
- 제한적 효과
- 모델 배포는 여전히 수동

**권장 시나리오:**
- 모델 업데이트 드묾
- 소규모 팀
- 실험 중심 (프로덕션 아님)

---

## 아키�ecture 변경

### 현재 아키텍처

```
┌─────────────┐
│  Operation  │
│   Server    │
└──────┬──────┘
       │
       ▼
┌──────────────┐      ┌─────────────┐
│   Celery     │─────▶│   Analysis  │
│   Queue      │      │   Worker    │
└──────────────┘      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │Model Factory│
                      │(File System)│
                      └─────────────┘
```

### MLflow 적용 후

```
┌─────────────┐
│  Operation  │
│   Server    │
└──────┬──────┘
       │
       ▼
┌──────────────┐      ┌─────────────┐      ┌─────────────┐
│   Celery     │─────▶│   Analysis  │─────▶│   MLflow    │
│   Queue      │      │   Worker    │      │   Server    │
└──────────────┘      └──────┬──────┘      └──────┬──────┘
                             │                    │
                             ▼                    ▼
                      ┌─────────────┐      ┌─────────────┐
                      │Model Factory│      │  Artifact   │
                      │(MLflow API) │      │   Store     │
                      └─────────────┘      │(S3/MinIO)   │
                                           └─────────────┘
```

### Docker Compose 변경

```yaml
services:
  # 기존 서비스들...

  mlflow:
    image: mlflow/mlflow:latest
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://admin:admin123@postgres:5432/mlflow
      - DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    command: >
      mlflow server
      --backend-store-uri postgresql://admin:admin123@postgres:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts
      --host 0.0.0.0
      --port 5000
    depends_on:
      - postgres
      - minio
    networks:
      - webnet

  minio:
    image: minio/minio:latest
    container_name: minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    networks:
      - webnet

volumes:
  minio_data:
```

---

## 구현 예시

### 1. Analysis Worker 수정

#### Before (현재)
```python
# analysis-server/tasks.py
from core.model_factory import model_factory

@celery_app.task
def inference_task(job_id, model_name, data, config, metadata):
    # 모델 로드
    model = model_factory.get_model(model_name)

    # 추론
    predictions = model.predict(data)

    # 메트릭 계산
    metrics = {
        "inference_time": time.time() - start_time,
        "mse": calculate_mse(predictions, data)
    }

    # PostgreSQL 저장
    postgres_client.save_result(job_id, predictions, metrics)
```

#### After (MLflow 적용)
```python
# analysis-server/tasks.py
import mlflow
import mlflow.pytorch
from core.mlflow_model_loader import MLflowModelLoader

model_loader = MLflowModelLoader()

@celery_app.task
def inference_task(job_id, model_name, data, config, metadata):
    # MLflow Run 시작
    with mlflow.start_run(run_name=f"inference_{job_id}"):
        # 파라미터 로깅
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_points", len(data))
        mlflow.log_param("job_id", job_id)

        # 프로덕션 모델 로드
        model = model_loader.load_production_model(model_name)
        model_version = model_loader.get_model_version(model_name)

        mlflow.log_param("model_version", model_version)

        # 추론
        start_time = time.time()
        predictions = model.predict(data)
        inference_time = time.time() - start_time

        # 메트릭 계산 및 로깅
        mse = calculate_mse(predictions, data)
        mae = calculate_mae(predictions, data)

        mlflow.log_metric("inference_time", inference_time)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("data_size", len(data))

        # 예측 결과 artifact로 저장 (선택적)
        if config.get("save_predictions"):
            mlflow.log_dict({
                "predictions": predictions.tolist(),
                "input_data": data
            }, "predictions.json")

        # 기존 저장 로직 유지
        metrics = {
            "inference_time": inference_time,
            "mse": mse,
            "mae": mae
        }
        postgres_client.save_result(job_id, predictions, metrics)
```

### 2. Model Loader 구현

```python
# analysis-server/core/mlflow_model_loader.py
import mlflow
import mlflow.pytorch
from functools import lru_cache
from typing import Any

class MLflowModelLoader:
    def __init__(self, mlflow_tracking_uri: str = "http://mlflow:5000"):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        self._model_cache = {}

    @lru_cache(maxsize=10)
    def load_production_model(self, model_name: str) -> Any:
        """프로덕션 스테이지의 최신 모델 로드"""
        try:
            # 캐시 확인
            cache_key = f"{model_name}_production"
            if cache_key in self._model_cache:
                cached = self._model_cache[cache_key]
                # 버전 확인
                latest_version = self._get_latest_production_version(model_name)
                if cached["version"] == latest_version:
                    return cached["model"]

            # MLflow에서 로드
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pytorch.load_model(model_uri)

            # 캐시 업데이트
            version = self._get_latest_production_version(model_name)
            self._model_cache[cache_key] = {
                "model": model,
                "version": version
            }

            return model

        except Exception as e:
            # Fallback: 기존 방식으로 로드
            logger.warning(f"MLflow load failed, using fallback: {e}")
            from core.model_factory import model_factory
            return model_factory.get_model(model_name)

    def load_model_version(self, model_name: str, version: str) -> Any:
        """특정 버전의 모델 로드"""
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pytorch.load_model(model_uri)

    def _get_latest_production_version(self, model_name: str) -> str:
        """프로덕션 스테이지의 최신 버전 반환"""
        versions = self.client.get_latest_versions(
            model_name,
            stages=["Production"]
        )
        if not versions:
            raise ValueError(f"No production model for {model_name}")
        return versions[0].version

    def get_model_version(self, model_name: str) -> str:
        """현재 로드된 모델 버전 반환"""
        return self._get_latest_production_version(model_name)

    def list_models(self) -> list:
        """등록된 모델 목록 반환"""
        return [m.name for m in self.client.list_registered_models()]
```

### 3. 모델 학습 및 등록

```python
# scripts/train_and_register.py
import mlflow
import mlflow.pytorch
import torch

def train_lstm_model(params):
    """LSTM 모델 학습"""
    # MLflow Experiment 설정
    mlflow.set_experiment("lstm_timeseries_training")

    with mlflow.start_run(run_name=f"lstm_lr{params['learning_rate']}"):
        # 파라미터 로깅
        mlflow.log_params(params)

        # 모델 초기화
        model = LSTMModel(
            input_size=params['input_size'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers']
        )

        # 학습
        for epoch in range(params['epochs']):
            train_loss = train_epoch(model)
            val_loss = validate_epoch(model)

            # 에포크별 메트릭 로깅
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # 최종 평가
        test_metrics = evaluate_model(model)
        mlflow.log_metrics({
            "test_mse": test_metrics['mse'],
            "test_mae": test_metrics['mae'],
            "test_r2": test_metrics['r2']
        })

        # 모델 저장
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="lstm_timeseries",
            pip_requirements=["torch==2.0.0", "numpy==1.24.0"]
        )

        # 학습 그래프 저장
        fig = plot_training_curves(train_losses, val_losses)
        mlflow.log_figure(fig, "training_curves.png")

        return model

# 실행
params = {
    'learning_rate': 0.001,
    'hidden_size': 64,
    'num_layers': 2,
    'input_size': 10,
    'epochs': 100
}

model = train_lstm_model(params)
```

### 4. 모델 승격 (Staging → Production)

```python
# scripts/promote_model.py
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_to_staging(model_name: str, version: str):
    """모델을 Staging으로 승격"""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f"Model {model_name} v{version} promoted to Staging")

def promote_to_production(model_name: str, version: str):
    """모델을 Production으로 승격"""
    # 기존 Production 모델을 Archived로 이동
    current_prod = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )
    for model_ver in current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=model_ver.version,
            stage="Archived"
        )

    # 새 모델을 Production으로 승격
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Model {model_name} v{version} promoted to Production")

# 사용 예시
promote_to_staging("lstm_timeseries", "5")
# 스테이징 테스트 수행...
promote_to_production("lstm_timeseries", "5")
```

### 5. 대시보드 연동

```python
# operation-server/api/routes/mlflow_dashboard.py
from fastapi import APIRouter
from mlflow.tracking import MlflowClient

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

@router.get("/models")
async def get_registered_models():
    """등록된 모델 목록"""
    client = MlflowClient()
    models = client.list_registered_models()

    return {
        "models": [
            {
                "name": m.name,
                "latest_version": m.latest_versions[0].version if m.latest_versions else None,
                "description": m.description
            }
            for m in models
        ]
    }

@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """모델의 모든 버전"""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    return {
        "versions": [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "created_at": v.creation_timestamp
            }
            for v in versions
        ]
    }

@router.get("/models/{model_name}/metrics")
async def get_model_metrics(model_name: str, stage: str = "Production"):
    """프로덕션 모델의 메트릭"""
    client = MlflowClient()

    # 프로덕션 모델 버전 가져오기
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return {"error": f"No {stage} model found"}

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    return {
        "version": versions[0].version,
        "stage": stage,
        "metrics": run.data.metrics,
        "params": run.data.params
    }
```

---

## 비용 분석

### 리소스 요구사항

#### 개발 환경 (Docker Compose)
```
MLflow Server:
- CPU: 1 코어
- 메모리: 2GB
- 스토리지: 10GB

MinIO (Artifact Store):
- CPU: 0.5 코어
- 메모리: 1GB
- 스토리지: 50GB (모델 10개 버전 기준)

추가 PostgreSQL 데이터:
- ~500MB (실험 1000개 기준)

총 증가: CPU +1.5, 메모리 +3GB, 스토리지 +60GB
```

#### 프로덕션 환경 (AWS 기준)
```
MLflow Server (EC2 t3.medium):
- 비용: $35/월

RDS PostgreSQL (추가 스토리지):
- 비용: $5/월

S3 (Artifact Storage):
- 50GB: $1.15/월
- Request: $1/월

총 비용: ~$42/월
```

### 시간 투자

```
초기 설정: 3-5일
- Docker 설정: 1일
- MLflow 서버 구축: 1일
- 통합 테스트: 1-2일
- 문서화: 1일

점진적 도입 (권장):
- Phase 1 (Tracking): 1주
- Phase 2 (Models): 2주
- Phase 3 (Registry): 2주
총: 5주

학습 시간:
- 팀원 1인당: 2-3일
```

### ROI 분석

#### 비용
- 월 운영 비용: $42
- 초기 개발 비용: 5주 * 개발자 1명

#### 절감 효과
- 모델 배포 시간: 30분 → 5분 (월 4회 = 100분 절감)
- 실험 추적 시간: 10분/실험 → 0분 (월 20회 = 200분 절감)
- 롤백 시간: 30분 → 2분 (문제 발생 시)
- 모델 성능 비교: 1시간 → 5분 (월 2회 = 110분 절감)

월 절감 시간: ~7시간

#### Break-even
약 2-3개월 후 투자 회수

---

## 최종 권장사항

### 추천: 옵션 2 (점진적 도입)

#### 이유

**1. 현재 시스템 분석**
- 모델 2개 (LSTM, Moving Average)
- 업데이트 빈도: 중간 (월 1-2회 예상)
- 팀 규모: 소규모 (1-3명)
- 실험 빈도: 높음 (하이퍼파라미터 튜닝)

**2. MLflow 도입 가치**
- 모델 버전 관리: ★★★★★ (매우 중요)
- 실험 추적: ★★★★★ (매우 중요)
- 메트릭 시각화: ★★★★☆ (중요)
- 배포 자동화: ★★★☆☆ (현재 수동이지만 빈도 낮음)

**3. 리스크 평가**
- 복잡도 증가: 관리 가능
- 비용: 낮음 ($42/월)
- 마이그레이션: 점진적 도입으로 리스크 완화

### 구체적 실행 계획

#### Week 1-2: Phase 1 - MLflow Tracking
```python
# 목표: 실험 추적만 도입
# 변경: Analysis Worker에 mlflow.log_* 추가
# 리스크: 매우 낮음 (기존 코드 그대로 유지)
```

**작업:**
1. MLflow Server + MinIO 컨테이너 추가
2. `inference_task`에 MLflow 로깅 추가
3. MLflow UI에서 메트릭 확인

**성공 기준:**
- 추론 메트릭이 MLflow UI에 표시됨
- 기존 기능 정상 작동

---

#### Week 3-4: Phase 2 - MLflow Models
```python
# 목표: 모델을 MLflow에 저장
# 변경: 학습 스크립트 수정
# 리스크: 낮음 (로딩은 아직 기존 방식)
```

**작업:**
1. 학습 스크립트에 `mlflow.pytorch.log_model` 추가
2. 모델 등록 자동화
3. MLflow UI에서 모델 확인

**성공 기준:**
- 새 모델이 MLflow에 등록됨
- 버전 히스토리 확인 가능

---

#### Week 5-6: Phase 3 - MLflow Model Registry
```python
# 목표: 프로덕션 배포 MLflow로 전환
# 변경: Model Factory를 MLflowModelLoader로 교체
# 리스크: 중간 (철저한 테스트 필요)
```

**작업:**
1. `MLflowModelLoader` 구현
2. Staging/Production 승격 프로세스 구축
3. Analysis Worker를 MLflow 로딩으로 전환
4. Fallback 메커니즘 구현

**성공 기준:**
- 프로덕션 모델이 MLflow에서 로드됨
- 롤백 가능
- 성능 저하 없음

---

### 대안: 시작하지 않아도 되는 경우

다음 조건에 모두 해당하면 MLflow 도입을 보류:

1. 모델이 거의 변경되지 않음 (연 1-2회 이하)
2. 실험이 거의 없음 (하이퍼파라미터 고정)
3. 1명만 모델 관리
4. 현재 시스템에 만족

이 경우 현재의 단순한 시스템 유지를 권장합니다.

---

### 결론

**권장: 점진적 도입 (5-6주)**

**예상 효과:**
- 모델 관리 효율성 300% 향상
- 실험 추적 자동화
- 프로덕션 배포 신뢰성 향상

**투자 대비 효과:**
- 초기 투자: 5주 개발 시간
- 월 운영 비용: $42
- ROI: 2-3개월 후 회수
- 장기적 가치: 시스템 확장 시 필수 인프라

MLflow는 현재 시스템의 성숙도를 한 단계 높이는 데 매우 적합한 도구입니다.
