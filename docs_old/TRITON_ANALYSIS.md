# Nvidia Triton Inference Server 적용 분석

## 목차
1. [현재 시스템 아키텍처 분석](#현재-시스템-아키텍처-분석)
2. [Triton Server 개요](#triton-server-개요)
3. [적용 시 장점](#적용-시-장점)
4. [적용 시 단점 및 고려사항](#적용-시-단점-및-고려사항)
5. [아키텍처 변경 사항](#아키텍처-변경-사항)
6. [구현 복잡도 분석](#구현-복잡도-분석)
7. [비용-편익 분석](#비용-편익-분석)
8. [최종 권장사항](#최종-권장사항)

---

## 현재 시스템 아키텍처 분석

### 현재 추론 파이프라인

```
[클라이언트]
    ↓ HTTP POST
[Operation Server - FastAPI]
    ↓ Celery Task (Redis Queue)
[Analysis Worker - Celery Consumer]
    ↓
[Model Loader (Factory Pattern)]
    ↓
[PyTorch Model (GPU)]
    ↓
[결과 저장 - Redis/PostgreSQL/Kafka]
```

### 현재 구조의 특징

1. **모델 관리**: Factory Pattern으로 모델 등록 및 로딩
2. **배치 처리**: `batch_manager`를 통한 요청 배칭
3. **비동기 처리**: Celery를 통한 분산 워커 처리
4. **GPU 활용**: PyTorch 모델이 직접 CUDA 사용
5. **모델 종류**: LSTM, Moving Average 등 시계열 모델

### 현재 코드 구조

```python
# analysis-server/tasks.py
@celery_app.task(bind=True, base=InferenceTask)
def run_inference(self, job_id, model_name, data, config, metadata):
    # 1. 배치 큐에 추가
    batch_manager.add_to_batch(...)

    # 2. 모델 로드
    model = model_loader.load_model(model_name, config)

    # 3. 추론 실행
    results = model.infer(data)

    # 4. 결과 저장
    redis/postgres/kafka
```

---

## Triton Server 개요

### Triton Inference Server란?

Nvidia가 개발한 **오픈소스 추론 서빙 플랫폼**으로, 다양한 프레임워크의 모델을 고성능으로 서빙할 수 있습니다.

### 주요 기능

1. **Multi-Framework 지원**
   - PyTorch, TensorFlow, ONNX, TensorRT, Python Backend
   - OpenVINO, Custom Backend

2. **고급 추론 최적화**
   - Dynamic Batching (자동 배칭)
   - Concurrent Model Execution (모델 병렬 실행)
   - Model Pipelining (모델 파이프라인)
   - Model Ensembling (모델 앙상블)

3. **성능 최적화**
   - GPU 메모리 효율적 관리
   - Multi-GPU 지원
   - CPU/GPU 자동 선택
   - Request Scheduling 최적화

4. **모니터링 및 관리**
   - Prometheus Metrics
   - Model Repository (동적 로딩/언로딩)
   - Health Check API
   - gRPC/HTTP/REST API

---

## 적용 시 장점

### 1. 추론 성능 향상

**Dynamic Batching**
- 현재: 수동 배치 관리 (`batch_manager`)
- Triton: 자동 배치 최적화 (지연시간 vs 처리량 자동 조절)
- **예상 성능 향상**: 2-5배 처리량 증가

```yaml
# Triton 설정 예시
dynamic_batching {
  preferred_batch_size: [8, 16]
  max_queue_delay_microseconds: 100000
}
```

**GPU 활용도 개선**
- 현재: 단일 요청당 GPU 사용 → 비효율적
- Triton: 여러 요청 병합하여 GPU 활용도 극대화
- **예상 GPU 활용도**: 30-40% → 70-90%

### 2. 멀티 프레임워크 지원

**현재 문제점**
- PyTorch 모델만 지원
- 새로운 프레임워크 추가 시 큰 코드 변경 필요

**Triton 도입 후**
```
├── model_repository/
│   ├── lstm_pytorch/          # PyTorch 모델
│   │   ├── config.pbtxt
│   │   └── 1/model.pt
│   ├── moving_avg_onnx/       # ONNX 모델
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx
│   ├── ensemble_model/        # 앙상블 모델
│   │   └── config.pbtxt
│   └── custom_python/         # Python Backend
│       ├── config.pbtxt
│       └── 1/model.py
```

### 3. 모델 버전 관리 개선

**현재 구조**
- Factory Pattern으로 모델 등록
- 버전 관리 미흡

**Triton 구조**
```
model_repository/
└── lstm_timeseries/
    ├── config.pbtxt
    ├── 1/model.pt          # v1
    ├── 2/model.pt          # v2
    └── 3/model.pt          # v3 (latest)
```

- 동시에 여러 버전 서빙 가능
- A/B 테스팅 용이
- 롤백 간편

### 4. 운영 효율성 향상

**모니터링**
```python
# Prometheus metrics 자동 제공
triton_inference_request_success
triton_inference_request_duration_us
triton_inference_queue_duration_us
triton_model_inference_count
triton_gpu_utilization
```

**Health Check**
```bash
# 자동 제공되는 헬스체크
curl http://triton:8000/v2/health/live
curl http://triton:8000/v2/health/ready
curl http://triton:8000/v2/models/lstm_timeseries/ready
```

### 5. 확장성

**현재**
- Celery Worker 수평 확장만 가능
- 모델별 확장 불가능

**Triton**
- 모델별 인스턴스 수 조절 가능
- Multi-GPU 자동 분산
- Auto-scaling 가능

```yaml
# config.pbtxt
instance_group [
  {
    count: 2              # 모델 인스턴스 2개
    kind: KIND_GPU
    gpus: [0, 1]          # GPU 0, 1 사용
  }
]
```

---

## 적용 시 단점 및 고려사항

### 1. 시스템 복잡도 증가

**추가 컴포넌트**
- Triton Server 컨테이너 추가
- Model Repository 관리
- Config 파일 작성 (.pbtxt)

**학습 곡선**
- Triton 설정 학습 필요
- Protobuf 설정 파일 이해
- gRPC/HTTP API 학습

### 2. 기존 코드 대폭 수정 필요

**변경 범위**
```diff
- analysis-server/core/model_loader.py      (삭제)
- analysis-server/core/batch_manager.py     (삭제)
- analysis-server/models/                   (전체 재구성)
+ triton_client/ (새로운 클라이언트 코드)
+ model_repository/ (Triton 모델 저장소)
```

**작업량 추정**
- 모델 변환: 2-3일
- Triton 설정: 1-2일
- 클라이언트 재작성: 3-5일
- 테스트 및 디버깅: 3-5일
- **총 예상 작업**: 2-3주

### 3. 리소스 오버헤드

**메모리**
- Triton Server 자체 메모리: ~500MB-1GB
- 모델 캐싱으로 추가 메모리 필요

**컨테이너**
- 현재: 2개 (operation-server, analysis-worker)
- Triton 도입 후: 3개 (operation-server, triton-server, [선택] celery-worker)

### 4. 현재 시스템 특성상 불필요할 수 있음

**현재 모델 특성**
- LSTM, Moving Average: 경량 모델
- 추론 시간: 밀리초 단위
- 복잡한 전처리/후처리 필요

**Triton이 유용한 경우**
- 대규모 딥러닝 모델 (BERT, ResNet, GPT 등)
- 추론 시간이 긴 모델 (수백ms ~ 수초)
- 높은 처리량 요구 (초당 수천 건)

### 5. Python Backend 제약

**현재 코드**
```python
class LSTMTimeSeriesModel(BaseModel):
    def preprocess(self, data):
        # 복잡한 전처리
        data_array = np.array(data)
        normalized = (data_array - self.mean) / self.std
        # ...

    def postprocess(self, predictions):
        # 복잡한 후처리
        confidence = [0.95 - (i * 0.02) for i in range(len(predictions))]
        # ...
```

**Triton Python Backend**
- 복잡한 전처리/후처리는 Python Backend로 가능
- 하지만 Python Backend는 성능상 이점이 적음
- 단순 모델 추론만으로는 Triton의 장점 제한적

---

## 아키텍처 변경 사항

### 옵션 A: Full Triton (Celery 제거)

```
[클라이언트]
    ↓ HTTP POST
[Operation Server - FastAPI]
    ↓ HTTP/gRPC
[Triton Inference Server]
    ↓
[Model Repository (GPU)]
    ↓
[결과 반환]
    ↓
[Operation Server - 결과 저장]
```

**장점**
- 아키텍처 단순화
- Celery 제거로 복잡도 감소
- 동기 추론으로 응답 빠름

**단점**
- 비동기 처리 능력 상실
- 긴 작업 처리 어려움
- Kafka 통합 복잡

### 옵션 B: Hybrid (Triton + Celery 병행)

```
[클라이언트]
    ↓ HTTP POST
[Operation Server - FastAPI]
    ↓ Celery Task
[Analysis Worker - Celery]
    ↓ gRPC
[Triton Inference Server]
    ↓
[Model Repository (GPU)]
```

**장점**
- 비동기 처리 유지
- Kafka/Redis 통합 유지
- 점진적 마이그레이션 가능

**단점**
- 복잡도 증가 (Celery + Triton)
- 추가 네트워크 홉

---

## 구현 복잡도 분석

### Phase 1: Triton Server 설정 (난이도: )

**작업 내용**
1. Docker Compose에 Triton 추가
2. Model Repository 구조 설정
3. GPU 설정

```yaml
# docker-compose.yml
triton-server:
  image: nvcr.io/nvidia/tritonserver:24.01-py3
  container_name: triton-server
  ports:
    - "8000:8000"    # HTTP
    - "8001:8001"    # gRPC
    - "8002:8002"    # Metrics
  volumes:
    - ./model_repository:/models
  command: tritonserver --model-repository=/models
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Phase 2: 모델 변환 (난이도: )

**PyTorch → Triton PyTorch Backend**

```python
# model_repository/lstm_timeseries/1/model.py
import triton_python_backend_utils as pb_utils
import torch
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model = torch.jit.load('model.pt')
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            data = input_tensor.as_numpy()

            # 추론
            output = self.model(torch.from_numpy(data))

            # 응답 생성
            output_tensor = pb_utils.Tensor("OUTPUT", output.numpy())
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses
```

**Config 작성**

```protobuf
# model_repository/lstm_timeseries/config.pbtxt
name: "lstm_timeseries"
backend: "python"
max_batch_size: 32

input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [-1, 10, 1]  # [batch, sequence, features]
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [-1, 5]  # [batch, forecast_steps]
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### Phase 3: 클라이언트 재작성 (난이도: )

```python
# analysis-server/core/triton_client.py
import tritonclient.grpc as grpcclient
import numpy as np

class TritonInferenceClient:
    def __init__(self, url="triton-server:8001"):
        self.client = grpcclient.InferenceServerClient(url=url)

    def infer(self, model_name: str, data: list, version: str = ""):
        # 입력 준비
        input_data = np.array(data, dtype=np.float32)
        inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        # 출력 설정
        outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

        # 추론 실행
        response = self.client.infer(
            model_name=model_name,
            model_version=version,
            inputs=inputs,
            outputs=outputs
        )

        # 결과 추출
        output_data = response.as_numpy("OUTPUT")
        return output_data
```

### Phase 4: 통합 및 테스트 (난이도: )

**변경 필요 파일**
- `analysis-server/tasks.py` - Triton 클라이언트 사용
- `operation-server/api/routes/inference.py` - API 응답 형식 유지
- `shared/schemas/` - 스키마 조정
- 테스트 코드 전체 재작성

---

## 비용-편익 분석

### 비용 (Effort)

| 항목 | 예상 시간 | 난이도 |
|------|----------|--------|
| Triton 설정 및 학습 | 3-5일 |  |
| 모델 변환 및 Config 작성 | 3-5일 |  |
| 클라이언트 재작성 | 2-3일 |  |
| 통합 및 디버깅 | 5-7일 |  |
| 문서화 및 테스트 | 2-3일 |  |
| **총 작업 시간** | **15-23일** | **** |

### 편익 (Benefit)

| 항목 | 현재 | Triton 도입 후 | 개선도 |
|------|------|---------------|--------|
| 처리량 (RPS) | 10-20 | 50-100 | 5-10배 ⬆️ |
| GPU 활용도 | 30-40% | 70-90% | 2배 ⬆️ |
| 지연시간 (p99) | 100ms | 80ms | 20% ⬇️ |
| 배치 처리 | 수동 | 자동 |  |
| 모니터링 | 제한적 | 풍부 |  |
| 멀티 프레임워크 |  |  |  |
| 모델 버전 관리 | 수동 | 자동 |  |

---

## 최종 권장사항

### 결론: **현재 시점에서는 Triton 도입을 권장하지 않음**

### 이유

#### 1. **현재 시스템 규모와 요구사항**
- 경량 모델 (LSTM, Moving Average)
- 추론 시간이 짧음 (밀리초 단위)
- 처리량 요구사항이 크지 않음
- **Triton의 장점을 충분히 활용하기 어려움**

#### 2. **구현 복잡도 대비 효과**
- 작업 시간: 3-4주
- 성능 향상: 제한적 (이미 빠른 모델)
- **ROI(투자 대비 효과)가 낮음**

#### 3. **현재 아키텍처의 강점**
- Celery: 비동기 처리, 재시도, 분산 처리 우수
- Factory Pattern: 간단하고 유연함
- Kafka/Redis 통합: 잘 동작 중

---

### Triton 도입을 고려할 만한 상황

다음 조건 중 **2개 이상 해당되면** Triton 도입을 재검토하세요:

 **대규모 딥러닝 모델 사용**
- BERT, GPT, ResNet, YOLO 등
- 모델 크기 > 500MB
- 추론 시간 > 100ms

 **높은 처리량 요구**
- 초당 1000+ 요청
- 실시간 응답 필요
- GPU 활용도 최대화 필요

 **멀티 프레임워크 필요**
- PyTorch, TensorFlow, ONNX 혼용
- 모델 앙상블 필요

 **모델 버전 관리 복잡**
- A/B 테스팅 빈번
- 여러 버전 동시 서빙 필요
- 카나리 배포 필요

 **프로덕션 SLA 엄격**
- 99.9% 가용성 요구
- 자동 장애 복구 필요
- 세밀한 모니터링 필요

---

### 대안: 현재 시스템 개선 방안

Triton 없이 현재 시스템을 개선할 수 있는 방법:

#### 1. **MLflow 통합** (추천 )
```python
# 모델 버전 관리 및 추적
import mlflow

with mlflow.start_run():
    model = load_model()
    predictions = model.predict(data)
    mlflow.log_metric("inference_time", time)
    mlflow.log_model(model, "model")
```

**장점**
- 모델 버전 관리 개선
- 실험 추적
- 모델 레지스트리
- **작업 시간: 3-5일**

#### 2. **배치 처리 최적화**
```python
# 현재 batch_manager 개선
class ImprovedBatchManager:
    def __init__(self):
        self.max_batch_size = 32
        self.max_wait_time = 0.1  # 100ms

    async def process_batch(self):
        # 동적 배치 크기 조절
        # GPU 활용도 모니터링
        # 자동 튜닝
```

**장점**
- Triton의 Dynamic Batching 효과 모방
- 기존 코드 재사용
- **작업 시간: 2-3일**

#### 3. **Prometheus + Grafana 모니터링**
```python
from prometheus_client import Counter, Histogram

inference_counter = Counter('inference_requests_total', 'Total inference requests')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')

@inference_duration.time()
def infer(data):
    inference_counter.inc()
    return model.predict(data)
```

**장점**
- 세밀한 모니터링
- Triton 수준의 메트릭
- **작업 시간: 1-2일**

#### 4. **모델 최적화**
```python
# PyTorch JIT 컴파일
model = torch.jit.script(lstm_model)
model.save("model_optimized.pt")

# 또는 ONNX 변환
torch.onnx.export(model, dummy_input, "model.onnx")
```

**장점**
- 추론 속도 향상 (20-50%)
- 메모리 사용량 감소
- **작업 시간: 1-2일**

---

### 로드맵 제안

#### Phase 1: 현재 시스템 최적화 (1-2주)
1. MLflow 통합  **[진행 중]**
2. Prometheus 모니터링 추가
3. 배치 처리 개선
4. 모델 JIT 컴파일

#### Phase 2: 스케일 테스트 (1주)
1. 부하 테스트 수행
2. 병목 지점 파악
3. GPU 활용도 측정

#### Phase 3: 재평가 (필요 시)
**만약** 다음 조건이 발생하면 Triton 재검토:
- 처리량이 10배 이상 증가 필요
- 대규모 모델로 전환
- 멀티 프레임워크 필요

---

## 부록: Triton 도입 시 구현 예시

### 1. Docker Compose 설정

```yaml
services:
  triton-server:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    container_name: triton-server
    ports:
      - "8000:8000"   # HTTP
      - "8001:8001"   # gRPC
      - "8002:8002"   # Metrics
    volumes:
      - ./model_repository:/models
      - ./triton-config:/config
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: |
      tritonserver
        --model-repository=/models
        --strict-model-config=false
        --log-verbose=1
        --metrics-port=8002
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - webnet
    depends_on:
      - postgres
      - redis

  # Operation Server는 Triton 클라이언트로 변경
  operation-server:
    environment:
      - TRITON_SERVER_URL=triton-server:8001
    depends_on:
      - triton-server
```

### 2. Model Repository 구조

```
model_repository/
├── lstm_timeseries/
│   ├── config.pbtxt
│   ├── 1/
│   │   ├── model.py          # Python backend
│   │   └── model.pt          # PyTorch 모델
│   └── 2/
│       ├── model.py
│       └── model.pt
├── moving_average/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── ensemble_model/
    └── config.pbtxt
```

### 3. Triton 클라이언트 통합

```python
# analysis-server/core/triton_inference.py
from typing import Dict, Any, List
import tritonclient.grpc as grpcclient
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class TritonInferenceClient:
    """Triton Inference Server 클라이언트"""

    def __init__(self, url: str = "triton-server:8001"):
        self.url = url
        self.client = None

    def connect(self):
        """Triton 서버 연결"""
        try:
            self.client = grpcclient.InferenceServerClient(url=self.url)
            if self.client.is_server_live():
                logger.info(f"Connected to Triton server at {self.url}")
            else:
                raise ConnectionError("Triton server is not live")
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            raise

    def infer(
        self,
        model_name: str,
        data: List[float],
        model_version: str = "",
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        추론 실행

        Args:
            model_name: 모델 이름
            data: 입력 데이터
            model_version: 모델 버전 (빈 문자열이면 최신 버전)
            timeout: 타임아웃 (초)

        Returns:
            추론 결과
        """
        if not self.client:
            self.connect()

        try:
            # 입력 데이터 준비
            input_data = np.array(data, dtype=np.float32)
            input_data = input_data.reshape(1, -1, 1)  # [batch, seq, features]

            # Triton 입력 객체 생성
            inputs = [
                grpcclient.InferInput(
                    "INPUT",
                    input_data.shape,
                    "FP32"
                )
            ]
            inputs[0].set_data_from_numpy(input_data)

            # 출력 설정
            outputs = [
                grpcclient.InferRequestedOutput("OUTPUT"),
                grpcclient.InferRequestedOutput("CONFIDENCE")
            ]

            # 추론 실행
            response = self.client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs,
                client_timeout=timeout
            )

            # 결과 추출
            predictions = response.as_numpy("OUTPUT")
            confidence = response.as_numpy("CONFIDENCE")

            return {
                "predictions": predictions.tolist(),
                "confidence": confidence.tolist(),
                "model_name": model_name,
                "model_version": response.model_version
            }

        except Exception as e:
            logger.error(f"Triton inference error: {e}")
            raise

    def get_model_metadata(self, model_name: str, model_version: str = ""):
        """모델 메타데이터 조회"""
        if not self.client:
            self.connect()

        metadata = self.client.get_model_metadata(
            model_name=model_name,
            model_version=model_version
        )
        return metadata

    def is_model_ready(self, model_name: str, model_version: str = ""):
        """모델 준비 상태 확인"""
        if not self.client:
            self.connect()

        return self.client.is_model_ready(
            model_name=model_name,
            model_version=model_version
        )

# 싱글톤 인스턴스
triton_client = TritonInferenceClient()
```

### 4. Celery Task 수정

```python
# analysis-server/tasks.py (Triton 버전)
from core.triton_inference import triton_client

@celery_app.task(bind=True, base=InferenceTask)
def run_inference(self, job_id: str, model_name: str, data: list, config: dict, metadata: dict):
    """Triton을 사용한 추론 태스크"""
    try:
        logger.info(f"[Task] Job {job_id}: Using Triton for model '{model_name}'")

        # Triton 추론
        start_time = time.time()
        result = triton_client.infer(
            model_name=model_name,
            data=data,
            model_version=config.get("model_version", "")
        )
        inference_time = time.time() - start_time

        # 결과 데이터 구성
        result_data = {
            "job_id": job_id,
            "status": "completed",
            "model_name": model_name,
            "predictions": result["predictions"],
            "confidence": result["confidence"],
            "metrics": {
                "inference_time": inference_time,
                "model_version": result["model_version"]
            },
            "metadata": metadata,
            "completed_at": datetime.utcnow().isoformat()
        }

        # 결과 저장 (Redis/PostgreSQL/Kafka)
        save_results(job_id, result_data)

        logger.info(f"[Task] Job {job_id}: Completed in {inference_time:.3f}s")
        return result_data

    except Exception as e:
        logger.error(f"[Task] Job {job_id}: Error: {str(e)}")
        raise
```

---

## 요약

###  Triton 도입이 적합한 경우
- 대규모 딥러닝 모델 (BERT, ResNet 등)
- 높은 처리량 요구 (1000+ RPS)
- 멀티 프레임워크 필요
- 복잡한 모델 버전 관리 필요

###  현재 시스템에는 과도함
- 경량 모델 (LSTM, MA)
- 낮은/중간 처리량
- 단일 프레임워크 (PyTorch)
- 간단한 버전 관리

###  추천 접근법
1. **현재**: MLflow 통합 (진행 중) 
2. **단기**: 배치 처리 최적화, 모니터링 강화
3. **중기**: 부하 테스트 및 병목 분석
4. **장기**: 필요 시 Triton 재검토

###  핵심 메시지
> **"Right tool for the right job"**
> Triton은 훌륭한 도구이지만, 현재 시스템에는 과도한 엔지니어링입니다.
> MLflow + 최적화된 Celery가 현재 요구사항에 더 적합합니다.

---

**작성일**: 2025-10-15
**버전**: 1.0
**작성자**: Analysis Team
