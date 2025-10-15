# 🚀 TensorRT 기반 Triton Inference Server 배포 가이드

## 📌 시스템 개요

이 프로젝트는 **NVIDIA Triton Inference Server**와 **TensorRT**를 사용하여 시계열 예측 모델을 GPU에서 고성능으로 서빙하는 시스템입니다.

### 핵심 특징

- ✅ **RTX 5060 (sm_120) 완전 지원**: TensorRT 엔진 사용으로 최신 GPU 호환성 확보
- ✅ **범용 모델 변환 도구**: 어떤 PyTorch 모델이든 TensorRT로 변환 가능
- ✅ **사용자 친화적**: 모델 학습 → 변환 → 배포까지 자동화
- ✅ **고성능**: FP16 정밀도, Dynamic Batching, CUDA Graph 최적화

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 워크플로우                          │
└─────────────────────────────────────────────────────────────┘
           │
           ├─ 1. PyTorch 모델 학습
           │    └─ model.pth 생성
           │
           ├─ 2. TensorRT 변환
           │    └─ python convert_model_to_tensorrt.py
           │         ├─ ONNX 변환
           │         └─ TensorRT 엔진 빌드 (.plan)
           │
           ├─ 3. Triton 배포
           │    └─ model_repository/
           │         └─ my_model/
           │              ├─ config.pbtxt
           │              └─ 1/
           │                   └─ model.plan
           │
           └─ 4. 추론 서비스
                └─ HTTP/gRPC API

┌─────────────────────────────────────────────────────────────┐
│                 Triton Inference Server                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ VAE Model    │  │ Transformer  │  │ Custom Model │      │
│  │ (TensorRT)   │  │ (TensorRT)   │  │ (TensorRT)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Dynamic Batching Engine                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         NVIDIA GPU (RTX 5060, sm_120)              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 구성 요소

### 1. 모델 변환 도구

**파일**: `scripts/convert_model_to_tensorrt.py`

**기능**:
- PyTorch `.pth` 파일 로드
- ONNX 변환
- TensorRT 엔진 빌드
- Triton 설정 자동 생성

**사용법**:
```bash
python scripts/convert_model_to_tensorrt.py \
    --model-path YOUR_MODEL.pth \
    --model-class YourModelClass \
    --model-module your_model.py \
    --model-name model_name \
    --input-shape 20,1 \
    --output-shape 10
```

### 2. Triton Server

**컨테이너**: `nvcr.io/nvidia/tritonserver:24.10-py3`

**설정**:
- TensorRT 백엔드
- Dynamic Batching
- GPU 인스턴스
- CUDA Graph 최적화

### 3. 모델 예제

#### VAE (Variational Autoencoder)

```python
# model_repository/vae_timeseries/1/train_and_export.py
class TimeSeriesVAE(nn.Module):
    - 입력: [batch, 1, 20] (1채널, 20 timesteps)
    - 출력: [batch, 10] (10 forecast steps)
    - Latent Dim: 32
    - Architecture: Encoder-Decoder with reparameterization
```

#### Transformer

```python
# model_repository/transformer_timeseries/1/train_and_export.py
class TimeSeriesTransformer(nn.Module):
    - 입력: [batch, 20, 1] (20 timesteps, 1 feature)
    - 출력: [batch, 10] (10 forecast steps)
    - d_model: 64
    - nhead: 4
    - num_layers: 3
    - Architecture: Multi-head self-attention
```

---

## 📋 빠른 시작

### Step 1: 모델 학습

```python
import torch
import torch.nn as nn

# 모델 정의
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 모델 정의 ...

    def forward(self, x):
        # ... forward 로직 ...
        return output

# 학습
model = MyModel()
# ... 학습 코드 ...

# 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {...}  # 옵션
}, 'my_model.pth')
```

### Step 2: TensorRT 변환

```bash
python scripts/convert_model_to_tensorrt.py \
    --model-path my_model.pth \
    --model-class MyModel \
    --model-module my_model.py \
    --model-name my_timeseries \
    --input-shape 20,1 \
    --output-shape 10 \
    --output-dir ./model_repository
```

**출력**:
```
model_repository/
└── my_timeseries/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

### Step 3: Triton 배포

```bash
# 1. 모델 복사
cp -r ./model_repository/my_timeseries /path/to/satellite/model_repository/

# 2. Triton Server 시작
cd /path/to/satellite
docker compose up -d triton-server

# 3. 모델 확인
curl http://localhost:8500/v2/models/my_timeseries
```

### Step 4: 추론 테스트

```bash
# Python 클라이언트 사용
python scripts/test_simulator.py
```

---

## 📂 프로젝트 구조

```
satellite/
├── Dockerfile.triton                      # Triton Server Dockerfile (간소화)
├── docker-compose.yml                     # Docker Compose 설정
│
├── model_repository/                      # Triton 모델 리포지토리
│   ├── vae_timeseries/
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       ├── model.plan                 # TensorRT 엔진
│   │       ├── model.onnx                 # ONNX (중간 파일, 옵션)
│   │       └── train_and_export.py        # 학습/변환 스크립트
│   │
│   └── transformer_timeseries/
│       ├── config.pbtxt
│       └── 1/
│           ├── model.plan
│           └── train_and_export.py
│
├── scripts/
│   ├── convert_model_to_tensorrt.py       # ⭐ 핵심 변환 도구
│   ├── demo_create_simple_model.py        # 테스트 모델 생성 예제
│   └── README_MODEL_CONVERSION.md         # ⭐ 상세 가이드
│
├── shared/
│   └── schemas/
│       └── inference_response.py          # API 스키마
│
└── MODEL_DEPLOYMENT_SUMMARY.md            # 이 파일
```

---

## 🔧 고급 설정

### 1. 배치 크기 최적화

```bash
python scripts/convert_model_to_tensorrt.py \
    --min-batch 1 \
    --opt-batch 16 \  # 가장 많이 사용할 배치 크기
    --max-batch 64
```

### 2. 정밀도 선택

```bash
# FP16 (기본, 권장)
--fp16

# FP32 (정확도 우선)
--no-fp16

# INT8 (최고 성능, 캘리브레이션 필요)
--int8
```

### 3. GPU 인스턴스 조정

```bash
# 경량 모델: 여러 인스턴스
--instance-count 4

# 무거운 모델 (Transformer): 1개
--instance-count 1
```

### 4. config.pbtxt 수동 수정

```protobuf
# Dynamic Batching 조정
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100000  # 100ms
}

# CUDA Graph 최적화
optimization {
  cuda {
    graphs: true
    graph_spec {
      batch_size: 8
      input { ... }
    }
  }
}
```

---

## 🚨 문제 해결

### 1. RTX 5060 호환성

**문제**: PyTorch가 sm_120을 지원하지 않음

**해결**: ✅ TensorRT 사용으로 해결됨
- TensorRT는 빌드 시 현재 GPU에 최적화된 엔진 생성
- RTX 5060에서 완벽 작동

### 2. trtexec 명령 없음

**해결**:
```bash
# Docker 컨테이너 내에서 실행
docker run --gpus all -v $(pwd):/workspace \
    nvcr.io/nvidia/tritonserver:24.10-py3 \
    bash -c "cd /workspace && python scripts/convert_model_to_tensorrt.py ..."
```

### 3. 메모리 부족

**해결**:
```bash
# 배치 크기 줄이기
--max-batch 16

# 인스턴스 개수 줄이기
--instance-count 1

# FP16 사용
--fp16
```

### 4. 모델 로드 실패

**확인사항**:
```python
# 모델 클래스가 올바르게 정의되었는지 확인
# state_dict 키가 일치하는지 확인
checkpoint = torch.load('model.pth')
print(checkpoint.keys())
```

---

## 📊 성능 벤치마크

### 예상 성능 (RTX 5060)

| 모델 | Batch Size | Latency (ms) | Throughput (RPS) |
|------|-----------|--------------|------------------|
| VAE (FP16) | 16 | ~5-10 | ~1000-2000 |
| Transformer (FP16) | 8 | ~10-20 | ~400-800 |
| Simple LSTM (FP16) | 32 | ~3-8 | ~2000-4000 |

*실제 성능은 모델 크기와 복잡도에 따라 다릅니다*

### 최적화 팁

1. **Dynamic Batching 활용**: 여러 요청을 자동으로 배치 처리
2. **FP16 정밀도**: 약 2배 성능 향상, 정확도 거의 동일
3. **CUDA Graph**: 커널 실행 오버헤드 감소
4. **Optimal Batch Size**: 시스템 부하에 맞게 조정

---

## 🔄 워크플로우 비교

### ❌ 이전 방식 (PyTorch 백엔드)

```
PyTorch 모델 (.pth)
    ↓
Triton Python Backend
    ↓
GPU 호환성 문제 (sm_120 미지원)
    ↓
❌ 실패
```

### ✅ 현재 방식 (TensorRT 백엔드)

```
PyTorch 모델 (.pth)
    ↓
ONNX 변환
    ↓
TensorRT 엔진 빌드 (.plan)
    ↓
Triton TensorRT Backend
    ↓
✅ RTX 5060에서 완벽 작동
```

---

## 📚 추가 리소스

### 공식 문서

- [TensorRT 문서](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [ONNX](https://onnx.ai/)

### 내부 가이드

- `scripts/README_MODEL_CONVERSION.md` - 상세한 변환 가이드
- `model_repository/vae_timeseries/1/train_and_export.py` - VAE 예제
- `model_repository/transformer_timeseries/1/train_and_export.py` - Transformer 예제

---

## ✅ 체크리스트

새 모델 배포 시:

- [ ] PyTorch 모델 학습 완료
- [ ] `.pth` 파일 저장 (state_dict + config 권장)
- [ ] `convert_model_to_tensorrt.py` 실행
- [ ] 생성된 `model.plan` 확인
- [ ] `config.pbtxt` 검토 (필요시 수정)
- [ ] Triton 모델 리포지토리로 복사
- [ ] Triton Server 재시작
- [ ] 추론 테스트 (`curl` 또는 클라이언트 스크립트)
- [ ] 성능 벤치마크 실행

---

## 🎯 결론

이 시스템은 **유연성**과 **성능**을 모두 갖춘 프로덕션 레벨 추론 서버입니다:

1. **범용성**: 어떤 PyTorch 모델이든 TensorRT로 변환 가능
2. **최신 GPU 지원**: RTX 5060 (sm_120) 완벽 지원
3. **사용자 친화성**: 한 줄 명령으로 변환 완료
4. **고성능**: FP16, Dynamic Batching, CUDA Graph 최적화
5. **확장성**: 새 모델 추가가 매우 간단

**핵심 명령 한 줄**:

```bash
python scripts/convert_model_to_tensorrt.py \
    --model-path YOUR_MODEL.pth \
    --model-class YOUR_CLASS \
    --model-module YOUR_MODULE.py \
    --model-name YOUR_NAME \
    --input-shape X,Y \
    --output-shape Z
```

이제 새로운 알고리즘이나 모델이 생기면, 위 명령 하나로 즉시 Triton에 배포할 수 있습니다! 🚀
