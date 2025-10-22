# Triton 추론 플랫폼 - 모델 변환 가이드

이 디렉토리는 **외부 학습 서버에서 받은 모델**을 Triton Inference Server용으로 변환하는 도구를 포함합니다.

##  파일 구조

```
scripts/
├── convert_model.py          # 범용 모델 변환 도구 (ONNX/TensorRT)
├── convert_example.sh         # 변환 예시 스크립트
├── test_simulator.py          # 전체 테스트 스위트
├── test_single_model.py       # 단일 모델 테스트
└── README.md                  # 이 문서
```

##  워크플로우

### 1. 외부 학습 서버에서 모델 받기

외부 학습 팀으로부터 다음을 받습니다:
- `.pth` 파일 (PyTorch 가중치)
- 모델 클래스 정의 (`.py` 파일)
- 입력/출력 shape 정보

### 2. 모델 변환

#### Option A: ONNX Runtime 백엔드 (권장 - 최고 GPU 호환성)

```bash
python3 convert_model.py \
    --model-path /path/to/model.pth \
    --model-class YourModelClass \
    --model-module your_model_definition.py \
    --model-name your_model_name \
    --input-shape <batch,channels,sequence> \
    --output-shape <forecast_steps> \
    --backend onnx \
    --output-dir /workspace/model_repository
```

#### Option B: TensorRT 백엔드 (최고 성능, GPU 호환성 주의)

```bash
python3 convert_model.py \
    --model-path /path/to/model.pth \
    --model-class YourModelClass \
    --model-module your_model_definition.py \
    --model-name your_model_name \
    --input-shape <batch,channels,sequence> \
    --output-shape <forecast_steps> \
    --backend tensorrt \
    --output-dir /workspace/model_repository
```

### 3. Triton Server 재시작

```bash
docker compose restart triton-server
```

### 4. 테스트

```bash
# 단일 모델 빠른 테스트
python3 test_single_model.py

# 전체 테스트 스위트
python3 test_simulator.py
```

##  convert_model.py 파라미터 설명

| 파라미터 | 설명 | 예시 |
|---------|------|------|
| `--model-path` | PyTorch 모델 파일 경로 | `vae_model.pth` |
| `--model-class` | 모델 클래스 이름 | `TimeSeriesVAE` |
| `--model-module` | 모델 정의 파이썬 파일 | `vae_model.py` |
| `--model-name` | Triton에서 사용할 모델 이름 | `vae_timeseries` |
| `--input-shape` | 모델 입력 shape (배치 제외) | `1,20` |
| `--output-shape` | 모델 출력 shape (배치 제외) | `10` |
| `--backend` | 백엔드 선택 (`onnx` 또는 `tensorrt`) | `onnx` |
| `--output-dir` | 출력 디렉토리 | `../model_repository` |

##  실제 사용 예시

### VAE 모델 변환

```bash
python3 convert_model.py \
    --model-path /external/vae_timeseries.pth \
    --model-class TimeSeriesVAE \
    --model-module vae_definition.py \
    --model-name vae_timeseries \
    --input-shape 1,20 \
    --output-shape 10 \
    --backend onnx \
    --output-dir ../model_repository
```

### Transformer 모델 변환

```bash
python3 convert_model.py \
    --model-path /external/transformer_timeseries.pth \
    --model-class TimeSeriesTransformer \
    --model-module transformer_definition.py \
    --model-name transformer_timeseries \
    --input-shape 20,1 \
    --output-shape 10 \
    --backend onnx \
    --output-dir ../model_repository
```

##  GPU 호환성

### ONNX Runtime (권장)
-  RTX 5060 (sm_100) 호환
-  대부분의 NVIDIA GPU 지원
-  CPU fallback 지원

### TensorRT
-  GPU별 엔진 빌드 필요
-  RTX 5060 (sm_100)은 최신 TensorRT 필요
-  최고 성능

##  참고 문서

- [Triton Inference Server 공식 문서](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [ONNX Runtime 문서](https://onnxruntime.ai/docs/)
- [TensorRT 문서](https://docs.nvidia.com/deeplearning/tensorrt/)

##  중요 사항

1. **이 플랫폼은 추론 전용입니다** - 모델 학습은 외부 서버에서 수행
2. **모델 정의 파일 필요** - `.pth` 파일만으로는 변환 불가능
3. **GPU 호환성 확인** - TensorRT 사용 시 GPU compute capability 확인
4. **입력/출력 shape 정확성** - 변환 시 정확한 shape 정보 필수
