# 버전 정보

## Triton Inference Server

### 현재 사용 버전: 25.08

**NGC 이미지**: `nvcr.io/nvidia/tritonserver:25.08-trtllm-python-py3`

**포함된 구성요소**:

| 구성요소 | 버전 |
|---------|------|
| Triton Server | 2.51.0 |
| Python | 3.12.3 |
| PyTorch | 2.8.0a0+5289986c39.nv25.5 |
| TensorRT | 10.11.0.33 |
| CUDA | 12.x |
| ONNX Runtime | GPU acceleration 지원 |

**참고**: [NVIDIA NGC Triton Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/)

## 프로젝트 의존성

### Backend Services

**Python**: 3.10+

**주요 라이브러리**:
- FastAPI: 최신 안정 버전
- Celery: 최신 안정 버전
- PostgreSQL Driver: psycopg2-binary
- Redis: redis-py
- Kafka: confluent-kafka-python
- Triton Client: tritonclient[grpc]

### Frontend

**Node.js**: 18.x LTS

**주요 라이브러리**:
- React: 18.x
- React Router: 최신 버전

## Docker Images

### Infrastructure

| 서비스 | 이미지 | 버전 |
|--------|--------|------|
| Kafka | confluentinc/cp-kafka | latest |
| Redis | redis/redis-stack | latest |
| PostgreSQL | postgres | latest (16.x) |
| Elasticsearch | docker.elastic.co/elasticsearch/elasticsearch | 8.5.0 |
| Nginx | nginx | alpine |

### Monitoring

| 서비스 | 이미지 | 버전 |
|--------|--------|------|
| Kafka UI | provectuslabs/kafka-ui | latest |
| Flower | (custom build) | latest |

## GPU 요구사항

### 최소 요구사항
- NVIDIA GPU with CUDA Compute Capability 7.0 이상
- CUDA Toolkit 12.x
- NVIDIA Driver 550.x 이상

### 권장 사양
- GPU: RTX 3060 이상 (8GB+ VRAM)
- CUDA Compute Capability: 8.6 이상 (Ampere 아키텍처)
- Driver: 최신 stable 버전

### 테스트 환경
- GPU: RTX 5060 (sm_100, Compute Capability 10.0)
- VRAM: 8GB
- Driver: 최신 버전
- 백엔드: ONNX Runtime (GPU) ✅ 완전 호환

## 호환성 매트릭스

### TensorRT 지원

| GPU 아키텍처 | Compute Capability | TensorRT 10.11.0.33 |
|-------------|-------------------|---------------------|
| Ampere (RTX 30x0) | 8.6 | ✅ 완전 지원 |
| Ada Lovelace (RTX 40x0) | 8.9 | ✅ 완전 지원 |
| Blackwell (RTX 50x0) | 10.0 | ⚠️ 최신 TensorRT 필요 |

### ONNX Runtime 지원

| GPU 아키텍처 | ONNX Runtime |
|-------------|--------------|
| All NVIDIA GPUs | ✅ 완전 지원 (CUDA Backend) |
| RTX 50x0 (sm_100) | ✅ 완전 호환 |

**권장**: RTX 50x0 시리즈는 ONNX Runtime 백엔드 사용 권장

## 업그레이드 가이드

### Triton Server 업그레이드

```bash
# 1. 새 이미지 pull
docker pull nvcr.io/nvidia/tritonserver:25.08-trtllm-python-py3

# 2. 기존 컨테이너 중지
docker compose down triton-server

# 3. 이미지 빌드
docker compose build triton-server

# 4. 재시작
docker compose up -d triton-server
```

### 주의사항
- TensorRT 엔진(.plan)은 GPU 아키텍처별로 재빌드 필요
- ONNX 모델(.onnx)은 GPU 간 호환 가능
- 버전 업그레이드 시 모델 재테스트 필수

## 버전 확인 명령어

### Triton Server
```bash
docker compose exec triton-server cat /opt/tritonserver/TRITON_VERSION
```

### Python
```bash
docker compose exec triton-server python3 --version
```

### PyTorch
```bash
docker compose exec triton-server python3 -c "import torch; print(torch.__version__)"
```

### TensorRT
```bash
docker compose exec triton-server python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### CUDA
```bash
docker compose exec triton-server nvcc --version
```

## 변경 이력

### 2025-10-15
- Triton Server 24.10 → 25.08 업그레이드
- PyTorch 2.8.0 지원
- TensorRT 10.11.0.33 지원
- Python 3.12.3 사용
- RTX 5060 (sm_100) GPU 테스트 완료
