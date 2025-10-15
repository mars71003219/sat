# Deprecated Files

이 디렉토리의 파일들은 **Triton Inference Server** 도입으로 더 이상 사용되지 않습니다.

## 변경 사항 (2025-10-15)

### 삭제된 컴포넌트

| 파일 | 역할 | 대체 방법 |
|------|------|-----------|
| `batch_manager.py` | 수동 배치 처리 | **Triton Dynamic Batching** (자동) |
| `model_loader.py` | 모델 로딩 및 LRU 캐싱 | **Triton Model Repository** (자동) |
| `model_factory.py` | 모델 등록 (팩토리 패턴) | **Triton Model Repository** 구조 |

### Triton이 제공하는 기능

1. **Dynamic Batching** (batch_manager.py 대체)
   - C++ 기반 고성능 배치 처리
   - 실시간 GPU 활용률 최적화
   - 자동 배치 크기 조절

2. **Model Repository** (model_loader.py + model_factory.py 대체)
   - 표준화된 모델 저장 구조
   - 자동 모델 로딩/언로딩
   - 모델 버전 관리 기본 지원
   - 여러 프레임워크 지원 (PyTorch, TensorFlow, ONNX 등)

3. **추가 장점**
   - Prometheus 메트릭 자동 제공
   - Health Check API
   - 동시 모델 실행
   - GPU 메모리 자동 관리

## 새로운 아키텍처

```
[Analysis Worker - Celery]
    ↓
[Triton Client (전/후처리)]
    ↓ gRPC
[Triton Server (추론 + Dynamic Batching)]
    ↓
[Model Repository]
    ├── lstm_timeseries/
    └── moving_average/
```

## 참고 문서

- Triton 적용 상세 분석: `/docs/TRITON_ANALYSIS.md`
- Triton 공식 문서: https://docs.nvidia.com/deeplearning/triton-inference-server/
