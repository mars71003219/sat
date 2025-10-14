# Analysis Server - AI 추론 서버

GPU 기반 배치 처리를 지원하는 분산 AI 추론 서버

## 주요 기능

### 1. 팩토리 패턴 기반 모델 관리
- 모델 동적 등록 및 로딩
- 새로운 모델 추가 시 데코레이터만 사용
- LRU 캐싱으로 메모리 효율적 관리

### 2. 배치 처리 (Batch Processing)
- GPU 활용성 향상을 위한 자동 배치 큐잉
- **배치 크기 트리거**: 8개 요청 도달 시 즉시 처리
- **시간 트리거**: 2초 경과 시 자동 처리
- 모델별 독립적인 배치 큐 관리

### 3. Celery 기반 분산 처리
- 비동기 작업 큐
- 다중 워커 지원
- 독립적 스케일 아웃

## 현재 지원 모델

### 1. LSTM Time Series Model (`lstm_timeseries`)
딥러닝 기반 시계열 예측

```json
{
  "model_name": "lstm_timeseries",
  "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2],
  "config": {
    "forecast_steps": 5,
    "sequence_length": 10,
    "hidden_size": 64
  }
}
```

### 2. Moving Average Model (`moving_average`)
통계 기반 이동평균 예측

```json
{
  "model_name": "moving_average",
  "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2],
  "config": {
    "forecast_steps": 5,
    "window_size": 5,
    "include_trend": true
  }
}
```

## 새 모델 추가 방법

### 1. 모델 클래스 작성

```python
# analysis-server/models/my_models/new_model.py
from analysis_server.models.base_model import BaseModel
from analysis_server.core.model_factory import model_factory

@model_factory.register_decorator("new_model")
class NewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # 설정 초기화
    
    def load(self):
        # 모델 로드
        self.model = load_your_model()
        self.is_loaded = True
    
    def preprocess(self, data):
        # 전처리
        return processed_data
    
    def predict(self, data):
        # 추론
        return predictions
    
    def postprocess(self, predictions):
        # 후처리
        return {
            "predictions": predictions.tolist(),
            "confidence": [0.95] * len(predictions)
        }
    
    @classmethod
    def get_description(cls):
        return "새로운 모델 설명"
    
    @classmethod
    def get_input_schema(cls):
        return {"type": "object", "properties": {...}}
    
    @classmethod
    def get_output_schema(cls):
        return {"type": "object", "properties": {...}}
```

### 2. tasks.py에서 import

```python
# analysis-server/tasks.py
from analysis_server.models.my_models import NewModel
```

끝! 자동으로 팩토리에 등록되고 사용 가능합니다.

## 배치 처리 작동 방식

```
요청 1 → 배치 큐 [1]         → 타이머 시작 (2초)
요청 2 → 배치 큐 [1,2]       → 대기중
요청 3 → 배치 큐 [1,2,3]     → 대기중
...
요청 8 → 배치 큐 [1,2,..,8]  → 즉시 처리 (배치 크기 도달)

또는

요청 1 → 배치 큐 [1]         → 타이머 시작
요청 2 → 배치 큐 [1,2]       → 대기중
(2초 경과)                    → 타이머 트리거, [1,2] 처리
```

### 배치 설정 조정

`analysis-server/core/batch_manager.py`:

```python
batch_manager = BatchManager(
    max_batch_size=8,      # 배치 크기
    max_wait_time=2.0      # 대기 시간 (초)
)
```

## 구조

```
analysis-server/
├── core/
│   ├── model_factory.py      # 팩토리 패턴
│   ├── model_loader.py        # 모델 로더 + LRU 캐싱
│   └── batch_manager.py       # 배치 처리 매니저
├── models/
│   ├── base_model.py          # 추상 베이스 클래스
│   └── timeseries/            # 시계열 모델들
│       ├── lstm_model.py
│       └── moving_average_model.py
├── tasks.py                   # Celery 워커 태스크
└── Dockerfile
```

## 개발

```bash
# 의존성 설치
pip install -r requirements.txt

# Celery 워커 실행
celery -A analysis_server.tasks worker --loglevel=info --concurrency=2 --queue=inference

# 모델 목록 확인
python -c "from analysis_server.core.model_factory import model_factory; print(model_factory.get_registered_models())"
```

## 모니터링

### 배치 큐 상태 확인

```python
from analysis_server.core.batch_manager import batch_manager
print(batch_manager.get_queue_status())
```

### 모델 캐시 정보

```python
from analysis_server.core.model_loader import model_loader
print(model_loader.get_cache_info())
```
