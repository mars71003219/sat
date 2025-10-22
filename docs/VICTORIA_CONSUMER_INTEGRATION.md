# Victoria Consumer 통합 가이드

## 현재 구조

```
victoria-consumer/          # 별도 컨테이너
├── consumer.py            # Kafka → VictoriaMetrics
├── Dockerfile
└── requirements.txt
```

## 통합 옵션

### 옵션 1: operation-server에 백그라운드 태스크로 통합 ✅ 추천

**장점:**
- 컨테이너 수 감소 (13개 → 12개)
- 같은 코드베이스 관리
- 공유 설정 및 의존성
- 리소스 효율

**단점:**
- API 서버와 consumer가 같은 프로세스
- 장애 전파 가능성

**구현 방법:**

```python
# operation-server/background/victoria_consumer.py
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start victoria consumer as background task
    consumer_task = asyncio.create_task(run_victoria_consumer())

    yield

    # Cleanup
    consumer_task.cancel()

# main.py
app = FastAPI(lifespan=lifespan)
```

### 옵션 2: 현재 구조 유지 (별도 컨테이너) ✅ 현재

**장점:**
- 명확한 책임 분리
- 독립적 확장 (여러 consumer 인스턴스)
- 장애 격리
- 쉬운 모니터링

**단점:**
- 추가 컨테이너 오버헤드
- 별도 관리 필요

**언제 유용한가:**
- 대규모 데이터 처리
- 높은 처리량 요구
- 프로덕션 환경

### 옵션 3: Celery Worker로 통합

**구현:**
```python
# operation-server/celery_tasks/victoria_consumer.py
@celery.task
def consume_victoria_metrics():
    # Kafka consumer logic
    pass
```

**장점:**
- 기존 Celery 인프라 활용
- 작업 스케줄링 가능

**단점:**
- Celery는 장시간 실행 태스크에 부적합
- Kafka consumer는 무한 루프 필요

## 결론 및 권장사항

### 현재 프로젝트 규모에서는:

**별도 컨테이너 유지를 권장합니다!**

**이유:**

1. **마이크로서비스 학습 목적**
   - 실제 프로덕션 아키텍처 경험
   - 서비스 간 통신 패턴 학습

2. **확장성**
   - 향후 여러 위성 데이터 처리 가능
   - Consumer 인스턴스 추가 용이

3. **모니터링**
   - 독립적인 로그/메트릭
   - 장애 추적 용이

4. **배포 유연성**
   - Consumer만 업데이트 가능
   - API 서버 중단 없이 재배포

### 소규모 프로젝트/프로토타입이라면:

**operation-server 통합을 권장합니다!**

```python
# operation-server/main.py
import threading

def start_victoria_consumer():
    # consumer.py 로직을 함수로 변환
    pass

# 앱 시작 시 백그라운드 스레드 실행
consumer_thread = threading.Thread(
    target=start_victoria_consumer,
    daemon=True
)
consumer_thread.start()
```

## 비교표

| 측면 | 별도 컨테이너 | 통합 |
|------|-------------|------|
| 컨테이너 수 | 13개 | 12개 |
| 메모리 사용 | ~100MB 추가 | 동일 |
| 관리 복잡도 | 높음 | 낮음 |
| 확장성 | 우수 | 보통 |
| 장애 격리 | 우수 | 보통 |
| 학습 가치 | 높음 (MSA) | 보통 |
| 프로덕션 준비 | 우수 | 보통 |

## 실제 프로덕션 사례

### Confluent (Kafka 회사)
- 별도 Consumer 애플리케이션 사용
- 스트림 처리는 독립 서비스

### Datadog
- 메트릭 수집기는 별도 에이전트
- API와 분리

### 우리 프로젝트
- **현재**: Triton (추론), Operation (API), Analysis (Worker), Victoria-Consumer
- **각자 명확한 역할**
- **마이크로서비스 아키텍처 실습에 적합**

## 결정 기준

**별도 유지가 나은 경우:**
- ✅ 학습 목적 (현재)
- ✅ 높은 처리량
- ✅ 독립 확장 필요
- ✅ 프로덕션 환경

**통합이 나은 경우:**
- ✅ 빠른 프로토타입
- ✅ 리소스 제약
- ✅ 간단한 데이터 흐름
- ✅ 개발 환경만 사용
