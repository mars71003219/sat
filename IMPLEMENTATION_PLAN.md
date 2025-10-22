# 인공위성 시계열 데이터 시뮬레이션 및 트렌드 모니터링 시스템 구현 계획

## 📋 요구사항 분석

### 1. 데이터 생성 및 저장
- ✅ 현재: data_simulator.py가 랜덤 시계열 데이터 생성
- 🎯 목표: 인공위성 데이터 특성을 반영한 시계열 생성 (온도, 고도, 속도 등)
- 🎯 목표: Kafka를 통해 VictoriaMetrics 시계열 DB에 저장

### 2. 시계열 데이터베이스
- 🎯 추가: VictoriaMetrics (고성능 시계열 DB)
- 🎯 기능: 장기 데이터 보관, 빠른 범위 쿼리, 메트릭 집계

### 3. 트렌드 시각화
- 🎯 원본 데이터 트렌드 디스플레이 (기간 조회)
- 🎯 예측 결과 트렌드 디스플레이 (기간 조회)
- 🎯 원본 vs 예측 동시 비교 뷰
- 🎯 다크 테마 기반 현대적 UI (Stitch 컨셉 참고)

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     Satellite Data Simulator                     │
│  (인공위성 센서 데이터: 온도, 고도, 속도, 배터리, 태양광 등)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Kafka Topic                             │
│              satellite-telemetry-data (원본 데이터)              │
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐     ┌────────────────────────────────┐
│   VictoriaMetrics      │     │   Operation Server API         │
│  (시계열 데이터 저장)   │     │  (추론 요청 제출)              │
└────────────────────────┘     └────────────┬───────────────────┘
             │                               │
             │                               ▼
             │                  ┌────────────────────────────────┐
             │                  │    Triton Inference Server     │
             │                  │   (VAE/Transformer 예측)       │
             │                  └────────────┬───────────────────┘
             │                               │
             │                               ▼
             │                  ┌────────────────────────────────┐
             │                  │      PostgreSQL                │
             │                  │   (예측 결과 저장)             │
             │                  └────────────┬───────────────────┘
             │                               │
             └───────────────────────────────┼────────────────────┐
                                             ▼                    │
                              ┌──────────────────────────────┐   │
                              │   Trend Visualization API    │   │
                              │  - 원본 데이터 트렌드 조회   │◄──┘
                              │  - 예측 결과 트렌드 조회     │
                              │  - 비교 분석                 │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │   React Dashboard (다크 테마) │
                              │  - 실시간 데이터 차트        │
                              │  - 원본 vs 예측 비교         │
                              │  - 기간 선택 필터            │
                              └──────────────────────────────┘
```

## 📦 추가 구성요소

### 1. VictoriaMetrics
```yaml
# docker-compose.yml 추가
victoria-metrics:
  image: victoriametrics/victoria-metrics:latest
  ports:
    - "8428:8428"
  volumes:
    - victoria_data:/victoria-metrics-data
  command:
    - '--storageDataPath=/victoria-metrics-data'
    - '--retentionPeriod=1y'
```

### 2. Kafka Consumer (VictoriaMetrics Writer)
```python
# victoria-consumer/consumer.py
# Kafka에서 시계열 데이터를 읽어 VictoriaMetrics에 저장
```

### 3. 인공위성 데이터 시뮬레이터 업데이트
```python
# tests/data_simulator.py 업데이트
class SatelliteDataSimulator:
    """
    인공위성 텔레메트리 데이터 생성:
    - 온도 (Temperature): -50°C ~ 50°C, 열 사이클 패턴
    - 고도 (Altitude): 400km ~ 450km, 궤도 변동
    - 속도 (Velocity): 7.6km/s ~ 7.8km/s
    - 배터리 전압 (Battery): 3.0V ~ 4.2V, 충/방전 사이클
    - 태양광 패널 출력 (Solar): 0W ~ 100W, 지구 그림자 영향
    """
```

### 4. 트렌드 조회 API
```python
# operation-server/api/routes/trends.py
@router.get("/trends/raw")
async def get_raw_data_trend(
    metric: str,
    start_time: datetime,
    end_time: datetime
):
    """VictoriaMetrics에서 원본 데이터 조회"""

@router.get("/trends/prediction")
async def get_prediction_trend(
    model_name: str,
    start_time: datetime,
    end_time: datetime
):
    """PostgreSQL에서 예측 결과 조회"""

@router.get("/trends/compare")
async def compare_trends(...):
    """원본 vs 예측 비교"""
```

### 5. React 대시보드 (다크 테마)
```typescript
// frontend/src/components/TrendDashboard.tsx
// - Recharts 또는 Chart.js 사용
// - 다크 테마: background #191a1f, text #fff
// - 시간 범위 선택 (1h, 6h, 1d, 1w, 1m, custom)
// - 실시간 업데이트 (WebSocket)
```

## 🎨 UI 디자인 컨셉 (Stitch 참고)

### 색상 스킴
```css
:root {
  --bg-primary: #191a1f;      /* 다크 배경 */
  --bg-secondary: #25262b;    /* 카드/패널 배경 */
  --text-primary: #ffffff;    /* 메인 텍스트 */
  --text-secondary: #a0a0a0;  /* 서브 텍스트 */
  --accent-blue: #4a9eff;     /* 액센트 컬러 (원본 데이터) */
  --accent-green: #4ade80;    /* 예측 데이터 */
  --accent-red: #f87171;      /* 경고/이상치 */
  --border-color: #3a3a3f;    /* 테두리 */
}
```

### 레이아웃
```
┌────────────────────────────────────────────────────────────┐
│  [🛰️ Satellite Monitor]  [1h] [6h] [1d] [1w] [Custom▼]   │ ← 헤더 (48px)
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │ Temperature     │  │ Altitude        │                │
│  │ [실시간 차트]   │  │ [실시간 차트]   │  ← 메트릭 카드
│  │ Raw | Pred     │  │ Raw | Pred     │                │
│  └─────────────────┘  └─────────────────┘                │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │           Temperature Trend Comparison               │ │
│  │  [━━━━ Raw Data]  [- - - - Prediction]             │ │
│  │  [차트 영역: 시간별 온도 변화 + 예측값 오버레이]    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │             Model Performance Metrics                │ │
│  │  MAE: 0.85  RMSE: 1.23  R²: 0.92                   │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

## 📅 구현 단계

### Phase 1: 인프라 구축 ✅ 완료
- [x] VictoriaMetrics Docker 서비스 추가
- [x] Kafka Consumer (VictoriaMetrics Writer) 구현
- [x] docker-compose.yml 업데이트 및 테스트

### Phase 2: 데이터 시뮬레이터 ✅ 완료
- [x] 인공위성 센서 데이터 생성 로직 구현
- [x] Kafka 메시지 포맷 정의
- [x] 시뮬레이터 → Kafka → VictoriaMetrics 파이프라인 구현

### Phase 3: API 구현 ✅ 완료
- [x] 트렌드 조회 API 엔드포인트 구현
  - `/api/v1/trends/raw`
  - `/api/v1/trends/prediction`
  - `/api/v1/trends/compare`
  - `/api/v1/trends/metrics`
  - `/api/v1/trends/satellites`
- [x] 시간 범위 필터링 로직
- [x] 데이터 집계 (평균, 최대, 최소)

### Phase 4: 프론트엔드 ✅ 완료
- [x] React 대시보드 컴포넌트 구조 설계
- [x] 다크 테마 CSS (#191a1f 배경, #fff 텍스트)
- [x] 차트 라이브러리 통합 (Recharts)
- [x] 실시간 업데이트 (30초 폴링)
- [x] 시간 범위 선택 UI (1h, 6h, 1d, 1w)
- [x] 원본 vs 예측 비교 차트

### Phase 5: 통합 테스트 (다음 단계)
- [ ] 전체 파이프라인 End-to-End 테스트
- [ ] 성능 테스트 (대량 데이터 처리)
- [ ] UI/UX 최종 조정

## 📊 데이터 스키마

### Kafka Message (satellite-telemetry-data)
```json
{
  "timestamp": "2025-10-15T12:00:00Z",
  "satellite_id": "SAT-001",
  "metrics": {
    "temperature": 23.5,
    "altitude": 425.3,
    "velocity": 7.66,
    "battery_voltage": 3.8,
    "solar_power": 85.2
  },
  "location": {
    "latitude": 37.5,
    "longitude": 127.0
  }
}
```

### VictoriaMetrics (Time Series)
```
satellite_temperature{satellite_id="SAT-001"} 23.5 @timestamp
satellite_altitude{satellite_id="SAT-001"} 425.3 @timestamp
satellite_velocity{satellite_id="SAT-001"} 7.66 @timestamp
...
```

### PostgreSQL (Predictions)
```sql
-- 기존 inference_results 테이블에 추가 컬럼
ALTER TABLE inference_results
ADD COLUMN satellite_id VARCHAR(50),
ADD COLUMN metric_type VARCHAR(50),
ADD COLUMN input_timestamp TIMESTAMP;
```

## 🚀 빠른 시작 (구현 후)

```bash
# 1. 전체 서비스 시작 (VictoriaMetrics 포함)
docker compose up -d

# 2. 인공위성 데이터 시뮬레이터 실행
cd tests
python3 data_simulator.py --mode satellite --interval 5

# 3. 대시보드 접속
http://localhost/trends
```

## 📝 다음 스텝

1. VictoriaMetrics 추가부터 시작
2. Kafka Consumer 구현
3. 시뮬레이터 업데이트
4. API 개발
5. 프론트엔드 구현

진행 순서대로 하나씩 구현하겠습니다!
