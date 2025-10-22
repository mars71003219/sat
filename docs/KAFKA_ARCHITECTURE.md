# Kafka 아키텍처 설명

## 질문에 대한 답변

**지금 카프카 서버는 1개입니다. 그 안에 3개의 토픽이 있습니다.**

## 현재 구조

```
┌─────────────────────────────────────────────────────────────┐
│                  Kafka Broker (1개)                          │
│                 Container: kafka                             │
│                 Port: 9092                                   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Topic 1: satellite-telemetry                       │    │
│  │ - Partition: 1개                                   │    │
│  │ - 용도: 위성 텔레메트리 데이터                     │    │
│  │ - Producer: satellite_simulator.py                 │    │
│  │ - Consumer: victoria-consumer                      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Topic 2: inference.results                         │    │
│  │ - Partition: 1개                                   │    │
│  │ - 용도: Triton 추론 결과                          │    │
│  │ - Producer: analysis-server                        │    │
│  │ - Consumer: operation-server (아마도)              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Topic 3: __consumer_offsets (시스템 토픽)         │    │
│  │ - Partition: 50개                                  │    │
│  │ - 용도: Consumer offset 추적 (자동 생성)          │    │
│  │ - Kafka 내부 사용                                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Kafka 기본 개념

### 1. Kafka Broker (서버)
- **현재 개수: 1개**
- **역할**: 메시지를 저장하고 전달하는 서버
- **컨테이너 이름**: `kafka`
- **포트**: 9092

### 2. Topic (토픽)
- **현재 개수: 3개**
  1. `satellite-telemetry` - 우리가 만든 것
  2. `inference.results` - 기존에 있던 것
  3. `__consumer_offsets` - Kafka 시스템이 자동 생성

- **비유**: 우체통 또는 메시지 큐
- **역할**: 메시지를 카테고리별로 분류

### 3. Partition (파티션)
- **satellite-telemetry**: 1개 파티션
- **inference.results**: 1개 파티션
- **__consumer_offsets**: 50개 파티션

- **비유**: 우체통 안의 칸막이
- **역할**: 병렬 처리 및 확장성

## 데이터 흐름

### 위성 데이터 파이프라인

```
satellite_simulator.py
       |
       | (produce)
       v
┌──────────────────┐
│ Kafka Broker     │
│  Topic:          │
│  satellite-      │
│  telemetry       │
└──────────────────┘
       |
       | (consume)
       v
victoria-consumer
       |
       v
VictoriaMetrics
```

### 추론 결과 파이프라인 (기존)

```
analysis-server (Triton 추론)
       |
       | (produce)
       v
┌──────────────────┐
│ Kafka Broker     │
│  Topic:          │
│  inference.      │
│  results         │
└──────────────────┘
       |
       | (consume?)
       v
operation-server (추정)
```

## Producer vs Consumer

### Producer (메시지 생산자)
```python
# satellite_simulator.py
producer = Producer({'bootstrap.servers': 'kafka:9092'})
producer.produce(
    topic='satellite-telemetry',
    value=json.dumps(telemetry_data)
)
```

### Consumer (메시지 소비자)
```python
# victoria-consumer/consumer.py
consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'victoria-consumer-group'
})
consumer.subscribe(['satellite-telemetry'])
```

## 실제 메시지 확인

### satellite-telemetry 토픽의 메시지

```bash
docker exec kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic satellite-telemetry \
    --from-beginning \
    --max-messages 1
```

**출력 예시:**
```json
{
  "timestamp": "2025-10-22T03:58:50.973550+00:00",
  "satellite_id": "SAT-001",
  "metrics": {
    "temperature": -20.81,
    "altitude": 449.95,
    "velocity": 7.655,
    "battery_voltage": 3.59,
    "solar_power": 0.0
  },
  "location": {
    "latitude": 0.2044,
    "longitude": -29.8243
  }
}
```

## Kafka 클러스터 vs 단일 브로커

### 현재 구조 (단일 브로커)
```
┌──────────────┐
│ Kafka Broker │
│   (Node 1)   │
│   Port 9092  │
└──────────────┘
```

**특징:**
- ✅ 간단한 설정
- ✅ 개발/테스트에 적합
- ❌ 장애 복구 불가
- ❌ 확장성 제한

### 프로덕션 구조 (클러스터)
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Kafka Broker │  │ Kafka Broker │  │ Kafka Broker │
│   (Node 1)   │  │   (Node 2)   │  │   (Node 3)   │
│   Port 9092  │  │   Port 9093  │  │   Port 9094  │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                    Kafka Cluster
```

**특징:**
- ✅ 고가용성 (HA)
- ✅ 장애 복구 (Failover)
- ✅ 수평 확장
- ✅ 데이터 복제 (Replication)

## Replication Factor

### 현재 설정
```
satellite-telemetry: ReplicationFactor: 1
```

**의미**: 데이터 복사본이 1개만 존재 (원본만)

### 프로덕션 권장
```
ReplicationFactor: 3
```

**의미**: 데이터가 3개 브로커에 복제됨
- 브로커 2개가 죽어도 데이터 유지

## 토픽별 용도 정리

| 토픽 | 용도 | Producer | Consumer | 메시지 예시 |
|------|------|----------|----------|-------------|
| **satellite-telemetry** | 위성 센서 데이터 | satellite_simulator.py | victoria-consumer | 온도, 고도, 속도 등 |
| **inference.results** | Triton 추론 결과 | analysis-server | operation-server? | 예측값, 모델 출력 |
| **__consumer_offsets** | Consumer 오프셋 추적 | Kafka 내부 | Kafka 내부 | offset=1234 |

## 확인 명령어

### 1. 토픽 목록 보기
```bash
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```

### 2. 토픽 상세 정보
```bash
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --describe --topic satellite-telemetry
```

### 3. 메시지 보기 (처음부터)
```bash
docker exec kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic satellite-telemetry \
    --from-beginning
```

### 4. 메시지 개수 확인
```bash
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 \
    --topic satellite-telemetry
```

## 결론

**정답:**
- **Kafka 서버 (Broker): 1개**
- **토픽 (Topic): 3개**
  - satellite-telemetry (우리가 만든 것)
  - inference.results (기존)
  - __consumer_offsets (시스템)

**비유:**
- Kafka Broker = 우체국 (1개)
- Topic = 우체통 (3개)
- Partition = 우체통 내부 칸막이
- Message = 편지

**1개의 Kafka 서버가 여러 개의 토픽을 관리하고 있습니다!**
