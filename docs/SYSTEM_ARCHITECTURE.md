# 위성 텔레메트리 분석 시스템 - 시스템 아키텍처

## 1. 전체 시스템 구성도

```mermaid
graph TB
    subgraph "데이터 소스 계층"
        SAT1[SAT-001 Simulator]
        SAT2[SAT-002 Simulator]
        SAT3[SAT-003 Simulator]
    end

    subgraph "메시지 큐 계층"
        KAFKA[Apache Kafka KRaft<br/>Topic: satellite-telemetry<br/>Port: 9092]
    end

    subgraph "데이터 소비 계층"
        VC[Victoria Consumer<br/>시계열 저장]
        KIT[Kafka Inference Trigger<br/>추론 트리거]
    end

    subgraph "데이터베이스 & 스토리지 계층"
        VM[VictoriaMetrics<br/>시계열 DB<br/>Port: 8428]
        RMQ[RabbitMQ AMQP<br/>Queue: inference<br/>Port: 5672, 15672]
    end

    subgraph "AI 추론 계층"
        AW[Analysis Worker<br/>Celery Consumer<br/>Concurrency: 2]
        subgraph TS[Triton Inference Server<br/>HTTP: 8500, gRPC: 8501<br/>GPU: NVIDIA CUDA 13.0]
            M1[LSTM Timeseries<br/>→ EPS 서브시스템]
            M2[LSTM Timeseries<br/>→ Thermal 서브시스템]
            M3[LSTM Timeseries<br/>→ AOCS 서브시스템]
            M4[LSTM Timeseries<br/>→ Comm 서브시스템]
        end
    end

    subgraph "결과 저장 & API 계층"
        subgraph PG[PostgreSQL Database<br/>Port: 5432]
            T1[inference_jobs<br/>job_id, satellite_id, status]
            T2[subsystem_inferences<br/>subsystem, predictions, anomaly_score]
        end
        OPS[Operation Server<br/>FastAPI<br/>Port: 8000]
    end

    subgraph "웹 서비스 계층"
        NGX[Nginx Reverse Proxy<br/>Port: 80]
        FE[React Frontend<br/>Dashboard & Monitoring]
    end

    subgraph "모니터링 & 관리 도구"
        KUI[Kafka UI<br/>Port: 8080]
        RMUI[RabbitMQ Management<br/>Port: 15672]
        FLW[Flower Celery Monitor<br/>Port: 5555]
        ES[Elasticsearch<br/>Port: 9200]
    end

    SAT1 -->|텔레메트리 데이터<br/>JSON/3초| KAFKA
    SAT2 -->|텔레메트리 데이터<br/>JSON/3초| KAFKA
    SAT3 -->|텔레메트리 데이터<br/>JSON/3초| KAFKA

    KAFKA -->|Poll| VC
    KAFKA -->|Poll| KIT

    VC -->|Write| VM
    KIT -->|Celery Tasks| RMQ

    RMQ -->|Consume| AW
    AW -->|gRPC 8001| TS

    TS -->|추론 결과| PG
    T1 -.->|FK| T2

    PG -->|SQL Queries| OPS
    OPS -->|HTTP/WebSocket| NGX
    NGX -->|Route| FE

    KAFKA -.->|Monitor| KUI
    RMQ -.->|Monitor| RMUI
    AW -.->|Monitor| FLW

    style TS fill:#e1f5ff
    style PG fill:#fff4e1
    style KAFKA fill:#ffe1f5
    style RMQ fill:#ffe1f5
```

## 2. 네트워크 구성

```mermaid
graph LR
    subgraph Docker Network webnet
        KAFKA[kafka:9092]
        RMQ[rabbitmq:5672]
        PG[postgres:5432]
        TRITON[triton:8001]

        KAFKA ---|Internal<br/>Communication| RMQ
        RMQ ---|Internal<br/>Communication| PG
        PG ---|Internal<br/>Communication| TRITON
    end

    subgraph External Access
        P80[":80 → Nginx"]
        P8000[":8000 → Operation Server"]
        P8080[":8080 → Kafka UI"]
        P8428[":8428 → VictoriaMetrics"]
        P8500[":8500 → Triton HTTP"]
        P8501[":8501 → Triton gRPC"]
        P9092[":9092 → Kafka"]
        P5432[":5432 → PostgreSQL"]
        P5555[":5555 → Flower"]
        P15672[":15672 → RabbitMQ Management"]
    end

    P9092 -.->|Port Mapping| KAFKA
    P5432 -.->|Port Mapping| PG
```

## 3. 데이터 플로우

### 3.1 실시간 텔레메트리 수집 & 저장

```mermaid
sequenceDiagram
    participant SAT as Satellite Simulator
    participant KAFKA as Kafka Topic
    participant VC as Victoria Consumer
    participant VM as VictoriaMetrics TSDB

    SAT->>KAFKA: JSON (3초 간격)
    loop Poll
        VC->>KAFKA: Poll messages
        KAFKA->>VC: Telemetry data
        VC->>VM: Write metrics
    end
```

### 3.2 AI 추론 파이프라인 (자동 트리거)

```mermaid
flowchart TD
    KAFKA[Kafka Topic]
    KIT[Kafka Inference Trigger]

    subgraph Trigger Conditions
        TC1[Battery SoC < 30%]
        TC2[Temperature < -30°C or > 50°C]
        TC3[Thruster Active]
        TC4[Periodic Check 10%]
    end

    subgraph Parallel Processing
        EPS[EPS Task]
        THERMAL[Thermal Task]
        AOCS[AOCS Task]
        COMM[Comm Task]
    end

    RMQ[RabbitMQ Celery Queue]
    AW[Analysis Worker<br/>Celery Consumer]
    TRITON[Triton Inference Server<br/>GPU]

    subgraph Triton Processing
        PRE[Preprocess]
        INFER[LSTM Inference]
        POST[Postprocess]
    end

    PG[(PostgreSQL<br/>Jobs & Results)]
    KAFKA_OUT[Kafka<br/>결과 전송]

    KAFKA -->|Poll| KIT
    KIT -->|분석 조건| TC1 & TC2 & TC3 & TC4
    TC1 & TC2 & TC3 & TC4 -->|트리거 결정| EPS & THERMAL & AOCS & COMM

    EPS --> RMQ
    THERMAL --> RMQ
    AOCS --> RMQ
    COMM --> RMQ

    RMQ --> AW
    AW -->|gRPC| TRITON

    TRITON --> PRE
    PRE --> INFER
    INFER --> POST

    POST -->|결과 반환| PG
    PG --> KAFKA_OUT

    style TRITON fill:#e1f5ff
    style PG fill:#fff4e1
```

### 3.3 수동 추론 요청 (API)

```mermaid
sequenceDiagram
    participant CLIENT as Client (Web)
    participant OPS as Operation Server (API)
    participant RMQ as RabbitMQ
    participant AW as Analysis Worker
    participant TRITON as Triton Server
    participant PG as PostgreSQL

    CLIENT->>OPS: POST /api/v1/inference/submit
    OPS->>PG: INSERT status='pending'
    OPS->>RMQ: Celery Task
    OPS-->>CLIENT: 200 OK<br/>{job_id, status: pending}

    RMQ->>AW: Consume task
    AW->>TRITON: gRPC Infer
    TRITON-->>AW: predictions, confidence

    AW->>PG: UPDATE status='completed'<br/>저장 predictions, metrics

    CLIENT->>OPS: GET /api/v1/inference/result/{job_id}
    OPS->>PG: SELECT result
    PG-->>OPS: result data
    OPS-->>CLIENT: 200 OK<br/>{predictions, confidence, metrics}

    Note over CLIENT,OPS: WebSocket 실시간 업데이트
    OPS->>CLIENT: WS: {status: completed}
```

## 4. 서브시스템별 모델 매핑

```mermaid
graph LR
    subgraph Subsystems
        EPS[EPS 전력<br/>12개 특징<br/>5 스텝]
        THERMAL[Thermal<br/>6개 특징<br/>5 스텝]
        AOCS[AOCS 자세제어<br/>12개 특징<br/>3 스텝]
        COMM[Comm 통신<br/>3개 특징<br/>3 스텝]
    end

    subgraph Models
        LSTM[LSTM Timeseries Model]
    end

    EPS -->|배터리, 태양전지판<br/>전압/전류| LSTM
    THERMAL -->|각종 온도<br/>센서| LSTM
    AOCS -->|자이로, 반작용휠<br/>GPS| LSTM
    COMM -->|RSSI, 데이터<br/>백로그| LSTM

    style LSTM fill:#e1f5ff
```

### 입력 특징 상세

**EPS (12개 특징):**
```
battery_voltage, battery_soc_percent, battery_current, battery_temperature,
solar_panel_1_voltage, solar_panel_1_current,
solar_panel_2_voltage, solar_panel_2_current,
solar_panel_3_voltage, solar_panel_3_current,
total_power_consumption, total_power_generation
```

**Thermal (6개 특징):**
```
battery_temp, obc_temp, comm_temp, payload_temp,
solar_panel_temp, external_temp
```

**AOCS (12개 특징):**
```
gyro_x, gyro_y, gyro_z, sun_sensor_angle,
magnetometer_x, magnetometer_y, magnetometer_z,
reaction_wheel_1_rpm, reaction_wheel_2_rpm, reaction_wheel_3_rpm,
gps_altitude_km, gps_velocity_kmps
```

**Comm (3개 특징):**
```
rssi_dbm, data_backlog_mb, last_contact_seconds_ago
```

## 5. 주요 기술 스택

```mermaid
mindmap
  root((기술 스택))
    백엔드
      FastAPI
      Celery
      PostgreSQL
      RabbitMQ
      Apache Kafka
    AI/ML
      NVIDIA Triton
      PyTorch
      ONNX Runtime
    프론트엔드
      React
      Nginx
    데이터 저장
      VictoriaMetrics
      Elasticsearch
    모니터링
      Kafka UI
      Flower
      RabbitMQ Management
```

## 6. 컨테이너 배포 구성

```mermaid
graph TB
    subgraph "Infrastructure Services"
        KAFKA[kafka<br/>confluentinc/cp-kafka<br/>:9092]
        RMQ[rabbitmq<br/>rabbitmq:3-management<br/>:5672, 15672]
        PG[postgres<br/>postgres:latest<br/>:5432]
        ES[elasticsearch<br/>elasticsearch:8.5.0<br/>:9200]
    end

    subgraph "AI Services"
        TRITON[triton-server<br/>커스텀 빌드<br/>:8500-8502<br/>GPU]
        AW[analysis-worker-1<br/>커스텀 빌드<br/>Celery Worker]
        KIT[kafka-inference-trigger<br/>커스텀 빌드<br/>추론 트리거]
    end

    subgraph "API & Web Services"
        OPS[operation-server<br/>커스텀 빌드<br/>:8000]
        FE[frontend<br/>satlas-ui:latest<br/>React UI]
        NGX[nginx<br/>nginx:alpine<br/>:80]
    end

    subgraph "Monitoring Services"
        KUI[kafka-ui<br/>provectuslabs/kafka-ui<br/>:8080]
        FLW[flower<br/>커스텀 빌드<br/>:5555]
    end

    subgraph "Data Storage Services"
        VM[victoria-metrics<br/>victoriametrics/victoria-metrics<br/>:8428]
        VC[victoria-consumer<br/>커스텀 빌드<br/>시계열 저장]
    end

    KAFKA --> KIT
    KAFKA --> VC
    KIT --> RMQ
    RMQ --> AW
    AW --> TRITON
    AW --> PG
    VC --> VM
    OPS --> PG
    OPS --> ES
    NGX --> OPS
    NGX --> FE

    style TRITON fill:#e1f5ff
    style PG fill:#fff4e1
```

| 서비스명 | 이미지 | 포트 | 역할 |
|---------|--------|------|------|
| kafka | confluentinc/cp-kafka | 9092 | 메시지 스트리밍 |
| rabbitmq | rabbitmq:3-management | 5672, 15672 | 작업 큐 |
| postgres | postgres:latest | 5432 | 데이터베이스 |
| triton-server | 커스텀 빌드 | 8500-8502 | GPU 추론 |
| analysis-worker-1 | 커스텀 빌드 | - | Celery Worker |
| kafka-inference-trigger | 커스텀 빌드 | - | 추론 트리거 |
| operation-server | 커스텀 빌드 | 8000 | API 서버 |
| victoria-metrics | victoriametrics/victoria-metrics | 8428 | 시계열 DB |
| victoria-consumer | 커스텀 빌드 | - | 시계열 저장 |
| frontend | satlas-ui:latest | - | React UI |
| nginx | nginx:alpine | 80 | 웹 서버 |
| flower | 커스텀 빌드 | 5555 | Celery 모니터 |
| kafka-ui | provectuslabs/kafka-ui | 8080 | Kafka 모니터 |
| elasticsearch | elasticsearch:8.5.0 | 9200 | 검색 엔진 |

## 7. 확장성 및 성능

```mermaid
graph LR
    subgraph Horizontal Scaling
        AW1[Analysis Worker 1]
        AW2[Analysis Worker 2]
        AW3[Analysis Worker 3]
        AW4[Analysis Worker 4]
    end

    subgraph Kafka Partitions
        P1[Partition 0]
        P2[Partition 1]
        P3[Partition 2]
        P4[Partition 3]
    end

    subgraph Load Balancing
        RMQ[RabbitMQ<br/>Task Distribution]
    end

    P1 & P2 & P3 & P4 --> RMQ
    RMQ --> AW1 & AW2 & AW3 & AW4
    AW1 & AW2 & AW3 & AW4 -->|gRPC| TRITON[Triton Server<br/>Dynamic Batching<br/>GPU: CUDA 13.0]

    style TRITON fill:#e1f5ff
```

### 수평 확장
- **Analysis Worker**: `docker compose up --scale analysis-worker-1=4`로 워커 수 증가
- **Kafka Partitions**: 토픽 파티션 수 증가로 처리량 향상
- **Celery Concurrency**: 워커당 동시 처리 수 조정

### GPU 활용
- Triton Server는 Dynamic Batching으로 여러 요청을 자동으로 배치 처리
- CUDA 13.0 기반 최적화된 추론

### 캐싱 & 최적화
- PostgreSQL 인덱스: job_id, satellite_id, subsystem, created_at
- RabbitMQ persistent delivery mode
- Celery task acks_late=True로 안정성 확보

## 8. 장애 복구 & 안정성

```mermaid
graph TD
    subgraph Health Checks
        HC1[Triton Server<br/>HTTP health endpoint]
        HC2[RabbitMQ<br/>rabbitmq-diagnostics ping]
        HC3[PostgreSQL<br/>자동 재시작]
    end

    subgraph Data Persistence
        V1[PostgreSQL<br/>postgres_data 볼륨]
        V2[Kafka<br/>kafka_data 볼륨]
        V3[VictoriaMetrics<br/>victoria_data 볼륨]
        V4[Elasticsearch<br/>es_data 볼륨]
    end

    subgraph Error Handling
        E1[Celery task retry]
        E2[PostgreSQL ON CONFLICT<br/>동시성 해결]
        E3[restart: unless-stopped<br/>자동 재시작]
    end

    HC1 & HC2 & HC3 --> MONITOR[Monitoring System]
    V1 & V2 & V3 & V4 --> STORAGE[Persistent Storage]
    E1 & E2 & E3 --> RELIABILITY[System Reliability]

    style MONITOR fill:#e1ffe1
    style STORAGE fill:#fff4e1
    style RELIABILITY fill:#ffe1e1
```

### Health Checks
- Triton Server: HTTP health endpoint
- RabbitMQ: rabbitmq-diagnostics ping
- PostgreSQL: 자동 재시작

### 데이터 영속성
- PostgreSQL: `postgres_data` 볼륨
- Kafka: `kafka_data` 볼륨
- VictoriaMetrics: `victoria_data` 볼륨
- Elasticsearch: `es_data` 볼륨

### 에러 핸들링
- Celery task retry 메커니즘
- PostgreSQL ON CONFLICT 처리로 동시성 문제 해결
- 모든 서비스에 restart: unless-stopped 정책
