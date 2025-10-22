# UML 다이어그램

이 문서는 시스템의 구조와 동작을 시각화한 UML 다이어그램을 포함합니다.

## 1. 시퀀스 다이어그램

### 1.1 추론 요청 시퀀스

```mermaid
sequenceDiagram
    participant C as Client
    participant N as Nginx
    participant O as Operation Server
    participant R as Redis (Broker)
    participant A as Analysis Worker
    participant P as PostgreSQL
    participant K as Kafka

    C->>N: POST /api/v1/inference/submit
    N->>O: Forward request

    Note over O: Validate request
    O->>O: Generate job_id

    O->>R: publish task (queue: inference)
    R-->>O: task_id

    O->>P: INSERT job status (pending)
    P-->>O: Success

    O-->>N: 200 OK {job_id, status: queued}
    N-->>C: Response

    Note over R,A: Async Task Processing

    R->>A: Fetch task from queue
    Note over A: Task received

    A->>A: Load model (LSTM/MA)
    A->>A: Run inference
    A->>A: Calculate metrics

    par Save to PostgreSQL
        A->>P: UPDATE inference_results
        P-->>A: Success
    and Cache to Redis
        A->>R: SET cache (DB 0)
        R-->>A: OK
    and Publish to Kafka
        A->>K: SEND inference_results topic
        K-->>A: Acknowledged
    end

    Note over A: Task completed
```

### 1.2 대시보드 실시간 업데이트 시퀀스

```mermaid
sequenceDiagram
    participant B as Browser
    participant N as Nginx
    participant O as Operation Server
    participant P as PostgreSQL
    participant R as Redis

    B->>N: WebSocket: ws://localhost/api/v1/dashboard/ws
    N->>O: Upgrade to WebSocket
    O-->>N: WebSocket Connected
    N-->>B: WebSocket Connected

    Note over B: Display [CONNECTED]

    loop Every 2 seconds
        Note over O: Fetch latest data

        O->>P: SELECT recent_results (LIMIT 10)
        P-->>O: results[]

        O->>P: SELECT statistics
        P-->>O: {total, completed, failed, success_rate}

        O->>O: Convert datetime to string
        O->>O: Convert Decimal to float

        O->>N: send_json({type: update, ...})
        N->>B: WebSocket message

        Note over B: Update dashboard UI
        B->>B: Update stats cards
        B->>B: Update results table
    end

    B->>N: Close connection
    N->>O: WebSocket disconnect
    Note over O: Remove from active_connections
```

### 1.3 데이터 시뮬레이터 플로우

```mermaid
sequenceDiagram
    participant S as Data Simulator
    participant O as Operation Server
    participant A1 as Analysis Worker
    participant A2 as Analysis Worker
    participant P as PostgreSQL

    loop Every N seconds
        S->>S: Generate time series data
        S->>S: Select random pattern

        Note over S: Submit to both models

        par LSTM Model
            S->>O: POST /api/v1/inference/submit<br/>(model: lstm_timeseries)
            O->>A1: Celery task
            A1->>A1: LSTM inference
            A1->>P: Save result
        and Moving Average Model
            S->>O: POST /api/v1/inference/submit<br/>(model: moving_average)
            O->>A2: Celery task
            A2->>A2: MA calculation
            A2->>P: Save result
        end

        Note over S: Wait random interval (2-5s)
    end
```

## 2. 클래스 다이어그램

### 2.1 Operation Server 클래스 구조

```mermaid
classDiagram
    class FastAPIApp {
        +app: FastAPI
        +include_router()
        +run()
    }

    class InferenceRouter {
        +prefix: /api/v1/inference
        +submit_inference(request)
        +get_result(job_id)
        +get_history()
    }

    class DashboardRouter {
        +prefix: /api/v1/dashboard
        +dashboard_websocket(websocket)
        +get_recent_results(limit)
        +get_live_statistics()
        +get_model_comparison()
        +get_pattern_distribution()
    }

    class DashboardManager {
        -active_connections: List[WebSocket]
        +connect(websocket)
        +disconnect(websocket)
        +broadcast(message)
    }

    class PostgresClient {
        -connection: Connection
        -pool: ConnectionPool
        +save_inference_result(result)
        +get_inference_result(job_id)
        +get_results_history(limit)
        +get_recent_results(limit)
        +get_statistics()
        +query(sql, params)
    }

    class RedisClient {
        -client: Redis
        +set(key, value, ttl)
        +get(key)
        +delete(key)
        +publish(channel, message)
    }

    class KafkaProducer {
        -producer: Producer
        +send(topic, message)
        +flush()
        +close()
    }

    class CeleryClient {
        -app: Celery
        +send_task(name, args, kwargs)
        +get_result(task_id)
    }

    FastAPIApp --> InferenceRouter
    FastAPIApp --> DashboardRouter
    InferenceRouter --> CeleryClient
    InferenceRouter --> PostgresClient
    DashboardRouter --> PostgresClient
    DashboardRouter --> DashboardManager
    InferenceRouter --> RedisClient
    InferenceRouter --> KafkaProducer
```

### 2.2 Analysis Worker 클래스 구조

```mermaid
classDiagram
    class CeleryApp {
        +app: Celery
        +task()
        +worker_main()
    }

    class InferenceTask {
        +inference_task(model_name, data, metadata)
        -_load_model(model_name)
        -_run_inference(model, data)
        -_calculate_metrics(predictions, data)
    }

    class LSTMModel {
        -model: torch.nn.Module
        -device: torch.device
        +load_weights(path)
        +predict(data)
        +preprocess(data)
        +postprocess(output)
    }

    class MovingAverageModel {
        -window_size: int
        +predict(data)
        +calculate(data, window)
    }

    class ModelRegistry {
        -models: Dict[str, Model]
        +register(name, model)
        +get(name)
        +list_models()
    }

    class ResultSaver {
        +save_to_postgres(result)
        +save_to_redis(result)
        +save_to_kafka(result)
        +save_all(result)
    }

    CeleryApp --> InferenceTask
    InferenceTask --> ModelRegistry
    InferenceTask --> ResultSaver
    ModelRegistry --> LSTMModel
    ModelRegistry --> MovingAverageModel
    ResultSaver --> PostgresClient
    ResultSaver --> RedisClient
    ResultSaver --> KafkaProducer
```

### 2.3 데이터 모델 클래스

```mermaid
classDiagram
    class InferenceRequest {
        +model_name: str
        +data: List[float]
        +metadata: Dict[str, Any]
        +validate()
    }

    class InferenceResult {
        +job_id: str
        +model_name: str
        +status: str
        +predictions: List[float]
        +confidence: Dict[str, float]
        +metrics: Dict[str, float]
        +metadata: Dict[str, Any]
        +created_at: datetime
        +completed_at: datetime
        +to_dict()
        +from_dict(data)
    }

    class DashboardUpdate {
        +type: str
        +timestamp: str
        +recent_results: List[InferenceResult]
        +statistics: Statistics
        +to_json()
    }

    class Statistics {
        +total_jobs: int
        +completed: int
        +failed: int
        +success_rate: float
        +models_used: Dict[str, int]
        +throughput_per_minute: int
        +calculate()
    }

    class TimeSeriesData {
        +pattern: str
        +length: int
        +values: List[float]
        +generate(pattern)
        +validate()
    }

    InferenceRequest --> TimeSeriesData
    InferenceResult --> TimeSeriesData
    DashboardUpdate --> InferenceResult
    DashboardUpdate --> Statistics
```

## 3. 컴포넌트 다이어그램

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Browser[Web Browser]
        Dashboard[AI Dashboard]
        KafkaUI[Kafka UI]
        RedisUI[Redis Insight]
        FlowerUI[Flower UI]
    end

    subgraph Proxy["Proxy Layer"]
        Nginx[Nginx<br/>Port 80]
    end

    subgraph Application["Application Layer"]
        OpServer[Operation Server<br/>FastAPI<br/>Port 8000]
        Worker1[Analysis Worker 1<br/>Celery + GPU]
        Worker2[Analysis Worker 2<br/>Celery + GPU]
    end

    subgraph Infrastructure["Infrastructure Layer"]
        Redis[(Redis<br/>Port 6379)]
        Postgres[(PostgreSQL<br/>Port 5432)]
        Kafka[(Kafka<br/>Port 9092)]
        Elastic[(Elasticsearch<br/>Port 9200)]
    end

    subgraph Models["AI Models"]
        LSTM[LSTM Model<br/>PyTorch]
        MA[Moving Average]
    end

    Browser --> Nginx
    Dashboard --> Nginx
    KafkaUI --> Kafka
    RedisUI --> Redis
    FlowerUI --> Redis

    Nginx --> OpServer

    OpServer --> Redis
    OpServer --> Postgres
    OpServer --> Kafka
    OpServer --> Elastic

    Redis --> Worker1
    Redis --> Worker2

    Worker1 --> Postgres
    Worker1 --> Redis
    Worker1 --> Kafka
    Worker1 --> LSTM
    Worker1 --> MA

    Worker2 --> Postgres
    Worker2 --> Redis
    Worker2 --> Kafka
    Worker2 --> LSTM
    Worker2 --> MA
```

## 4. 상태 다이어그램

### 4.1 추론 작업 상태 전이

```mermaid
stateDiagram-v2
    [*] --> Submitted: Client submits request

    Submitted --> Queued: Task queued in Celery

    Queued --> Processing: Worker picks up task

    Processing --> Running: Model inference starts

    Running --> Completed: Success
    Running --> Failed: Error occurs

    Completed --> [*]: Results saved
    Failed --> [*]: Error logged

    Processing --> Retrying: Temporary failure
    Retrying --> Processing: Retry attempt
    Retrying --> Failed: Max retries exceeded
```

### 4.2 WebSocket 연결 상태

```mermaid
stateDiagram-v2
    [*] --> Disconnected: Initial state

    Disconnected --> Connecting: Client initiates connection

    Connecting --> Connected: WebSocket handshake success
    Connecting --> Disconnected: Connection failed

    Connected --> Sending: Server sends update
    Sending --> Connected: Message sent
    Sending --> Error: Send failed

    Connected --> Disconnected: Client closes
    Error --> Reconnecting: Auto reconnect
    Reconnecting --> Connecting: Retry connection

    Disconnected --> [*]
```

## 5. 배포 다이어그램

```mermaid
graph TB
    subgraph DockerHost["Docker Host"]
        subgraph Network["Docker Network: webnet"]
            subgraph Services["Services"]
                C1[kafka<br/>confluentinc/cp-kafka]
                C2[redis<br/>redis/redis-stack]
                C3[postgres<br/>postgres:latest]
                C4[elasticsearch<br/>elasticsearch:8.5.0]
                C5[operation-server<br/>Custom Build]
                C6[analysis-worker-1<br/>Custom Build<br/>GPU]
                C7[kafka-ui<br/>provectuslabs/kafka-ui]
                C8[flower<br/>Custom Build]
                C9[nginx<br/>nginx:alpine]
            end

            subgraph Volumes["Volumes"]
                V1[(kafka_data)]
                V2[(postgres_data)]
                V3[(es_data)]
            end
        end

        subgraph HostPorts["Host Ports"]
            P1[80]
            P2[5555]
            P3[6379]
            P4[8001]
            P5[8000]
            P6[8080]
            P7[9092]
            P8[5432]
            P9[9200]
        end
    end

    C1 --> V1
    C3 --> V2
    C4 --> V3

    P1 --> C9
    P2 --> C8
    P3 --> C2
    P4 --> C2
    P5 --> C5
    P6 --> C7
    P7 --> C1
    P8 --> C3
    P9 --> C4
```

## 6. 활동 다이어그램

### 6.1 데이터 처리 파이프라인

```mermaid
flowchart TD
    Start([Start]) --> Generate[Generate Time Series Data]
    Generate --> SelectPattern{Select Pattern}

    SelectPattern -->|Linear| Linear[Linear Pattern]
    SelectPattern -->|Seasonal| Seasonal[Seasonal Pattern]
    SelectPattern -->|Exponential| Exponential[Exponential Pattern]
    SelectPattern -->|Cyclical| Cyclical[Cyclical Pattern]
    SelectPattern -->|Random Walk| Random[Random Walk Pattern]

    Linear --> Submit[Submit to API]
    Seasonal --> Submit
    Exponential --> Submit
    Cyclical --> Submit
    Random --> Submit

    Submit --> Queue[Add to Celery Queue]

    Queue --> WorkerPick[Worker Picks Task]

    WorkerPick --> LoadModel[Load Model]

    LoadModel --> ModelType{Model Type}

    ModelType -->|LSTM| LSTM[LSTM Inference]
    ModelType -->|MA| MA[Moving Average Calc]

    LSTM --> CalcMetrics[Calculate Metrics]
    MA --> CalcMetrics

    CalcMetrics --> SaveDB[Save to PostgreSQL]
    SaveDB --> CacheRedis[Cache to Redis]
    CacheRedis --> PublishKafka[Publish to Kafka]

    PublishKafka --> UpdateDash[Update Dashboard]

    UpdateDash --> Wait[Wait Interval]
    Wait --> Generate
```

## 다이어그램 사용 방법

이 문서의 다이어그램들은 Mermaid 문법을 사용합니다. 다음 도구들을 통해 시각화할 수 있습니다:

1. **GitHub**: GitHub에서 자동으로 렌더링
2. **VS Code**: Mermaid Preview 확장 설치
3. **Online**: https://mermaid.live/
4. **Markdown Viewers**: 대부분의 모던 마크다운 뷰어 지원

## 다이어그램 업데이트

시스템 구조 변경 시 해당 다이어그램을 함께 업데이트해야 합니다:
- 새로운 서비스 추가
- API 엔드포인트 변경
- 데이터 플로우 수정
- 상태 전이 변경
