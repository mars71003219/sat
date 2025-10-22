# Triton Inference Server 통합 테스트 가이드

## 목차
1. [테스트 환경 준비](#테스트-환경-준비)
2. [단계별 테스트](#단계별-테스트)
3. [성능 테스트](#성능-테스트)
4. [문제 해결](#문제-해결)

---

## 테스트 환경 준비

### 1. 시스템 시작

```bash
# 1. Kafka 초기화 (처음 한 번만)
./init-kafka.sh

# 2. Docker Compose로 시스템 시작
docker-compose up -d

# 3. 모든 서비스 시작 대기 (약 30초)
sleep 30

# 4. 서비스 상태 확인
docker-compose ps
```

### 2. 서비스 확인

```bash
# 모든 서비스가 "Up" 상태여야 함
docker-compose ps

# 기대 결과:
# NAME                  STATUS
# triton-server         Up (healthy)
# analysis-worker-1     Up
# operation-server      Up
# kafka                 Up
# redis                 Up
# postgres              Up
```

---

## 단계별 테스트

### Step 1: Triton Server Health Check

#### 1.1 서버 Live 확인
```bash
curl http://localhost:8500/v2/health/live
```

**기대 결과**:
```json
{"status": "live"}
```

#### 1.2 서버 Ready 확인
```bash
curl http://localhost:8500/v2/health/ready
```

**기대 결과**:
```json
{"status": "ready"}
```

#### 1.3 모델 목록 확인
```bash
curl http://localhost:8500/v2 | jq
```

**기대 결과**:
```json
{
  "name": "triton",
  "version": "24.01",
  "extensions": ["classification", "sequence", "model_repository", ...]
}
```

---

### Step 2: 모델 상태 확인

#### 2.1 LSTM 모델 확인
```bash
curl http://localhost:8500/v2/models/lstm_timeseries | jq
```

**기대 결과**:
```json
{
  "name": "lstm_timeseries",
  "versions": ["1"],
  "platform": "python",
  "inputs": [
    {
      "name": "INPUT",
      "datatype": "FP32",
      "shape": [-1, 1]
    }
  ],
  "outputs": [
    {
      "name": "OUTPUT",
      "datatype": "FP32",
      "shape": [-1]
    }
  ]
}
```

#### 2.2 Moving Average 모델 확인
```bash
curl http://localhost:8500/v2/models/moving_average | jq
```

**기대 결과**:
```json
{
  "name": "moving_average",
  "versions": ["1"],
  "platform": "python",
  ...
}
```

#### 2.3 모델 READY 상태 확인
```bash
curl http://localhost:8500/v2/models/lstm_timeseries/ready
curl http://localhost:8500/v2/models/moving_average/ready
```

**기대 결과** (둘 다):
```json
{"ready": true}
```

---

### Step 3: Analysis Worker 연결 테스트

#### 3.1 Worker 로그 확인
```bash
docker-compose logs analysis-worker-1 | grep -i triton
```

**기대 로그**:
```
[INFO] Analysis Server: Using Triton Inference Server
[INFO] [Triton Client] Connected to triton-server:8001
```

#### 3.2 환경변수 확인
```bash
docker-compose exec analysis-worker-1 env | grep TRITON
```

**기대 결과**:
```
TRITON_SERVER_URL=triton-server:8001
```

#### 3.3 네트워크 연결 테스트
```bash
docker-compose exec analysis-worker-1 ping -c 3 triton-server
```

**기대 결과**:
```
3 packets transmitted, 3 received, 0% packet loss
```

---

### Step 4: End-to-End 추론 테스트

#### 4.1 LSTM 모델 추론 테스트

```bash
# 1. 추론 요청 제출
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "config": {
      "sequence_length": 10,
      "forecast_steps": 5
    }
  }')

echo $RESPONSE | jq

# 2. job_id 추출
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 3. 결과 대기 (3초)
sleep 3

# 4. 결과 조회
curl -s http://localhost:8000/api/v1/inference/result/$JOB_ID | jq
```

**기대 결과**:
```json
{
  "job_id": "...",
  "status": "completed",
  "model_name": "lstm_timeseries",
  "predictions": [10.5, 11.0, 11.5, 12.0, 12.5],
  "confidence": [0.95, 0.93, 0.91, 0.89, 0.87],
  "metrics": {
    "inference_time": 0.035,
    "model_type": "LSTM",
    "forecast_steps": 5,
    "sequence_length": 10
  },
  "metadata": {
    "processed_by": "triton_server",
    "triton_client": "grpc"
  }
}
```

#### 4.2 Moving Average 모델 추론 테스트

```bash
# 1. 추론 요청
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "moving_average",
    "data": [10, 12, 15, 14, 16, 18, 17, 19, 21, 20],
    "config": {
      "window_size": 5,
      "forecast_steps": 5
    }
  }')

echo $RESPONSE | jq

# 2. job_id 추출 및 결과 조회
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
sleep 3
curl -s http://localhost:8000/api/v1/inference/result/$JOB_ID | jq
```

**기대 결과**:
```json
{
  "job_id": "...",
  "status": "completed",
  "model_name": "moving_average",
  "predictions": [20.5, 21.0, 21.5, 22.0, 22.5],
  "confidence": [0.90, 0.87, 0.84, 0.81, 0.78],
  "upper_bound": [22.0, 22.5, 23.0, 23.5, 24.0],
  "lower_bound": [19.0, 19.5, 20.0, 20.5, 21.0],
  "metrics": {
    "model_type": "Moving Average",
    "window_size": 5,
    "forecast_steps": 5
  }
}
```

---

### Step 5: 동시 요청 테스트 (배치 처리)

#### 5.1 동시 10개 요청 전송

```bash
#!/bin/bash
# test_concurrent.sh

echo "Sending 10 concurrent requests..."

for i in {1..10}; do
  (
    RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/inference/submit \
      -H "Content-Type: application/json" \
      -d "{
        \"model_name\": \"lstm_timeseries\",
        \"data\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        \"config\": {
          \"sequence_length\": 10,
          \"forecast_steps\": 5
        }
      }")

    JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
    echo "Request $i: Job ID = $JOB_ID"
  ) &
done

wait

echo "All requests sent!"
```

**실행**:
```bash
chmod +x test_concurrent.sh
./test_concurrent.sh
```

#### 5.2 Triton 배치 처리 확인

```bash
# Triton 로그에서 배치 처리 확인
docker-compose logs triton-server | grep "batch"
```

**기대 로그**:
```
[INFO] Dynamic batching: combined 8 requests into batch size 8
[INFO] Model lstm_timeseries: infer batch size 8 completed in 35ms
```

---

## 성능 테스트

### 1. 부하 테스트 (Apache Bench)

#### 1.1 설치
```bash
sudo apt-get install apache2-utils  # Ubuntu
# 또는
brew install apache2  # macOS
```

#### 1.2 100개 요청, 동시성 10
```bash
# 테스트 데이터 준비
cat > test_payload.json <<EOF
{
  "model_name": "lstm_timeseries",
  "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "config": {
    "sequence_length": 10,
    "forecast_steps": 5
  }
}
EOF

# 부하 테스트 실행
ab -n 100 -c 10 -p test_payload.json -T application/json \
  http://localhost:8000/api/v1/inference/submit
```

**기대 결과**:
```
Requests per second:    80-100 [#/sec]
Time per request:       100-125 [ms] (mean)
Time per request:       10-12 [ms] (mean, across all concurrent requests)
```

### 2. GPU 사용률 모니터링

#### 실시간 GPU 모니터링
```bash
# 터미널 1: 부하 테스트 실행
ab -n 1000 -c 50 -p test_payload.json -T application/json \
  http://localhost:8000/api/v1/inference/submit

# 터미널 2: GPU 사용률 모니터링
watch -n 0.5 nvidia-smi
```

**기대 GPU 사용률**: 70-90%

### 3. Triton Metrics 확인

```bash
# 메트릭 조회
curl http://localhost:8502/metrics | grep nv_inference

# 주요 메트릭 필터
curl http://localhost:8502/metrics | grep -E "(nv_inference_count|nv_inference_request_success|nv_gpu_utilization)"
```

**기대 메트릭**:
```
nv_inference_count{model="lstm_timeseries",version="1"} 1250
nv_inference_request_success{model="lstm_timeseries",version="1"} 1250
nv_gpu_utilization{gpu_uuid="..."} 0.85
```

---

## 문제 해결

### 문제 1: Triton Server가 시작되지 않음

**증상**:
```bash
docker-compose ps
# triton-server: Exit 1
```

**진단**:
```bash
docker-compose logs triton-server
```

**일반적인 원인 및 해결**:

#### 1. Model Repository 경로 문제
```bash
# 경로 확인
ls -la model_repository/

# 권한 확인 및 수정
chmod -R 755 model_repository/
```

#### 2. GPU 드라이버 문제
```bash
# GPU 확인
nvidia-smi

# NVIDIA Docker Runtime 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### 3. Config 파일 문법 오류
```bash
# config.pbtxt 문법 확인
cat model_repository/lstm_timeseries/config.pbtxt
```

---

### 문제 2: 모델이 READY 상태가 아님

**증상**:
```bash
curl http://localhost:8500/v2/models/lstm_timeseries/ready
# {"ready": false}
```

**진단**:
```bash
# Triton 로그에서 에러 확인
docker-compose logs triton-server | grep -A 10 "lstm_timeseries"
```

**일반적인 원인 및 해결**:

#### 1. Python 의존성 누락
```bash
# Triton 컨테이너 내부 확인
docker-compose exec triton-server bash

# Python 패키지 확인
pip list | grep torch
pip list | grep numpy

# 누락 시 설치
pip install torch numpy
```

#### 2. model.py 코드 오류
```bash
# Python 문법 확인
python -m py_compile model_repository/lstm_timeseries/1/model.py

# 에러 확인
docker-compose logs triton-server | grep "Python error"
```

---

### 문제 3: Analysis Worker에서 Triton 연결 실패

**증상**:
```bash
docker-compose logs analysis-worker-1 | grep -i error
# [ERROR] [Triton Client] Connection failed
```

**진단**:
```bash
# Worker에서 Triton 연결 테스트
docker-compose exec analysis-worker-1 bash

# gRPC 포트 확인
nc -zv triton-server 8001

# 환경변수 확인
echo $TRITON_SERVER_URL
```

**해결 방법**:
```yaml
# docker-compose.yml에서 확인
analysis-worker-1:
  environment:
    - TRITON_SERVER_URL=triton-server:8001  # 올바른 URL
  depends_on:
    - triton-server  # 의존성 확인
```

---

### 문제 4: 추론 결과가 이상함

**증상**:
```bash
# 예측값이 NaN 또는 0
{
  "predictions": [null, null, null, null, null]
}
```

**진단 및 해결**:

#### 1. 입력 데이터 확인
```python
# triton_client.py에 디버그 로깅 추가
logger.info(f"[DEBUG] Input data: {data}")
logger.info(f"[DEBUG] Normalized: {normalized}")
logger.info(f"[DEBUG] Input shape: {input_data.shape}")
```

#### 2. Triton 모델 출력 확인
```python
# model.py에 디버그 로깅 추가
print(f"[Triton Model] Input shape: {input_data.shape}")
print(f"[Triton Model] Output shape: {output_numpy.shape}")
print(f"[Triton Model] Output sample: {output_numpy[0][:5]}")
```

#### 3. 전/후처리 로직 확인
```python
# 정규화 파라미터 확인
logger.info(f"[DEBUG] Mean: {mean}, Std: {std}")

# 역정규화 확인
logger.info(f"[DEBUG] Predictions (normalized): {predictions_normalized}")
logger.info(f"[DEBUG] Predictions (denormalized): {predictions}")
```

---

## 성능 벤치마크 체크리스트

### 기본 성능 확인

- [ ] 처리량: 80+ RPS (100개 요청, 동시성 10)
- [ ] 지연시간 (p50): < 40ms
- [ ] 지연시간 (p99): < 100ms
- [ ] GPU 활용률: > 70%

### Dynamic Batching 확인

- [ ] Triton 로그에서 배치 처리 확인
- [ ] 배치 크기 8, 16, 32로 자동 조절되는지 확인
- [ ] 큐 대기 시간 < 100ms

### 안정성 확인

- [ ] 1000개 요청 모두 성공 (에러율 0%)
- [ ] 메모리 누수 없음 (장시간 실행 후 메모리 증가 없음)
- [ ] GPU OOM 발생하지 않음

---

## 테스트 자동화 스크립트

### 전체 테스트 실행

```bash
#!/bin/bash
# run_all_tests.sh

echo "=========================================="
echo "Triton Inference Server Integration Test"
echo "=========================================="

# Step 1: Health Check
echo -e "\n[Step 1] Triton Server Health Check"
if curl -s http://localhost:8500/v2/health/ready | grep -q "ready"; then
  echo "Triton Server is ready"
else
  echo "Triton Server is not ready"
  exit 1
fi

# Step 2: Model Check
echo -e "\n[Step 2] Model Ready Check"
if curl -s http://localhost:8500/v2/models/lstm_timeseries/ready | grep -q "true"; then
  echo "LSTM model is ready"
else
  echo "LSTM model is not ready"
  exit 1
fi

if curl -s http://localhost:8500/v2/models/moving_average/ready | grep -q "true"; then
  echo "Moving Average model is ready"
else
  echo "Moving Average model is not ready"
  exit 1
fi

# Step 3: Inference Test
echo -e "\n[Step 3] Inference Test"
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "config": {"sequence_length": 10, "forecast_steps": 5}
  }')

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

if [ "$JOB_ID" != "null" ]; then
  echo "Inference request submitted: $JOB_ID"
else
  echo "Inference request failed"
  exit 1
fi

sleep 3

RESULT=$(curl -s http://localhost:8000/api/v1/inference/result/$JOB_ID)
STATUS=$(echo $RESULT | jq -r '.status')

if [ "$STATUS" == "completed" ]; then
  echo "Inference completed successfully"
  echo $RESULT | jq '.predictions'
else
  echo "Inference failed: $STATUS"
  exit 1
fi

# Step 4: Performance Test
echo -e "\n[Step 4] Performance Test (100 requests, concurrency 10)"
ab -n 100 -c 10 -q -p test_payload.json -T application/json \
  http://localhost:8000/api/v1/inference/submit \
  | grep "Requests per second"

echo -e "\n=========================================="
echo "All tests passed!"
echo "=========================================="
```

**실행**:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

---

**작성일**: 2025-10-15
**버전**: 1.0
**다음 단계**: 프로덕션 배포 전 성능 벤치마크 수행
