# 위성 텔레메트리 분석 시스템 - 문서

이 디렉토리에는 위성 텔레메트리 분석 시스템의 전체 문서가 포함되어 있습니다.

## 📚 문서 목록

### 1. [시스템 아키텍처](./SYSTEM_ARCHITECTURE.md)

시스템의 전체 구성과 아키텍처를 상세히 설명합니다.

**주요 내용**:
- ✅ 전체 시스템 구성도 (계층별 도식화)
- ✅ 네트워크 구성 및 포트 매핑
- ✅ 데이터 플로우 (실시간 텔레메트리, AI 추론 파이프라인)
- ✅ 서브시스템별 모델 매핑 (EPS, Thermal, AOCS, Comm)
- ✅ 기술 스택 및 컨테이너 배포 구성
- ✅ 확장성 및 장애 복구 전략

**대상**: 시스템 관리자, 개발자, 아키텍트

---

### 2. [UML 다이어그램](./UML_DIAGRAMS.md)

시스템의 동작과 구조를 UML로 표현합니다.

**주요 내용**:
- ✅ **시퀀스 다이어그램**:
  - 자동 추론 트리거 시퀀스 (Kafka → Triton)
  - 수동 추론 API 시퀀스
  - 실시간 대시보드 WebSocket 시퀀스
- ✅ **클래스 다이어그램**:
  - 핵심 도메인 모델 (InferenceJob, SubsystemInference)
  - 서비스 계층 (TritonClient, PostgresClient, CeleryApp)
  - API 라우터 구조
- ✅ **컴포넌트 다이어그램**: 전체 컴포넌트 아키텍처
- ✅ **상태 다이어그램**: Job 및 Inference 상태 전이
- ✅ **배포 다이어그램**: Docker 컨테이너 배포 구성

**대상**: 개발자, 시스템 설계자

---

### 3. [API 레퍼런스](./API_REFERENCE.md)

REST API 및 WebSocket API의 전체 명세입니다.

**주요 내용**:
- ✅ **Inference API**: 추론 작업 제출, 상태 조회, 결과 조회, WebSocket
- ✅ **Dashboard API**: 실시간 모니터링, 통계, 모델 비교
- ✅ **Trends API**: 시계열 데이터 조회, 예측 vs 실제 비교
- ✅ **Query API**: 고급 쿼리 및 필터링
- ✅ **Search API**: Elasticsearch 기반 전체 텍스트 검색
- ✅ **예제 코드**: Python, JavaScript, cURL

**대상**: API 사용자, 프론트엔드 개발자

---

### 4. [사용자 매뉴얼](./USER_MANUAL.md)

시스템 운영 및 관리를 위한 실무 가이드입니다.

**주요 내용**:
- ✅ **시스템 시작 및 종료**: Docker Compose 명령어, 로그 확인
- ✅ **메시지 큐 상태 확인**:
  - Kafka UI 및 CLI 사용법
  - RabbitMQ Management UI 및 CLI
  - Celery Flower 모니터링
- ✅ **데이터베이스 확인**:
  - PostgreSQL 접속 및 쿼리
  - 추론 결과 조회
  - 통계 및 분석 쿼리
  - 데이터 정리 및 백업
- ✅ **시스템 모니터링**:
  - VictoriaMetrics 메트릭 조회
  - Triton Server Health Check
  - Docker 리소스 모니터링
  - GPU 모니터링
- ✅ **추론 작업 실행**: 수동/자동 추론, 배치 추론
- ✅ **트러블슈팅**: 일반적인 문제 해결
- ✅ **성능 튜닝**: 최적화 가이드

**대상**: 시스템 관리자, DevOps 엔지니어, 운영팀

---

## 🚀 빠른 시작

### 시스템 시작

```bash
cd /mnt/c/projects/satellite

# 1. Kafka 초기화 (최초 1회)
./init-kafka.sh

# 2. 전체 시스템 시작
docker compose up -d

# 3. 상태 확인
docker compose ps
```

### 모니터링 대시보드 접속

- **웹 프론트엔드**: http://localhost
- **Operation Server API**: http://localhost:8000
- **Kafka UI**: http://localhost:8080
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)
- **Flower (Celery)**: http://localhost:5555
- **VictoriaMetrics**: http://localhost:8428

### 추론 작업 실행

```bash
# API로 추론 제출
curl -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
    "config": {"forecast_horizon": 5}
  }'

# 시뮬레이터로 자동 추론 트리거
cd tests
python3 satellite_simulator.py --kafka kafka:9092 --satellites 2 --interval 3 --duration 30
```

---

## 📊 시스템 개요

### 아키텍처 계층

```
┌─────────────────────────────────────┐
│     Web Layer (Nginx + React)      │  ← 사용자 인터페이스
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│   API Layer (Operation Server)     │  ← REST API & WebSocket
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│  Message Queue (Kafka + RabbitMQ)  │  ← 비동기 작업 큐
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│  AI Layer (Triton + Analysis Worker)│  ← GPU 기반 추론
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│  Storage (PostgreSQL + VictoriaMetrics)│  ← 데이터 저장
└─────────────────────────────────────┘
```

### 주요 기능

1. **실시간 텔레메트리 수집**: Kafka를 통한 스트리밍 데이터 수집
2. **자동 이상 감지**: 4개 서브시스템 (EPS, Thermal, AOCS, Comm) 독립 분석
3. **GPU 가속 추론**: NVIDIA Triton Server를 통한 고성능 LSTM 추론
4. **시계열 저장**: VictoriaMetrics에 실시간 데이터 저장
5. **실시간 대시보드**: WebSocket 기반 실시간 모니터링

---

## 🔧 기술 스택

### 백엔드
- **FastAPI** - 고성능 비동기 API 서버
- **Celery** - 분산 작업 큐 시스템
- **PostgreSQL** - 관계형 데이터베이스
- **RabbitMQ** - AMQP 메시지 브로커
- **Apache Kafka** - 실시간 스트리밍 플랫폼

### AI/ML
- **NVIDIA Triton Inference Server** - GPU 추론 엔진
- **PyTorch** - 딥러닝 프레임워크
- **ONNX Runtime** - 모델 추론 최적화

### 프론트엔드
- **React** - UI 라이브러리
- **Nginx** - 웹 서버 & 리버스 프록시

### 데이터 저장
- **VictoriaMetrics** - 시계열 데이터베이스
- **Elasticsearch** - 검색 엔진

### 모니터링
- **Kafka UI** - Kafka 모니터링
- **Flower** - Celery 작업 모니터링
- **RabbitMQ Management** - 큐 상태 모니터링

---

## 📖 문서 읽는 순서

### 초급 사용자
1. [사용자 매뉴얼](./USER_MANUAL.md) - 시스템 시작 및 기본 사용법
2. [API 레퍼런스](./API_REFERENCE.md) - API 사용 방법

### 중급 개발자
1. [시스템 아키텍처](./SYSTEM_ARCHITECTURE.md) - 전체 구조 이해
2. [UML 다이어그램](./UML_DIAGRAMS.md) - 시퀀스 및 클래스 구조
3. [API 레퍼런스](./API_REFERENCE.md) - API 상세 명세

### 고급 아키텍트
1. [시스템 아키텍처](./SYSTEM_ARCHITECTURE.md) - 전체 설계
2. [UML 다이어그램](./UML_DIAGRAMS.md) - 상세 설계
3. [사용자 매뉴얼](./USER_MANUAL.md) - 운영 및 튜닝

---

## 🆘 지원 및 문의

### 문제 해결
1. [사용자 매뉴얼 - 트러블슈팅](./USER_MANUAL.md#6-트러블슈팅) 참조
2. [시스템 로그 확인](./USER_MANUAL.md#16-로그-확인)
3. GitHub Issues 등록

### 성능 최적화
- [사용자 매뉴얼 - 성능 튜닝](./USER_MANUAL.md#7-성능-튜닝)

---

## 📝 변경 이력

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| 1.0.0 | 2025-10-30 | 초기 문서 작성 (시스템 아키텍처, UML, API, 사용자 매뉴얼) |

---

## 📄 라이선스

이 프로젝트는 [MIT License](../LICENSE)를 따릅니다.

---

## 👥 기여자

- 시스템 설계 및 구현
- 문서 작성

---

## 🌟 주요 특징

✅ **전문적인 도식화**: 모든 다이어그램은 ASCII 아트로 명확하게 표현
✅ **반듯한 박스 및 화살표**: 정렬된 구조로 가독성 최대화
✅ **실무 중심 가이드**: 실제 운영에 필요한 모든 정보 포함
✅ **완전한 예제 코드**: Python, JavaScript, cURL 예제 제공
✅ **상세한 트러블슈팅**: 일반적인 문제와 해결 방법

---

**시스템에 대한 궁금한 점이 있으시면 각 문서를 참조하거나 문의해 주세요!**
