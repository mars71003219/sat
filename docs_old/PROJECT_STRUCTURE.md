# 프로젝트 구조

##  디렉토리 개요

```
satellite/
├── analysis-server/          # Celery Worker - Triton 추론 실행
│   ├── core/                 # 핵심 로직 (Triton Client)
│   ├── utils/                # 유틸리티
│   └── tasks.py              # Celery 작업 정의
│
├── operation-server/         # FastAPI - API 서버 및 작업 관리
│   ├── api/                  # API 라우트
│   ├── database/             # 데이터베이스 모델
│   └── main.py               # 서버 엔트리포인트
│
├── shared/                   # 공통 스키마 및 유틸리티
│   └── schemas/              # Pydantic 스키마
│
├── frontend/                 # React 프론트엔드
│   ├── public/               # 정적 파일
│   │   └── dashboard.html    # 실시간 모니터링 대시보드
│   ├── src/                  # React 소스
│   └── package.json
│
├── model_repository/         # Triton 모델 저장소
│   ├── vae_timeseries/       # VAE 모델
│   │   ├── 1/                # 버전 1
│   │   │   ├── model.onnx    # ONNX 모델 파일
│   │   │   └── model.py      # Python 백엔드 (선택)
│   │   └── config.pbtxt      # Triton 설정
│   │
│   └── transformer_timeseries/  # Transformer 모델
│       ├── 1/
│       │   ├── model.onnx
│       │   └── model.py
│       └── config.pbtxt
│
├── scripts/                  # 모델 변환 도구
│   ├── convert_model.py      # 범용 PyTorch → ONNX/TensorRT 변환
│   ├── convert_example.sh    # 변환 예시 스크립트
│   └── README.md             # 사용 가이드
│
├── tests/                    # 테스트 및 데모
│   ├── data_simulator.py     # 데이터 생성 시뮬레이터 
│   ├── test_simulator.py     # 전체 성능 테스트
│   ├── test_single_model.py  # 단일 모델 테스트
│   └── README.md             # 테스트 가이드
│
├── nginx/                    # Nginx 리버스 프록시
│   └── conf.d/
│
├── docs/                     # 문서
│   ├── TRITON_IMPLEMENTATION.md
│   ├── TRITON_TESTING_GUIDE.md
│   └── TRITON_ANALYSIS.md
│
├── docker-compose.yml        # Docker Compose 설정
├── Dockerfile.triton         # Triton Server Dockerfile
└── CLAUDE.md                 # Claude Code 작업 가이드
```

##  각 폴더의 역할

### 추론 서비스
- **analysis-server**: Celery Worker로 Triton Server에 추론 요청을 보내고 결과를 처리
- **operation-server**: API 서버로 외부 요청을 받아 큐에 전달하고 결과를 반환
- **model_repository**: Triton Server가 로드하는 모델 파일들 (ONNX/TensorRT)

### 프론트엔드
- **frontend**: React 기반 웹 인터페이스
  - `/`: 메인 애플리케이션
  - `/dashboard.html`: 실시간 모니터링 대시보드

### 개발 도구
- **scripts**: 외부에서 학습된 모델을 ONNX/TensorRT로 변환하는 도구
- **tests**: 시스템 테스트 및 데모 도구

### 인프라
- **nginx**: 리버스 프록시 (프론트엔드 → API 라우팅)
- **shared**: 마이크로서비스 간 공유 코드

##  데이터 흐름

```
외부 학습 서버
    ↓
[.pth 모델 파일]
    ↓
scripts/convert_model.py (ONNX/TensorRT 변환)
    ↓
model_repository/ (모델 배포)
    ↓
Triton Server (GPU 추론)
    ↓
analysis-server (Celery Worker)
    ↓
RabbitMQ / Redis
    ↓
operation-server (FastAPI)
    ↓
PostgreSQL (결과 저장)
    ↓
frontend (웹 UI)
```

##  테스트 흐름

```
tests/data_simulator.py (데이터 생성)
    ↓
operation-server (API)
    ↓
RabbitMQ (메시지 큐)
    ↓
analysis-server (Worker)
    ↓
Triton Server (GPU 추론)
    ↓
PostgreSQL / Redis (저장)
    ↓
frontend/dashboard.html (실시간 모니터링)
    ↓
Flower (Celery 모니터링)
```

##  주요 파일

### 설정 파일
- `docker-compose.yml`: 모든 서비스 정의
- `model_repository/*/config.pbtxt`: Triton 모델 설정
- `nginx/conf.d/default.conf`: Nginx 라우팅 설정

### 코어 로직
- `analysis-server/core/triton_client.py`: Triton 추론 클라이언트
- `analysis-server/tasks.py`: Celery 작업 정의
- `operation-server/main.py`: FastAPI 애플리케이션

### 테스트 도구
- `tests/data_simulator.py`: 실제 운영 환경 시뮬레이션 
- `tests/test_simulator.py`: 성능 벤치마크
- `tests/test_single_model.py`: 빠른 동작 확인

##  빠른 시작

### 1. 모든 서비스 시작
```bash
docker compose up -d
```

### 2. 모델 변환 (외부 학습 모델 받은 경우)
```bash
cd scripts
python3 convert_model.py \
    --model-path /path/to/model.pth \
    --model-class YourModel \
    --backend onnx
```

### 3. 시스템 테스트
```bash
cd tests
python3 test_single_model.py     # 빠른 확인
python3 data_simulator.py         # 전체 시스템 테스트
```

### 4. 모니터링
- 대시보드: http://localhost
- 실시간 모니터링: http://localhost/dashboard.html
- Flower: http://localhost:5555
- Kafka UI: http://localhost:8080

##  개발 가이드

### 새 모델 추가
1. 외부 학습 서버에서 `.pth` 파일 받기
2. `scripts/convert_model.py`로 ONNX/TensorRT 변환
3. `model_repository/`에 모델 배포
4. `analysis-server/core/triton_client.py`에 추론 메서드 추가
5. Triton Server 재시작: `docker compose restart triton-server`

### 테스트 실행
```bash
cd tests
python3 data_simulator.py --interval 5 --max-batch 3
```

### 로그 확인
```bash
# 전체 로그
docker compose logs -f

# 특정 서비스
docker compose logs -f triton-server
docker compose logs -f analysis-worker-1
docker compose logs -f operation-server
```

##  시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (CUDA 지원)
  - 권장: RTX 3060 이상
  - 메모리: 8GB 이상
- **RAM**: 16GB 이상
- **디스크**: 50GB 이상

### 소프트웨어
- Docker & Docker Compose
- NVIDIA Docker Runtime
- Python 3.10+

##  참고 문서

- `scripts/README.md`: 모델 변환 가이드
- `tests/README.md`: 테스트 및 데모 가이드
- `docs/TRITON_IMPLEMENTATION.md`: Triton 구현 상세
- `CLAUDE.md`: Claude Code 작업 가이드
