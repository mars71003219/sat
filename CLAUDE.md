# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

이 프로젝트는 마이크로서비스 기반 아키텍처로 구성된 위성 이미지 초해상화 시스템입니다. Docker Compose를 통해 여러 서비스를 오케스트레이션하며, GPU 기반 AI 서비스와 React 프론트엔드를 포함합니다.

## Architecture

### Service Components

프로젝트는 세 가지 주요 계층으로 구성됩니다:

1. **Infrastructure Services**: Kafka (KRaft mode), RabbitMQ, Redis, PostgreSQL, Elasticsearch
2. **AI Service**: FastAPI 기반 GPU 가속 이미지 처리 서비스 (포트 8002)
3. **Web Layer**: React 프론트엔드 + Nginx 리버스 프록시 (포트 80)

### Service Communication

- Nginx는 `/api` 경로를 AI 서비스(`ai:8000`)로 프록시합니다
- 모든 서비스는 `webnet` bridge 네트워크에서 통신합니다
- AI 서비스는 NVIDIA GPU를 사용합니다 (CUDA 13.0 기반)

## Development Commands

### Docker Environment

**프로덕션 환경 (미리 빌드된 이미지 사용):**
```bash
docker-compose up -d
```

**개발 환경 (Dockerfile에서 빌드):**
```bash
docker-compose -f docker-compose-dev.yml up -d
```

**서비스 중지:**
```bash
docker-compose down
```

**로그 확인:**
```bash
docker-compose logs -f [service_name]
```

### Kafka Setup

Kafka는 KRaft 모드로 실행되며, 첫 실행 전 클러스터 ID가 필요합니다:

```bash
./init-kafka.sh
```

이 스크립트는 `.env` 파일에 `KAFKA_CLUSTER_ID`를 생성합니다. `.env` 파일이 이미 존재하면 새로운 ID를 생성하지 않습니다.

### Frontend Development

프론트엔드는 Create React App 기반입니다 (`frontend/` 디렉토리):

```bash
cd frontend
npm install              # 의존성 설치
npm start                # 개발 서버 실행 (포트 3000)
npm test                 # 테스트 실행
npm run build            # 프로덕션 빌드
```

**Note**: Docker 환경에서는 `frontend/src` 디렉토리가 컨테이너에 마운트되어 hot-reload가 지원됩니다.

### AI Service Development

AI 서비스는 FastAPI 기반입니다 (`ai/` 디렉토리):

```bash
cd ai
pip3 install -r requirements.txt  # 의존성 설치
uvicorn main:app --reload         # 개발 서버 실행
```

**Dependencies**: PyTorch 2.1.0, ONNX Runtime (GPU), BasicSR, CLIP 등 이미지 처리 및 딥러닝 라이브러리를 사용합니다.

## Docker Images

### Building Images

**AI 서비스 이미지 빌드:**
```bash
docker build -t satlas-ai:latest ./ai
```

**프론트엔드 이미지 빌드:**
```bash
docker build -t satlas-ui:latest ./frontend
```

## Service Ports

- **Nginx**: 80 (HTTP)
- **Frontend** (dev): 3000
- **AI Service**: 8002 → 8000 (컨테이너 내부)
- **Kafka**: 9092
- **RabbitMQ**: 5672 (AMQP), 15672 (Management UI)
- **Redis**: 6379 (Redis), 8001 (RedisInsight)
- **PostgreSQL**: 5432
- **Elasticsearch**: 9200

## Important Notes

### Kafka Configuration

- **KRaft Mode**: Zookeeper 없이 실행됩니다 (프로덕션 환경)
- **Dev Mode**: `docker-compose-dev.yml`은 Zookeeper를 포함합니다
- Kafka 컨테이너는 `kafka_data` 볼륨에 데이터를 영구 저장합니다

### GPU Requirements

AI 서비스는 NVIDIA GPU가 필요합니다. Docker에서 GPU를 사용하려면:
- NVIDIA Docker runtime 설치 필요
- `deploy.resources.reservations.devices` 설정이 GPU 할당을 관리합니다

### Environment Variables

`.env` 파일은 Kafka 클러스터 ID를 저장합니다. 이 파일은 `init-kafka.sh`로 자동 생성되며, 버전 관리에서 제외되어야 합니다.

## Code Structure

### Frontend (`frontend/`)
- 표준 Create React App 구조
- `src/App.js`: 메인 애플리케이션 컴포넌트
- Nginx 다단계 빌드로 프로덕션 이미지 생성

### AI Service (`ai/`)
- `main.py`: FastAPI 애플리케이션 엔트리포인트
- `/api/enhance`: 이미지 초해상화 엔드포인트 (POST)
- CUDA 13.0 기반 GPU 가속 환경

### Nginx (`nginx/`)
- `nginx.conf`: 메인 Nginx 설정
- `conf.d/default.conf`: 라우팅 규칙 (SPA fallback, API 프록시)