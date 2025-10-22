# 모니터링 UI 접속 가이드

브라우저에서 각 서비스를 모니터링할 수 있는 웹 UI가 제공됩니다.

## 1. Kafka UI
**접속 주소:** http://localhost:8080

**기능:**
- Kafka 클러스터 상태 모니터링
- 토픽 목록 및 상세 정보 조회
- 메시지 브라우징 및 검색
- 컨슈머 그룹 모니터링
- 파티션 및 레플리카 상태 확인

**사용 방법:**
1. 브라우저에서 http://localhost:8080 접속
2. 좌측 메뉴에서 "Topics" 클릭하여 토픽 목록 확인
3. 특정 토픽을 클릭하여 메시지 조회 가능
4. "Consumers" 메뉴에서 컨슈머 그룹 상태 확인

**주요 토픽:**
- `inference_results`: AI 추론 결과가 저장되는 토픽

## 2. Redis Insight
**접속 주소:** http://localhost:8001

**기능:**
- Redis 데이터베이스 모니터링
- 키-값 데이터 조회 및 수정
- 메모리 사용량 분석
- 명령어 실행 (CLI)
- 성능 메트릭 확인

**사용 방법:**
1. 브라우저에서 http://localhost:8001 접속
2. 처음 접속 시 데이터베이스 연결 설정
   - Host: redis
   - Port: 6379
   - Database: 0 (또는 원하는 DB 번호)
3. "Browser" 탭에서 키 목록 및 값 조회
4. "Workbench" 탭에서 Redis 명령어 실행
5. "Analysis Tools" 탭에서 메모리 분석

**Redis DB 구성:**
- DB 0: 일반 캐시 데이터
- DB 1: Celery 브로커 (작업 큐)
- DB 2: Celery 결과 백엔드

## 3. AI 추론 대시보드
**접속 주소:** http://localhost/

**기능:**
- 실시간 추론 결과 모니터링
- 통계 정보 (Total Jobs, Completed, Failed, Success Rate)
- 최근 추론 결과 테이블
- WebSocket 기반 자동 업데이트 (2초마다)

**표시 정보:**
- Time: 추론 시간
- Job ID: 작업 고유 식별자
- Model: 사용된 모델 (LSTM, Moving Average)
- Pattern: 데이터 패턴 유형
- Status: 작업 상태 (completed, failed, pending)
- Predictions: 예측 결과 (처음 3개 값)
- Inference Time: 추론 소요 시간 (ms)

## 4. Celery Flower
**접속 주소:** http://localhost:5555

**기능:**
- Celery 워커 모니터링
- 작업 큐 상태 확인
- 작업 실행 이력
- 워커 성능 메트릭

**사용 방법:**
1. 브라우저에서 http://localhost:5555 접속
2. "Workers" 탭에서 활성 워커 확인
3. "Tasks" 탭에서 작업 목록 및 상태 확인
4. 특정 작업 클릭하여 상세 정보 조회

## 전체 서비스 포트 요약

| 서비스 | 포트 | 접속 URL | 설명 |
|--------|------|----------|------|
| AI Dashboard | 80 | http://localhost/ | 실시간 추론 대시보드 |
| Kafka UI | 8080 | http://localhost:8080 | Kafka 모니터링 |
| Redis Insight | 8001 | http://localhost:8001 | Redis 모니터링 |
| Celery Flower | 5555 | http://localhost:5555 | Celery 작업 모니터링 |
| Operation API | 8000 | http://localhost:8000/docs | FastAPI 문서 |
| PostgreSQL | 5432 | - | 데이터베이스 (외부 클라이언트 필요) |
| Redis | 6379 | - | 캐시 서버 (외부 클라이언트 필요) |
| Kafka | 9092 | - | 메시지 브로커 (외부 클라이언트 필요) |
| Elasticsearch | 9200 | - | 검색 엔진 (외부 클라이언트 필요) |

## 접속 확인 방법

모든 서비스가 정상적으로 실행 중인지 확인:
```bash
docker compose ps
```

특정 서비스 로그 확인:
```bash
docker compose logs -f [service_name]
```

예시:
```bash
docker compose logs -f kafka-ui
docker compose logs -f redis
```
