# 빠른 시작 가이드

## 1. 시스템 시작

```bash
docker compose up -d
```

## 2. 웹 대시보드 열기

브라우저에서 다음 주소로 접속:

```
http://localhost/dashboard.html
```

또는

```
http://127.0.0.1/dashboard.html
```

## 3. 데이터 시뮬레이터 실행

새 터미널을 열고:

```bash
# 기본 실행 (5초마다 1-5개 데이터 생성)
python3 data_simulator.py

# 빠른 테스트 (3초마다, 랜덤 주기)
python3 data_simulator.py --interval 3 --random-interval

# 대량 생성 (2초마다 5-10개)
python3 data_simulator.py --interval 2 --min-batch 5 --max-batch 10
```

## 4. 실시간 모니터링

### 웹 대시보드
- URL: http://localhost/dashboard.html
- 실시간 통계, 최근 추론 결과 확인

### Celery 모니터링 (Flower)
- URL: http://localhost:5555
- Worker 상태, 작업 큐 확인

### API 직접 호출
```bash
# 헬스 체크
curl http://localhost:8000/health

# 최근 결과 조회
curl http://localhost:8000/api/v1/dashboard/recent?limit=10

# 실시간 통계
curl http://localhost:8000/api/v1/dashboard/live-stats
```

## 5. 중지

```bash
# 시뮬레이터 중지
Ctrl + C

# 시스템 중지
docker compose down
```

## 접속 주소 요약

| 서비스 | URL | 설명 |
|--------|-----|------|
| 웹 대시보드 | http://localhost/dashboard.html | 실시간 추론 결과 대시보드 |
| Operation Server API | http://localhost:8000 | REST API 엔드포인트 |
| Flower (Celery) | http://localhost:5555 | Celery 작업 모니터링 |
| RedisInsight | http://localhost:8001 | Redis 데이터 확인 |

## 문제 해결

### 대시보드가 연결 안될 때
```bash
# 컨테이너 상태 확인
docker compose ps

# 로그 확인
docker compose logs nginx
docker compose logs operation-server

# 재시작
docker compose restart nginx operation-server
```

### 시뮬레이터 연결 오류
```bash
# Operation Server 상태 확인
curl http://localhost:8000/health

# 실행 중인지 확인
docker compose ps operation-server
```

### 추론이 실패할 때
```bash
# Worker 로그 확인
docker compose logs analysis-worker-1 --tail 50

# Celery 상태 확인 (Flower)
# 브라우저: http://localhost:5555
```

## 추가 정보

자세한 내용은 다음 문서를 참고하세요:
- `SIMULATOR_README.md` - 시뮬레이터 상세 가이드
- `CLAUDE.md` - 프로젝트 전체 구조
