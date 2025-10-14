#!/usr/bin/env python3
"""
시스템 전체 데모 스크립트
"""
import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60 + "\n")

def demo_health_check():
    """1. 헬스 체크"""
    print_section("1. System Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    time.sleep(1)

def demo_submit_inferences():
    """2. 추론 작업 제출"""
    print_section("2. Submit Inference Jobs")
    
    # LSTM 모델 추론
    lstm_request = {
        "model_name": "lstm_timeseries",
        "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5, 5.1, 4.8],
        "config": {"forecast_steps": 5}
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/inference/submit",
        json=lstm_request
    )
    lstm_job_id = response.json()["job_id"]
    print(f"LSTM Job submitted: {lstm_job_id}")
    print(json.dumps(response.json(), indent=2))
    
    # Moving Average 모델 추론
    ma_request = {
        "model_name": "moving_average",
        "data": [10, 12, 11, 13, 15, 14, 16, 18, 17, 19],
        "config": {"forecast_steps": 5, "window_size": 5}
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/inference/submit",
        json=ma_request
    )
    ma_job_id = response.json()["job_id"]
    print(f"\nMoving Average Job submitted: {ma_job_id}")
    print(json.dumps(response.json(), indent=2))
    
    return lstm_job_id, ma_job_id

def demo_check_status(job_ids):
    """3. 작업 상태 확인"""
    print_section("3. Check Job Status")
    
    for i, job_id in enumerate(job_ids, 1):
        print(f"\nJob {i}: {job_id}")
        
        # 작업 완료 대기
        for attempt in range(10):
            response = requests.get(f"{BASE_URL}/api/v1/inference/status/{job_id}")
            status = response.json()["status"]
            print(f"  Attempt {attempt+1}: Status = {status}")
            
            if status == "completed":
                break
            time.sleep(1)

def demo_get_results(job_ids):
    """4. 결과 조회"""
    print_section("4. Get Inference Results")
    
    for i, job_id in enumerate(job_ids, 1):
        print(f"\nJob {i} Result:")
        response = requests.get(f"{BASE_URL}/api/v1/inference/result/{job_id}")
        result = response.json()
        
        print(f"  Model: {result.get('model_name')}")
        print(f"  Predictions: {result.get('predictions')}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Inference Time: {result.get('metrics', {}).get('inference_time')} seconds")

def demo_query_history():
    """5. 히스토리 조회"""
    print_section("5. Query Inference History")
    
    response = requests.get(f"{BASE_URL}/api/v1/results/history?limit=10")
    data = response.json()
    
    print(f"Total results: {data['count']}")
    for i, result in enumerate(data['results'][:3], 1):
        print(f"\n  Result {i}:")
        print(f"    Job ID: {result.get('job_id')}")
        print(f"    Model: {result.get('model_name')}")
        print(f"    Status: {result.get('status')}")

def demo_statistics():
    """6. 통계 조회"""
    print_section("6. Get Statistics")
    
    response = requests.get(f"{BASE_URL}/api/v1/results/stats/summary")
    stats = response.json()
    
    print(f"Total Jobs: {stats['total_jobs']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.2f}%")
    print(f"\nModels Used:")
    for model, count in stats['models_used'].items():
        print(f"  - {model}: {count} times")

def demo_search():
    """7. Elasticsearch 검색"""
    print_section("7. Search with Elasticsearch")
    
    try:
        # 모델별 검색
        response = requests.get(
            f"{BASE_URL}/api/v1/search/inference",
            params={"model_name": "lstm_timeseries", "limit": 5}
        )
        data = response.json()
        print(f"LSTM Results: {data['count']} found")
        
        # 모델 분석
        response = requests.get(f"{BASE_URL}/api/v1/search/analytics/models")
        analytics = response.json()
        print(f"\nModel Analytics:")
        for model in analytics.get('model_statistics', [])[:3]:
            print(f"  - {model['model_name']}: {model['total_inferences']} inferences")
    except Exception as e:
        print(f"Elasticsearch demo skipped (service may not be ready): {e}")

def main():
    """메인 데모 실행"""
    print("\n" + "#"*60)
    print("#" + " "*18 + "SYSTEM DEMO" + " "*18 + "#")
    print("#"*60)
    print("\nNote: PostgreSQL is pre-seeded with sample data")
    print("      (sensor data, timeseries, metrics, activity logs)")
    
    try:
        demo_health_check()
        
        job_ids = demo_submit_inferences()
        time.sleep(2)
        
        demo_check_status(job_ids)
        time.sleep(1)
        
        demo_get_results(job_ids)
        
        demo_query_history()
        
        demo_statistics()
        
        demo_search()
        
        print_section("Demo Completed Successfully!")
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server.")
        print("Make sure the system is running: docker compose up -d")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == '__main__':
    main()
