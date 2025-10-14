#!/usr/bin/env python3
"""
데이터 시뮬레이터
지속적으로 시계열 데이터를 생성하고 추론 작업을 제출합니다.
"""
import requests
import time
import random
import json
from datetime import datetime
from typing import List, Dict, Any
import argparse


class DataSimulator:
    """시계열 데이터 시뮬레이터"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.patterns = ["linear", "seasonal", "random_walk", "exponential", "cyclical"]
        self.models = ["lstm_timeseries", "moving_average"]

    def generate_time_series(self, pattern: str, length: int = 10) -> List[float]:
        """패턴에 따라 시계열 데이터 생성"""

        if pattern == "linear":
            # 선형 증가 패턴
            base = random.uniform(10, 20)
            slope = random.uniform(0.5, 2.0)
            return [base + i * slope + random.gauss(0, 0.3) for i in range(length)]

        elif pattern == "seasonal":
            # 계절성 패턴
            base = random.uniform(40, 60)
            amplitude = random.uniform(10, 20)
            period = random.randint(3, 5)
            import math
            return [base + amplitude * math.sin(2 * math.pi * i / period) + random.gauss(0, 0.5)
                    for i in range(length)]

        elif pattern == "exponential":
            # 지수 증가 패턴
            base = random.uniform(80, 120)
            rate = random.uniform(0.02, 0.08)
            return [base * ((1 + rate) ** i) + random.gauss(0, 0.5) for i in range(length)]

        elif pattern == "cyclical":
            # 순환 패턴
            base = random.uniform(0.02, 0.08)
            amplitude = random.uniform(0.02, 0.05)
            import math
            return [base + amplitude * math.cos(2 * math.pi * i / 6) + random.gauss(0, 0.005)
                    for i in range(length)]

        else:  # random_walk
            # 랜덤워크 패턴
            base = random.uniform(1000, 1020)
            values = [base]
            for _ in range(length - 1):
                change = random.gauss(0, 0.5)
                values.append(values[-1] + change)
            return values

    def submit_inference(self, model_name: str, data: List[float],
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """추론 작업 제출"""
        request_data = {
            "model_name": model_name,
            "data": data,
            "config": {
                "forecast_steps": 5,
                "window_size": 5 if model_name == "moving_average" else None
            },
            "metadata": metadata or {}
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/inference/submit",
                json=request_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error submitting inference: {e}")
            return None

    def get_job_result(self, job_id: str, max_attempts: int = 15) -> Dict[str, Any]:
        """작업 결과 조회 (완료될 때까지 대기)"""
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.base_url}/api/v1/inference/result/{job_id}",
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "completed":
                        return result
                    elif result.get("status") == "failed":
                        print(f"Job {job_id} failed: {result.get('error_message')}")
                        return result
                time.sleep(1)
            except Exception as e:
                print(f"Error getting result: {e}")
                time.sleep(1)

        return {"status": "timeout", "job_id": job_id}

    def run_simulation(self, interval: int = 5, random_interval: bool = True,
                      min_batch: int = 1, max_batch: int = 5):
        """시뮬레이션 실행"""
        print("="*70)
        print("데이터 시뮬레이터 시작")
        print("="*70)
        print(f"기본 주기: {interval}초")
        print(f"랜덤 주기: {'활성화' if random_interval else '비활성화'}")
        print(f"배치 크기: {min_batch}-{max_batch}개")
        print(f"사용 모델: {', '.join(self.models)}")
        print(f"생성 패턴: {', '.join(self.patterns)}")
        print("="*70)
        print()

        iteration = 0

        try:
            while True:
                iteration += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 랜덤하게 생성할 데이터 개수 결정
                batch_size = random.randint(min_batch, max_batch)

                print(f"\n[{timestamp}] Iteration #{iteration} - Batch Size: {batch_size}")
                print("-" * 70)

                jobs = []

                for i in range(batch_size):
                    # 랜덤 패턴 선택
                    pattern = random.choice(self.patterns)
                    data_length = random.randint(10, 20)

                    # 데이터 생성
                    data = self.generate_time_series(pattern, data_length)

                    print(f"\n  [{i+1}/{batch_size}] Pattern: {pattern}, Length: {data_length}")
                    print(f"  Data sample: [{data[0]:.2f}, {data[1]:.2f}, ..., {data[-1]:.2f}]")

                    # 두 모델로 추론 제출
                    for model_name in self.models:
                        metadata = {
                            "pattern": pattern,
                            "iteration": iteration,
                            "batch_index": i,
                            "timestamp": timestamp,
                            "simulator": "data_simulator"
                        }

                        result = self.submit_inference(model_name, data, metadata)

                        if result:
                            job_id = result.get("job_id")
                            print(f"    - {model_name}: Job {job_id} submitted")
                            jobs.append({
                                "job_id": job_id,
                                "model": model_name,
                                "pattern": pattern,
                                "data": data
                            })
                        else:
                            print(f"    - {model_name}: Failed to submit")

                # 결과 수집 (비동기적으로)
                print(f"\n  Submitted {len(jobs)} jobs")

                # 다음 주기까지 대기
                if random_interval:
                    wait_time = random.uniform(interval * 0.5, interval * 1.5)
                else:
                    wait_time = interval

                print(f"\n  Next batch in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\n\n시뮬레이터 중지")
            print(f"총 {iteration}회 반복 실행")


def main():
    parser = argparse.ArgumentParser(description="데이터 시뮬레이터")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="데이터 생성 주기 (초, 기본값: 5)"
    )
    parser.add_argument(
        "--random-interval",
        action="store_true",
        help="랜덤 주기 활성화 (interval의 50%%-150%% 범위)"
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="최소 배치 크기 (기본값: 1)"
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=5,
        help="최대 배치 크기 (기본값: 5)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Operation Server URL (기본값: http://localhost:8000)"
    )

    args = parser.parse_args()

    simulator = DataSimulator(base_url=args.url)
    simulator.run_simulation(
        interval=args.interval,
        random_interval=args.random_interval,
        min_batch=args.min_batch,
        max_batch=args.max_batch
    )


if __name__ == "__main__":
    main()
