#!/usr/bin/env python3
"""
Triton Inference System 시뮬레이터 및 성능 테스트
"""
import requests
import time
import json
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import sys


class InferenceSimulator:
    """추론 시스템 시뮬레이터"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.results = []

    def check_health(self) -> bool:
        """서비스 헬스 체크"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Operation Server is healthy")
                return True
            else:
                print(f"Operation Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Cannot connect to Operation Server: {e}")
            return False

    def submit_inference(
        self,
        model_name: str,
        data: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """추론 요청 제출"""
        payload = {
            "model_name": model_name,
            "data": data,
            "config": config
        }

        try:
            response = requests.post(
                f"{self.api_url}/inference/submit",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Submit failed: {response.status_code} - {response.text}")
                return {"error": response.text}

        except Exception as e:
            print(f"Submit request failed: {e}")
            return {"error": str(e)}

    def get_result(self, job_id: str, max_retries: int = 10) -> Dict[str, Any]:
        """추론 결과 조회 (재시도 포함)"""
        for i in range(max_retries):
            try:
                response = requests.get(
                    f"{self.api_url}/inference/result/{job_id}",
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "completed":
                        return result
                    elif result.get("status") == "failed":
                        return result

                # 아직 완료되지 않음 - 대기
                time.sleep(0.5)

            except Exception as e:
                if i == max_retries - 1:
                    return {"error": str(e)}
                time.sleep(0.5)

        return {"error": "Timeout waiting for result"}

    def single_inference_test(self, model_name: str) -> Dict[str, Any]:
        """단일 추론 테스트"""
        print(f"\n{'='*60}")
        print(f"Single Inference Test: {model_name}")
        print(f"{'='*60}")

        # 테스트 데이터 준비 (20 steps input for VAE and Transformer)
        if model_name == "vae_timeseries":
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
            config = {
                "sequence_length": 20,
                "forecast_steps": 10
            }
        elif model_name == "transformer_timeseries":
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
            config = {
                "sequence_length": 20,
                "forecast_steps": 10
            }
        # Legacy models (if still present)
        elif model_name == "lstm_timeseries":
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            config = {
                "sequence_length": 10,
                "forecast_steps": 5
            }
        else:  # moving_average
            data = [10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 20.0]
            config = {
                "window_size": 5,
                "forecast_steps": 5
            }

        # 추론 요청
        start_time = time.time()
        submit_response = self.submit_inference(model_name, data, config)

        if "error" in submit_response:
            print(f"Test failed: {submit_response['error']}")
            return {"success": False, "error": submit_response['error']}

        job_id = submit_response.get("job_id")
        print(f"Job submitted: {job_id}")

        # 결과 대기
        result = self.get_result(job_id)
        total_time = time.time() - start_time

        if "error" in result:
            print(f"Result retrieval failed: {result['error']}")
            return {"success": False, "error": result['error']}

        # 결과 출력
        print(f"\n Results:")
        print(f"  - Status: {result.get('status')}")
        print(f"  - Total Time: {total_time:.3f}s")
        print(f"  - Inference Time: {result.get('metrics', {}).get('inference_time', 'N/A')}s")
        print(f"  - Predictions: {result.get('predictions', [])[:5]}...")
        print(f"  - Confidence: {result.get('confidence', [])[:5]}...")
        print(f"  - Processed By: {result.get('metadata', {}).get('processed_by', 'N/A')}")

        return {
            "success": True,
            "job_id": job_id,
            "total_time": total_time,
            "inference_time": result.get('metrics', {}).get('inference_time'),
            "predictions": result.get('predictions'),
            "model_name": model_name
        }

    def concurrent_inference_test(
        self,
        model_name: str,
        num_requests: int = 20,
        max_workers: int = 10
    ) -> Dict[str, Any]:
        """동시 추론 테스트"""
        print(f"\n{'='*60}")
        print(f"Concurrent Inference Test: {model_name}")
        print(f"Requests: {num_requests}, Concurrency: {max_workers}")
        print(f"{'='*60}")

        # 테스트 데이터 준비 (20 steps input for VAE and Transformer)
        if model_name in ["vae_timeseries", "transformer_timeseries"]:
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
            config = {"sequence_length": 20, "forecast_steps": 10}
        elif model_name == "lstm_timeseries":
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            config = {"sequence_length": 10, "forecast_steps": 5}
        else:  # moving_average
            data = [10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 20.0]
            config = {"window_size": 5, "forecast_steps": 5}

        results = []
        start_time = time.time()

        def send_request(request_id: int):
            """단일 요청 전송"""
            req_start = time.time()

            # 요청 제출
            submit_response = self.submit_inference(model_name, data, config)
            if "error" in submit_response:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": submit_response['error']
                }

            job_id = submit_response.get("job_id")

            # 결과 대기
            result = self.get_result(job_id, max_retries=20)

            req_time = time.time() - req_start

            if "error" in result:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": result['error']
                }

            return {
                "request_id": request_id,
                "success": True,
                "job_id": job_id,
                "total_time": req_time,
                "inference_time": result.get('metrics', {}).get('inference_time', 0)
            }

        # 병렬 요청 실행
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(send_request, i)
                for i in range(num_requests)
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                if result['success']:
                    print(f"✅ [{completed}/{num_requests}] Request {result['request_id']}: "
                          f"{result['total_time']:.3f}s")
                else:
                    print(f"❌ [{completed}/{num_requests}] Request {result['request_id']} failed")

        total_time = time.time() - start_time

        # 통계 계산
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        if successful:
            total_times = [r['total_time'] for r in successful]
            inference_times = [r['inference_time'] for r in successful if r.get('inference_time')]

            stats = {
                "total_requests": num_requests,
                "successful": len(successful),
                "failed": len(failed),
                "total_time": total_time,
                "throughput": len(successful) / total_time,
                "latency_mean": statistics.mean(total_times),
                "latency_median": statistics.median(total_times),
                "latency_min": min(total_times),
                "latency_max": max(total_times),
                "latency_p95": statistics.quantiles(total_times, n=20)[18] if len(total_times) > 1 else total_times[0],
                "latency_p99": statistics.quantiles(total_times, n=100)[98] if len(total_times) > 1 else total_times[0],
                "inference_time_mean": statistics.mean(inference_times) if inference_times else 0
            }

            # 결과 출력
            print(f"\n📊 Performance Statistics:")
            print(f"  - Total Requests: {stats['total_requests']}")
            print(f"  - Successful: {stats['successful']}")
            print(f"  - Failed: {stats['failed']}")
            print(f"  - Total Time: {stats['total_time']:.3f}s")
            print(f"  - Throughput: {stats['throughput']:.2f} RPS")
            print(f"\n  Latency:")
            print(f"    - Mean: {stats['latency_mean']:.3f}s")
            print(f"    - Median: {stats['latency_median']:.3f}s")
            print(f"    - Min: {stats['latency_min']:.3f}s")
            print(f"    - Max: {stats['latency_max']:.3f}s")
            print(f"    - P95: {stats['latency_p95']:.3f}s")
            print(f"    - P99: {stats['latency_p99']:.3f}s")
            print(f"\n  Inference Time (Triton):")
            print(f"    - Mean: {stats['inference_time_mean']:.3f}s")

            return stats
        else:
            print(f"\n❌ All requests failed")
            return {"total_requests": num_requests, "successful": 0, "failed": len(failed)}

    def run_full_test_suite(self):
        """전체 테스트 스위트 실행"""
        print("\n" + "="*60)
        print("Triton Inference System - Full Test Suite")
        print("="*60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Health Check
        print("\n[1/5] Health Check")
        if not self.check_health():
            print("\n❌ System is not ready. Exiting.")
            sys.exit(1)

        # 2. Single VAE Test
        print("\n[2/5] Single VAE Test")
        vae_single = self.single_inference_test("vae_timeseries")

        # 3. Single Transformer Test
        print("\n[3/5] Single Transformer Test")
        transformer_single = self.single_inference_test("transformer_timeseries")

        # 4. Concurrent VAE Test
        print("\n[4/5] Concurrent VAE Test")
        vae_concurrent = self.concurrent_inference_test(
            "vae_timeseries",
            num_requests=50,
            max_workers=10
        )

        # 5. Concurrent Transformer Test
        print("\n[5/5] Concurrent Transformer Test")
        transformer_concurrent = self.concurrent_inference_test(
            "transformer_timeseries",
            num_requests=50,
            max_workers=10
        )

        # 최종 리포트
        self.generate_report(vae_single, transformer_single, vae_concurrent, transformer_concurrent)

    def generate_report(self, vae_single, transformer_single, vae_concurrent, transformer_concurrent):
        """성능 리포트 생성"""
        print("\n" + "="*60)
        print("Performance Report")
        print("="*60)

        # 단일 요청 성능
        print("\n📊 Single Request Performance:")
        print(f"  VAE:")
        if vae_single.get('success'):
            print(f"    - Total Time: {vae_single.get('total_time'):.3f}s")
            print(f"    - Inference Time: {vae_single.get('inference_time'):.3f}s")
        else:
            print(f"    - Failed: {vae_single.get('error', 'Unknown error')}")

        print(f"\n  Transformer:")
        if transformer_single.get('success'):
            print(f"    - Total Time: {transformer_single.get('total_time'):.3f}s")
            print(f"    - Inference Time: {transformer_single.get('inference_time'):.3f}s")
        else:
            print(f"    - Failed: {transformer_single.get('error', 'Unknown error')}")

        # 동시 요청 성능
        print("\n📊 Concurrent Request Performance (50 requests, 10 workers):")
        print(f"  VAE:")
        if isinstance(vae_concurrent, dict) and vae_concurrent.get('successful', 0) > 0:
            print(f"    - Throughput: {vae_concurrent.get('throughput', 0):.2f} RPS")
            print(f"    - Latency (Mean): {vae_concurrent.get('latency_mean', 0):.3f}s")
            print(f"    - Latency (P95): {vae_concurrent.get('latency_p95', 0):.3f}s")
            print(f"    - Success Rate: {vae_concurrent.get('successful', 0)}/{vae_concurrent.get('total_requests', 0)}")

        print(f"\n  Transformer:")
        if isinstance(transformer_concurrent, dict) and transformer_concurrent.get('successful', 0) > 0:
            print(f"    - Throughput: {transformer_concurrent.get('throughput', 0):.2f} RPS")
            print(f"    - Latency (Mean): {transformer_concurrent.get('latency_mean', 0):.3f}s")
            print(f"    - Latency (P95): {transformer_concurrent.get('latency_p95', 0):.3f}s")
            print(f"    - Success Rate: {transformer_concurrent.get('successful', 0)}/{transformer_concurrent.get('total_requests', 0)}")

        # 성능 목표 달성 여부
        print("\n🎯 Performance Goals:")

        vae_throughput = vae_concurrent.get('throughput', 0)
        transformer_throughput = transformer_concurrent.get('throughput', 0)

        goals = [
            ("Throughput > 30 RPS", vae_throughput > 30 or transformer_throughput > 30),
            ("P95 Latency < 200ms", vae_concurrent.get('latency_p95', 1) < 0.2),
            ("Success Rate > 95%",
             vae_concurrent.get('successful', 0) / vae_concurrent.get('total_requests', 1) > 0.95)
        ]

        for goal, achieved in goals:
            status = "✅" if achieved else "❌"
            print(f"  {status} {goal}")

        print("\n" + "="*60)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")


if __name__ == "__main__":
    simulator = InferenceSimulator()

    # 전체 테스트 스위트 실행
    simulator.run_full_test_suite()
