#!/usr/bin/env python3
"""단일 모델 테스트"""
import requests
import time
import json

def test_single_model(model_name: str):
    """단일 모델 테스트"""
    base_url = "http://localhost:8000"

    # 테스트 데이터 (20 steps for VAE/Transformer)
    if model_name in ["vae_timeseries", "transformer_timeseries"]:
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        config = {"sequence_length": 20, "forecast_steps": 10}
    else:
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        config = {"sequence_length": 10, "forecast_steps": 5}

    payload = {
        "model_name": model_name,
        "data": data,
        "config": config
    }

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Submit job
    print("Submitting job...")
    start = time.time()
    response = requests.post(f"{base_url}/api/v1/inference/submit", json=payload)

    if response.status_code != 200:
        print(f"❌ Submit failed: {response.status_code}")
        print(response.text)
        return

    job_id = response.json()["job_id"]
    print(f"✅ Job ID: {job_id}")

    # Wait for result
    print("Waiting for result...")
    for i in range(30):
        time.sleep(1)
        result_response = requests.get(f"{base_url}/api/v1/inference/result/{job_id}")

        if result_response.status_code == 200:
            result = result_response.json()
            if result.get("status") == "completed":
                total_time = time.time() - start
                print(f"\n✅ Success!")
                print(f"   Total Time: {total_time:.3f}s")
                print(f"   Inference Time: {result['metrics']['inference_time']:.3f}s")
                print(f"   Model Type: {result['metrics']['model_type']}")
                print(f"   Predictions: {result['predictions'][:5]}...")
                print(f"   Confidence: {result['confidence'][:5]}...")
                return
            elif result.get("status") == "failed":
                print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
                return

        if i % 5 == 0:
            print(f"   Waiting... ({i}s)")

    print(f"❌ Timeout after 30s")


if __name__ == "__main__":
    # Test VAE
    test_single_model("vae_timeseries")

    # Test Transformer
    test_single_model("transformer_timeseries")
