#!/bin/bash
"""
외부 학습 서버에서 받은 .pth 파일을 ONNX 또는 TensorRT로 변환하는 예시

사용법:
    # ONNX 변환
    ./convert_example.sh /path/to/model.pth onnx

    # TensorRT 변환
    ./convert_example.sh /path/to/model.pth tensorrt

전제조건:
    - 외부 학습 서버에서 학습된 .pth 파일
    - 모델 클래스 정의 파일 (.py)
"""

set -e

# 인자 확인
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model.pth> <backend> [model_name]"
    echo ""
    echo "Arguments:"
    echo "  model.pth  - Path to PyTorch model file"
    echo "  backend    - onnx or tensorrt"
    echo "  model_name - Optional model name (default: extracted from filename)"
    echo ""
    echo "Example:"
    echo "  $0 /models/vae_model.pth onnx vae_timeseries"
    exit 1
fi

MODEL_PATH=$1
BACKEND=$2
MODEL_NAME=${3:-$(basename "$MODEL_PATH" .pth)}

echo "========================================"
echo "Model Conversion Tool"
echo "========================================"
echo "Model Path: ${MODEL_PATH}"
echo "Backend: ${BACKEND}"
echo "Model Name: ${MODEL_NAME}"
echo "========================================"

# 모델 파일 존재 확인
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model file not found: ${MODEL_PATH}"
    exit 1
fi

# 백엔드 확인
if [ "${BACKEND}" != "onnx" ] && [ "${BACKEND}" != "tensorrt" ]; then
    echo "Error: Backend must be 'onnx' or 'tensorrt'"
    exit 1
fi

echo ""
echo "Starting conversion..."
echo ""

# Docker 컨테이너 내에서 변환 실행
docker run --rm \
    -v /mnt/c/projects/satellite:/workspace \
    -v $(dirname "${MODEL_PATH}"):/external_models \
    -w /workspace/scripts \
    nvcr.io/nvidia/tritonserver:24.10-py3 \
    bash -c "
        set -e

        pip3 install -q torch torchvision onnx

        # 실제 변환 명령어는 모델 구조에 따라 조정 필요
        # 예시:
        # python3 convert_model.py \
        #     --model-path /external_models/$(basename ${MODEL_PATH}) \
        #     --model-class YourModelClass \
        #     --model-module your_model_definition.py \
        #     --model-name ${MODEL_NAME} \
        #     --input-shape <your_input_shape> \
        #     --output-shape <your_output_shape> \
        #     --backend ${BACKEND} \
        #     --output-dir /workspace/model_repository

        echo ''
        echo '⚠️  Please customize this script with your model specifics:'
        echo '   - model-class: Your PyTorch model class name'
        echo '   - model-module: Python file with model definition'
        echo '   - input-shape: Model input dimensions'
        echo '   - output-shape: Model output dimensions'
    "

echo ""
echo "========================================"
echo "Conversion completed!"
echo "========================================"
