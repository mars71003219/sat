#!/usr/bin/env python3
"""
범용 PyTorch 모델 변환 도구
ONNX Runtime 또는 TensorRT 백엔드 선택 가능

사용법:
    # ONNX Runtime (기본, GPU 호환성 좋음)
    python convert_model.py \
        --model-path model.pth \
        --model-class MyModel \
        --model-module my_model.py \
        --model-name my_model \
        --input-shape 1,20 \
        --output-shape 10 \
        --backend onnx

    # TensorRT (최고 성능, GPU 제한적)
    python convert_model.py \
        --model-path model.pth \
        --model-class MyModel \
        --model-module my_model.py \
        --model-name my_model \
        --input-shape 1,20 \
        --output-shape 10 \
        --backend tensorrt
"""

import argparse
import os
import sys
import torch
import importlib.util
from pathlib import Path


def load_pytorch_model(model_path, model_class_name, model_module_path=None):
    """PyTorch 모델 로드"""
    print(f"[1/4] Loading PyTorch model from {model_path}")

    if model_module_path:
        spec = importlib.util.spec_from_file_location("model_module", model_module_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        ModelClass = getattr(model_module, model_class_name)
    else:
        ModelClass = globals().get(model_class_name)
        if ModelClass is None:
            raise ValueError(f"Model class '{model_class_name}' not found. Use --model-module to specify.")

    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_config = checkpoint.get('model_config', {})
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            model_config = {}
    else:
        state_dict = checkpoint
        model_config = {}

    if model_config:
        model = ModelClass(**model_config)
    else:
        model = ModelClass()

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded: {model_class_name}")
    return model


def export_to_onnx(model, input_shape, output_path, dynamic_batch=True):
    """PyTorch 모델을 ONNX로 변환"""
    print(f"\n[2/4] Exporting to ONNX: {output_path}")

    dummy_input = torch.randn(1, *input_shape)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes['INPUT'] = {0: 'batch_size'}
        dynamic_axes['OUTPUT'] = {0: 'batch_size'}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['INPUT'],
        output_names=['OUTPUT'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    print(f"✅ ONNX model exported: {output_path}")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
    except ImportError:
        print("⚠️  ONNX package not installed, skipping verification")

    return output_path


def build_tensorrt_engine(
    onnx_path,
    engine_path,
    input_shape,
    min_batch=1,
    opt_batch=16,
    max_batch=64,
    fp16=True,
):
    """ONNX를 TensorRT 엔진으로 변환"""
    print(f"\n[3/4] Building TensorRT engine: {engine_path}")

    import subprocess
    import shutil

    trtexec_path = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"

    if not os.path.exists(trtexec_path):
        print(f"❌ trtexec not found at {trtexec_path}")
        print("⚠️  TensorRT conversion requires running inside Triton container")
        sys.exit(1)

    input_shape_str = "x".join(map(str, input_shape))

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--shapes=INPUT:1x{input_shape_str}",
        f"--minShapes=INPUT:{min_batch}x{input_shape_str}",
        f"--optShapes=INPUT:{opt_batch}x{input_shape_str}",
        f"--maxShapes=INPUT:{max_batch}x{input_shape_str}",
        "--verbose",
    ]

    if fp16:
        cmd.append("--fp16")

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ TensorRT engine build failed!")
        print(result.stderr)
        sys.exit(1)

    log_path = engine_path.replace(".plan", "_build.log")
    with open(log_path, 'w') as f:
        f.write(result.stdout)

    print(f"✅ TensorRT engine built: {engine_path}")
    print(f"✅ Build log saved: {log_path}")

    return engine_path


def create_triton_config(
    model_name,
    model_file,
    backend,
    output_dir,
    input_shape,
    output_shape,
    max_batch_size=64,
    instance_count=2,
    use_gpu=True
):
    """Triton 모델 리포지토리 구조 및 config 생성"""
    print(f"\n[4/4] Creating Triton model repository structure")

    model_dir = Path(output_dir) / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # 모델 파일 복사
    import shutil
    if backend == "onnx":
        target_file = version_dir / "model.onnx"
        platform = "onnxruntime_onnx"
    else:  # tensorrt
        target_file = version_dir / "model.plan"
        platform = "tensorrt_plan"

    shutil.copy(model_file, target_file)

    print(f"✅ Model directory created: {model_dir}")
    print(f"✅ Model file copied to: {target_file}")

    # config.pbtxt 생성
    config_path = model_dir / "config.pbtxt"

    input_dims = ", ".join(map(str, input_shape))
    output_dims = ", ".join(map(str, output_shape)) if isinstance(output_shape, (list, tuple)) else str(output_shape)

    # Instance 설정
    if use_gpu:
        instance_config = f'''instance_group [
  {{
    count: {instance_count}
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]'''
    else:
        instance_config = f'''instance_group [
  {{
    count: {instance_count}
    kind: KIND_CPU
  }}
]'''

    # 백엔드별 최적화 설정
    if backend == "onnx" and use_gpu:
        optimization = '''optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "cuda"
    } ]
  }
}'''
    elif backend == "tensorrt":
        optimization = '''optimization {
  cuda {
    graphs: true
  }
}'''
    else:
        optimization = ""

    config_content = f'''name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

# 입력 정의
input [
  {{
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ {input_dims} ]
  }}
]

# 출력 정의
output [
  {{
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ {output_dims} ]
  }}
]

# Dynamic Batching 설정
dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100000
}}

# 인스턴스 설정
{instance_config}

# 최적화 설정
{optimization}
'''

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ Config file created: {config_path}")

    return model_dir


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch 모델 변환 도구 (ONNX Runtime / TensorRT 선택)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ONNX Runtime (GPU 호환성 좋음, 권장)
  python convert_model.py \\
      --model-path vae.pth \\
      --model-class TimeSeriesVAE \\
      --model-module train_vae.py \\
      --model-name vae_timeseries \\
      --input-shape 1,20 \\
      --output-shape 10 \\
      --backend onnx

  # TensorRT (최고 성능, GPU 제한적)
  python convert_model.py \\
      --model-path transformer.pth \\
      --model-class TimeSeriesTransformer \\
      --model-module train_transformer.py \\
      --model-name transformer_timeseries \\
      --input-shape 20,1 \\
      --output-shape 10 \\
      --backend tensorrt \\
      --fp16
        """
    )

    # 필수 인자
    parser.add_argument('--model-path', required=True, help='PyTorch 모델 파일 (.pth)')
    parser.add_argument('--model-class', required=True, help='모델 클래스 이름')
    parser.add_argument('--model-name', required=True, help='Triton 모델 이름')
    parser.add_argument('--input-shape', required=True, help='입력 shape (쉼표 구분, 예: 1,20)')
    parser.add_argument('--output-shape', required=True, help='출력 shape (예: 10)')

    # 백엔드 선택
    parser.add_argument('--backend', choices=['onnx', 'tensorrt'], default='onnx',
                        help='백엔드 선택 (기본: onnx)')

    # 옵션 인자
    parser.add_argument('--model-module', help='모델 클래스가 정의된 Python 파일')
    parser.add_argument('--output-dir', default='./model_repository', help='모델 리포지토리 디렉토리')
    parser.add_argument('--min-batch', type=int, default=1, help='최소 배치 크기 (TensorRT)')
    parser.add_argument('--opt-batch', type=int, default=16, help='최적 배치 크기 (TensorRT)')
    parser.add_argument('--max-batch', type=int, default=64, help='최대 배치 크기')
    parser.add_argument('--instance-count', type=int, default=2, help='인스턴스 개수')
    parser.add_argument('--fp16', action='store_true', help='FP16 정밀도 사용 (TensorRT)')
    parser.add_argument('--cpu', action='store_true', help='CPU 모드 사용')

    args = parser.parse_args()

    # Shape 파싱
    input_shape = list(map(int, args.input_shape.split(',')))
    output_shape = list(map(int, args.output_shape.split(',')))
    if len(output_shape) == 1:
        output_shape = output_shape[0]

    backend_name = "ONNX Runtime" if args.backend == "onnx" else "TensorRT"
    device_name = "CPU" if args.cpu else "GPU"

    print("="*70)
    print(f"PyTorch Model Conversion Tool")
    print("="*70)
    print(f"Model Path:    {args.model_path}")
    print(f"Model Class:   {args.model_class}")
    print(f"Model Name:    {args.model_name}")
    print(f"Input Shape:   {input_shape}")
    print(f"Output Shape:  {output_shape}")
    print(f"Backend:       {backend_name}")
    print(f"Device:        {device_name}")
    if args.backend == "tensorrt":
        print(f"FP16:          {args.fp16}")
    print("="*70)

    # 1. PyTorch 모델 로드
    model = load_pytorch_model(args.model_path, args.model_class, args.model_module)

    # 2. ONNX 변환 (공통)
    onnx_path = f"{args.model_name}.onnx"
    export_to_onnx(model, input_shape, onnx_path)

    # 3. 백엔드별 처리
    if args.backend == "tensorrt":
        # TensorRT 엔진 빌드
        engine_path = f"{args.model_name}.plan"
        build_tensorrt_engine(
            onnx_path,
            engine_path,
            input_shape,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            fp16=args.fp16,
        )
        model_file = engine_path
    else:
        # ONNX 그대로 사용
        model_file = onnx_path
        print(f"\n[3/4] Using ONNX model directly (skipping TensorRT)")

    # 4. Triton 모델 리포지토리 생성
    model_dir = create_triton_config(
        args.model_name,
        model_file,
        args.backend,
        args.output_dir,
        input_shape,
        output_shape,
        max_batch_size=args.max_batch,
        instance_count=args.instance_count,
        use_gpu=not args.cpu
    )

    # 완료
    print(f"\n✅ Conversion completed successfully!")
    print("="*70)
    print(f"Model Repository: {model_dir}")
    print(f"  ├── config.pbtxt (platform: {args.backend})")
    print(f"  └── 1/")
    if args.backend == "onnx":
        print(f"      └── model.onnx")
    else:
        print(f"      └── model.plan")
    print("="*70)
    print("\nNext steps:")
    print(f"1. Restart Triton: docker compose restart triton-server")
    print(f"2. Test model: curl http://localhost:8500/v2/models/{args.model_name}")
    print("="*70)


if __name__ == "__main__":
    main()
