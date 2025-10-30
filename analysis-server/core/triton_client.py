"""
Triton Inference Client
전처리 → Triton 추론 → 후처리를 담당
"""
import numpy as np
import tritonclient.grpc as grpcclient
from typing import Dict, Any, List, Optional
import os
from utils.logger import get_logger

logger = get_logger(__name__)


class TritonInferenceClient:
    """
    Triton Inference Server 클라이언트

    역할:
    - 전처리: 원본 데이터를 모델 입력 형식으로 변환
    - 추론: Triton Server에 gRPC 요청
    - 후처리: 모델 출력을 최종 결과로 변환
    """

    def __init__(self, server_url: Optional[str] = None):
        """
        Args:
            server_url: Triton Server URL (기본값: 환경변수 TRITON_SERVER_URL)
        """
        self.server_url = server_url or os.getenv("TRITON_SERVER_URL", "triton-server:8001")
        self.client = None
        # Lazy initialization - connect on first use

    def _connect(self):
        """Triton Server 연결"""
        try:
            self.client = grpcclient.InferenceServerClient(url=self.server_url)

            if self.client.is_server_live():
                logger.info(f"[Triton Client] Connected to {self.server_url}")
            else:
                raise ConnectionError(f"Triton server at {self.server_url} is not live")

        except Exception as e:
            logger.error(f"[Triton Client] Connection failed: {e}")
            raise

    def infer_lstm(
        self,
        data: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LSTM 모델 추론 (전처리 + 추론 + 후처리)

        Args:
            data: 시계열 데이터 리스트
            config: 모델 설정 (sequence_length, forecast_steps 등)

        Returns:
            결과 딕셔너리 (predictions, confidence, metrics)
        """
        # 설정 추출
        sequence_length = config.get("sequence_length", 10)
        forecast_steps = config.get("forecast_steps", 5)

        # ========================================
        # 1. 전처리
        # ========================================
        data_array = np.array(data, dtype=np.float32)

        # 정규화
        mean = np.mean(data_array)
        std = np.std(data_array) + 1e-8
        normalized = (data_array - mean) / std

        # 시퀀스 생성
        if len(normalized) < sequence_length:
            # 패딩
            padded = np.pad(
                normalized,
                (sequence_length - len(normalized), 0),
                mode='edge'
            )
            input_sequence = padded
        else:
            # 마지막 sequence_length개만 사용
            input_sequence = normalized[-sequence_length:]

        # Reshape: [1, sequence_length, 1] → [batch, seq, features]
        input_data = input_sequence.reshape(1, sequence_length, 1).astype(np.float32)

        # ========================================
        # 2. Triton 추론
        # ========================================
        try:
            # Triton 입력 생성
            inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)

            # 출력 요청
            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

            # 추론 실행
            response = self.client.infer(
                model_name="lstm_timeseries",
                inputs=inputs,
                outputs=outputs
            )

            # 결과 추출
            output_data = response.as_numpy("OUTPUT")  # [1, forecast_steps]
            predictions_normalized = output_data[0]  # [forecast_steps]

        except Exception as e:
            logger.error(f"[Triton Client] LSTM inference failed: {e}")
            raise

        # ========================================
        # 3. 후처리
        # ========================================
        # 역정규화
        predictions = predictions_normalized * std + mean

        # 신뢰도 계산 (시간이 지날수록 감소)
        confidence = [0.95 - (i * 0.02) for i in range(len(predictions))]

        return {
            "predictions": predictions.tolist(),
            "confidence": confidence,
            "metrics": {
                "model_type": "LSTM",
                "forecast_steps": forecast_steps,
                "sequence_length": sequence_length,
                "mean": float(mean),
                "std": float(std)
            }
        }

    def infer_moving_average(
        self,
        data: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Moving Average 모델 추론 (전처리 + 추론 + 후처리)

        Args:
            data: 시계열 데이터 리스트
            config: 모델 설정 (window_size, forecast_steps 등)

        Returns:
            결과 딕셔너리 (predictions, confidence, upper_bound, lower_bound)
        """
        # 설정 추출
        window_size = config.get("window_size", 5)
        forecast_steps = config.get("forecast_steps", 5)

        # ========================================
        # 1. 전처리
        # ========================================
        data_array = np.array(data, dtype=np.float32)

        # Reshape: [1, len(data)] → [batch, sequence_length]
        input_data = data_array.reshape(1, -1).astype(np.float32)

        # ========================================
        # 2. Triton 추론
        # ========================================
        try:
            # Triton 입력 생성
            inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)

            # 출력 요청
            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

            # 추론 실행
            response = self.client.infer(
                model_name="moving_average",
                inputs=inputs,
                outputs=outputs
            )

            # 결과 추출
            output_data = response.as_numpy("OUTPUT")  # [1, forecast_steps]
            predictions = output_data[0]  # [forecast_steps]

        except Exception as e:
            logger.error(f"[Triton Client] Moving Average inference failed: {e}")
            raise

        # ========================================
        # 3. 후처리
        # ========================================
        # 신뢰도 계산
        base_confidence = 0.90
        confidence = [base_confidence - (i * 0.03) for i in range(len(predictions))]

        # 신뢰구간 계산 (95% CI)
        std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
        upper_bound = predictions + (1.96 * std_dev)
        lower_bound = predictions - (1.96 * std_dev)

        return {
            "predictions": predictions.tolist(),
            "confidence": confidence,
            "upper_bound": upper_bound.tolist(),
            "lower_bound": lower_bound.tolist(),
            "metrics": {
                "model_type": "Moving Average",
                "window_size": window_size,
                "forecast_steps": forecast_steps,
                "std_dev": float(std_dev)
            }
        }

    def infer_vae(
        self,
        data: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        VAE 모델 추론 (전처리 + 추론 + 후처리)

        Args:
            data: 시계열 데이터 리스트 (20 steps)
            config: 모델 설정 (sequence_length=20, forecast_steps=10)

        Returns:
            결과 딕셔너리 (predictions, confidence, metrics)
        """
        # 설정 추출
        sequence_length = config.get("sequence_length", 20)
        forecast_steps = config.get("forecast_steps", 10)

        # ========================================
        # 1. 전처리
        # ========================================
        data_array = np.array(data, dtype=np.float32)

        # 정규화
        mean = np.mean(data_array)
        std = np.std(data_array) + 1e-8
        normalized = (data_array - mean) / std

        # 시퀀스 생성
        if len(normalized) < sequence_length:
            # 패딩
            padded = np.pad(
                normalized,
                (sequence_length - len(normalized), 0),
                mode='edge'
            )
            input_sequence = padded
        else:
            # 마지막 sequence_length개만 사용
            input_sequence = normalized[-sequence_length:]

        # Reshape: [1, 1, 20] → [batch, channels, sequence]
        input_data = input_sequence.reshape(1, 1, sequence_length).astype(np.float32)

        # ========================================
        # 2. Triton 추론
        # ========================================
        try:
            # Triton 입력 생성
            inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)

            # 출력 요청
            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

            # 추론 실행
            response = self.client.infer(
                model_name="vae_timeseries",
                inputs=inputs,
                outputs=outputs
            )

            # 결과 추출
            output_data = response.as_numpy("OUTPUT")  # [1, forecast_steps]
            predictions_normalized = output_data[0]  # [forecast_steps]

        except Exception as e:
            logger.error(f"[Triton Client] VAE inference failed: {e}")
            raise

        # ========================================
        # 3. 후처리
        # ========================================
        # 역정규화
        predictions = predictions_normalized * std + mean

        # 신뢰도 계산 (VAE는 확률적 모델)
        confidence = [0.92 - (i * 0.015) for i in range(len(predictions))]

        return {
            "predictions": predictions.tolist(),
            "confidence": confidence,
            "metrics": {
                "model_type": "VAE",
                "forecast_steps": forecast_steps,
                "sequence_length": sequence_length,
                "mean": float(mean),
                "std": float(std)
            }
        }

    def infer_transformer(
        self,
        data: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transformer 모델 추론 (전처리 + 추론 + 후처리)

        Args:
            data: 시계열 데이터 리스트 (20 steps)
            config: 모델 설정 (sequence_length=20, forecast_steps=10)

        Returns:
            결과 딕셔너리 (predictions, confidence, metrics)
        """
        # 설정 추출
        sequence_length = config.get("sequence_length", 20)
        forecast_steps = config.get("forecast_steps", 10)

        # ========================================
        # 1. 전처리
        # ========================================
        data_array = np.array(data, dtype=np.float32)

        # 정규화
        mean = np.mean(data_array)
        std = np.std(data_array) + 1e-8
        normalized = (data_array - mean) / std

        # 시퀀스 생성
        if len(normalized) < sequence_length:
            # 패딩
            padded = np.pad(
                normalized,
                (sequence_length - len(normalized), 0),
                mode='edge'
            )
            input_sequence = padded
        else:
            # 마지막 sequence_length개만 사용
            input_sequence = normalized[-sequence_length:]

        # Reshape: [1, 20, 1] → [batch, sequence, features]
        input_data = input_sequence.reshape(1, sequence_length, 1).astype(np.float32)

        # ========================================
        # 2. Triton 추론
        # ========================================
        try:
            # Triton 입력 생성
            inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)

            # 출력 요청
            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

            # 추론 실행
            response = self.client.infer(
                model_name="transformer_timeseries",
                inputs=inputs,
                outputs=outputs
            )

            # 결과 추출
            output_data = response.as_numpy("OUTPUT")  # [1, forecast_steps]
            predictions_normalized = output_data[0]  # [forecast_steps]

        except Exception as e:
            logger.error(f"[Triton Client] Transformer inference failed: {e}")
            raise

        # ========================================
        # 3. 후처리
        # ========================================
        # 역정규화
        predictions = predictions_normalized * std + mean

        # 신뢰도 계산 (Transformer는 attention 기반)
        confidence = [0.94 - (i * 0.012) for i in range(len(predictions))]

        return {
            "predictions": predictions.tolist(),
            "confidence": confidence,
            "metrics": {
                "model_type": "Transformer",
                "forecast_steps": forecast_steps,
                "sequence_length": sequence_length,
                "mean": float(mean),
                "std": float(std)
            }
        }

    def infer(
        self,
        model_name: str,
        data: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        범용 추론 메서드

        Args:
            model_name: 모델 이름 ("lstm_timeseries", "moving_average", "vae_timeseries", "transformer_timeseries")
            data: 입력 데이터
            config: 모델 설정

        Returns:
            추론 결과
        """
        if model_name == "lstm_timeseries":
            return self.infer_lstm(data, config)
        elif model_name == "moving_average":
            return self.infer_moving_average(data, config)
        elif model_name == "vae_timeseries":
            return self.infer_vae(data, config)
        elif model_name == "transformer_timeseries":
            return self.infer_transformer(data, config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def is_server_ready(self) -> bool:
        """Triton Server 상태 확인"""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            logger.error(f"[Triton Client] Server health check failed: {e}")
            return False

    def is_model_ready(self, model_name: str) -> bool:
        """모델 준비 상태 확인"""
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            logger.error(f"[Triton Client] Model '{model_name}' health check failed: {e}")
            return False

    def get_model_metadata(self, model_name: str) -> Dict:
        """모델 메타데이터 조회"""
        try:
            metadata = self.client.get_model_metadata(model_name)
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": list(inp.shape)
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": list(out.shape)
                    }
                    for out in metadata.outputs
                ]
            }
        except Exception as e:
            logger.error(f"[Triton Client] Failed to get metadata for '{model_name}': {e}")
            raise


# 싱글톤 인스턴스
# Lazy singleton - will be initialized on first use
triton_client = None

def get_triton_client():
    """Get or create Triton client singleton"""
    global triton_client
    if triton_client is None:
        triton_client = TritonInferenceClient()
        triton_client._connect()
    return triton_client
