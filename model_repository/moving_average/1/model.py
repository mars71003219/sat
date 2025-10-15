"""
Triton Python Backend - Moving Average Model
핵심 추론 로직만 포함 (전/후처리는 클라이언트에서 처리)
"""
import numpy as np
import triton_python_backend_utils as pb_utils
import json


class TritonPythonModel:
    """Triton Python Backend Model - Moving Average"""

    def initialize(self, args):
        """
        모델 초기화

        Args:
            args: Triton에서 제공하는 초기화 인자 (dict)
        """
        # 모델 설정 파라미터
        self.model_config = json.loads(args['model_config'])

        # 기본 파라미터
        self.window_size = 5
        self.forecast_steps = 5
        self.include_trend = True

        # 파라미터 로딩 (config.pbtxt의 parameters 섹션에서)
        if 'parameters' in self.model_config:
            params = self.model_config['parameters']
            if 'window_size' in params:
                self.window_size = int(params['window_size']['string_value'])
            if 'forecast_steps' in params:
                self.forecast_steps = int(params['forecast_steps']['string_value'])
            if 'include_trend' in params:
                trend_val = params['include_trend']['string_value'].lower()
                self.include_trend = trend_val == 'true'

        print(f"[Triton MA] Model initialized")
        print(f"[Triton MA] Config: window={self.window_size}, "
              f"forecast={self.forecast_steps}, trend={self.include_trend}")

    def execute(self, requests):
        """
        추론 실행 (배치 처리 지원)

        Args:
            requests: Triton inference requests (list)

        Returns:
            responses: Triton inference responses (list)
        """
        responses = []

        for request in requests:
            # 입력 텐서 추출
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()

            # 배치 처리
            batch_predictions = []

            for data in input_data:
                # 단일 시계열에 대한 예측
                predictions = self._predict_single(data)
                batch_predictions.append(predictions)

            # NumPy 배열로 변환
            output_numpy = np.array(batch_predictions, dtype=np.float32)

            # Triton 출력 텐서 생성
            output_tensor = pb_utils.Tensor("OUTPUT", output_numpy)

            # 응답 생성
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def _predict_single(self, data: np.ndarray) -> np.ndarray:
        """
        단일 시계열에 대한 이동평균 예측

        Args:
            data: 입력 시계열 데이터 (1D numpy array)

        Returns:
            predictions: 예측 결과 (1D numpy array)
        """
        predictions = []
        current_data = data.copy()

        for _ in range(self.forecast_steps):
            # 윈도우 선택
            if len(current_data) >= self.window_size:
                window = current_data[-self.window_size:]
            else:
                window = current_data

            # 이동평균 계산
            ma_pred = np.mean(window)

            # 트렌드 추가
            if self.include_trend and len(current_data) >= 2:
                window_len = min(len(current_data), self.window_size)
                trend = (current_data[-1] - current_data[-window_len]) / window_len
                ma_pred += trend

            predictions.append(ma_pred)

            # 다음 예측을 위해 데이터 업데이트
            current_data = np.append(current_data, ma_pred)

        return np.array(predictions, dtype=np.float32)

    def finalize(self):
        """모델 종료 시 정리"""
        print("[Triton MA] Model finalized")
