"""
Triton Python Backend - LSTM Time Series Model
핵심 추론 로직만 포함 (전/후처리는 클라이언트에서 처리)
"""
import numpy as np
import torch
import torch.nn as nn
import triton_python_backend_utils as pb_utils
import json


class LSTMNetwork(nn.Module):
    """LSTM 네트워크"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RTX 5060 (sm_120) 호환성을 위해 proj_size=0으로 설정 (flatten_parameters 우회)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, proj_size=0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class TritonPythonModel:
    """Triton Python Backend Model"""

    def initialize(self, args):
        """
        모델 초기화

        Args:
            args: Triton에서 제공하는 초기화 인자 (dict)
        """
        # 모델 설정 파라미터
        self.model_config = json.loads(args['model_config'])

        # 파라미터 추출
        self.sequence_length = 10
        self.forecast_steps = 5
        self.hidden_size = 64
        self.num_layers = 2

        # 파라미터 로딩 (config.pbtxt의 parameters 섹션에서)
        if 'parameters' in self.model_config:
            params = self.model_config['parameters']
            if 'sequence_length' in params:
                self.sequence_length = int(params['sequence_length']['string_value'])
            if 'forecast_steps' in params:
                self.forecast_steps = int(params['forecast_steps']['string_value'])
            if 'hidden_size' in params:
                self.hidden_size = int(params['hidden_size']['string_value'])
            if 'num_layers' in params:
                self.num_layers = int(params['num_layers']['string_value'])

        # GPU 설정 - RTX 5060 (sm_120) 호환성 처리
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LSTM 모델 생성
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        )

        # GPU로 이동 시도, 실패하면 CPU로 폴백
        try:
            self.model = self.model.to(self.device)
            # 테스트 실행하여 GPU 호환성 확인
            test_input = torch.randn(1, self.sequence_length, 1).to(self.device)
            with torch.no_grad():
                _ = self.model(test_input)
            print(f"[Triton LSTM] Model initialized on {self.device}")
        except RuntimeError as e:
            if "sm_120" in str(e) or "no kernel image" in str(e):
                print(f"[Triton LSTM] WARNING: GPU (sm_120) not compatible, falling back to CPU")
                print(f"[Triton LSTM] Error: {str(e)[:200]}")
                self.device = torch.device("cpu")
                self.model = self.model.cpu()
                print(f"[Triton LSTM] Model successfully loaded on CPU")
            else:
                raise e

        self.model.eval()

        print(f"[Triton LSTM] Config: seq_len={self.sequence_length}, "
              f"forecast={self.forecast_steps}, hidden={self.hidden_size}")

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

            # NumPy → PyTorch Tensor
            # input_data shape: [batch_size, sequence_length, features]
            input_torch = torch.from_numpy(input_data).float().to(self.device)

            # 추론 실행
            with torch.no_grad():
                predictions = []
                current_seq = input_torch

                # Autoregressive prediction
                for _ in range(self.forecast_steps):
                    pred = self.model(current_seq)
                    predictions.append(pred)

                    # 다음 시퀀스 생성 (마지막 예측값을 추가)
                    current_seq = torch.cat([
                        current_seq[:, 1:, :],  # 첫 번째 타임스텝 제거
                        pred.unsqueeze(2)        # 예측값 추가
                    ], dim=1)

                # 예측 결과 병합
                output = torch.cat(predictions, dim=1)  # [batch_size, forecast_steps]

            # PyTorch Tensor → NumPy
            output_numpy = output.cpu().numpy().astype(np.float32)

            # Triton 출력 텐서 생성
            output_tensor = pb_utils.Tensor("OUTPUT", output_numpy)

            # 응답 생성
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """모델 종료 시 정리"""
        print("[Triton LSTM] Model finalized")
        del self.model
