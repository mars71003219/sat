"""
Triton Python Backend - VAE Time Series Model
"""
import numpy as np
import torch
import torch.nn as nn
import triton_python_backend_utils as pb_utils
import json


class TimeSeriesVAE(nn.Module):
    """VAE for Time Series Forecasting"""

    def __init__(self, sequence_length=20, forecast_steps=10, latent_dim=32, hidden_dim=64):
        super(TimeSeriesVAE, self).__init__()

        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder (simplified for ONNX compatibility)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, stride=2, padding=1),  # 20 -> 10
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # 10 -> 5
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(hidden_dim * 2 * 5, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 * 5, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * 2 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        self.forecast_projection = nn.Linear(8, forecast_steps)

    def encode(self, x):
        h = self.encoder(x)  # Already flattened
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.hidden_dim * 2, 8)
        h = self.decoder(h)
        h = h.squeeze(1)
        forecast = self.forecast_projection(h)
        return forecast

    def forward(self, x):
        """For inference - deterministic (using mu only)"""
        mu, _ = self.encode(x)
        forecast = self.decode(mu)
        return forecast


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

        # 기본 파라미터
        self.sequence_length = 20
        self.forecast_steps = 10
        self.latent_dim = 32
        self.hidden_dim = 64

        # 파라미터 로딩 (config.pbtxt의 parameters 섹션에서)
        if 'parameters' in self.model_config:
            params = self.model_config['parameters']
            if 'sequence_length' in params:
                self.sequence_length = int(params['sequence_length']['string_value'])
            if 'forecast_steps' in params:
                self.forecast_steps = int(params['forecast_steps']['string_value'])
            if 'latent_dim' in params:
                self.latent_dim = int(params['latent_dim']['string_value'])
            if 'hidden_dim' in params:
                self.hidden_dim = int(params['hidden_dim']['string_value'])

        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # VAE 모델 생성
        self.model = TimeSeriesVAE(
            sequence_length=self.sequence_length,
            forecast_steps=self.forecast_steps,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        )

        # GPU로 이동 시도
        try:
            self.model = self.model.to(self.device)
            # 테스트 실행하여 GPU 호환성 확인
            test_input = torch.randn(1, 1, self.sequence_length).to(self.device)
            with torch.no_grad():
                _ = self.model(test_input)
            print(f"[Triton VAE] Model initialized on {self.device}")
        except RuntimeError as e:
            print(f"[Triton VAE] WARNING: GPU initialization failed, falling back to CPU")
            print(f"[Triton VAE] Error: {str(e)[:200]}")
            self.device = torch.device("cpu")
            self.model = self.model.cpu()
            print(f"[Triton VAE] Model successfully loaded on CPU")

        self.model.eval()

        print(f"[Triton VAE] Config: seq_len={self.sequence_length}, "
              f"forecast={self.forecast_steps}, latent_dim={self.latent_dim}, "
              f"hidden_dim={self.hidden_dim}")

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
            # input_data shape: [batch_size, 1, sequence_length]
            input_torch = torch.from_numpy(input_data).float().to(self.device)

            # 추론 실행
            with torch.no_grad():
                output = self.model(input_torch)  # [batch_size, forecast_steps]

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
        print("[Triton VAE] Model finalized")
        del self.model
