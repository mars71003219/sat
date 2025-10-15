"""
Triton Python Backend - Transformer Time Series Model
"""
import numpy as np
import torch
import torch.nn as nn
import triton_python_backend_utils as pb_utils
import json
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer for Time Series Forecasting"""

    def __init__(
        self,
        sequence_length=20,
        forecast_steps=10,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model * sequence_length, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, forecast_steps),
        )

    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        encoded_flat = encoded.reshape(encoded.size(0), -1)
        forecast = self.forecast_head(encoded_flat)
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
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 3
        self.dim_feedforward = 256
        self.dropout = 0.1

        # 파라미터 로딩 (config.pbtxt의 parameters 섹션에서)
        if 'parameters' in self.model_config:
            params = self.model_config['parameters']
            if 'sequence_length' in params:
                self.sequence_length = int(params['sequence_length']['string_value'])
            if 'forecast_steps' in params:
                self.forecast_steps = int(params['forecast_steps']['string_value'])
            if 'd_model' in params:
                self.d_model = int(params['d_model']['string_value'])
            if 'nhead' in params:
                self.nhead = int(params['nhead']['string_value'])
            if 'num_layers' in params:
                self.num_layers = int(params['num_layers']['string_value'])
            if 'dim_feedforward' in params:
                self.dim_feedforward = int(params['dim_feedforward']['string_value'])
            if 'dropout' in params:
                self.dropout = float(params['dropout']['string_value'])

        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformer 모델 생성
        self.model = TimeSeriesTransformer(
            sequence_length=self.sequence_length,
            forecast_steps=self.forecast_steps,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )

        # GPU로 이동 시도
        try:
            self.model = self.model.to(self.device)
            # 테스트 실행하여 GPU 호환성 확인
            test_input = torch.randn(1, self.sequence_length, 1).to(self.device)
            with torch.no_grad():
                _ = self.model(test_input)
            print(f"[Triton Transformer] Model initialized on {self.device}")
        except RuntimeError as e:
            print(f"[Triton Transformer] WARNING: GPU initialization failed, falling back to CPU")
            print(f"[Triton Transformer] Error: {str(e)[:200]}")
            self.device = torch.device("cpu")
            self.model = self.model.cpu()
            print(f"[Triton Transformer] Model successfully loaded on CPU")

        self.model.eval()

        print(f"[Triton Transformer] Config: seq_len={self.sequence_length}, "
              f"forecast={self.forecast_steps}, d_model={self.d_model}, "
              f"nhead={self.nhead}, num_layers={self.num_layers}")

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
            # input_data shape: [batch_size, sequence_length, 1]
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
        print("[Triton Transformer] Model finalized")
        del self.model
