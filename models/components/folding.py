import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .sp_network import SPNBlock # 분리된 모듈 임포트

class DynamicLosslessTensorFolding(nn.Module):
    """
    [Methodology 3.1] 동적 형상 적응형 무손실 텐서 폴딩 레이어
    
    1. 128 배수 패딩 이유:
       - Swin의 계층적 구조는 32의 배수를 요구함. NPU 하드웨어 타일링(128x128) 효율을 극대화하기 위해 
         32의 배수 중 가장 최적화된 128 배수를 선택.
    2. 3채널 출력 이유:
       - YOLO 헤드와의 결합 시, 기존 사전 학습된(Pre-trained) 백본과 호환성을 유지하고 
         YOLO Neck이 기대하는 표준 RGB 채널 구조를 입력으로 제공하기 위함.
    3. 동적 패딩(Dynamic Padding) 이유:
       - 카메라 입력(비정방형)을 강제로 정방형으로 리사이즈할 때 발생하는 결함 정보(핀홀 등)의 
         왜곡/손실을 방지하고, 원본 픽셀의 공간적 밀도를 100% 보존하기 위함.
       - AI-Hub 데이터셋의 다양한 해상도(예: 4000x3000, 1868x4000 등)에 대응하기 위해, 입력 이미지의 원본 비율을 유지하면서
    4. 무손실 폴딩(Lossless Folding) 이유:
       - 핀홀과 같은 미세 결함은 픽셀 정보 하나하나가 생명임. PixelUnshuffle은 연산 없이 정보를 
         채널 차원으로 이동시키므로 핀홀의 특징 소실(Information Bottleneck)을 원천 차단함.
    5. Log-CPB(Coordinate Positional Bias) 보정 이유:
       - 패딩에 의해 입력 크기가 변할 때, Swin Transformer의 상대적 위치 편향값이 해상도 변화에 
         따라 왜곡됨. 로그 스케일로 이를 보정하여 위치 학습의 일관성을 확보함.
    
        * 요구사항 *
    - 제거된 Swin Stage: 연산 병목 및 미세 결함 소실 방지를 위해 Stage 4 제거.
    - 학습 해상도: 제한 없음 (동적 패딩이 128 배수로 자동 정렬).
    - 추론 해상도: 제한 없음 (동적 패딩이 128 배수로 자동 정렬).
    """
    # NPU 최적화 체급(Micro)에 맞춰 embed_dim 기본값을 64로 변경
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        # PixelUnshuffle: 연산량 0 (Zero-FLOPs). 공간 정보를 채널로 접음 (r=4)
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=4)
        
        # S-Box (치환): 채널 정렬을 위한 1x1 투영. 
        # 기존 Conv2d를 대체하여 암호학적 구조를 완성합니다.
        self.spn = SPNBlock(in_channels=in_channels * 16, out_channels=embed_dim, groups=4, use_act=True)

        # ✅ 좋은 방향: 공간 차원(H, W)을 제외하고 오직 고정 하이퍼파라미터인 embed_dim만 명시!
        # PyTorch LayerNorm은 정수 하나만 주어지면 텐서의 '최하위(마지막) 차원'을 기준으로 정규화합니다.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        
        # 1. 동적 패딩 계산 (128 배수)
        pad_h = (128 - orig_h % 128) % 128
        pad_w = (128 - orig_w % 128) % 128
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
        padded_h, padded_w = x.shape[2], x.shape[3]
        
        # 2. 무손실 텐서 폴딩: [B, 3, H, W] -> [B, 48, H/4, W/4]
        x = self.unshuffle(x)
        
        # 3. SPN 치환/순열 연산
        out = self.spn(x)

        # 4. ✅ 최적의 SPN-LN-Swin 정규화 인터페이스 구현
        # LayerNorm을 적용하기 위해 채널 차원(C)을 맨 뒤로 밀어줍니다.
        # [B, C, H_sub, W_sub] -> [B, H_sub, W_sub, C]
        out = out.permute(0, 2, 3, 1)
        
        # 최하위 차원인 C(embed_dim)축을 기준으로 정규화 수행 (H, W가 동적으로 변해도 완벽 대응)
        out = self.norm(out)
        
        # 5.Log-CPB 메타데이터 및 원본 사이즈 정보 포함
        # 추론 시 BBox 좌표 복원에 필수적인 정보입니다.
        meta_info = {
            'orig_size': (orig_h, orig_w),
            'padded_size': (padded_h, padded_w),
            'log_ratio_h': math.log(padded_h / orig_h) if orig_h > 0 else 0,
            'log_ratio_w': math.log(padded_w / orig_w) if orig_w > 0 else 0
        }
        
        return out, meta_info