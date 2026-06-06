import torch
import torch.nn as nn
from .sp_network import SPNModule # 구현하신 SPN 모듈 재사용

class FeatureAlignmentBridge(nn.Module):
    """
    [Methodology 3.4] 하이브리드 차원 정렬 브릿지 (Hybrid Dimensionality Alignment Bridge)
    
    이질적인 두 아키텍처(Swin Transformer 백본과 YOLO26 Neck) 사이의 
    '공간적(Spatial)' 및 '채널적(Channel)' 텐서 형상 충돌을 해결하는 핵심 연결 모듈입니다.
    
    1. 공간 정렬 (Spatial Alignment):
       - Swin-Micro의 출력 Stride는 [4, 8, 16] (각각 C1, C2, C3)입니다.
       - YOLO26 Neck(PANet)이 기대하는 입력 Stride는 [8, 16, 32] (각각 P3, P4, P5)입니다.
       - 연산량 0(Zero-FLOPs)인 2x2 Average Pooling을 통해 정보의 손실을 최소화하면서 
         공간 해상도를 YOLO 규격으로 강제 동기화합니다.
         
    2. 채널 정렬 (Channel Projection via Cryptographic SPN):
       - 기존 하이브리드 모델들이 사용하는 1x1 Conv는 채널 차원 투영 시 O(C^2)의 연산 병목을 유발합니다.
       - 이를 해결하기 위해 앞서 정의한 '암호학적 SPN 모듈'을 재사용합니다.
       - Group Conv(S-Box)와 Channel Shuffle(P-Box)을 통해 연산량을 1/Groups로 줄이면서도
         결함 특징(Feature)의 확산(Diffusion)을 보장합니다.
         
    3. NPU 친화성 (Edge-Device Optimization):
       - Average Pooling과 SPN 모듈은 아이폰 ANE(Apple Neural Engine) 등 
         모바일 NPU에서 하드웨어 가속을 가장 잘 받는 연산 패턴입니다.

    4. 분포 정규화 (Cross-Domain Distribution Alignment):
       - Swin Transformer의 LayerNorm 기반 출력 분포와 YOLO 넥의 BatchNorm 기반 수용 분포 간의 불일치를 해소합니다.
       - 브릿지 출력단에 BatchNorm2d를 적용하여, 이질적인 도메인 간의 데이터 통계량을 동기화함으로써
         학습 초기 불안정성을 제거하고 NPU에서의 Conv-BN Fusion 최적화를 극대화합니다.
    """
    def __init__(self, swin_channels=[64, 128, 256], yolo_channels=[256, 512, 512], spn_groups=4):
        """
        Args:
            swin_channels: Swin-Micro 백본의 출력 채널 리스트 (C1, C2, C3)
            yolo_channels: YOLO26 Medium Neck이 요구하는 입력 채널 리스트 (P3, P4, P5)
                           * yaml의 width_multiple에 의해 실제 채널 수는 동적으로 계산되어 주입됨.
            spn_groups: SPN 모듈의 그룹 컨볼루션 분할 수
        """
        super().__init__()
        
        # 1. 공간 정렬을 위한 무손실 풀링 (Stride 4,8,16 -> 8,16,32)
        # NPU 타일링을 깨지 않기 위해 Padding=0, Kernel=2, Stride=2 사용
        self.spatial_align = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 2. 채널 정렬을 위한 암호학적 SPN 투영 (기존 1x1 Conv 대체)
        # C1 -> P3 투영
        self.bridge_p3 = self._make_bridge_block(swin_channels[0], yolo_channels[0], spn_groups)
        
        # C2 -> P4 투영
        self.bridge_p4 = self._make_bridge_block(swin_channels[1], yolo_channels[1], spn_groups)
        
        # C3 -> P5 투영
        self.bridge_p5 = self._make_bridge_block(swin_channels[2], yolo_channels[2], spn_groups)

    def _make_bridge_block(self, in_c, out_c, groups):
        """
        NPU 최적화를 위해 Conv(SPN) - BN - Act를 하나의 블록으로 묶습니다.
        """
        return nn.Sequential(
            SPNModule(in_channels=in_c, out_channels=out_c, groups=groups),
            nn.BatchNorm2d(out_c), # ★ 핵심 정규화 계층 추가
            nn.SiLU(inplace=True)
        )
    
    def forward(self, features):
        """
        Input: 
            features: Swin 백본에서 넘어온 피처맵 리스트 [C1, C2, C3]
                      - C1: [B, 64,  H/4,  W/4]
                      - C2: [B, 128, H/8,  W/8]
                      - C3: [B, 256, H/16, W/16]
        Output:
            aligned_features: YOLO 넥에 주입될 피처맵 리스트 [P3, P4, P5]
                      - P3: [B, yolo_C[0], H/8,  W/8]
                      - P4: [B, yolo_C[1], H/16, W/16]
                      - P5: [B, yolo_C[2], H/32, W/32]
        """
        c1, c2, c3 = features
        
        # Phase 1: Spatial & Channel Alignment for P3
        p3 = self.bridge_p3(self.spatial_align(c1))
        
        # Phase 2: Spatial & Channel Alignment for P4
        p4 = self.bridge_p4(self.spatial_align(c2))
        
        # Phase 3: Spatial & Channel Alignment for P5
        p5 = self.bridge_p5(self.spatial_align(c3))
        
        return [p3, p4, p5]