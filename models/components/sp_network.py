import torch
import torch.nn as nn

class SPNModule(nn.Module):
    """
    [Methodology 3.2] Substitution-Permutation Network (SPN) 모듈
    
    암호학의 SPN 구조를 딥러닝 피처 맵 변환에 이식하여, 1x1 Conv의 연산량을 
    그룹화(Substitution)와 셔플링(Permutation)으로 최적화한 모듈입니다.
    
    1. Substitution (치환, S-Box): 
       - 전체 채널을 Group으로 나누어 Group Conv를 수행합니다.
       - 연산량(FLOPs)을 1/groups 비율로 직접적으로 줄입니다.
    2. Permutation (순열, P-Box): 
       - Channel Shuffle 연산을 통해 그룹화로 인해 끊어진 채널 간의 정보 흐름(Information Flow)을 
         연산량 0(Zero-FLOPs)으로 복구합니다.
    
    [입/출력 요구사항]
    - Input : [Batch, in_channels, Height, Width]
    - Output: [Batch, out_channels, Height, Width]
    - 제약조건: in_channels는 groups로 나누어떨어져야 함.
    """
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        
        # S-Box: Group Conv를 사용하여 파라미터와 연산량을 대폭 절감 (1/groups)
        # 1x1 Conv는 공간 정보를 유지하면서 채널만 변환함
        self.s_box = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        [Tensor Flow]
        Input:  (B, in_C, H, W)
        Output: (B, out_C, H, W)
        """
        # S-Box 연산
        x = self.act(self.s_box(x))
        # P-Box 연산: 채널 셔플 (Zero-FLOPs)
        x = self._channel_shuffle(x, self.groups)
        return x

    def _channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous().view(b, -1, h, w)
        return x