import torch
import torch.nn as nn
import torch.nn.functional as F
from .sp_network import SPNBlock # SPN 모듈 재사용

class SPN_PANet(nn.Module):
    """
    [Methodology 3.5] SPN-based Path Aggregation Network (SPN-PANet)
    
    기존 YOLO Neck의 무거운 C3k2 및 Dense 1x1 Conv를 암호학적 SPN 구조로 대체하여,
    연산량(FLOPs)을 극단적으로 최소화한 양방향(Top-Down & Bottom-Up) 피처 융합 모듈.
    """
    def __init__(self, channels=[256, 512, 512], spn_groups=4):
        """
        Args:
            channels: Bridge에서 넘어온 P3, P4, P5의 채널 수 (예: Medium 체급 [256, 512, 512])
            spn_groups: SPN 그룹 분할 수 (기본 4)
        """
        super().__init__()
        self.spn_groups = spn_groups
        c3, c4, c5 = channels  # 256, 512, 512

        # ---------------------------------------------------------
        # [1] Top-Down Pathway (하향식 피처 융합: 의미론적 정보 전달)
        # ---------------------------------------------------------
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # FLOPs = 0

        # ---------------------------------------------------------
        # [2] Bottom-Up Pathway (상향식 피처 융합: 공간적 로컬 정보 전달)
        # ---------------------------------------------------------
        # FLOP-Free 공간 차원 축소 (Conv stride=2 대신 AvgPool2d 사용)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # 각 Concat 이후의 채널 수를 고려하여 정의
        self.td_spn_p4 = self._make_fusion_block(c5 + c4, c4)
        self.td_spn_p3 = self._make_fusion_block(c4 + c3, c3)
        self.bu_spn_p4 = self._make_fusion_block(c3 + c4, c4)
        self.bu_spn_p5 = self._make_fusion_block(c4 + c5, c5)

    def _make_fusion_block(self, in_c, out_c):
        """
        [NPU 최적화] SPN 연산 후 BN + SiLU를 묶어 커널 퓨전 유도
        """
        return nn.Sequential(
            SPNBlock(in_channels=in_c, out_channels=out_c, groups=self.spn_groups, use_act=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, features):
        """
        Input: Bridge에서 출력된 [p3, p4, p5]
        Output: YOLO Head로 전달할 융합된 [out_p3, out_p4, out_p5]
        """
        p3, p4, p5 = features

        # ==========================================
        # Phase 1: Top-Down Path (P5 -> P4 -> P3)
        # ==========================================
        # 1. P5를 2배 키우고 P4와 결합
        up_p5 = self.upsample(p5)
        cat_p4 = torch.cat([up_p5, p4], dim=1)
        td_p4 = self.td_spn_p4(cat_p4)  # 기존 C3k2를 대체하는 SPN 융합

        # 2. 융합된 P4를 2배 키우고 P3와 결합
        up_p4 = self.upsample(td_p4)
        cat_p3 = torch.cat([up_p4, p3], dim=1)
        out_p3 = self.td_spn_p3(cat_p3) # 최종 P3 출력 (Head로 전달)

        # ==========================================
        # Phase 2: Bottom-Up Path (P3 -> P4 -> P5)
        # ==========================================
        # 3. 완성된 P3의 크기를 반으로 줄이고, Top-Down P4와 결합
        down_p3 = self.downsample(out_p3)
        cat_bu_p4 = torch.cat([down_p3, td_p4], dim=1)
        out_p4 = self.bu_spn_p4(cat_bu_p4) # 최종 P4 출력 (Head로 전달)

        # 4. 완성된 P4의 크기를 반으로 줄이고, 원본 P5와 결합
        down_p4 = self.downsample(out_p4)
        cat_bu_p5 = torch.cat([down_p4, p5], dim=1)
        out_p5 = self.bu_spn_p5(cat_bu_p5) # 최종 P5 출력 (Head로 전달)

        return [out_p3, out_p4, out_p5]