import torch
import torch.nn as nn

# =====================================================================
# [Phase 2 ~ 6] 모듈화된 컴포넌트 Import
# 폴더 구조: models/components/...
# =====================================================================
from .components.folding import DynamicLosslessTensorFolding
from .components.swin_backbone import SwinMicroBackbone
from .components.feature_alignment_bridge import FeatureAlignmentBridge
from .components.spn_panet_neck import SPN_PANet
from .components.nms_free_head import YOLO26NMSFreeHead


class SwinYOLO26(nn.Module):
    """
    [Swin-YOLO26 Hybrid Architecture Main Wrapper]
    프로젝트: 엣지 비전 AI 기반 선박 도장 품질 감리 시스템
    
    이 클래스는 분할 설계된 각 컴포넌트들을 하나로 묶어주는 E2E 파이프라인입니다.
    사용자는 이 래퍼 클래스만 호출하면 내부의 복잡한 텐서 흐름을 신경 쓸 필요가 없습니다.
    """
    def __init__(self, nc=8, embed_dim=64, yolo_ch=(256, 512, 512), max_det=300):
        """
        Args:
            nc: 클래스 개수 (양품 1 + 불량 7 = 8)
            embed_dim: Swin-Micro 백본의 초기 임베딩 차원
            yolo_ch: YOLO Neck이 요구하는 채널 리스트 [P3, P4, P5]
            max_det: NMS-Free 추론 시 추출할 최대 객체 수
        """
        super().__init__()
        self.nc = nc
        
        print("🚀 [Swin-YOLO26] Initializing Edge-AI Hybrid Pipeline...")
        
        # [Phase 2] 동적 형상 적응형 무손실 폴딩 (Zero-Padding & PixelUnshuffle)
        self.folding = DynamicLosslessTensorFolding(in_channels=3, embed_dim=embed_dim)
        
        # [Phase 3] 미세 결함 타겟팅 Swin-Micro 백본 (Swin Transformer Blocks)
        self.backbone = SwinMicroBackbone(embed_dim=embed_dim, depths=[2, 2, 2])
        
        # [Phase 4] 텐서 충돌 방지 차원 정렬 브릿지 (Spatial & Channel Alignment)
        # Swin 출력 Stride별 채널 [64, 128, 256] -> YOLO Neck [256, 512, 512]
        self.bridge = FeatureAlignmentBridge(
            swin_channels=[embed_dim, embed_dim * 2, embed_dim * 4],
            yolo_channels=yolo_ch
        )
        
        # [Phase 5] 연산량 최소화 양방향 융합 넥 (SPN-PANet)
        self.neck = SPN_PANet(channels=yolo_ch)
        
        # [Phase 6] NMS-Free & STAL 최적화 헤드 (Decoupled Head & DFL)
        self.head = YOLO26NMSFreeHead(nc=nc, ch=yolo_ch, max_det=max_det)


    def forward(self, x):
        """
        [전체 데이터 흐름 (Data Flow)]
        """
        # 1. Input -> Folding
        # x: 비정방형 고해상도 이미지 [B, 3, H, W]
        # x_folded: 128 배수 패딩 및 압축 완료 텐서, meta_info: 복원용 메타데이터
        x_folded, meta_info = self.folding(x)
        
        # 2. Folding -> Backbone
        # swin_features: 멀티 스케일 피처 맵 리스트 [C1, C2, C3]
        swin_features = self.backbone(x_folded)
        
        # 3. Backbone -> Bridge
        # aligned_features: YOLO 규격으로 번역된 피처 맵 리스트 [P3_in, P4_in, P5_in]
        aligned_features = self.bridge(swin_features)
        
        # 4. Bridge -> Neck
        # fused_features: Top-Down & Bottom-Up으로 융합된 피처 맵 [P3_out, P4_out, P5_out]
        fused_features = self.neck(aligned_features)
        
        # 5. Neck -> Head
        # preds: 
        #  - Training 모드: Loss 계산을 위한 Raw 텐서 리스트
        #  - Eval 모드: NMS-Free 연산이 끝난 Top-K 객체 정보 (boxes, scores, labels)
        preds = self.head(fused_features)
        
        return preds

    
    def get_flops_params(self):
        """
        논문 작성을 위한 하위 모듈별 파라미터 수 측정 편의 함수 (선택 사항)
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
            
        return {
            "Folding": count_params(self.folding),
            "Backbone": count_params(self.backbone),
            "Bridge": count_params(self.bridge),
            "Neck": count_params(self.neck),
            "Head": count_params(self.head),
            "Total": count_params(self)
        }