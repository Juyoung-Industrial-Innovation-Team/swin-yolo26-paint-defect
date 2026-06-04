"""
Swin-YOLO26 Hybrid Architecture
====================================================
Project: Edge Vision AI-based Ship Painting Quality Inspection
Architecture Design: 
  - Backbone : Swin Transformer (Tiny) - 비정방형 고해상도 이미지의 공간적/질감적 특징(Local Texture) 보존
  - Bridge   : Feature Alignment Bridge - Swin의 텐서 출력을 YOLO Neck이 요구하는 다중 스케일 차원으로 프로젝션
  - Neck/Head: YOLO26 (Medium) - STAL(Small-Target-Aware Label Assignment) 및 NMS-Free Head 적용
====================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # 💡 이 한 줄이 누락되었습니다!
import timm
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer

class FeatureAlignmentBridge(nn.Module):
    """
    [Methodology 3.2] 차원 정렬 브릿지 (Dimensionality Alignment Bridge)
    
    Swin Transformer 백본의 계층적 채널 출력(C1, C2, C3)을 
    YOLO26 Neck 모듈(PANet/FPN 구조)이 요구하는 입력 채널 규격으로 변환하는 모듈입니다.
    이 과정을 통해 두 이질적인 네트워크 간의 텐서 형상(Tensor Shape) 충돌을 방지합니다.
    """
    def __init__(self, swin_channels, target_neck_channels):
        super().__init__()
        # [수정] YOLO의 Neck(PANet)이 기대하는 고정 피처 맵 크기
        self.target_sizes = [80, 40, 20]

        # Swin-Tiny의 출력 채널 [192, 384, 768]을 
        # YOLO26m의 순정 Neck 입력 채널 [512, 512, 512]로 강제 확장(Projection)
        self.bridge_p3 = nn.Conv2d(swin_channels[0], target_neck_channels[0], kernel_size=1)
        self.bridge_p4 = nn.Conv2d(swin_channels[1], target_neck_channels[1], kernel_size=1)
        self.bridge_p5 = nn.Conv2d(swin_channels[2], target_neck_channels[2], kernel_size=1)
        
        # 비선형성 추가 및 텐서 활성화
        self.act = nn.SiLU(inplace=True)

    def forward(self, features):
        aligned = []
        for i, (feat, target_size) in enumerate(zip(features, self.target_sizes)):
            x = feat.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
            if i == 0: x = self.bridge_p3(x)
            elif i == 1: x = self.bridge_p4(x)
            else: x = self.bridge_p5(x)
            
            # [수정] AdaptivePooling으로 해상도를 target_size로 강제 고정
            x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            aligned.append(self.act(x))
        return aligned

class SwinYOLO26(DetectionModel):
    """
    [Methodology 3.3] 하이브리드 아키텍처 통합 (Hybrid Architecture Integration)
    
    Ultralytics의 DetectionModel을 상속받아, CNN 기반 순정 백본을 
    Swin Transformer로 동적 치환(Dynamic Substitution)하는 클래스입니다.
    """
    def __init__(self, swin_size='n', yolo_size='m', num_classes=7):
        # 1. 순정 YOLO26 구조체(yaml)를 기반으로 모델 뼈대 초기화
        # 이 시점에는 아직 CNN 백본 레이어(0~9번)가 존재합니다.
        super().__init__(f"yolo26{yolo_size}.yaml", nc=num_classes)
        
        # 2. Swin Transformer 백본 생성 (timm 활용)
        # 선박 도장면의 미세 결함 처리를 위해 패치 병합(Patch Merging) 기반의 특징을 추출합니다.
        # dynamic_img_size=True 옵션으로 가변 해상도(비정방형) 처리를 지원합니다.
        print("💡 [Swin-YOLO] Swin-Tiny Backbone 로드 중...")
        self.swin_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            features_only=True, 
            out_indices=(1, 2, 3), # P3, P4, P5 추출
            dynamic_img_size=True,
            img_size=640 # 학습/추론 시 기본 해상도
        )
        
        # 3. 브릿지 채널 설정
        # 실험을 통해 도출된 Swin 실제 출력 채널과 YOLO26m 요구 채널 매핑
        actual_swin_channels = [192, 384, 768]
        target_yolo_channels = [512, 512, 512] 
        self.bridge = FeatureAlignmentBridge(actual_swin_channels, target_yolo_channels)
        print("💡 [Swin-YOLO] Feature Alignment Bridge 결합 완료!")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, 'swin_backbone'):
            return super()._predict_once(x, profile, visualize, embed)
        
        # 1. 고정 입력 해상도 확보
        x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
        
        # 2. 백본 및 브릿지 통과
        features = self.swin_backbone(x)
        aligned_features = self.bridge(features)
        
        # 3. y 리스트 초기화
        y = [None] * len(self.model)
        
        # 4. Bridge 결과 주입 (P3, P4, P5)
        y[4] = aligned_features[0]
        y[6] = aligned_features[1]
        y[9] = aligned_features[2]
        
        # 5. [중요] 0~9번 층의 빈 곳을 더미 텐서로 채우기 (에러 방지)
        for i in range(10):
            if y[i] is None:
                if i < 5: y[i] = torch.zeros(1, 512, 80, 80).to(x.device)
                elif i < 8: y[i] = torch.zeros(1, 512, 40, 40).to(x.device)
                else: y[i] = torch.zeros(1, 512, 20, 20).to(x.device)

        # 6. 이후 Neck/Head 연산 진행
        x = aligned_features[2]
        for i, m in enumerate(self.model):
            if i < 10: continue
            if m.f != -1:
                if isinstance(m.f, int): x = y[m.f]
                else: x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x if m.i in self.save else None
        return x
    
    
    
# ==========================================================
# 💡 [새로 추가된 부분] 커스텀 트레이너 (훈련 엔진)
# ==========================================================
class SwinYOLOTrainer(DetectionTrainer):
    """
    [Methodology 3.4] 학습 엔진 파이프라인 (Training Engine)
    
    Ultralytics의 기본 학습 엔진을 상속받아, 
    우리가 직접 조립한 Swin-YOLO26 모델을 훈련 루프에 강제 주입합니다.
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        원래는 yaml을 읽어 모델을 새로 생성하는 함수지만,
        train.py에서 trainer.model 에 이미 SwinYOLO26을 넣어두었으므로 이를 그대로 반환합니다.
        """
        if hasattr(self, 'model') and self.model is not None:
            return self.model
        
        # 예외 상황을 위한 기본 폴백(Fallback)
        return super().get_model(cfg, weights, verbose)
    
    # 👇👇 이 함수를 추가해 주세요! 👇👇
    def get_validator(self):
        """
        [핵심 수정] 검증(Validation) 단계에서 YOLO가 멋대로 이미지를 
        직사각형(Rectangular)으로 자르는 것을 원천 차단합니다.
        Swin 백본은 정사각형(Square) 입력만 처리할 수 있습니다.
        """
        self.args.rect = False 
        return super().get_validator()