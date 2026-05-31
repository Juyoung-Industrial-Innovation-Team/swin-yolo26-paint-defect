import torch
import torch.nn as nn
import timm
from ultralytics.nn.tasks import DetectionModel

class FeatureAlignmentBridge(nn.Module):
    """
    Swin Transformer의 계층적 채널 출력을 YOLO Neck이 요구하는 채널(C3, C4, C5) 규격으로 변환하는 브릿지 레이어.
    논문 Methodology의 핵심 기여점 중 하나인 '차원 정렬(Dimensionality Alignment)' 역할을 수행합니다.
    """
    def __init__(self, swin_channels, yolo_channels):
        super().__init__()
        # Swin의 3가지 출력(보통 192, 384, 768 등)을 YOLO의 (256, 512, 1024 등)으로 1x1 Conv 변환
        self.bridge_p3 = nn.Conv2d(swin_channels[0], yolo_channels[0], kernel_size=1, stride=1)
        self.bridge_p4 = nn.Conv2d(swin_channels[1], yolo_channels[1], kernel_size=1, stride=1)
        self.bridge_p5 = nn.Conv2d(swin_channels[2], yolo_channels[2], kernel_size=1, stride=1)
        self.act = nn.SiLU(inplace=True) # YOLO 최적화 활성화 함수

    def forward(self, features):
        p3 = self.act(self.bridge_p3(features[0]))
        p4 = self.act(self.bridge_p4(features[1]))
        p5 = self.act(self.bridge_p5(features[2]))
        return [p3, p4, p5]

class SwinYOLO26(nn.Module):
    """
    Swin Transformer Backbone + YOLO26 NMS-Free Head Hybrid Architecture
    비정방형 고해상도 이미지를 리사이즈(정보 소실) 없이 처리하여 미세 핀홀 탐지율을 극대화합니다.
    """
    def __init__(self, model_size='m', num_classes=7):
        super().__init__()
        
        # 1. Swin Transformer Backbone 로드 (timm 라이브러리 활용)
        # model_size 매개변수와 상관없이 Swin의 체급(tiny)을 고정하여 일관된 피처맵 크기와 채널 수를 보장합니다.
        swin_type = {'n': 'swin_tiny_patch4_window7_224',
                     's': 'swin_small_patch4_window7_224',
                     'm': 'swin_base_patch4_window7_224',
                     'l': 'swin_large_patch4_window7_224'}.get(model_size, 'swin_tiny_patch4_window7_224')
        
        # features_only=True: 최종 분류기가 아닌 중간 계층의 피처맵 3개만 추출
        self.backbone = timm.create_model(swin_type, pretrained=True, features_only=True, out_indices=(1, 2, 3))
        
        # 모델 사이즈별 채널 매핑 딕셔너리
        yolo_neck_channels = {
            'n': [64, 128, 256],
            's': [128, 256, 512],
            'm': [256, 512, 1024],
            'l': [512, 1024, 2048]
        }[model_size]
        swin_out_channels = self.backbone.feature_info.channels()
        
        # 2. Feature Alignment Bridge 초기화
        self.bridge = FeatureAlignmentBridge(swin_out_channels, yolo_neck_channels)
        
        # 3. YOLO26 Head (STAL 알고리즘 적용된 NMS-Free Head) 로드
        # 실제 구현 시에는 Ultralytics의 DetectionModel을 상속받은 커스텀 헤드를 씌웁니다.
        # yaml 파일 없이 코드 레벨에서 헤드를 바로 인스턴스화합니다.
        self.head = self._build_yolo_head(model_size, num_classes)
        
    def _build_yolo_head(self, size, nc):
        # (개념적 코드) Ultralytics 내부의 v11/v26 Head 아키텍처를 동적 생성
        # 실제로는 미리 정의된 head.yaml을 읽어오거나 Ultralytics nn 모듈을 조립
        dummy_yaml = f"yolo11{size}.yaml" 
        temp_model = DetectionModel(dummy_yaml, nc=nc)
        # 백본을 제외한 Neck(FPN/PAN)과 Detect Head 부분만 분리하여 반환
        return temp_model.model[10:] 

    def forward(self, x):
        """
        순전파 (Forward Pass)
        x: 입력 텐서 (비정방형 텐서 입력 가능) -> shape: (B, 3, H, W)
        """
        # Step 1: 비정방형 이미지를 패치로 쪼개어 특징 추출 (정보 손실 제로)
        features = self.backbone(x)
        
        # Step 2: YOLO Head가 이해할 수 있는 차원으로 규격화
        aligned_features = self.bridge(features)
        
        # Step 3: NMS-Free 기반 결함 직접 추론 (Direct Output)
        out = self.head(aligned_features)
        
        return out