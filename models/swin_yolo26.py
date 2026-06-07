import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer

# 💡 [핵심] Ultralytics의 가장 근본적이고 범용적인 Detect 헤드 임포트
from ultralytics.nn.modules.head import Detect  

# =====================================================================
# 모듈화된 컴포넌트 Import
# =====================================================================
from .components.folding import DynamicLosslessTensorFolding
from .components.swin_backbone import SwinMicroBackbone
from .components.feature_alignment_bridge import FeatureAlignmentBridge
from .components.spn_panet_neck import SPN_PANet

class SwinYOLO26(DetectionModel):
    """
    [Swin-YOLO26 Hybrid Architecture]
    Ultralytics의 순정 Detect 헤드를 활용하여 프레임워크와 100% 호환되는 E2E 파이프라인
    """
    def __init__(self, nc=8, cfg='yolo26m.yaml', verbose=True, embed_dim=64):
        # 1. 부모 클래스 초기화 (tasks.py 내부의 에러 유발 로직을 빈 껍데기로 무사 통과)
        super().__init__(cfg=cfg, nc=nc, verbose=False)

        # 2. 💡 [아키텍처 인젝션] 
        # 우리의 독자적인 하이브리드 파이프라인을 구축하고, 마지막에 순정 Detect를 장착합니다.
        self.model = nn.Sequential(
            DynamicLosslessTensorFolding(),
            SwinMicroBackbone(pretrained=True, embed_dim=embed_dim),
            FeatureAlignmentBridge(
                swin_channels=[embed_dim, embed_dim * 2, embed_dim * 4], 
                yolo_channels=[256, 512, 512]
            ),
            SPN_PANet(in_channels=[256, 512, 512]),
            
            # 💡 [핵심] 순정 Detect를 NMS-Free 모드로 동작시키기 위해 end2end=True 부여!
            # 만약 기존 YOLO 방식의 NMS 후처리가 필요하다면 end2end=False로 변경하면 즉시 전환됩니다.
            Detect(nc=nc, ch=(256, 512, 512), end2end=True) 
        )

        # 3. 모델 순차 루프(Sequential Loop)를 위한 메타데이터 강제 주입
        self.save = []  
        for i, m in enumerate(self.model):
            m.i = i
            m.f = -1  # 직전 모듈의 출력을 그대로 입력으로 받음
            m.type = m.__class__.__name__
            m.np = sum(x.numel() for x in m.parameters())

        # 4. Detect 헤드 필수 속성 동기화
        head = self.model[-1]
        head.inplace = self.inplace
        head.stride = torch.tensor([8.0, 16.0, 32.0]) # P3, P4, P5 스트라이드 부여
        self.stride = head.stride
        
        # 💡 [필수] Detect 레이어의 초기 편향(Bias) 값 세팅 (Loss 발산 방지)
        head.bias_init()  

        # 5. 하이브리드 구조 Summary 출력
        if verbose:
            self.info()

    # forward() 오버라이딩은 제거합니다. 부모 클래스(BaseModel)의 로직이 완벽히 제어합니다.


class SwinYOLOTrainer(DetectionTrainer):
    """
    [Swin-YOLO26 Custom Trainer]
    Ultralytics의 순정 DetectionTrainer를 상속받아 데이터 증강, Loss 계산 등의
    거대한 훈련 파이프라인을 그대로 재사용하면서, 모델(뇌)만 Swin-YOLO26으로 교체합니다.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        순정 DetectionModel 대신 우리의 SwinYOLO26을 인스턴스화하여 반환합니다.
        """
        # 1. 데이터셋 yaml 파일에서 파싱된 클래스 수(nc)를 가져옵니다.
        nc = self.data["nc"] if self.data else (self.args.nc or 80)
        
        if verbose:
            print(f"🚀 [SwinYOLOTrainer] 하이브리드 아키텍처(Swin-YOLO26) 조립 중... (Classes: {nc})")

        # 2. 우리의 하이브리드 모델 생성 (내부적으로 순정 Detect Head와 연결됨)
        model = SwinYOLO26(nc=nc, cfg=cfg, verbose=verbose)

        # 3. Pre-trained 가중치가 전달되었다면 로드 (주로 YOLO Head 가중치 전이 학습용)
        if weights:
            model.load(weights)

        return model