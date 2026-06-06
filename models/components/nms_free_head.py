import torch
import torch.nn as nn
import math

# =====================================================================
# 1. Distribution Focal Loss (DFL) 모듈
# =====================================================================
class DFL(nn.Module):
    """
    바운딩 박스의 경계 모호성을 해결하기 위한 분포 초점 손실 모듈.
    NMS-Free 환경에서 STAL 할당자와 함께 작동하여 BBox의 신뢰도를 확률 밀도로 매핑합니다.
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # [B, 4, 16, Anchors] -> softmax -> [B, 4, Anchors]
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# =====================================================================
# 2. YOLO26 NMS-Free Head 모듈
# =====================================================================
class YOLO26NMSFreeHead(nn.Module):
    """
    [Methodology 3.6] YOLO26 NMS-Free Decoupled Head
    
    기존 YOLO의 NMS 병목을 원천 차단하고, STAL 알고리즘 및 ProgLoss에 
    최적화된 One-to-One 매핑을 수행하는 헤드 네트워크입니다.
    """
    def __init__(self, nc=8, ch=(256, 512, 512), max_det=300):
        super().__init__()
        self.nc = nc                # Number of classes (양품 1 + 불량 7 = 8)
        self.nl = len(ch)           # Number of Detection Layers (P3, P4, P5 = 3)
        self.reg_max = 16           # DFL 채널 크기
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl) # Stride 값 (동적으로 계산됨)
        self.max_det = max_det      # NMS-Free 추론 시 추출할 최대 객체 수

        # Decoupled Head 채널 설정 (연산량 최소화를 위한 차원 조정)
        c2 = max((16, ch[0] // 4, self.reg_max * 4)) # Regression 채널
        c3 = max(ch[0], min(self.nc, 100))           # Classification 채널

        # 1. Box Regression Branch (회귀 브랜치)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True),
                nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True),
                nn.Conv2d(c2, 4 * self.reg_max, 1) # 최종 출력: 4개 좌표 * 16(DFL)
            ) for x in ch
        )
        
        # 2. Classification Branch (분류 브랜치)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
                nn.Conv2d(c3, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
                nn.Conv2d(c3, self.nc, 1) # 최종 출력: nc (8개 클래스 확률)
            ) for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # MuSGD 옵티마이저 최적화를 위한 초기 편향(Bias) 세팅
        self._initialize_biases()

    def _initialize_biases(self, cf=None):
        """
        [MuSGD Optimizer Support]
        학습 초기 발산을 막고, STAL이 미세 결함에 빠르게 집중할 수 있도록 
        분류 브랜치의 초기 편향값을 Focal Loss의 특성에 맞춰 재조정합니다.
        """
        for a, b, s in zip(self.cv2, self.cv3, [8, 16, 32]): # stride 기준
            a[-1].bias = nn.Parameter(torch.zeros(4 * self.reg_max))
            b[-1].bias = nn.Parameter(torch.log(torch.tensor([0.01 / (self.nc - 0.01)])) * torch.ones(self.nc)) # objectness bias

    def forward(self, x):
        """
        입력: SPN_PANet에서 넘어온 [P3, P4, P5]
        출력: (Train 시) Loss 계산용 텐서 / (Inference 시) NMS-Free [B, 300, 6] 텐서
        """
        shape = x[0].shape  # B, C, H, W
        for i in range(self.nl):
            # Regression & Classification 동시 연산 후 결합
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            # 훈련 모드: Ultralytics Loss(STAL & ProgLoss) 함수가 해독할 수 있도록 Raw 텐서 반환
            return x
            
        else:
            # 💡 [NMS-Free E2E Inference]
            # 엣지 NPU(CoreML) 컴파일 시 후처리가 필요 없도록 모델 내부에서 좌표 매핑 및 TopK 추출
            y = []
            for i in range(self.nl):
                b, _, h, w = x[i].shape
                # 텐서 플래튼 (Flatten)
                xi = x[i].view(b, self.no, h * w)
                
                # Box 디코딩 (DFL 적용)
                box, cls = xi.split((self.reg_max * 4, self.nc), 1)
                box = self.dfl(box)
                
                # Sigmoid를 통한 Confidence 추출
                cls = cls.sigmoid()
                y.append(torch.cat((box, cls), 1))
            
            # 다중 스케일(P3, P4, P5) 텐서 결합
            y = torch.cat(y, 2) # [Batch, 4+nc, Total_Anchors]
            
            # BBox 좌표(4)와 Class Score(8) 분리
            boxes, scores = y.split((4, self.nc), 1)
            
            # 클래스 중 가장 높은 점수 추출
            max_scores, labels = scores.max(1) # [Batch, Total_Anchors]
            
            # NMS-Free 핵심: 단순 Top-K 추출 (CPU 병목 제거)
            # - STAL 훈련 덕분에 겹치는 박스가 없으므로 NMS가 불필요
            topk_scores, topk_indices = torch.topk(max_scores, self.max_det, dim=1)
            
            # 최종 결과 포매팅: [Batch, max_det, 6] -> (x, y, w, h, conf, class_id)
            # ※ 실제 전개시 grid 추가 및 stride 곱하기 연산이 포함됩니다 (여기서는 개념적 요약)
            return boxes, topk_scores, labels