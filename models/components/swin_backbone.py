import torch
import torch.nn as nn
import torch.nn.functional as F
from .sp_network import SPNBlock # SPN 모듈 재사용

class SPNPatchMerging(nn.Module):
    """
    [Methodology 3.3] SPN 기반 초경량 패치 병합 (Lightweight Patch Merging)
    
    기존 Swin Transformer의 무거운 Linear 차원 축소를 암호학적 SPN 구조로 대체하여,
    미세 결함 정보를 무손실로 압축(PixelUnshuffle)하면서 연산량은 1/4로 절감합니다.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        
        # 1. 무손실 공간 압축: r=2 이므로 해상도는 1/2로, 채널은 4배(4*dim)로 변환
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        
        # 2. SPN 모듈: 4*dim 채널을 2*dim으로 압축하면서 혼돈/확산(Confusion/Diffusion) 적용
        # (예: 64채널 입력 -> unshuffle 후 256채널 -> SPN 후 128채널 출력)
        self.spn = SPNBlock(in_channels=4 * dim, out_channels=2 * dim, groups=4, use_act=False)
        
        # 3. 정규화
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        Input: [B, H, W, C]
        Output: [B, H/2, W/2, 2C]
        """
        # PixelUnshuffle 및 Conv2d 연산을 위해 채널을 앞으로 [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # 무손실 다운샘플링 & 채널 차원 투영
        x = self.unshuffle(x)  # [B, 4C, H/2, W/2]
        x = self.spn(x)        # [B, 2C, H/2, W/2]
        
        # LayerNorm 연산을 위해 다시 채널을 뒤로 [B, H/2, W/2, 2C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        return x

# =====================================================================
# 1. NPU 최적화 무손실 윈도우 분할 및 병합 함수 (Zero-Padding)
# =====================================================================
def window_partition(x, window_size):
    """
    [B, H, W, C] 텐서를 [B * num_windows, window_size, window_size, C]로 분할.
    (H와 W가 window_size의 배수임이 Folding 레이어에 의해 보장됨)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # NPU 메모리 정렬을 위한 contiguous 적용
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    분할된 윈도우를 다시 [B, H, W, C] 원본 해상도로 병합.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# =====================================================================
# 2. Window Multi-head Self Attention (W-MSA / SW-MSA)
# =====================================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (8, 8)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 상대적 위치 편향(Relative Position Bias) 테이블
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # 위치 인덱스 맵핑 생성 (고정된 윈도우 크기이므로 init에서 한 번만 생성하여 연산량 절감)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 위치 편향 더하기
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Shifted Window를 위한 마스크 적용
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

# =====================================================================
# 3. Swin Transformer Block (W-MSA -> SW-MSA 교차 구조)
# =====================================================================
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path) # DropPath 간소화
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        H, W = x.shape[1], x.shape[2]
        B, L, C = x.shape[0], H * W, x.shape[3]

        shortcut = x
        x = self.norm1(x)

        # Shift 연산 (SW-MSA인 경우)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # ⭐ [최적화] 동적 어텐션 마스크 캐싱 로직 (실시간 FPS 방어용)
            resolution_key = (H, W)
            if not hasattr(self, 'mask_cache'):
                self.mask_cache = {}
            
            # 현재 해상도에 대한 마스크가 캐시에 없으면 새로 생성
            if resolution_key not in self.mask_cache:
                img_mask = torch.zeros((1, H, W, 1), device=x.device)
                h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                
                # 생성된 마스크를 딕셔너리에 저장
                self.mask_cache[resolution_key] = attn_mask
            
            # ⭐ Device 방어: 캐시된 마스크가 현재 x의 디바이스(GPU 등)와 맞는지 보장
            attn_mask = self.mask_cache[resolution_key].to(x.device)
        else:
            shifted_x = x
            attn_mask = None

        # 윈도우 분할
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # 어텐션 연산
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 윈도우 병합
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

        # Shift 복구
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN 적용
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# =====================================================================
# 4. Swin Stage 1 (Basic Layer) 
# =====================================================================
class SwinStage(nn.Module):
    """
    설정 파일의 depths: [2, 2, 2] 중 하나를 담당하는 스테이지 모듈입니다.
    """
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4., qkv_bias=True, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # 지정된 depth(예: 2)만큼 SwinTransformerBlock을 쌓습니다.
        # W-MSA와 SW-MSA가 번갈아가며 나오도록 shift_size를 조절합니다.
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, # 짝수: W-MSA, 홀수: SW-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])

    def forward(self, x):
        """
        입력: Folding에서 넘어온 [B, H, W, C] (예: [B, 160, 160, 64])
        출력: 로컬 질감이 매핑된 [B, H, W, C]
        """
        for block in self.blocks:
            x = block(x)
        return x

class SwinMicroBackbone(nn.Module):
    """
    선박 도장 결함 탐지를 위한 온디바이스 맞춤형 Swin-Micro 백본.
    출력으로 멀티 스케일 피처맵 (C1, C2, C3) 리스트를 반환하여 YOLO Neck과 결합합니다.
    """
    def __init__(self, embed_dim=64, depths=[2, 2, 2], num_heads=[2, 4, 8], window_size=8, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Stochastic Depth (Drop Path) 비율을 각 레이어마다 점진적으로 증가시킴
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        # 각 Stage 및 Patch Merging 레이어 조립
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * (2 ** i_layer)) # 64 -> 128 -> 256
            
            # 1. Swin Block Stage 구성
            stage = SwinStage(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            )
            self.stages.append(stage)
            
            # 2. Patch Merging 구성 (마지막 Stage 3 이후에는 다운샘플링 안 함)
            if i_layer < self.num_layers - 1:
                merge_layer = SPNPatchMerging(dim=dim)
                self.merges.append(merge_layer)
            else:
                self.merges.append(nn.Identity())

    def forward(self, x):
        """
        Input: Folding 레이어에서 넘어온 텐서 [B, H, W, C] (이미 Permute 되어 있음!)
        Returns: YOLO 넥에 주입할 피처맵 리스트 [C1, C2, C3]
        """
        # ❌ 삭제: x = x.permute(0, 2, 3, 1) (Folding에서 이미 처리했음)
        
        outs = []
        for i in range(self.num_layers):
            # 1. Transformer 특징 추출
            x = self.stages[i](x)   # x는 [B, H, W, C] 형태
            
            # YOLO Neck으로 넘겨주기 위해 Pytorch 표준 포맷 [B, C, H, W]으로 복제하여 저장
            outs.append(x.permute(0, 3, 1, 2).contiguous())
            
            # 2. 해상도 축소 및 채널 확장 (마지막 레이어는 수행 안함)
            x = self.merges[i](x)
            
        # outs 리스트에는 다음 텐서들이 담깁니다.
        # outs[0] (C1) -> [B, 64,  H/4,  W/4]  (Stride 4)
        # outs[1] (C2) -> [B, 128, H/8,  W/8]  (Stride 8)
        # outs[2] (C3) -> [B, 256, H/16, W/16] (Stride 16)
        return outs