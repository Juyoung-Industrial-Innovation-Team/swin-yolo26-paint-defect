"""
Swin-YOLO26 Hybrid Model Training Script
====================================================
Project: Edge Vision AI-based Ship Painting Quality Inspection
Description: 
  이 스크립트는 제안하는 하이브리드(Swin-YOLO26) 모델을
  순정 Ultralytics 환경과 동일한 조건에서 공정하게 학습시키기 위한 통합 훈련 엔진입니다.
====================================================
"""

import os
import sys
import argparse
import yaml
import torch

# 💡 OMP 다중 라이브러리 충돌 에러 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 💡 프로젝트 절대 경로 강제 주입
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    # 1. 터미널 명령어 매개변수 설정
    parser = argparse.ArgumentParser(description="선박 도장 결함 탐지 모델 학습 스크립트")
    parser.add_argument('--config', type=str, required=True, help='학습 설정 파일 경로 (예: configs/swin_micro_yolo_26m.yaml)')
    args = parser.parse_args()

    # 2. YAML 설정 파일 읽기
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 절대 경로 고정 및 설정값 매핑
    model_name_or_path = config.pop('model_path') # 순정 yaml 타겟 (yolo26m.yaml)
    config_name = os.path.basename(args.config).replace('.yaml', '')
    
    # 🚨 데이터셋 및 결과물 저장 절대 경로 강제 할당
    YAML_PATH = os.path.join(PROJECT_ROOT, "data", "ship_paint_data.yaml")
    RUNS_PATH = os.path.join(PROJECT_ROOT, "runs")
    
    # config에 절대 경로 주입
    config['data'] = YAML_PATH
    config['project'] = RUNS_PATH
    config['name'] = f"exp_{config_name}"
    config['model'] = model_name_or_path # 더미 구조체 우회용
    
    print("\n" + "=" * 70)
    print("🚢 선박 도장 품질 결함 탐지 - 하이브리드 학습 파이프라인 가동")
    print("=" * 70)
    print(f"📄 읽어들인 설정 파일: {config_path}")
    print(f"🧠 베이스라인 구조 타겟: {model_name_or_path}")
    print(f"📊 할당된 데이터 경로: {YAML_PATH}")
    print(f"📁 결과물 저장 폴더: {RUNS_PATH}\\{config['name']}")
    print("=" * 70 + "\n")

    # =========================================================================
    # 4. 모델 분기 처리 및 훈련 시작
    # =========================================================================
    
    if "swin" in config_name.lower():
        print("🛠️ [Ours] Swin-YOLO26 아키텍처 주입 (Architecture Injection) 시작...")
        
        # 💡 [핵심 해결] Ultralytics가 모르는 커스텀 파라미터들을 제거(pop)하여 충돌을 막습니다.
        # (이 값들은 yaml에서 빼내어 백업해두고, 트레이너에게는 순정 인자만 넘깁니다.)
        custom_backbone_cfg = config.pop('backbone_config', {})
        custom_folding_cfg = config.pop('folding_config', {})
        custom_yolo_cfg = config.pop('yolo_config', {})

        # ⭐ [추가/수정] yaml에서 freeze_epochs 값을 가져오고, config에서 제거
        # yaml에 설정이 없으면 기본값 20 사용
        freeze_epochs = config.pop('freeze_epochs', 20)

        # 커스텀 트레이너 로드
        from models.swin_yolo26 import SwinYOLOTrainer
        
        # 💡 [핵심] 트레이너 인스턴스화 (이때 overrides로 설정값 묶음을 전달)
        # 트레이너 내부에서 get_model()이 호출되며 자동으로 우리의 SwinYOLO26이 장착됩니다.
        trainer = SwinYOLOTrainer(overrides=config)

        # 2. 💡 [에러 해결] cfg 인자를 명시하여 모델 생성! (NoneType 에러 원천 차단)
        target_cfg = config.get('model', 'yolo26m.yaml') 
        model = trainer.get_model(cfg=target_cfg)
        
        # 3. 💡 사전 학습 가중치 메모리에 로드
        pretrained_path = r".\runs\baseline_yolo26m\weights\best.pt"
        if os.path.exists(pretrained_path):
            print(f"📥 사전 학습 가중치 로드 중: {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')['model'].float().state_dict()
            model_dict = model.state_dict()
            
            # 4. 💡 수동 이식 (Weight Injection) 로직
            matched_layers = 0
            for pre_k, pre_v in pretrained_dict.items():
                for mod_k in model_dict.keys():
                    # 이름의 맨 끝 2마디(예: 'cv2', 'weight')와 텐서 모양이 일치하면 덮어씌움
                    if pre_k.split('.')[-2:] == mod_k.split('.')[-2:] and pre_v.shape == model_dict[mod_k].shape:
                        model_dict[mod_k] = pre_v
                        matched_layers += 1
                        break # 짝을 찾았으니 다음 가중치로

            # 이식된 딕셔너리를 모델에 최종 업데이트
            model.load_state_dict(model_dict, strict=False)
            print(f"🎯 수동 가중치 이식 성공: 총 {matched_layers}개의 레이어가 매핑되었습니다!")
        else:
            print("⚠️ 사전 학습 가중치 파일이 없습니다. 무작위 초기화(Random Init)로 시작합니다.")

        # =====================================================================
        # ⭐ [추가 로직] Swin Backbone & Bridge 동결 (Freeze) 로직
        # =====================================================================
        print(f"❄️ 초기 {freeze_epochs} 에포크 동안 Swin Backbone 및 Bridge 레이어를 동결합니다.")
        
        frozen_layers = []
        for name, param in model.named_parameters():
            # 'swin_backbone' 이나 'bridge' 이름이 포함된 파라미터는 기울기 계산을 끕니다.
            if 'swin_backbone' in name or 'bridge' in name:
                param.requires_grad = False
                frozen_layers.append(name)
        
        print(f"🔒 총 {len(frozen_layers)}개의 파라미터 텐서가 동결되었습니다.")
        
        # Ultralytics 훈련 루프 내에서 특정 시점에 동결을 푸는 콜백(Callback) 함수 정의
        def unfreeze_callback(trainer):
            # 현재 에포크가 freeze_epochs에 도달하면
            if trainer.epoch == freeze_epochs:
                print(f"\n🔥 [Epoch {freeze_epochs}] Swin Backbone 및 Bridge의 동결을 해제합니다. 전체 모델 학습 시작!")
                unfrozen_count = 0
                for name, param in trainer.model.named_parameters():
                    if 'swin_backbone' in name or 'bridge' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                print(f"🔓 총 {unfrozen_count}개의 파라미터 텐서가 활성화되었습니다.")

        # 트레이너에 매 에포크 시작 시 작동할 콜백 함수 등록
        trainer.add_callback("on_train_epoch_start", unfreeze_callback)
        # =====================================================================

        # 5. 💡 가중치가 꽉 찬 모델을 트레이너에 장착!
        trainer.model = model

        # 🚀 훈련 시작 (이 한 줄로 DDP, AMP, 데이터로더, Loss 계산이 전부 자동으로 돌아갑니다)
        trainer.train()
        
    else:
        print("⚙️ [Baseline] 순정 YOLO 훈련 모드 가동...")
        from ultralytics import YOLO
        
        model = YOLO(model_name_or_path)
        model.train(**config)

if __name__ == '__main__':
    main()