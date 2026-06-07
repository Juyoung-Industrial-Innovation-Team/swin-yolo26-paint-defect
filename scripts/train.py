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
        custom_freeze_epochs = config.pop('freeze_epochs', {})

        # 커스텀 트레이너 로드
        from models.swin_yolo26 import SwinYOLOTrainer
        
        # 💡 [핵심] 트레이너 인스턴스화 (이때 overrides로 설정값 묶음을 전달)
        # 트레이너 내부에서 get_model()이 호출되며 자동으로 우리의 SwinYOLO26이 장착됩니다.
        trainer = SwinYOLOTrainer(overrides=config)
        
        # 🚀 훈련 시작 (이 한 줄로 DDP, AMP, 데이터로더, Loss 계산이 전부 자동으로 돌아갑니다)
        trainer.train()
        
    else:
        print("⚙️ [Baseline] 순정 YOLO 훈련 모드 가동...")
        from ultralytics import YOLO
        
        model = YOLO(model_name_or_path)
        model.train(**config)

if __name__ == '__main__':
    main()