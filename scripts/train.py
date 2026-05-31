"""
Swin-YOLO26 Hybrid Model Training Script
====================================================
Project: Edge Vision AI-based Ship Painting Quality Inspection
Description: 
  이 스크립트는 베이스라인(YOLO11m) 모델과 제안하는 하이브리드(Swin-YOLO26) 모델을
  동일한 환경에서 공정하게 학습시키기 위한 통합 훈련 엔진입니다.
====================================================
"""

import os
import sys
import argparse
import yaml
import torch

# 💡 [핵심 해결 1] OMP 다중 라이브러리 충돌 에러 방지 (반드시 맨 위에 위치)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 💡 [핵심 해결 2] 프로젝트 루트 경로를 가장 먼저 시스템에 강제 주입
# 실행 위치에 상관없이 항상 'swin-yolo26-paint-defect' 폴더를 기준으로 삼도록 만듭니다.
# __file__은 현재 스크립트(scripts/train.py)의 절대 경로를 가리킵니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # append 대신 insert(0)을 써서 최우선 순위로 만듭니다.

def main():
    # 1. 터미널 명령어 매개변수 설정
    parser = argparse.ArgumentParser(description="선박 도장 결함 탐지 모델 학습 스크립트")
    parser.add_argument('--config', type=str, required=True, help='학습 설정 파일 경로 (예: configs/model_swin_yolo26m.yaml)')
    args = parser.parse_args()

    # 2. YAML 설정 파일 읽기
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 모델 파일명과 저장 폴더명 추출
    model_name_or_path = config.pop('model_path') # yaml 안의 model_path 값을 가져오고 config에서 제거
    config_name = os.path.basename(args.config).replace('.yaml', '')
    
    # 🚨 데이터셋 명세서의 절대 경로를 강제 할당합니다.
    YAML_PATH = os.path.join(PROJECT_ROOT, "data", "ship_paint_data.yaml")
    
    # 저장될 폴더의 절대 경로 설정
    RUNS_PATH = os.path.join(PROJECT_ROOT, "runs")
    
    # config 딕셔너리에 절대 경로를 직접 주입 (기존 값이 있으면 덮어씁니다)
    config['data'] = YAML_PATH
    
    print("=" * 60)
    print(f"🚀 읽어들인 설정 파일: {config_path}")
    print(f"🧠 로드할 모델 타겟: {model_name_or_path}")
    print(f"📊 할당된 데이터 경로: {YAML_PATH}")
    print(f"📁 결과물 저장 폴더: {RUNS_PATH}/exp_{config_name}")
    print("=" * 60)

    # =========================================================================
    # 4. 커스텀 파이썬 모델 분기 처리 (Ablation Study 대응)
    # =========================================================================
    
    # 제안 모델: Swin-YOLO26 (하이브리드 아키텍처)
    # config 파일명이 'swin'을 포함하거나, model_path가 커스텀 yaml을 가리키는 경우
    if "swin" in config_name.lower():
        print("🛠️ [Ours] 하이브리드 아키텍처(Swin-YOLO26) 훈련 모드 가동...")
        
        # 앞서 구현한 커스텀 모델과 트레이너를 로드합니다.
        # sys.path.insert(0, PROJECT_ROOT) 덕분에 models 폴더를 직접 찾을 수 있습니다.
        from models.swin_yolo26 import SwinYOLO26, SwinYOLOTrainer
        
        # 모델 객체를 직접 인스턴스화 (이 과정에서 Bridge 채널 맵핑이 완료됨)
        # 클래스 개수(nc)는 ship_paint_data.yaml에 정의된 7개로 고정
        model_instance = SwinYOLO26(swin_size='n', yolo_size='m', num_classes=7)
        
        # 훈련 인자 구성
        train_args = config.copy()
        # 💡 [핵심] Ultralytics 내부 검증 로직을 통과하기 위해 순정 yaml 파일명 명시
        # trainer는 이 문자열을 보고 내부 레이어 구조를 그리지만, 
        # 우리가 직후에 trainer.model = model_instance로 덮어씌울 것이므로 문제없습니다.
        train_args['model'] = "yolo26m.yaml" 
        train_args['project'] = RUNS_PATH
        train_args['name'] = f"exp_{config_name}"
        
        # 💡 [핵심 연동] 트레이너 객체 생성 
        trainer = SwinYOLOTrainer(overrides=train_args)
        
        # 💡 [가장 중요한 부분] 생성된 트레이너에 우리가 수술한 하이브리드 모델 객체를 강제 이식
        trainer.model = model_instance 
        
        # 학습 시작
        trainer.train()
        
    # 베이스라인 모델: YOLO11m (비교 대조군)
    else:
        print("⚙️ [Baseline] 순정 YOLO 훈련 모드 가동...")
        from ultralytics import YOLO
        
        # 순정 가중치(.pt) 로드 (model_name_or_path 에는 'yolo11m.pt' 등이 들어있어야 함)
        model = YOLO(model_name_or_path)
        
        # DFL 등 순정 학습 시작
        model.train(project=RUNS_PATH, name=f"exp_{config_name}", **config)

if __name__ == '__main__':
    main()