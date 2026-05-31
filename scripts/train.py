import os
import sys
import argparse
import yaml

# 💡 [핵심 해결 1] OMP 다중 라이브러리 충돌 에러 방지 (반드시 맨 위에 위치)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

def main():
    # 1. 터미널 명령어 매개변수 설정
    parser = argparse.ArgumentParser(description="선박 도장 결함 탐지 모델 학습 스크립트")
    parser.add_argument('--config', type=str, required=True, help='학습 설정 파일 경로 (예: configs/model_yolo11m.yaml)')
    args = parser.parse_args()

    # 2. YAML 설정 파일 읽기
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 모델 파일명과 저장 폴더명 추출
    model_path = config.pop('model_path') 
    config_name = os.path.basename(args.config).replace('.yaml', '')
    
    # 💡 [핵심] 프로젝트 루트 및 각종 절대 경로 설정
    # 현재 스크립트(scripts/train.py)의 부모(..) 디렉토리가 프로젝트 루트입니다.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 🚨 데이터셋 명세서의 절대 경로를 강제 할당합니다.
    YAML_PATH = os.path.join(PROJECT_ROOT, "data", "ship_paint_data.yaml")

    # Dry Run 테스트용 더미 데이터 경로로 임시 교체 💡
    #YAML_PATH = os.path.join(PROJECT_ROOT, "dummy_data", "dummy.yaml")
    
    # 저장될 폴더의 절대 경로 설정
    RUNS_PATH = os.path.join(PROJECT_ROOT, "runs")
    
    # config 딕셔너리에 절대 경로를 직접 주입 (기존 값이 있으면 덮어씁니다)
    config['data'] = YAML_PATH
    
    print(f"🚀 읽어들인 설정 파일: {config_path}")
    print(f"🧠 로드할 모델 가중치: {model_path}")
    print(f"📊 강제 할당된 데이터 경로: {YAML_PATH}")
    print(f"📁 결과물 저장 폴더: {RUNS_PATH}/exp_{config_name}")
    print("-" * 50)

    # 4. 모델 초기화 및 커스텀 구조 연동
    if model_path.endswith('.py'):
        print("🛠️ 커스텀 파이썬 모델(.py) 로드를 감지했습니다.")
        # models 폴더를 시스템 경로에 추가
        models_dir = os.path.dirname(os.path.abspath(model_path))
        if models_dir not in sys.path:
            sys.path.append(models_dir)
        
        # swin_yolo26.py 파일에서 SwinYOLO26 클래스 임포트
        module_name = os.path.basename(model_path).replace('.py', '')
        custom_module = __import__(module_name)
        SwinYOLO26 = getattr(custom_module, 'SwinYOLO26')
        
        # 💡 [핵심 하이재킹] DetectionTrainer 가로채기
        class SwinYOLOTrainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                print("\n" + "🔥" * 25)
                print("🚀 [트레이너 하이재킹 성공] 원본 YOLO 대신 Swin-YOLO26 아키텍처를 강제 주입합니다!")
                print("🔥" * 25 + "\n")
                
                # 1. 원본 YOLO 껍데기 로드 (Loss 함수, 정답 매칭 등 Ultralytics 생태계 활용 목적)
                shell_model = super().get_model(cfg, weights, verbose)
                
                # 2. 우리의 커스텀 신경망(Swin-YOLO) 생성
                num_cls = self.data.get('nc', 7)
                custom_net = SwinYOLO26(model_size='m', num_classes=num_cls)
                
                # 3. 껍데기의 핵심 순전파(Forward) 함수를 커스텀 신경망으로 덮어쓰기
                # (args, kwargs를 무시하여 호환성 에러 방지)
                def custom_forward(x, *args, **kwargs):
                    return custom_net(x)
                
                shell_model._forward_once = custom_forward
                shell_model.model = custom_net # Summary 출력용 속임수
                
                return shell_model

        # 딕셔너리로 인자 묶기
        train_args = config.copy()
        train_args['model'] = "yolo11m.yaml" # 껍데기 생성을 위한 더미
        train_args['project'] = RUNS_PATH
        train_args['name'] = f"exp_{config_name}"
        
        # 하이재킹된 트레이너로 직접 학습 시작!
        trainer = SwinYOLOTrainer(overrides=train_args)
        trainer.train()
        
    else:
        # 기존 베이스라인 모델용
        model = YOLO(model_path)
        model.train(project=RUNS_PATH, name=f"exp_{config_name}", **config)

if __name__ == '__main__':
    main()