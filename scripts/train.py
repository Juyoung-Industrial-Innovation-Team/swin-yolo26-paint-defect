import os
import sys
import argparse
import yaml

# 💡 [핵심 해결 1] OMP 다중 라이브러리 충돌 에러 방지 (반드시 맨 위에 위치)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

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
        
        # 커스텀 PyTorch 모델 인스턴스화
        custom_model = SwinYOLO26(model_size='m', num_classes=config.get('nc', 7))
        
        # 💡 [핵심 트릭] Ultralytics 학습 파이프라인을 타기 위해 껍데기 YOLO 객체 생성 후 알맹이 교체
        model = YOLO("yolo11m.yaml") # 베이스라인 껍데기 로드
        model.model = custom_model   # 우리가 만든 Swin-YOLO로 신경망 완벽 교체
        
        # Ultralytics 엔진이 요구하는 필수 메타데이터 주입
        model.model.names = config.get('names', ['워터스포팅', '흐름', '도막분리', '핀홀', '균열', '부풀음', '이물질포함'])
        model.model.nc = len(model.model.names)
        
    else:
        # 기존 방식 (.pt 또는 .yaml)
        model = YOLO(model_path)

    # 5. 학습 시작
    # **config는 data, epochs, batch, imgsz 등을 한 번에 딕셔너리로 풀어 넣는 파이썬 문법입니다.
    results = model.train(
        project=RUNS_PATH,
        name=f"exp_{config_name}",  
        **config                    
    )

if __name__ == '__main__':
    main()