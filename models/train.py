import os
# 💡 [핵심 해결 1] OMP 다중 라이브러리 충돌 에러 방지 (반드시 맨 위에 위치)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

def main():
    # 프로젝트 루트 및 경로 설정
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    YAML_PATH = os.path.join(PROJECT_ROOT, "data", "ship_paint_data.yaml")
    RUNS_PATH = os.path.join(PROJECT_ROOT, "runs")
    
    print(f"🚀 학습 데이터 경로: {YAML_PATH}")
    print(f"📁 결과물 저장 경로: {RUNS_PATH}")

    # 모델 로드
    model = YOLO("yolo11m.pt") 

    # 학습 시작
    results = model.train(
        data=YAML_PATH,     
        epochs=100,
        imgsz=640,          # Co-sight 팀 결과와 비교를 위해 640으로 테스트.
        
        # 💡 [핵심 해결 2] 하드웨어 한계 수동 할당 (OOM 방지)
        batch=32,            # 오토배치(-1)가 실패했으므로, 3090에 딱 맞는 안전한 최대치(8) 수동 부여
        workers=4,          # 윈도우 환경의 안정성을 위해 4로 조정
        cache=False,        # RAM/Disk 캐싱 끄기
        
        device=0,           # GPU 0번 사용
        project=RUNS_PATH,  
        name="baseline_yolo11m_640",
        val=True,           
        plots=True,         
    )

if __name__ == '__main__':
    main()