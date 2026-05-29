# 🚢 Swin-YOLO26 기반 선박 도장 결함 탐지 온디바이스 모델 연구

본 저장소는 **'선박 도장 품질 데이터 활용 온디바이스 감리 지원 시스템'** 구축을 위한 D-8 학술대회 투고용 프로젝트(주영산업 혁신팀 & UNIST 산학협력)입니다. 
비정방형 고해상도 이미지에서 미세 결함(핀홀, 균열 등)을 손실 없이 탐지하기 위해 **Swin Transformer 백본과 YOLO26의 STAL 헤드를 결합한 하이브리드 아키텍처**를 제안하며, 최종적으로 iPhone 16(CoreML) 환경에서의 온디바이스 실시간 추론을 목표로 합니다.

## 📁 Repository Structure
```text
swin-yolo26-paint-defect/
├── data/               # ⚠️ 빈 폴더 (로컬의 423GB 데이터셋 마운트 경로)
├── models/             # ⚠️ 빈 폴더 (학습된 가중치 .pt, .mlpackage 저장 경로)
├── notebooks/          # EDA, 데이터 불균형 시각화, 실험용 주피터 노트북
├── src/
│   ├── preprocess.py   # AI-Hub JSON 파싱 및 11개 클래스 YOLO 포맷 변환
│   ├── train.py        # 하이브리드 모델 학습 스크립트
│   └── export.py       # CoreML (INT8/FP16) 변환 스크립트
├── requirements.txt    # 파이썬 의존성 패키지 목록
│
├── 📁 ios-app/        # [추가] 아이폰 16 구현물용 폴더
│   ├── ShipPaintApp/  # Xcode 프로젝트 (.xcodeproj)
│   ├── Models/        # CoreML 모델 파일 (.mlpackage)
│   ├── ViewControllers/# 앱의 UI 로직
│   └── README.md      # iOS 앱 빌드/배포 가이드
│
└── .gitignore          # 대용량 파일 업로드 방지 규칙, 'ios-app/ShipPaintApp/DerivedData/' 등을 추가하여 불필요한 빌드 파일 제외
```

## 📊 Dataset (AI-Hub 선박 도장 품질 데이터)
* 원본 규모: 약 423GB (양품 10만 장, 불량 4만 장)
* 클래스 정의 (총 11종):
  * 양품(4종): 외판, 선수, 선미, 갑판
  * 불량(7종): 흐름, 핀홀, 이물질포함, 워터스포팅, 부풀음, 도막분리, 균열
* 주의사항: 원본 데이터 및 전처리된 텍스트 라벨 파일은 용량 문제로 GitHub에 업로드하지 않습니다. 반드시 로컬 환경의 `data/` 폴더에 위치시켜야 합니다.

## 🤝 Collaborators
* 주영산업 혁신팀: 프로젝트 총괄, 도메인 지식 제공, 데이터셋 정제
* UNIST: AI 아키텍처 설계 (Swin-YOLO26), 모델 튜닝 및 자문

## 📜 Acknowledgments
본 연구는 과학기술정보통신부가 주관하고 한국지능정보사회진흥원(AI-Hub)이 지원한 '선박 도장 품질 측정 데이터'를 활용하여 수행되었습니다.
