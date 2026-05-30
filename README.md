# 🚢 Swin-YOLO26 기반 선박 도장 결함 탐지 온디바이스 모델 연구

본 저장소는 **'선박 도장 품질 데이터 활용 온디바이스 감리 지원 시스템'** 구축을 위한 2026 KISCP Spring 학술대회 투고용 프로젝트(주영산업 혁신팀 & UNIST 노바투스대학원 온디바이스AI 수업 Co-sight 팀)입니다. 
비정방형 고해상도 이미지에서 미세 결함(핀홀, 균열 등)을 손실 없이 탐지하기 위해 **Swin Transformer 백본과 YOLO26의 STAL 헤드를 결합한 하이브리드 아키텍처**를 제안하며, 최종적으로 iPhone 16(CoreML) 환경에서의 온디바이스 실시간 추론을 목표로 합니다.

## 🚀 Get Started: GitHub 저장소 연동하기 (팀원용)
GitHub에 익숙하지 않은 팀원들도 아래 순서대로 터미널에 입력하면 프로젝트 환경을 즉시 세팅할 수 있습니다.

### 1. 저장소 복제 (Clone)
먼저 본인의 컴퓨터에 프로젝트 폴더를 생성하고 저장소를 가져옵니다.
```bash
# 원하는 폴더로 이동 후 저장소 클론
git clone https://github.com/Juyoung-Industrial-Innovation-Team/swin-yolo26-paint-defect
cd swin-yolo26-paint-defect
```

#### ⚠️ 협업 시 주의사항
* **데이터셋 깃허브 업로드 금지**: `data/` 폴더는 `.gitignore`에 등록되어 있습니다. 데이터셋(`*.zip`, `.csv` 등)은 반드시 구글 드라이브 링크를 통해 배포 및 로컬 마운트 원칙을 지켜주세요.
* **커밋 전 확인:** `git status` 명령어를 통해 대용량 모델 가중치 파일(`*.pt`, `*.mlpackage`)이 포함되어 있는지 꼭 확인 후 `git add` 하세요.

### 2. 경량화 데이터셋 다운로드 (Google Drive)
본 프로젝트는 423GB 원본 대신, 로컬 디스크 및 메모리 부하를 방지하기 위해 불필요한 데이터를 제거하고 클래스 밸런싱을 맞춘 경량화 마스터 데이터셋(`ship_paint_dataset_light.zip`)을 사용합니다.
- [다운로드 링크 (Google Drive)]: (https://drive.google.com/file/d/1AL2rdw95PNnYLQvqwOupwpyhZE_9g8RI/view?usp=sharing)
- 다운로드한 .zip 파일을 방금 클론받은 프로젝트의 data/ 폴더 내부에 압축 해제합니다.
- 압축 해제 후, 파일 구조가 반드시 아래와 같이 되어야 파이프라인이 정상 작동합니다.
```
swin-yolo26-paint-defect/
└── data/
    ├── 01-1.정식개방데이터/    # 이 안에 Training, Validation 폴더가 있어야 함
    └── balanced_annotations.csv
```

## 🛠 환경 구축 및 실행 가이드 (Setup Guide)
본 프로젝트는 Miniconda를 기반으로 한 Python 3.10 환경을 표준으로 합니다. RTX 3090 또는 동급의 GPU 자원을 활용하기 위해 아래 절차를 따라주십시오. (Miniconda가 설치 되어 있지 않다면 먼저 설치 후 진행해주세요.)

```bash
# Conda 가상환경 생성 (environment.yml 사용)
conda env create -f environment.yml

# 가상환경 진입
conda activate swin-yolo

# GPU 가속 및 PyTorch 설치 확인
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```
* VS Code 사용자: `Ctrl + Shift + P` -> `Python: Select Interpreter`에서 `swin-yolo` 가상환경을 선택하십시오.

## 📁 Repository Structure
```text
swin-yolo26-paint-defect/
├── 📁 data/               # ⚠️ 빈 폴더 (로컬의 423GB 데이터셋 마운트 경로)
├── 📁 models/             # ⚠️ 빈 폴더 (학습된 가중치 .pt, .mlpackage 저장 경로)
├── 📁 utils/              # 데이터 전처리(csv 파일 생성) 모듈
├── 📁 notebooks/          # EDA, 데이터 불균형 시각화, 실험용 주피터 노트북
├── 📁 src/
│   ├── preprocess.py       # A메타데이터 기반 YOLO 포맷 변환 (On-the-fly)
│   ├── train.py            # Swin-YOLO26 하이브리드 모델 학습 스크립트
│   └── export.py           # CoreML (INT8/FP16) 양자화 변환 스크립트
├── environment.yml         # 프로젝트 전용 Conda 가상환경 정의 파일 (Python 3.10 기반)
│
├── 📁 ios-app/             # [추가] 아이폰 16 구현물용 폴더
│   ├── 📁 ShipPaintApp/    # Xcode 프로젝트 (.xcodeproj)
│   ├── 📁 Models/          # CoreML 모델 파일 (.mlpackage)
│   ├── 📁 ViewControllers/ # 앱의 UI 로직
│   └── README.md            # iOS 앱 빌드/배포 가이드
│
└── .gitignore               # 대용량 파일 및 빌드 산출물 업로드 방지 규칙
```

## 📊 Dataset (AI-Hub 선박 도장 품질 데이터)
* 원본 규모: 약 423GB (양품 10만 장, 불량 4만 장)
* 클래스 정의 (총 11종):
  * 양품(4종): 외판, 선수, 선미, 갑판
  * 불량(7종): 흐름, 핀홀, 이물질포함, 워터스포팅, 부풀음, 도막분리, 균열
* 주의사항: 원본 데이터 및 전처리된 텍스트 라벨 파일은 용량 문제로 GitHub에 업로드하지 않습니다. 반드시 로컬 환경의 `data/` 폴더에 위치시켜야 합니다.

## 🤝 Collaborators
* 주영산업 혁신팀: 프로젝트 총괄, 도메인 지식 제공, 데이터셋 정제, Swin-YOLO26 아키텍처 설계
* UNIST 노바투스대학원 온디바이스AI 수업 Co-sight 팀: 데이터셋 검수, YOLO11m, YOLO26 모델 테스트, ios-app 개발

## 📜 Acknowledgments
본 연구는 과학기술정보통신부가 주관하고 한국지능정보사회진흥원(AI-Hub)이 지원한 '선박 도장 품질 측정 데이터'를 활용하여 수행되었습니다.
