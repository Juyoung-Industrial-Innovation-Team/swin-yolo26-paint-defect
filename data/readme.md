# 📂 AI-Hub 선박 도장 품질 측정 데이터 (경량화 버전)

## 1. 개요 (Overview)
본 디렉토리는 AI-Hub의 원본 데이터(총 423GB) 중 **'Swin-YOLO26 하이브리드 모델을 활용한 온디바이스 감리 시스템'** 연구에 불필요한 데이터를 제거하여 경량화한 작업 폴더입니다. 

원본 데이터의 디렉토리 구조(`Training`, `Validation`)는 동일하게 유지하되, 연구 대상에서 제외된 특정 양품 부위 및 도막 손상(스크래치 등) 파일 및 **학습에 불필요한 `Other` 폴더 전체를 통째로 삭제**하여 로컬 디스크 및 메모리 부하를 최소화했습니다.

* **🔗 원본 데이터 출처:** [AI-Hub 공식 링크](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71447)

## 2. 데이터 선별 기준 (Target Classes)
본 연구(논문)에서 정의한 **8개 핵심 클래스**와 관련된 데이터만 보존했습니다.
* **✅ 보존 대상 (8종):**
  * 양품 (1종 통합): 선수, 선미, 갑판, 외판 → **정상(Normal)**으로 단일화
  * 도장 불량 (7종): 워터스포팅, 흐름, 도막분리, 핀홀, 균열, 부풀음, 이물질포함
* **❌ 삭제 대상:**
  * 미사용 양품: 선실, 기관실, 파이프, 탱크, 해치커버, 엔진커버
  * 미사용 손상: 용접손상, 스크래치, 도막떨어짐
  * **불필요 부가 정보: `Other` (메타데이터) 폴더 전체**

---

## 3. 삭제된 압축 파일 목록 (Deleted Files History)

### 📁 01-1.정식개방데이터 / Training
* **01.원천데이터 (TS_*)**
  * `TS_양품_해치커버.zip`, `TS_양품_파이프.zip`, `TS_양품_탱크.zip`, `TS_양품_엔진커버.zip`, `TS_양품_선실.zip`, `TS_양품_기관실.zip`
  * `TS_도막 손상_용접손상.zip`, `TS_도막 손상_스크래치.zip`, `TS_도막 손상_도막떨어짐.zip`
* **02.라벨링데이터 (TL_*)**
  * `TL_양품_해치커버.zip`, `TL_양품_파이프.zip`, `TL_양품_탱크.zip`, `TL_양품_엔진커버.zip`, `TL_양품_선실.zip`, `TL_양품_기관실.zip`
  * `TL_도막 손상_용접손상.zip`, `TL_도막 손상_스크래치.zip`, `TL_도막 손상_도막떨어짐.zip`

### 📁 01-1.정식개방데이터 / Validation
* **01.원천데이터 (VS_*)**
  * `VS_양품_해치커버.zip`, `VS_양품_파이프.zip`, `VS_양품_탱크.zip`, `VS_양품_엔진커버.zip`, `VS_양품_선실.zip`, `VS_양품_기관실.zip`
  * `VS_도막 손상_용접손상.zip`, `VS_도막 손상_스크래치.zip`, `VS_도막 손상_도막떨어짐.zip`
* **02.라벨링데이터 (VL_*)**
  * `VL_양품_해치커버.zip`, `VL_양품_파이프.zip`, `VL_양품_탱크.zip`, `VL_양품_엔진커버.zip`, `VL_양품_선실.zip`, `VL_양품_기관실.zip`
  * `VL_도막 손상_용접손상.zip`, `VL_도막 손상_스크래치.zip`, `VL_도막 손상_도막떨어짐.zip`

### 📁 01-1.정식개방데이터 / Other
* **01.메타데이터 (_Other_*)**
  * `_Other_01.메타데이터_양품_해치커버.zip`, `_Other_01.메타데이터_양품_파이프.zip`, `_Other_01.메타데이터_양품_탱크.zip`, `_Other_01.메타데이터_양품_엔진커버.zip`, `_Other_01.메타데이터_양품_선실.zip`, `_Other_01.메타데이터_양품_기관실.zip`
  * `_Other_01.메타데이터_도막 손상_용접손상.zip`, `_Other_01.메타데이터_도막 손상_스크래치.zip`, `_Other_01.메타데이터_도막 손상_도막떨어짐.zip`

### 💥 01-1.정식개방데이터 / Other (폴더 통째로 일괄 삭제)
* YOLO(객체 탐지) 학습 파이프라인에는 이미지(.jpg)와 바운딩 박스 라벨(.json/.txt)만 필요합니다. 따라서 촬영 조건 등 부가적인 텍스트 정보만 담겨 불필요하게 스토리지 용량을 차지하는 **`Other` 디렉토리와 그 하위의 모든 압축 파일(`_Other_01.메타데이터_*.zip`)은 디스크 최적화를 위해 폴더째로 완전 삭제(Wipe-out) 처리**하였습니다.

---

## 4. 데이터 전처리 및 밸런싱 (Data Preprocessing & Balancing)
본 연구는 모델 아키텍처(Swin-YOLO26 등) 간의 순수한 성능 비교를 위해 데이터 불균형(Class Imbalance)을 통제하였습니다. 또한 물리적인 원본 폴더 구조를 훼손하지 않는 **'메타데이터 드라이븐(Metadata-driven)'** 방식을 채택했습니다.

1. **마스터 CSV 추출:** 수만 개의 무거운 JSON 파일에서 객체 BBox 좌표, 해상도, 상대 경로만 추출하여 단일 메타데이터 파일(`balanced_annotations.csv`)로 압축 및 통합했습니다.
2. **언더샘플링 (Under-sampling):** 가장 수량이 적은 소수 클래스인 '균열(Crack)' 이미지(5,220장)를 기준으로, 8개 핵심 클래스의 이미지 수량을 1:1로 동일하게 맞췄습니다.
3. **5:5 Test-set 분할:** 객관적인 성능 평가를 위해 기존 Validation 데이터를 임의 분할(Random Split)하여 Test 셋을 자체 구축했습니다.
   * **Train:** 4,640장 / **Valid:** 290장 / **Test:** 290장 (클래스별 공통, 총 41,760장 활용)
4. **잔여 파일 삭제 (최적화):** 선택되지 않은 수십만 장의 잔여 `.jpg` 이미지와 더 이상 불필요해진 `.json` 라벨 파일들은 로컬 스토리지 최적화를 위해 물리적으로 모두 삭제되었습니다.

---

## 5. 학습 데이터 참조 및 활용 가이드 (Data Loading)
이제 복잡한 AI-Hub 폴더 구조를 순회하거나 무거운 JSON 파일을 다시 파싱할 필요가 없습니다. PyTorch Dataset 또는 YOLO DataLoader 구축 시 오직 `balanced_annotations.csv` 파일 하나만 읽어 들여 활용합니다.

**[데이터 로더 활용 예시 (Python/Pandas)]**
```python
import pandas as pd

# 1. 밸런싱된 마스터 메타데이터 로드
df = pd.read_csv('../data/balanced_annotations.csv')

# 2. 학습(Train) / 검증(Valid) / 평가(Test) 셋 분리
train_df = df[df['split'] == 'Train']
valid_df = df[df['split'] == 'Valid']
test_df  = df[df['split'] == 'Test']

# 3. 데이터 로딩 로직 예시 (YOLO 정규화 포함)
for index, row in train_df.iterrows():
    img_path = row['relative_path'] # 원본 이미지의 실제 상대 경로
    class_id = row['category_id']   # 타겟 클래스 ID
    
    # YOLO 0~1 스케일 정규화 (Center X, Center Y, Width, Height)
    x_center = (row['bbox_x'] + (row['bbox_w'] / 2)) / row['width']
    y_center = (row['bbox_y'] + (row['bbox_h'] / 2)) / row['height']
    w_norm = row['bbox_w'] / row['width']
    h_norm = row['bbox_h'] / row['height']
    
    # 이후 cv2.imread(img_path) 로드 및 Tensor 변환 수행...
```

**💡 협업자 가이드 (Collaborator Guide)**
> 이 환경을 넘겨받은 공동 연구자들은 별도의 필터링이나 JSON 파싱 작업 없이, 제공된 `balanced_annotations.csv`의 `relative_path`를 참조하여 곧바로 모델 학습 파이프라인 개발에 착수하시면 됩니다.