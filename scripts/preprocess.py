import os
import pandas as pd
from tqdm.auto import tqdm

# 1. 설정
CSV_PATH = '../data/balanced_annotations.csv'

# YOLO가 인식할 수 있도록 클래스 ID를 0부터 시작하는 연속된 정수로 재매핑합니다.
# (기존 101, 201.. 방식은 YOLO가 에러를 뿜습니다)
CLASS_MAP = {
    101: 0, # 정상
    201: 1, # 워터스포팅
    202: 2, # 흐름
    203: 3, # 도막분리
    204: 4, # 핀홀
    205: 5, # 균열
    206: 6, # 부풀음
    207: 7  # 이물질포함
}

print("🚀 YOLO 포맷 라벨링(.txt) 생성을 시작합니다...")

# 2. 데이터 로드
df = pd.read_csv(CSV_PATH)

# 진행 상황 시각화를 위해 파일명(이미지) 단위로 그룹화합니다.
grouped = df.groupby('file_name')

# 3. 이미지 단위 순회 및 .txt 생성
for file_name, group in tqdm(grouped, desc="라벨 파일 생성 중"):
    # 그룹 내 첫 번째 행에서 실제 경로를 가져옵니다.
    relative_path = group.iloc[0]['relative_path']
    
    # 이미지 파일(.jpg)이 있는 폴더 경로와 파일명(확장자 제외) 추출
    img_dir = os.path.dirname(relative_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 동일한 폴더에 같은 이름의 .txt 파일 경로 생성
    txt_path = os.path.join(img_dir, f"{base_name}.txt")
    
    # txt 파일 내용 작성
    yolo_lines = []

    # '정상(Normal)' 체크: 첫 번째 행의 카테고리가 101이면 바로 파일 건너뜀
    # 정상 이미지는 라벨 파일이 없어야 배경으로 학습됩니다.
    if group.iloc[0]['category_id'] == 101:
        if os.path.exists(txt_path): os.remove(txt_path)
        continue

    for _, row in group.iterrows():
        class_id = CLASS_MAP.get(row['category_id'])
        
        # 만약 매핑에 없는 이상한 ID가 섞여있다면 스킵
        if class_id is None:
            continue
            
        # 1. 원본 절대 좌표
        x_min, y_min, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
        img_w, img_h = row['width'], row['height']

        # x, y, w, h 가 0이면 이미지 전체를 덮는 박스로 설정
        if w == 0 or h == 0:
            x_center = 0.5
            y_center = 0.5
            norm_w = 1.0
            norm_h = 1.0
        else:
            # YOLO 중앙점 좌표 및 정규화 (핵심 변환 로직)
            x_center = (x_min + (w / 2)) / img_w
            y_center = (y_min + (h / 2)) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

        # YOLO 포맷 문자열 조합 (소수점 6자리까지)
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        yolo_lines.append(line)
        
    # 파일 쓰기
    if yolo_lines:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

print(f"\n🎉 완료! 총 {len(grouped):,}개의 .txt 라벨 파일이 이미지와 같은 폴더에 생성되었습니다.")