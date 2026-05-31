import pandas as pd
import os

CSV_PATH = '../data/balanced_annotations.csv'
# 1. 절대 경로의 뼈대 (기준점)
DATA_ROOT_ABS_PATH = os.path.abspath('../data/01-1.정식개방데이터')

df = pd.read_csv(CSV_PATH)
images_df = df[['file_name', 'relative_path', 'split']].drop_duplicates()

print("🚀 경로 리스트 재생성 중...")

for split_name in ['Train', 'Valid', 'Test']:
    split_data = images_df[images_df['split'] == split_name]
    list_file_path = f'../data/{split_name.lower()}_list.txt'
    
    with open(list_file_path, 'w', encoding='utf-8') as f:
        for _, row in split_data.iterrows():
            rel_path = row['relative_path']
            
            # 2. 앞부분의 지저분한 상대경로(./, ../data 등)를 무시하고 
            # 핵심 폴더인 'Training' 또는 'Validation'부터 문자열을 잘라냅니다.
            if 'Training' in rel_path:
                core_path = rel_path[rel_path.find('Training'):]
            elif 'Validation' in rel_path:
                core_path = rel_path[rel_path.find('Validation'):]
            else:
                core_path = rel_path # 예외 상황 대비
            
            # 3. 완벽한 절대경로 조립
            abs_img_path = os.path.join(DATA_ROOT_ABS_PATH, core_path)
            
            # 윈도우(\)와 리눅스(/) 슬래시 혼용 방지 (YOLO는 / 를 선호합니다)
            abs_img_path = abs_img_path.replace('\\', '/')
            
            f.write(f"{abs_img_path}\n")

print("✅ 중복 경로가 제거된 list.txt 파일 생성 완료!")