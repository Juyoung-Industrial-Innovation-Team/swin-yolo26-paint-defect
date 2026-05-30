import pandas as pd
import os

CSV_PATH = '../data/balanced_annotations.csv'
# 데이터셋이 있는 절대 경로 (로컬 환경에 맞게 수정 필요)
# 예시: C:/Projects/swin-yolo26/data/01-1.정식개방데이터
DATA_ROOT_ABS_PATH = os.path.abspath('../data/01-1.정식개방데이터')

df = pd.read_csv(CSV_PATH)
# 중복 제거 (이미지 한 장당 한 줄만 필요)
images_df = df[['file_name', 'relative_path', 'split']].drop_duplicates()

# Train, Valid, Test 별로 파일 생성
for split_name in ['Train', 'Valid', 'Test']:
    split_data = images_df[images_df['split'] == split_name]
    list_file_path = f'../data/{split_name.lower()}_list.txt'
    
    with open(list_file_path, 'w', encoding='utf-8') as f:
        for _, row in split_data.iterrows():
            # CSV의 상대 경로에서 앞의 '.' 같은 불필요한 부분 제거 후 결합
            # 예: './Training/원천데이터/...' -> 'Training/원천데이터/...'
            clean_rel_path = row['relative_path'].lstrip('./\\')
            abs_img_path = os.path.join(DATA_ROOT_ABS_PATH, clean_rel_path)
            
            # 슬래시 방향 통일 (윈도우 환경 대비)
            abs_img_path = abs_img_path.replace('\\', '/')
            f.write(f"{abs_img_path}\n")

print("✅ train_list.txt, valid_list.txt, test_list.txt 생성 완료!")