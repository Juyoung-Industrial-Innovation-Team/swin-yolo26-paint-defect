import os
from tqdm.auto import tqdm

# 1. 설정
TARGET_DIR = '../data/01-1.정식개방데이터'
# 주의: True일 때는 삭제하지 않고 목록만 보여줍니다. 
# 실제 삭제를 원하시면 False로 변경하세요.
DRY_RUN = True 

def cleanup_txt_files(directory):
    files_to_delete = []
    
    # 2. 모든 폴더 방문
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.txt'):
                files_to_delete.append(os.path.join(root, file))
    
    print(f"🔍 총 {len(files_to_delete):,}개의 .txt 파일을 찾았습니다.")
    
    if DRY_RUN:
        print("\n🚨 [Dry Run 모드] 실제 삭제가 진행되지 않습니다.")
        for f in files_to_delete[:10]: # 예시로 10개만 출력
            print(f"삭제 예정: {f}")
        print("... (중략) ...")
        print("\n실제 삭제를 원하시면 코드의 DRY_RUN 변수를 False로 변경하세요.")
    else:
        print("\n⚠️ 실제 삭제를 시작합니다...")
        for f in tqdm(files_to_delete, desc="삭제 중"):
            try:
                os.remove(f)
            except Exception as e:
                print(f"에러 발생: {f} - {e}")
        print("✅ 삭제 완료!")

if __name__ == "__main__":
    if os.path.exists(TARGET_DIR):
        cleanup_txt_files(TARGET_DIR)
    else:
        print(f"❌ 경로를 찾을 수 없습니다: {TARGET_DIR}")