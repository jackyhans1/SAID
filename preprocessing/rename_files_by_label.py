import os
import pandas as pd

# CSV 파일 경로 및 데이터 로드
csv_path = "/data/ALC/TABLE/SESSEXT.csv"
csv_data = pd.read_csv(csv_path)

# /data/alc_jihan 디렉토리 설정
base_dir = "/data/alc_jihan"
folders = ["h_json", "h_wav", "m_wav", "m_json", "par", "hlb", "phonetic", "textgrid"]  # 탐색할 폴더들

# CSV 데이터에서 SES와 LAB 매핑
ses_lab_map = {str(row["SES"]).zfill(4): str(row["LAB"]) for _, row in csv_data.iterrows()}

# 모든 폴더 순회
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    
    if not os.path.exists(folder_path):
        continue
    
    # 해당 폴더 내 파일 순회
    for filename in os.listdir(folder_path):
        # SES 값을 파일 이름에서 추출
        try:
            ses_id = filename.split("_")[0][3:7]
        except IndexError:
            print(f"Skipping file due to unexpected name format: {filename}")
            continue
        
        # CSV 데이터에서 SES와 일치하는 LAB 값 가져오기
        if ses_id in ses_lab_map:
            lab_value = ses_lab_map[ses_id]
            
            # 새로운 파일명 생성
            new_filename = f"{lab_value}_{filename}"
            
            # 파일명 변경
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Failed to rename {old_file_path} -> {new_file_path}: {e}")
