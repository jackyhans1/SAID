import os
import shutil

# 원본 데이터 폴더와 복사 대상 폴더 정의
source_dir = "/data/ALC/EMU/ALC_emuDB/"
destination_dir = "/data/alc_jihan/"

# 파일 종류별 복사 대상 폴더 설정
dest_h_json_dir = os.path.join(destination_dir, "h_json")
dest_m_json_dir = os.path.join(destination_dir, "m_json")

# 복사 대상 폴더가 없으면 생성
os.makedirs(dest_h_json_dir, exist_ok=True)
os.makedirs(dest_m_json_dir, exist_ok=True)

# 처리할 블록 목록 (블록 50과 60 제외)
valid_blocks = ["BLOCK10", "BLOCK11", "BLOCK20", "BLOCK30", "BLOCK40"]

# 세션 디렉토리를 탐색
for block_ses_dir in os.listdir(source_dir):  # e.g., BLOCK10_SES1006_ses
    # 블록 이름이 valid_blocks에 포함되어야 처리
    if any(block_ses_dir.startswith(block) for block in valid_blocks):
        block_ses_path = os.path.join(source_dir, block_ses_dir)
        
        if os.path.isdir(block_ses_path):  # 디렉토리 확인
            for bundle_dir in os.listdir(block_ses_path):  # 번들 디렉토리 탐색
                bundle_path = os.path.join(block_ses_path, bundle_dir)
                if os.path.isdir(bundle_path):  # 번들 디렉토리가 맞는지 확인
                    for file_name in os.listdir(bundle_path):  # 파일 탐색
                        file_path = os.path.join(bundle_path, file_name)
                        # 파일 이름에 따라 복사
                        if file_name.endswith("_h_00_annot.json") or \
                           file_name.endswith("_h_01_annot.json") or \
                           file_name.endswith("_h_02_annot.json") or \
                           file_name.endswith("_h_03_annot.json") or \
                           file_name.endswith("_h_04_annot.json") or \
                           file_name.endswith("_h_05_annot.json"):
                            shutil.copy(file_path, dest_h_json_dir)
                        elif file_name.endswith("_m_00_annot.json") or \
                             file_name.endswith("_m_01_annot.json") or \
                             file_name.endswith("_m_02_annot.json") or \
                             file_name.endswith("_m_03_annot.json") or \
                             file_name.endswith("_m_04_annot.json") or \
                             file_name.endswith("_m_05_annot.json"):
                            shutil.copy(file_path, dest_m_json_dir)
