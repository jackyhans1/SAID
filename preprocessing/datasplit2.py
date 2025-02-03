import os
import shutil

# 원본 데이터 폴더와 복사 대상 폴더 정의
source_dir = "/data/ALC/EMU/LAB/"
destination_dir = "/data/alc_jihan/"

# 파일 종류별 복사 대상 폴더 설정
dest_hlb_dir = os.path.join(destination_dir, "hlb")
dest_phonetic_dir = os.path.join(destination_dir, "phonetic")

# 복사 대상 폴더가 없으면 생성
os.makedirs(dest_hlb_dir, exist_ok=True)
os.makedirs(dest_phonetic_dir, exist_ok=True)

# 처리할 BLOCK 폴더 목록
block_list = [f"BLOCK{num}" for num in [10, 11, 20, 30, 40]]

# 각 BLOCK 폴더를 순회
for block_name in block_list:
    block_dir = os.path.join(source_dir, block_name)
    
    # BLOCK 폴더가 존재하는지 확인
    if os.path.exists(block_dir):
        # 세션 폴더(SES1006, SES1007 등)를 순회
        for ses_dir in os.listdir(block_dir):
            ses_path = os.path.join(block_dir, ses_dir)
            
            if os.path.isdir(ses_path):
                # 세션 폴더 내 파일들을 순회
                for file_name in os.listdir(ses_path):
                    file_path = os.path.join(ses_path, file_name)
                    
                    # 파일 확장자에 따라 복사
                    if file_name.endswith(".hlb"):
                        shutil.copy(file_path, dest_hlb_dir)
                    elif file_name.endswith(".phonetic"):
                        shutil.copy(file_path, dest_phonetic_dir)
