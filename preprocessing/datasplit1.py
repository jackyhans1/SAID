import os
import shutil

# 원본 데이터 폴더와 복사 대상 폴더 정의
source_dir = "/data/ALC/DATA/"
destination_dir = "/data/alc_jihan/"

# 파일 종류별 복사 대상 폴더 설정
dest_h_wav_dir = os.path.join(destination_dir, "h_wav")
dest_m_wav_dir = os.path.join(destination_dir, "m_wav")
dest_par_dir = os.path.join(destination_dir, "par")
dest_textgrid_dir = os.path.join(destination_dir, "textgrid")

# 복사 대상 폴더가 없으면 생성
os.makedirs(dest_h_wav_dir, exist_ok=True)
os.makedirs(dest_m_wav_dir, exist_ok=True)
os.makedirs(dest_par_dir, exist_ok=True)
os.makedirs(dest_textgrid_dir, exist_ok=True)

# 처리할 블록 이름 목록
block_list = [f"BLOCK{num}" for num in [10, 11, 20, 30, 40]]

# 각 블록을 순회
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
                    
                    # 파일 확장자 및 이름에 따라 복사 대상 폴더로 복사
                    if file_name.endswith("_h_00.wav"):
                        shutil.copy(file_path, dest_h_wav_dir)
                    elif file_name.endswith("_h_01.wav"):
                        shutil.copy(file_path, dest_h_wav_dir)
                    elif file_name.endswith("_h_02.wav"):
                        shutil.copy(file_path, dest_h_wav_dir)
                    elif file_name.endswith("_h_03.wav"):
                        shutil.copy(file_path, dest_h_wav_dir)
                    elif file_name.endswith("_h_04.wav"):
                        shutil.copy(file_path, dest_h_wav_dir)
                    elif file_name.endswith("_h_05.wav"):
                        shutil.copy(file_path, dest_h_wav_dir)
                    elif file_name.endswith("_m_00.wav"):
                        shutil.copy(file_path, dest_m_wav_dir)
                    elif file_name.endswith("_m_01.wav"):
                        shutil.copy(file_path, dest_m_wav_dir)
                    elif file_name.endswith("_m_02.wav"):
                        shutil.copy(file_path, dest_m_wav_dir)
                    elif file_name.endswith("_m_03.wav"):
                        shutil.copy(file_path, dest_m_wav_dir)
                    elif file_name.endswith("_m_04.wav"):
                        shutil.copy(file_path, dest_m_wav_dir)
                    elif file_name.endswith("_m_05.wav"):
                        shutil.copy(file_path, dest_m_wav_dir)
                    elif file_name.endswith(".par"):
                        shutil.copy(file_path, dest_par_dir)
                    elif file_name.endswith(".TextGrid"):
                        shutil.copy(file_path, dest_textgrid_dir)
