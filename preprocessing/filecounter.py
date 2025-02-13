import os

destination_dir = "/data/alc_jihan/"
dirs_to_count = {
    "h_wav": os.path.join(destination_dir, "h_wav"),
    "m_wav": os.path.join(destination_dir, "m_wav"),
    "par": os.path.join(destination_dir, "par"),
    "textgrid": os.path.join(destination_dir, "textgrid"),
    "hlb": os.path.join(destination_dir, "hlb"),
    "phonetic": os.path.join(destination_dir, "phonetic"),
    "m_json": os.path.join(destination_dir, "m_json"),
    "h_json": os.path.join(destination_dir, "h_json"),
    "h_wav_16K_sliced": os.path.join(destination_dir, "h_wav_16K_sliced"),
    "h_wav_16K_merged": os.path.join(destination_dir, "h_wav_16K_merged"),
    "h_wav_slided": os.path.join(destination_dir, "h_wav_slided"),
    "melspectrograms_for_swin": os.path.join(destination_dir, "melspectrograms_for_swin"),
    "extracted_features": os.path.join(destination_dir, "extracted_features"),
}

# 각 디렉토리 파일 수 계산 및 출력
file_counts = {}
for key, dir_path in dirs_to_count.items():
    if os.path.exists(dir_path):
        # 디렉토리 내 파일 수 세기
        file_counts[key] = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    else:
        # 디렉토리가 없으면 0으로 표시
        file_counts[key] = 0

# 결과 출력
for key, count in file_counts.items():
    print(f"{key}: {count} files")
