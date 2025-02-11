import os
import torch
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_DIR = "/data/alc_jihan/h_wav_16K_sliced"
TASK_CSV = "/data/alc_jihan/split_index/dataset_split_sliced.csv"  # Task 정보가 있는 CSV 파일
OUTPUT_CSV = "/data/alc_jihan/extracted_features_whisper_disfluency/all_data_Disfluency_features_more_added.csv"  # 출력 CSV 파일

# Whisper 모델 로드
model = whisper.load_model("large").to(device)

# Task 정보 불러오기
task_df = pd.read_csv(TASK_CSV)

# 파일명을 key로 하여 SubjectID, Class, Split, Task 저장
task_mapping = {row["FileName"]: (row["SubjectID"], row["Class"], row["Split"], row["Task"]) for _, row in task_df.iterrows()}

def extract_silence_features(audio_path):
    """Whisper 기반으로 음성과 침묵 구간을 분석 (맨 앞, 맨 뒤 묵음 제외)"""
    
    # Whisper 음성 인식 수행 (독일어 설정, GPU에서 실행)
    result = model.transcribe(audio_path, language="de", word_timestamps=True)

    # 음성(발화) 구간 추출
    speech_segments = [(segment["start"], segment["end"]) for segment in result["segments"]]

    if not speech_segments:
        return [0] * 9  # 음성이 전혀 없는 경우 기본값 반환
    
    # 첫 번째 발화 시작과 마지막 발화 끝을 기준으로 묵음 제거
    audio_start = speech_segments[0][0]  # 첫 번째 발화 시작점
    audio_end = speech_segments[-1][1]  # 마지막 발화 끝점

    # 중간 침묵 구간만 추출
    silence_segments = []
    prev_end = audio_start  # 첫 번째 발화 시작점 이후부터 분석

    for start, end in speech_segments:
        if start > prev_end:
            silence_segments.append((prev_end, start))  # 중간 침묵 구간 추가
        prev_end = end

    # 마지막 침묵(맨 뒤 묵음) 제거
    silence_segments = [seg for seg in silence_segments if seg[1] <= audio_end]

    # Feature 계산
    if not silence_segments:
        return [0] * 9  # 중간 침묵이 없는 경우 기본값 반환
    
    silence_durations = [end - start for start, end in silence_segments]
    len_s = sum(silence_durations)  # 전체 침묵 시간
    var_s = np.var(silence_durations)  # 침묵 길이 분산
    avg_s = np.mean(silence_durations)  # 침묵 길이 평균
    max_s = np.max(silence_durations)  # 최대 침묵 길이
    rat_s = len_s / (audio_end - audio_start)  # 전체 길이 대비 침묵 비율
    num_s = len(silence_segments)  # 침묵 구간 개수

    # 참가자(P) 간 침묵 구간 분석
    silent_diffs = np.diff([start for start, _ in silence_segments])
    num_ps = len(silent_diffs)
    avg_ps = np.mean(silent_diffs) if num_ps > 0 else 0
    var_ps = np.var(silent_diffs) if num_ps > 0 else 0
    
    return [len_s, var_s, avg_s, max_s, rat_s, num_s, num_ps, avg_ps, var_ps]

# 전체 WAV 파일 처리
all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
all_features = []

for file in tqdm(all_files, desc="Extracting Features"):
    file_path = os.path.join(AUDIO_DIR, file)
    file_name = os.path.splitext(file)[0]  # 확장자 제거한 파일명
    features = extract_silence_features(file_path)
    
    # Task 정보 추가
    subject_id, cls, split, task = task_mapping.get(file_name, ("Unknown", "Unknown", "Unknown", "Unknown"))
    
    all_features.append([file_name, subject_id, cls, split] + features + [task])

# 결과 저장 (Task 포함)
df = pd.DataFrame(all_features, columns=["Filename", "SubjectID", "Class", "Split", "Len_S", "Var_S", "Avg_S", "Max_S", "Rat_S", "Num_S", "Num_PS", "Avg_PS", "Var_PS", "Task"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"Feature extraction complete! Saved to {OUTPUT_CSV}")
