import os
import torch
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
from difflib import SequenceMatcher  # 유사도 계산용

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_DIR = "/data/alc_jihan/h_wav_16K_merged"
TASK_CSV = "/data/alc_jihan/split_index/merged_data.csv"
OUTPUT_CSV = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_whisper.csv"

model = whisper.load_model("large").to(device)

# Task 정보 불러오기: 파일명을 key로 하여 SubjectID, Class, Split, Task 저장
task_df = pd.read_csv(TASK_CSV)
task_mapping = {
    row["FileName"]: (row["SubjectID"], row["Class"], row["Split"], row["Task"])
    for _, row in task_df.iterrows()
}

def extract_features(audio_path):
    """
    Whisper를 이용하여 음성 파일로부터 feature들을 추출합니다.
    
    - Len_S: 전체 침묵 시간
    - Var_S: 침묵 길이 분산
    - Avg_S: 침묵 길이 평균
    - Max_S: 최대 침묵 길이
    - Rat_S: 전체 발화 구간 대비 침묵 비율
    - Num_S: 침묵 구간 개수
    - Num_PS: 연속 침묵 구간 간 시작 시간 차 개수
    - Avg_PS: 연속 침묵 구간 간 시작 시간 차 평균
    - Var_PS: 연속 침묵 구간 간 시작 시간 차 분산
    - Hesitation: 주저어("äh", "ähm", "hm" 등)의 등장 횟수
    - P_less_1: 1초 이하의 침묵 구간 개수
    - P_more_1: 1초 초과의 침묵 구간 개수
    - Repetition: 연속해서 반복된 단어 수 (발음이 약간 달라도 유사도가 높으면 count)
    - P_Sum: (P_less_1 + P_more_1)
    """
    try:
        # Whisper 음성 인식 수행 (독일어 설정)
        result = model.transcribe(audio_path, language="de", word_timestamps=True)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return [0]*9 + [0, 0, 0, 0, 0]

    # [1] 침묵 구간 관련 feature 추출
    speech_segments = [(segment["start"], segment["end"]) for segment in result["segments"]]
    
    if not speech_segments:
        silence_feats = [0]*9
        P_less_1 = 0
        P_more_1 = 0
        P_Sum = 0
    else:
        audio_start = speech_segments[0][0]
        audio_end = speech_segments[-1][1]
    
        silence_segments = []
        prev_end = audio_start
        for start, end in speech_segments:
            if start > prev_end:
                silence_segments.append((prev_end, start))
            prev_end = end
        # 맨 뒤의 침묵 제거
        silence_segments = [seg for seg in silence_segments if seg[1] <= audio_end]
    
        if not silence_segments:
            silence_feats = [0]*9
            P_less_1 = 0
            P_more_1 = 0
            P_Sum = 0
        else:
            silence_durations = [end - start for start, end in silence_segments]
            len_s = sum(silence_durations)
            var_s = np.var(silence_durations)
            avg_s = np.mean(silence_durations)
            max_s = np.max(silence_durations)
            rat_s = len_s / (audio_end - audio_start) if (audio_end - audio_start) > 0 else 0
            num_s = len(silence_segments)
    
            if num_s > 1:
                silent_diffs = np.diff([start for start, _ in silence_segments])
                num_ps = len(silent_diffs)
                avg_ps = np.mean(silent_diffs)
                var_ps = np.var(silent_diffs)
            else:
                num_ps = 0
                avg_ps = 0
                var_ps = 0
    
            silence_feats = [len_s, var_s, avg_s, max_s, rat_s, num_s, num_ps, avg_ps, var_ps]
    
            # 추가 침묵 feature: 1초 이하, 1초 초과, 합계
            P_less_1 = sum(1 for d in silence_durations if d <= 1.0)
            P_more_1 = sum(1 for d in silence_durations if d > 1.0)
            P_Sum = P_less_1 + P_more_1
    
    # [2] 텍스트 기반 추가 feature 추출
    # 포괄적인 주저어(filler) 리스트 (독일어)
    hesitation_fillers = {
        # "äh" 계열
        "äh", "ähh", "ähhh", "aäh", "ääh", "äääh", "aeh", "aehh", "aehhh",
        # "ähm" 계열
        "ähm", "ähmm", "ähmmm", "ähem", "äähm", "aehm", "aehmm", "aehmmm",
        # "hm" 계열
        "hm", "hmm", "hmmm", "mmm", "mhm",
        # "öhm" 계열
        "öhm", "ööh", "ööhmm", "oehm", "oohm", "oohmm"
    }
    
    hesitation_count = 0
    repetition_count = 0
    similarity_threshold = 0.5
    
    translator = str.maketrans('', '', string.punctuation)
    
    for segment in result["segments"]:
        if "words" in segment:
            words = [w["word"] for w in segment["words"]]
        else:
            words = segment["text"].split()
        cleaned_words = [w.lower().translate(translator) for w in words]
        
        for w in cleaned_words:
            if w in hesitation_fillers:
                hesitation_count += 1
        
        # 연속된 단어 비교 (약간의 변형도 고려)
        for i in range(1, len(cleaned_words)):
            if cleaned_words[i] and cleaned_words[i-1]:
                similarity = SequenceMatcher(None, cleaned_words[i], cleaned_words[i-1]).ratio()
                if similarity >= similarity_threshold:
                    repetition_count += 1

    additional_feats = [hesitation_count, P_less_1, P_more_1, repetition_count, P_Sum]
    
    return silence_feats + additional_feats

all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
all_features = []

for file in tqdm(all_files, desc="Extracting Features"):
    file_path = os.path.join(AUDIO_DIR, file)
    file_name = os.path.splitext(file)[0]
    features = extract_features(file_path)
    
    subject_id, cls, split, task = task_mapping.get(
        file_name, ("Unknown", "Unknown", "Unknown", "Unknown")
    )
    
    all_features.append([file_name, subject_id, cls, split] + features + [task])

columns = [
    "Filename", "SubjectID", "Class", "Split", "Task",
    "Len_S", "Var_S", "Avg_S", "Max_S", "Rat_S", "Num_S", "Num_PS", "Avg_PS", "Var_PS",
    "Hesitation", "P_less_1", "P_more_1", "Repetition", "P_Sum"
]

df = pd.DataFrame(all_features, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Feature extraction complete! Saved to {OUTPUT_CSV}")
