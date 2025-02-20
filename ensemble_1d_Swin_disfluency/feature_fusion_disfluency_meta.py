import os
import pandas as pd
import torch
import numpy as np

# ====== 경로 설정 ======
CSV_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_meta_data.csv"
HUBERT_PT_DIR = "/data/alc_jihan/HuBERT_feature_merged"
OUT_DIR = "/data/alc_jihan/hubert_meta_disfluency_feature_fusion"

# ====== 사용할 컬럼 지정 ======
DISFLUENCY_COLS = ["Repetition", "Rat_S", "Avg_S", "Var_S", "P_more_1"]
METADATA_COLS = ["WEI", "HEI"]
TASK_COL = "Task"
FILENAME_COL = "FileName"

# ====== 1) CSV 로드 ======
df = pd.read_csv(CSV_PATH)

# ====== 2) Disfluency & Metadata에 대한 min-max 계산 ======
disfluency_min = df[DISFLUENCY_COLS].min()
disfluency_max = df[DISFLUENCY_COLS].max()
metadata_min = df[METADATA_COLS].min()
metadata_max = df[METADATA_COLS].max()

# ====== 3) HuBERT Feature의 전역 통계량 (Z-score 정규화용) ======
#     - 모든 PT 파일의 모든 프레임을 누적하여 통계량을 구합니다.
sum_hubert = None
sum_hubert_sq = None
total_frames = 0  # 전체 프레임 수

print("==> [1차 루프] HuBERT Feature 전역 통계량(Mean, Std) 계산 중...")

for idx, row in df.iterrows():
    file_name = row[FILENAME_COL]  # 예: "0_0062014001_h_00"
    pt_path = os.path.join(HUBERT_PT_DIR, file_name + ".pt")
    
    if not os.path.exists(pt_path):
        continue
    
    # HuBERT Feature 로드: shape가 (1, T, 1024) 또는 (T, 1024)라고 가정
    hubert_feat = torch.load(pt_path).float()
    if hubert_feat.dim() == 3 and hubert_feat.shape[0] == 1:
        hubert_feat = hubert_feat.squeeze(0)  # → (T, 1024)
    
    if hubert_feat.dim() != 2 or hubert_feat.shape[1] != 1024:
        print(f"[WARNING] {file_name}의 shape이 예상과 다름: {hubert_feat.shape}")
        continue
    
    hubert_np = hubert_feat.cpu().numpy()  # (T, 1024)
    
    # 초기화
    if sum_hubert is None:
        sum_hubert = np.zeros(1024, dtype=np.float64)
        sum_hubert_sq = np.zeros(1024, dtype=np.float64)
    
    # 모든 프레임에 대해 각 차원의 합과 제곱합을 누적
    sum_hubert += hubert_np.sum(axis=0)         # (1024,)
    sum_hubert_sq += (hubert_np ** 2).sum(axis=0)  # (1024,)
    total_frames += hubert_np.shape[0]

if total_frames == 0:
    raise ValueError("HuBERT PT 파일을 찾지 못했거나 형식이 올바르지 않습니다.")

global_hubert_mean = sum_hubert / total_frames
global_hubert_var = (sum_hubert_sq / total_frames) - (global_hubert_mean ** 2)
global_hubert_std = np.sqrt(global_hubert_var)
global_hubert_std[global_hubert_std < 1e-9] = 1e-9  # division 방지

print(f"==> HuBERT Feature 전역 통계량 계산 완료: total_frames={total_frames}")
print(f"    global_mean[:5] = {global_hubert_mean[:5]}")
print(f"    global_std[:5]  = {global_hubert_std[:5]}")

# ====== 4) Feature Fusion (시퀀스 전체 유지 + zero padding) 및 저장 ======
print("==> [2차 루프] 시퀀스 전체 Feature Fusion 및 .pt 파일 생성 중...")

# monologue / dialogue 태스크만 Disfluency 적용
TARGET_TASKS = ["monologue", "dialogue"]

os.makedirs(OUT_DIR, exist_ok=True)

for idx, row in df.iterrows():
    file_name = row[FILENAME_COL]
    task = row[TASK_COL]
    
    pt_path = os.path.join(HUBERT_PT_DIR, file_name + ".pt")
    out_path = os.path.join(OUT_DIR, file_name + ".pt")
    
    if not os.path.exists(pt_path):
        continue
    
    # 1) HuBERT Feature 로드 (시퀀스 전체 유지)
    hubert_feat = torch.load(pt_path).float()
    if hubert_feat.dim() == 3 and hubert_feat.shape[0] == 1:
        hubert_feat = hubert_feat.squeeze(0)  # → (T, 1024)
    
    if hubert_feat.dim() != 2 or hubert_feat.shape[1] != 1024:
        print(f"[WARNING] {file_name}의 shape이 예상과 다름 (2차 루프): {hubert_feat.shape}")
        continue
    
    hubert_np = hubert_feat.cpu().numpy()  # (T, 1024)
    T = hubert_np.shape[0]
    
    # 2) 각 프레임에 대해 Z-score 정규화 (전역 통계량 사용)
    normalized_hubert = (hubert_np - global_hubert_mean) / global_hubert_std  # (T, 1024)
    
    # 3) Metadata (WEI, HEI) Min-Max 정규화 (static, shape: (2,))
    meta_vals = row[METADATA_COLS].values.astype(np.float32)
    meta_min_arr = metadata_min[METADATA_COLS].values.astype(np.float32)
    meta_max_arr = metadata_max[METADATA_COLS].values.astype(np.float32)
    meta_range = meta_max_arr - meta_min_arr
    meta_range[meta_range < 1e-9] = 1e-9
    meta_norm = (meta_vals - meta_min_arr) / meta_range  # (2,)
    # 각 프레임에 대해 복제 → (T, 2)
    meta_norm_expanded = np.tile(meta_norm, (T, 1))
    
    # 4) Disfluency (Repetition, Rat_S, Avg_S, Var_S, P_more_1) Min-Max 정규화 (static, (5,))
    disflu_vals = row[DISFLUENCY_COLS].values.astype(np.float32)
    disflu_min_arr = disfluency_min[DISFLUENCY_COLS].values.astype(np.float32)
    disflu_max_arr = disfluency_max[DISFLUENCY_COLS].values.astype(np.float32)
    disflu_range = disflu_max_arr - disflu_min_arr
    disflu_range[disflu_range < 1e-9] = 1e-9
    disflu_norm = (disflu_vals - disflu_min_arr) / disflu_range  # (5,)
    # Task에 따라 각 프레임에 대해 복제 또는 0으로 채움 → (T, 5)
    if task in TARGET_TASKS:
        disflu_norm_expanded = np.tile(disflu_norm, (T, 1))
    else:
        disflu_norm_expanded = np.zeros((T, len(DISFLUENCY_COLS)), dtype=np.float32)
    
    # 5) 모든 feature를 시간축(T) 기준으로 Concatenate  
    # fused_feature: [normalized_hubert (T,1024), meta_norm_expanded (T,2), disflu_norm_expanded (T,5)] → (T,1031)
    fused_vec = np.concatenate([normalized_hubert, meta_norm_expanded, disflu_norm_expanded], axis=1)
    
    # 6) fused_vec를 tensor로 변환 및 배치 차원 추가 (최종 shape: (1, T, 1031))
    fused_tensor = torch.from_numpy(fused_vec).float().unsqueeze(0)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(fused_tensor, out_path)

print("==> 모든 작업이 완료되었습니다!")
