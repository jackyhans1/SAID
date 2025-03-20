import os
import glob
import csv
import torch
import soundfile as sf
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_silence_segments(total_samples, speech_segments, sr=16000):
    """
    전체 파일 길이(total_samples)와 VAD로 얻은 음성 구간(speech_segments)을 바탕으로,
    침묵(silence) 구간을 [start_sec, end_sec] 리스트로 반환.
    - speech_segments: [{'start': int, 'end': int}, ...] (샘플 단위, 오름차순 정렬 가정)
    - return: list of (start_sec, end_sec) in seconds
    """
    silence_segments = []
    total_dur_sec = total_samples / sr

    # speech_segments가 비어있다면 -> 전체가 침묵
    if len(speech_segments) == 0:
        return [(0.0, total_dur_sec)]

    # 정렬이 안 되어 있으면 start 기준 정렬
    speech_segments = sorted(speech_segments, key=lambda x: x['start'])

    # 파일 시작~첫 번째 음성 구간 전
    first_start = speech_segments[0]['start'] / sr
    if first_start > 0:
        silence_segments.append((0.0, first_start))

    # 음성 구간 사이
    for i in range(len(speech_segments) - 1):
        end_i = speech_segments[i]['end'] / sr
        start_next = speech_segments[i+1]['start'] / sr
        if start_next > end_i:
            silence_segments.append((end_i, start_next))

    # 마지막 음성 구간 후~파일 끝
    last_end = speech_segments[-1]['end'] / sr
    if last_end < total_dur_sec:
        silence_segments.append((last_end, total_dur_sec))

    return silence_segments


def compute_silence_features(silence_segments, total_dur_sec):
    """
    침묵 구간 리스트(silence_segments)를 바탕으로
    Len_S, Var_S, Avg_S, Max_S, Rat_S, Num_S, Num_PS, Avg_PS, Var_PS, P_less_1, P_more_1, P_Sum
    등을 계산하여 dict로 반환.
    - silence_segments: [(start_sec, end_sec), ...]
    - total_dur_sec: 전체 파일 길이(초)
    """
    features = {
        "Len_S": 0.0,
        "Var_S": 0.0,
        "Avg_S": 0.0,
        "Max_S": 0.0,
        "Rat_S": 0.0,
        "Num_S": 0,
        "Num_PS": 0,
        "Avg_PS": 0.0,
        "Var_PS": 0.0,
        "P_less_1": 0,
        "P_more_1": 0,
        "P_Sum": 0
    }

    if len(silence_segments) == 0:
        # 침묵 구간이 없다면 모든 값 0
        return features

    # 각 침묵 구간의 길이(초)
    durations = []
    # 침묵 구간 시작 시간 리스트
    start_times = []

    for (st, ed) in silence_segments:
        dur = ed - st
        durations.append(dur)
        start_times.append(st)

    # 총 침묵 길이
    Len_S = sum(durations)
    Num_S = len(durations)
    Max_S = max(durations) if Num_S > 0 else 0.0
    Avg_S = Len_S / Num_S if Num_S > 0 else 0.0
    Var_S = np.var(durations, ddof=1) if Num_S > 1 else 0.0

    # 전체 길이 대비 침묵 비율
    Rat_S = Len_S / total_dur_sec if total_dur_sec > 0 else 0.0

    # 연속 침묵 구간 간 시작 시간 차
    if Num_S > 1:
        ps_diffs = []
        for i in range(len(start_times) - 1):
            ps_diffs.append(start_times[i+1] - start_times[i])
        Num_PS = len(ps_diffs)
        Avg_PS = np.mean(ps_diffs) if Num_PS > 0 else 0.0
        Var_PS = np.var(ps_diffs, ddof=1) if Num_PS > 1 else 0.0
    else:
        ps_diffs = []
        Num_PS = 0
        Avg_PS = 0.0
        Var_PS = 0.0

    # 침묵 길이가 1초 이하/초과
    P_less_1 = sum(1 for d in durations if d <= 1.0)
    P_more_1 = sum(1 for d in durations if d > 1.0)
    P_Sum = P_less_1 + P_more_1  # == Num_S

    features["Len_S"] = Len_S
    features["Var_S"] = Var_S
    features["Avg_S"] = Avg_S
    features["Max_S"] = Max_S
    features["Rat_S"] = Rat_S
    features["Num_S"] = Num_S
    features["Num_PS"] = Num_PS
    features["Avg_PS"] = Avg_PS
    features["Var_PS"] = Var_PS
    features["P_less_1"] = P_less_1
    features["P_more_1"] = P_more_1
    features["P_Sum"] = P_Sum

    return features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_ts,
     get_speech_ts_adaptive,
     read_audio,
     save_audio,
     get_number_ts) = utils

    vad_model = vad_model.to(device)
    vad_model.eval()

    source_dir = "/data/alc_jihan/h_wav_16K_merged"
    dest_dir   = "/data/alc_jihan/h_wav_16K_VAD"
    os.makedirs(dest_dir, exist_ok=True)

    csv_dir = "/data/alc_jihan/extracted_features_vad_dysfluency"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "vad_dysfluency.csv")

    sr = 16000

    header = [
        "FileName", "Len_S", "Var_S", "Avg_S", "Max_S",
        "Rat_S", "Num_S", "Num_PS", "Avg_PS", "Var_PS",
        "P_less_1", "P_more_1", "P_Sum"
    ]
    rows = []

    wav_files = glob.glob(os.path.join(source_dir, "*.wav"))
    wav_files.sort()

    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        file_id  = os.path.splitext(filename)[0]

        audio_torch = read_audio(wav_path, sampling_rate=sr)
        total_samples = audio_torch.shape[0]
        total_dur_sec = total_samples / sr

        audio_torch = audio_torch.to(device)

        with torch.no_grad():
            speech_segments = get_speech_ts(
                audio_torch, vad_model,
                sampling_rate=sr
            )
            
        silence_segments = get_silence_segments(
            total_samples=total_samples,
            speech_segments=speech_segments,
            sr=sr
        )
        
        feat_dict = compute_silence_features(silence_segments, total_dur_sec)
        
        if len(speech_segments) > 0:
            speech_only = []
            for seg in speech_segments:
                st_i = seg['start']
                ed_i = seg['end']
                speech_only.append(audio_torch[st_i:ed_i])
            speech_only = torch.cat(speech_only, dim=0)
        else:
            speech_only = torch.tensor([], dtype=audio_torch.dtype, device=device)
            
        speech_only_cpu = speech_only.cpu().numpy()
        out_wav_path = os.path.join(dest_dir, f"{file_id}.wav")
        sf.write(out_wav_path, speech_only_cpu, sr)

        print(f"[{filename}] -> {len(speech_segments)} segments, VAD file saved: {out_wav_path}")
        
        row_dict = {
            "FileName": file_id,
            "Len_S": feat_dict["Len_S"],
            "Var_S": feat_dict["Var_S"],
            "Avg_S": feat_dict["Avg_S"],
            "Max_S": feat_dict["Max_S"],
            "Rat_S": feat_dict["Rat_S"],
            "Num_S": feat_dict["Num_S"],
            "Num_PS": feat_dict["Num_PS"],
            "Avg_PS": feat_dict["Avg_PS"],
            "Var_PS": feat_dict["Var_PS"],
            "P_less_1": feat_dict["P_less_1"],
            "P_more_1": feat_dict["P_more_1"],
            "P_Sum": feat_dict["P_Sum"]
        }
        rows.append(row_dict)
        
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nDone! CSV saved at: {csv_path}")

if __name__ == "__main__":
    main()
