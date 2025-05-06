import os
import glob
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_silence_segments(total_samples, speech_segments, sr=16000):
    """
    전체 파일 길이(total_samples)와 VAD로 얻은 음성 구간(speech_segments)을 바탕으로,
    침묵(silence) 구간을 (start_sec, end_sec) 튜플 리스트로 반환.
    """
    silence_segments = []
    total_dur_sec = total_samples / sr

    if len(speech_segments) == 0:
        return [(0.0, total_dur_sec)]

    speech_segments = sorted(speech_segments, key=lambda x: x['start'])
    first_start = speech_segments[0]['start'] / sr
    if first_start > 0:
        silence_segments.append((0.0, first_start))
    for i in range(len(speech_segments) - 1):
        end_i = speech_segments[i]['end'] / sr
        start_next = speech_segments[i+1]['start'] / sr
        if start_next > end_i:
            silence_segments.append((end_i, start_next))
    last_end = speech_segments[-1]['end'] / sr
    if last_end < total_dur_sec:
        silence_segments.append((last_end, total_dur_sec))

    return silence_segments

def compute_correlation_matrix(mfcc_features: torch.Tensor) -> torch.Tensor:
    """
    mfcc_features: [n_mfcc, n_frames] (예: [13, T])에서 0차 제외 후 [12, T] 사용.
    각 프레임 벡터에 대해 평균 제거 및 L2 정규화를 수행한 뒤,
    각 프레임 벡터의 dot product를 계산하여 Pearson 상관계수 행렬([T, T])을 반환.
    """
    feats_t = mfcc_features.transpose(0, 1)  # [T, n_mfcc]
    feats_t = feats_t - feats_t.mean(dim=1, keepdim=True)
    norm = feats_t.norm(dim=1, keepdim=True)
    feats_t_norm = feats_t / (norm + 1e-8)
    ssm = torch.matmul(feats_t_norm, feats_t_norm.transpose(0, 1))
    return ssm

def main():
    source_dir = "/data/alc_jihan/h_wav_16K_merged"
    dest_dir = "/data/alc_jihan/silence_black_without_edge_imgae"
    os.makedirs(dest_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    frame_duration_ms = 30   # 30ms
    overlap_duration_ms = 20 # 20ms 오버랩 → 10ms hop
    frame_size = int(sr * (frame_duration_ms / 1000.0))   # 480 samples
    overlap_size = int(sr * (overlap_duration_ms / 1000.0)) # 320 samples
    hop_size = frame_size - overlap_size                  # 160 samples (10ms)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=13,
        melkwargs={
            'n_fft': frame_size,
            'win_length': frame_size,
            'hop_length': hop_size,
            'window_fn': torch.hamming_window
        }
    ).to(device)

    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_ts, get_speech_ts_adaptive, read_audio, save_audio, get_number_ts) = utils
    vad_model = vad_model.to(device)
    vad_model.eval()

    wav_files = glob.glob(os.path.join(source_dir, "*.wav"))
    wav_files.sort()

    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        file_id = os.path.splitext(filename)[0]

        waveform, sr_loaded = torchaudio.load(wav_path)
        if sr_loaded != sr:
            resampler = torchaudio.transforms.Resample(sr_loaded, sr).to(device)
            waveform = waveform.to(device)
            waveform = resampler(waveform)
        else:
            waveform = waveform.to(device)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        total_samples = waveform.shape[1]
        total_dur_sec = total_samples / sr

        with torch.no_grad():
            speech_segments = get_speech_ts(waveform, vad_model, sampling_rate=sr)

        silence_segments = get_silence_segments(total_samples, speech_segments, sr)

        with torch.no_grad():
            mfcc_all = mfcc_transform(waveform)  # [1, 13, n_frames]
        mfcc_use = mfcc_all[0, 1:, :]  # [12, n_frames]
        n_frames = mfcc_use.shape[1]

        if len(speech_segments) == 0:
            ssm = torch.zeros((1,1), dtype=torch.float32)
            ssm_norm = ssm.numpy()
            out_path = os.path.join(dest_dir, f"{file_id}.png")
            plt.imsave(out_path, ssm_norm, cmap='gray')
            print(f"[{filename}] -> All silence, saved blank image.")
            continue

        first_speech_sec = speech_segments[0]['start'] / sr
        last_speech_sec = speech_segments[-1]['end'] / sr

        keep_mask = torch.zeros(n_frames, dtype=torch.bool)
        for i in range(n_frames):
            frame_time = i * hop_size / sr
            if first_speech_sec <= frame_time < last_speech_sec:
                keep_mask[i] = True

        mid_mfcc = mfcc_use[:, keep_mask]
        mid_n_frames = mid_mfcc.shape[1]

        silent_frame_mask = torch.zeros(mid_n_frames, dtype=torch.bool)
        for i in range(mid_n_frames):
            frame_idx = torch.nonzero(keep_mask)[i].item()
            frame_time = frame_idx * hop_size / sr
            for (silence_start, silence_end) in silence_segments:
                if silence_start <= frame_time < silence_end:
                    silent_frame_mask[i] = True
                    break

        if silent_frame_mask.any():
            mid_mfcc[:, silent_frame_mask] = 0.0

        # silent frame에 해당하는 인덱스를 찾아, 나중에 상관계수 행렬에서 해당 행/열을 -1로 강제
        if silent_frame_mask.any() and mid_mfcc.shape[1] >= 2:
            silent_indices = torch.nonzero(silent_frame_mask).squeeze()
        
        if mid_mfcc.shape[1] < 2:
            ssm = torch.zeros((1,1), dtype=torch.float32)
        else:
            ssm = compute_correlation_matrix(mid_mfcc)  # [mid_frames, mid_frames]
            # silent frame에 해당하는 행과 열을 -1로 강제 설정하여, 정규화 시 0이 되도록 함
            if silent_frame_mask.any():
                ssm[silent_indices, :] = -1.0
                ssm[:, silent_indices] = -1.0

        ssm_cpu = ssm.cpu().numpy()

        # 전역 스케일: 상관계수가 [-1, 1]라고 가정 → [0, 1]로 매핑
        ssm_norm = (ssm_cpu + 1.0) / 2.0
        ssm_norm = np.clip(ssm_norm, 0.0, 1.0)

        out_filename = f"{file_id}.png"
        out_path = os.path.join(dest_dir, out_filename)
        plt.imsave(out_path, ssm_norm, cmap='gray')

        print(f"[{filename}] -> #SpeechSegments: {len(speech_segments)}, #MidFrames: {mid_mfcc.shape[1]}, saved: {out_filename}")

if __name__ == "__main__":
    main()
