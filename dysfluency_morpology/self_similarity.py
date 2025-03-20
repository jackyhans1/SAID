import os
import glob
import torch
import torchaudio
import matplotlib.pyplot as plt

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def compute_correlation_matrix(mfcc_features: torch.Tensor) -> torch.Tensor:
    """
    mfcc_features: shape [n_mfcc, n_frames] (예: [13, T])
      - 여기서는 0차 MFCC 제외 후 [12, T]가 될 것
    Pearson 상관계수 행렬을 GPU 상에서 직접 계산한 뒤 반환 (shape: [T, T]).
    
    절차:
      1) (n_mfcc, n_frames) → (n_frames, n_mfcc)로 전치
      2) 각 프레임 벡터를 평균 0으로 정규화
      3) 각 프레임 벡터를 L2 노멀라이즈
      4) (정규화된) 프레임 벡터 간 dot product = 상관계수
    """
    # (n_mfcc, n_frames) → (n_frames, n_mfcc)
    feats_t = mfcc_features.transpose(0, 1)  # [T, n_mfcc]

    # 평균 0으로 만들기
    feats_t = feats_t - feats_t.mean(dim=1, keepdim=True)  # 각 행(프레임)별 평균 제거

    # L2 정규화 (표준편차와 유사)
    norm = feats_t.norm(dim=1, keepdim=True)  # [T, 1]
    feats_t_norm = feats_t / (norm + 1e-8)    # NaN 방지용 작은 값 추가

    # dot product → [T, T] 상관계수 행렬
    ssm = torch.matmul(feats_t_norm, feats_t_norm.transpose(0, 1))
    # 값 범위: -1 ~ 1 (실제로는 완벽히 Pearson 상관과 동일)

    return ssm

def main():
    # 폴더 설정
    source_dir = "/data/alc_jihan/h_wav_16K_VAD"
    dest_dir   = "/data/alc_jihan/VAD_dysfluency_images"
    os.makedirs(dest_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr = 16000
    frame_duration_ms = 30   # 30ms
    overlap_duration_ms = 20 # 20ms 오버랩 → 10ms hop
    frame_size = int(sr * (frame_duration_ms / 1000.0))  # 480
    overlap_size = int(sr * (overlap_duration_ms / 1000.0))  # 320
    hop_size = frame_size - overlap_size  # 160 (10ms)

    # torchaudio.transforms.MFCC 설정
    # n_mfcc=13 (0차 포함), 이후 1~12만 사용
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=13,
        melkwargs={
            'n_fft': frame_size,
            'win_length': frame_size,
            'hop_length': hop_size,
            'window_fn': torch.hamming_window  # 해밍 윈도우
        }
    ).to(device)

    # 모든 WAV 파일에 대해 처리
    wav_files = glob.glob(os.path.join(source_dir, "*.wav"))
    wav_files.sort()

    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        file_id  = os.path.splitext(filename)[0]
        
        waveform, sr_loaded = torchaudio.load(wav_path)
        # 만약 sr_loaded != sr 이면 리샘플
        if sr_loaded != sr:
            resampler = torchaudio.transforms.Resample(sr_loaded, sr).to(device)
            waveform = waveform.to(device)
            waveform = resampler(waveform)
        else:
            waveform = waveform.to(device)

        # 이미 1채널이면 waveform.shape=(1, N), 2채널 이상이면 mean 등 처리
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # 스테레오 → 모노

        # MFCC 추출
        # mfcc_transform(waveform) shape: [batch=1, n_mfcc=13, time=frames]
        with torch.no_grad():
            mfcc_all = mfcc_transform(waveform)
        
        # 0차 MFCC 제외 → (12, n_frames)
        # 채널/배치 차원 1 → 제거
        mfcc_use = mfcc_all[0, 1:, :]  # shape [12, n_frames]

        # Pearson 상관계수 기반 Self-Similarity 행렬 계산
        ssm = compute_correlation_matrix(mfcc_use)  # shape [n_frames, n_frames]

        # numpy 변환 → min-max 정규화 → 이미지 저장
        ssm_cpu = ssm.cpu().numpy()
        # 전부 동일값이면 분모=0이 되어 NaN이 될 수 있으므로 예외 처리
        min_val, max_val = ssm_cpu.min(), ssm_cpu.max()
        if max_val > min_val:
            ssm_norm = (ssm_cpu - min_val) / (max_val - min_val)
        else:
            ssm_norm = np.zeros_like(ssm_cpu)

        out_filename = f"{file_id}.png"
        out_path = os.path.join(dest_dir, out_filename)
        plt.imsave(out_path, ssm_norm, cmap='gray')

        print(f"[{filename}] -> shape: {mfcc_use.shape}, saved: {out_filename}")

if __name__ == "__main__":
    main()
