import os
import torch
import torchaudio
import numpy as np
from PIL import Image

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 설정
input_dir = "/data/alc_jihan/h_wav_16K_sliced"
output_dir = "/data/alc_jihan/melspectrograms_for_swin_256"
os.makedirs(output_dir, exist_ok=True)

# GPU 기반 멜 필터 생성 함수
def create_mel_filter(sample_rate, n_fft, n_mels, fmin, fmax):
    mel_filter = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=fmin,
        f_max=fmax,
        n_stft=n_fft // 2 + 1  # STFT 결과 크기
    ).to(device)
    return mel_filter


# GPU 기반 로그 멜스펙트로그램 생성 함수
def compute_log_melspectrogram(wav_tensor, sample_rate, n_fft, hop_length, mel_filter):
    # STFT 계산 (GPU에서 수행)
    stft = torch.stft(
        wav_tensor, n_fft=n_fft, hop_length=hop_length,
        return_complex=True, normalized=False, center=True
    )
    # 복소수 스펙트럼의 크기 계산
    spectrogram = torch.abs(stft) ** 2  # Power spectrogram

    # 멜 스케일 변환 (GPU에서 수행)
    mel_spec = mel_filter(spectrogram)

    # 로그 스케일 변환 (GPU에서 수행)
    log_mel_spec = torch.log1p(mel_spec)

    # Normalize to [0, 255]
    log_mel_normalized = 255 * (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    log_mel_normalized = log_mel_normalized.to(dtype=torch.uint8)

    # RGB 3채널 생성 (동일한 값으로 채움)
    rgb_mel = torch.stack([log_mel_normalized] * 3, dim=0).permute(1, 2, 0)  # [H, W, 3]
    
    return rgb_mel

# 파일 처리 함수
def process_file(file_name, mel_filter, sample_rate, n_fft, hop_length):
    try:
        wav_path = os.path.join(input_dir, file_name)
        output_file = os.path.splitext(file_name)[0] + ".png"
        output_path = os.path.join(output_dir, output_file)

        # 음성 파일 로드 (GPU로 바로 이동)
        wav_tensor, sr = torchaudio.load(wav_path)
        wav_tensor = wav_tensor.to(device).squeeze(0)  # [C, T] -> [T]

        # 로그 멜스펙트로그램 생성
        rgb_mel = compute_log_melspectrogram(wav_tensor, sample_rate, n_fft, hop_length, mel_filter)

        # NumPy 배열로 변환
        rgb_mel = rgb_mel.cpu().numpy()

        # 배열이 (H, W, 3)인지 확인
        if rgb_mel.ndim != 3 or rgb_mel.shape[2] != 3:
            raise ValueError("RGB 배열이 올바르지 않습니다.")

        # 이미지 저장
        image = Image.fromarray(rgb_mel, mode="RGB")  # RGB 모드로 저장
        image.save(output_path)
        print(f"Processed: {file_name}, Image Size: {rgb_mel.shape[:2]}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# 메인 실행 함수
def main():
    # 설정값
    sample_rate = 16000
    n_fft = 2048
    hop_length = 512
    n_mels = 256
    fmin = 0
    fmax = 8000

    # GPU에서 멜 필터 생성
    mel_filter = create_mel_filter(sample_rate, n_fft, n_mels, fmin, fmax)

    # 처리할 파일 리스트
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    print(f"Processing {len(wav_files)} files...")

    # 파일 하나씩 처리
    for file_name in wav_files:
        process_file(file_name, mel_filter, sample_rate, n_fft, hop_length)

    print("모든 파일 처리가 완료되었습니다!")

if __name__ == "__main__":
    main()
